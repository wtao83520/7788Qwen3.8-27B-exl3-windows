#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""pm2 托管的推理服务：状态检查 + 启停验证。

专门用来验证 Windows + venv 最容易出问题的两点：
  1. pm2 只跟踪 venv 转发器的 PID；停止时必须确认**真正占显存的那个子进程**
     也退出了，否则 21.5 GiB 显存会一直卡着，下次启动直接 OOM。
  2. 启停必须走 pm2（pm2 stop / pm2 start），不能直接杀进程 ——
     进程一退出 pm2 就会自动拉起来。

    .\.venv\Scripts\python.exe _bench\pm2_status.py
    .\.venv\Scripts\python.exe _bench\pm2_status.py --stop-test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

import requests
import yaml

APP = "qwen38-server"


def _pm2_exe() -> str:
    """pm2 在 Windows 上是 pm2.cmd（PowerShell shim），subprocess 直接调
    "pm2" 会 FileNotFoundError，必须先解析成真实路径。"""
    import shutil

    for name in ("pm2.cmd", "pm2.exe", "pm2"):
        path = shutil.which(name)
        if path:
            return path
    return "pm2"


def run(cmd: list[str], timeout: float = 60) -> tuple[int, str]:
    if cmd and cmd[0] == "pm2":
        cmd = [_pm2_exe(), *cmd[1:]]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, shell=False)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def pm2_jlist() -> dict | None:
    rc, out = run(["pm2", "jlist"])
    if rc != 0:
        return None
    try:
        for app in json.loads(out):
            if app.get("name") == APP:
                return app
    except json.JSONDecodeError:
        return None
    return None


def service_pids() -> list[dict]:
    """所有含 server.py 的 python 进程（转发器 + 真解释器）。"""
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -and $_.CommandLine -like '*server.py*' } | "
        "Select-Object ProcessId,ParentProcessId,CommandLine | "
        "ConvertTo-Json -Compress -Depth 3"
    )
    p = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=60)
    text = (p.stdout or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    return [{"pid": d["ProcessId"], "ppid": d["ParentProcessId"]} for d in data]


def gpu_used() -> float | None:
    p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=20)
    try:
        return round(int(p.stdout.strip().splitlines()[0]) / 1024, 2)
    except Exception:
        return None


def report(tag: str) -> None:
    app = pm2_jlist()
    pids = service_pids()
    used = gpu_used()
    print(f"--- {tag} ---")
    if app:
        e = app.get("pm2_env", {})
        print(f"  pm2 状态   : {e.get('status')}  PID {app.get('pid')}  "
              f"重启次数 {e.get('restart_time')}  unstable {e.get('unstable_restarts')}")
    else:
        print("  pm2 状态   : 未注册")
    print(f"  服务进程   : {len(pids)} 个  {[p['pid'] for p in pids]}")
    print(f"  GPU 已用   : {used} GB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-test", action="store_true",
                    help="做一次 停止→确认显存释放→再启动 的完整验证")
    args = ap.parse_args()

    cfg = yaml.safe_load((BASE / "config.yaml").read_text(encoding="utf-8"))
    port = int(cfg["server"]["port"])

    print("=" * 74)
    print(f"pm2 托管状态检查（应用名 {APP}）")
    print("=" * 74)
    report("当前")

    try:
        h = requests.get(f"http://127.0.0.1:{port}/health", timeout=30).json()
        print(f"  /health    : ok  draft_kind={h.get('draft_kind')}  "
              f"KV={h.get('cache_k_bits')}/{h.get('cache_v_bits')}  "
              f"空闲 {(h.get('gpu') or {}).get('vram_free_gb')} GiB")
    except Exception as exc:
        print(f"  /health    : 不可用（{type(exc).__name__}）")

    if not args.stop_test:
        return 0

    # ---------------- 停止测试：这是 Windows + pm2 最大的坑 ----------------
    print()
    print("=" * 74)
    print("停止测试：pm2 stop 之后，真解释器（占显存的那个）必须也退出")
    print("=" * 74)
    before = gpu_used()
    print(f"  停止前 GPU 已用 {before} GB")

    rc, out = run(["pm2", "stop", APP], timeout=180)
    print(f"  pm2 stop → rc={rc}")
    if rc != 0:
        print(f"    {out.strip()[:300]}")

    # 给孤儿看门狗时间（它每 2s 轮询一次父进程）
    deadline = time.time() + 40
    freed = False
    while time.time() < deadline:
        time.sleep(2)
        pids = service_pids()
        used = gpu_used()
        if not pids and used is not None and used < before - 15:
            freed = True
            print(f"  ✓ {time.time() - (deadline - 40):.0f}s 后：进程全清，"
                  f"GPU 已用 {used} GB（释放了 {before - used:.1f} GB）")
            break
    if not freed:
        left = service_pids()
        print(f"  ✗ 40s 内未完全释放：残留进程 {[p['pid'] for p in left]}，"
              f"GPU {gpu_used()} GB")
        print("     → 说明孤儿看门狗没生效，停止时会有显存泄漏")
        for p in left:
            run(["taskkill", "/PID", str(p["pid"]), "/T", "/F"])
        print("     已手工清理残留进程")

    report("停止后")

    # ---------------- 再启动 ----------------
    print()
    print("=" * 74)
    print("重新启动")
    print("=" * 74)
    t0 = time.time()
    rc, out = run(["pm2", "start", APP], timeout=180)
    print(f"  pm2 start → rc={rc}")
    while time.time() - t0 < 300:
        time.sleep(3)
        try:
            h = requests.get(f"http://127.0.0.1:{port}/health", timeout=15).json()
            if h.get("status") == "ok":
                print(f"  ✓ {time.time() - t0:.0f}s 后 /health ok  "
                      f"draft_kind={h.get('draft_kind')}  "
                      f"空闲 {(h.get('gpu') or {}).get('vram_free_gb')} GiB")
                break
        except Exception:
            pass
    else:
        print(f"  ✗ 300s 内没起来，看 logs/pm2-err.log")
        return 1
    report("重启后")
    return 0


if __name__ == "__main__":
    sys.exit(main())
