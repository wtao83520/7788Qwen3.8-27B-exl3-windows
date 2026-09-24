#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""崩溃恢复测试：证明 pm2 的 autorestart 真的有效（「常驻」的核心价值）。

做法：直接杀掉服务进程（不是 pm2 stop，所以 pm2 应该把它当崩溃并自动拉起），
然后等服务恢复，并核对 /health 与显存。

    .\.venv\Scripts\python.exe _bench\pm2_crash_test.py
"""

from __future__ import annotations

import json
import shutil
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


def pm2_exe() -> str:
    for name in ("pm2.cmd", "pm2.exe", "pm2"):
        found = shutil.which(name)
        if found:
            return found
    return "pm2"


def app_info() -> dict:
    out = subprocess.run([pm2_exe(), "jlist"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    try:
        for a in json.loads(out):
            if a.get("name") == APP:
                return a
    except json.JSONDecodeError:
        pass
    return {}


def brief(tag: str) -> dict:
    a = app_info()
    e = a.get("pm2_env") or {}
    info = {
        "status": e.get("status"),
        "pid": a.get("pid"),
        "restarts": e.get("restart_time"),
        "unstable": e.get("unstable_restarts"),
    }
    print(f"  {tag:<8} status={info['status']:<8} pid={info['pid']:<8} "
          f"restarts={info['restarts']} unstable={info['unstable']}")
    return info


def service_pids() -> list[int]:
    script = (
        "$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -and $_.CommandLine -like '*server.py*' } | "
        "Select-Object -ExpandProperty ProcessId | ConvertTo-Json -Compress"
    )
    o = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                       capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.strip()
    if not o:
        return []
    try:
        data = json.loads(o)
    except json.JSONDecodeError:
        return []
    if isinstance(data, int):
        return [data]
    return [int(x) for x in data]


def gpu_used() -> float | None:
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip()
    try:
        return round(int(o.splitlines()[0]) / 1024, 2)
    except Exception:
        return None


def main() -> int:
    cfg = yaml.safe_load((BASE / "config.yaml").read_text(encoding="utf-8"))
    port = int(cfg["server"]["port"])
    url = f"http://127.0.0.1:{port}/health"

    print("=" * 74)
    print("pm2 崩溃恢复测试")
    print("=" * 74)
    print(f"  崩溃前 GPU = {gpu_used()} GB")
    before = brief("崩溃前")
    if before["status"] != "online":
        print("  服务不是 online，先退出")
        return 1

    pids = service_pids()
    print(f"\n  注入崩溃：taskkill 掉全部服务进程 {pids}")
    print("  （用 taskkill 而不是 pm2 stop —— pm2 stop 是「有意停止」，不会重启）")
    for pid in pids:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                       capture_output=True, text=True, timeout=30)

    print("\n  等待 pm2 自动拉起（退避起点 15s，加载约 7-15s）…")
    t0 = time.time()
    recovered = False
    while time.time() - t0 < 300:
        time.sleep(3)
        try:
            h = requests.get(url, timeout=8).json()
            if h.get("status") == "ok":
                print(f"  ✓ {time.time() - t0:.0f}s 后服务恢复："
                      f"draft={h.get('draft_kind')}  "
                      f"空闲={(h.get('gpu') or {}).get('vram_free_gb')} GiB")
                recovered = True
                break
        except Exception:
            pass

    print()
    print(f"  崩溃后 GPU = {gpu_used()} GB")
    after = brief("崩溃后")

    print()
    print("=" * 74)
    if not recovered:
        print("结论：✗ 没有自动恢复 —— autorestart 可能没生效")
        return 1
    if (after.get("restarts") or 0) > (before.get("restarts") or 0):
        print("结论：✓ 崩溃后 pm2 自动重启成功（autorestart 生效）")
        print(f"      restarts {before['restarts']} → {after['restarts']}，"
              f"unstable {before['unstable']} → {after['unstable']}")
        return 0
    print("结论：⚠ 服务恢复了，但 restart 计数没增加 —— 可能是我们自己重启的，"
          "不是 pm2 触发的，建议复核")
    return 1


if __name__ == "__main__":
    sys.exit(main())
