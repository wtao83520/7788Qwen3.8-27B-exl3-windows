#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""重启服务以套用当前 config.yaml，并打印生效配置。

    .\.venv\Scripts\python.exe _bench\apply_config.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

import requests
import yaml

PANEL = "http://127.0.0.1:8001"


def main() -> int:
    cfg = yaml.safe_load((BASE / "config.yaml").read_text(encoding="utf-8"))
    port = int(cfg["server"]["port"])
    service = f"http://127.0.0.1:{port}"

    print(f"目标配置：cache_quant={cfg['model']['cache_quant']}  "
          f"draft_model={cfg['model']['draft_model']}  "
          f"max_seq_len={cfg['model']['max_seq_len']}")

    print("停止服务…")
    try:
        requests.post(f"{PANEL}/api/stop?timeout=40", timeout=180)
    except Exception as exc:
        print(f"  stop 返回异常（继续）：{type(exc).__name__}")

    for _ in range(60):
        time.sleep(2)
        try:
            st = requests.get(f"{PANEL}/api/status", timeout=15).json()["service"]
            if st["status"] == "stopped":
                print("  已停止")
                break
        except Exception:
            pass

    print("按 config.yaml 启动…")
    r = requests.post(f"{PANEL}/api/start", timeout=180)
    print(f"  start HTTP {r.status_code}")

    t0 = time.time()
    st = {}
    while time.time() - t0 < 300:
        time.sleep(3)
        try:
            st = requests.get(f"{PANEL}/api/status", timeout=15).json()["service"]
        except Exception:
            continue
        if st.get("status") == "running":
            break
        if st.get("status") == "error":
            print("  ✗ 启动失败，见 logs/service.log")
            return 1
    print(f"  {st.get('status')}（{time.time() - t0:.0f}s）PID {st.get('pids')}")

    h = requests.get(f"{service}/health", timeout=60).json()
    g = h.get("gpu") or {}
    print()
    print("=== /health ===")
    for k in ("status", "model", "max_seq_len", "cache_quant", "cache_k_bits",
              "cache_v_bits", "draft_kind", "draft_tokens_per_window",
              "max_tokens", "uptime_s"):
        print(f"  {k:26}= {h.get(k)}")
    print(f"  {'vram_free_gb':26}= {g.get('vram_free_gb')} / {g.get('vram_total_gb')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
