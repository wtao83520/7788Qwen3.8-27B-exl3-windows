#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
并行下载测速：对比单连接 vs 多连接的聚合吞吐。

    .\.venv\Scripts\python.exe speed_test.py
    .\.venv\Scripts\python.exe speed_test.py --url <url> --conns 16 --seconds 15
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

import requests

PROBE = "https://hf-mirror.com/Qwen/Qwen3.8-27B/resolve/main/model-00001-of-00018.safetensors"


def worker(url: str, start: int, end: int, seconds: float, out: list, idx: int) -> None:
    got = 0
    try:
        proxies = {"http": None, "https": None}  # 测速时绕过本地代理，避免干扰判断
        with requests.get(
            url,
            headers={"Range": f"bytes={start}-{end}"},
            stream=True,
            timeout=seconds + 10,
            proxies=proxies,
        ) as r:
            if r.status_code >= 400:
                out.append((idx, got, f"HTTP {r.status_code}"))
                return
            t0 = time.time()
            for chunk in r.iter_content(262144):
                got += len(chunk)
                if time.time() - t0 > seconds:
                    break
        out.append((idx, got, "ok"))
    except Exception as exc:
        out.append((idx, got, f"{type(exc).__name__}"))


def run(url: str, conns: int, seconds: float) -> None:
    print(f"目标   : {url}")
    print(f"连接数 : {conns}    采样时长: {seconds}s")
    out: list = []
    threads = []
    # 每连接拿不同的 100 MB 区间，避免互相抢同一段
    span = 100 * 1024 * 1024
    for i in range(conns):
        start = i * span
        end = start + span - 1
        t = threading.Thread(target=worker, args=(url, start, end, seconds, out, i), daemon=True)
        threads.append(t)
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=seconds + 15)
    wall = time.time() - t0

    total = sum(g for _, g, _ in out)
    print(f"\n各连接：")
    for idx, got, status in sorted(out):
        print(f"  #{idx:<3} {got / 1024 / wall:>9.0f} kB/s   {status}")
    print(f"\n合计 {total / 1e6:.1f} MB / {wall:.1f}s = {total / 1024 / wall:.0f} kB/s"
          f"  ({total / 1024 / 1024 / wall:.1f} MB/s)")
    if conns > 1:
        avg = total / 1024 / wall / max(1, len(out))
        print(f"单连接平均 {avg:.0f} kB/s  →  {conns} 连接放大约 {total / 1024 / wall / avg:.1f} 倍")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=PROBE)
    ap.add_argument("--conns", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()
    run(args.url, args.conns, args.seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
