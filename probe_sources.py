#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对同一个模型文件，并发测试多个下载通道的速度，找出最快的。

    .\.venv\Scripts\python.exe probe_sources.py
    .\.venv\Scripts\python.exe probe_sources.py --seconds 12 --conns 4
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

# 目标：3.50bpw 分支的权重分片
REL = "turboderp/Qwen3.8-27B-exl3/resolve/3.50bpw/model-00001-of-00002.safetensors"

SOURCES = {
    "hf-mirror.com": f"https://hf-mirror.com/{REL}",
    "huggingface.co(代理)": f"https://huggingface.co/{REL}",
    "modelscope(HF式路径)": f"https://www.modelscope.cn/{REL}",
    "aifasthub.com": f"https://aifasthub.com/{REL}",
}


def fetch(url: str, start: int, end: int, seconds: float, out: list, use_proxy: bool) -> None:
    got = 0
    t0 = time.time()
    try:
        proxies = None if use_proxy else {"http": None, "https": None}
        with requests.get(
            url,
            headers={"Range": f"bytes={start}-{end}"},
            stream=True,
            timeout=seconds + 12,
            allow_redirects=True,
            proxies=proxies,
        ) as r:
            if r.status_code >= 400:
                out.append((got, f"HTTP {r.status_code}"))
                return
            for chunk in r.iter_content(262144):
                got += len(chunk)
                if time.time() - t0 > seconds:
                    break
            out.append((got, "ok"))
    except Exception as exc:
        out.append((got, type(exc).__name__))


def test(name: str, url: str, conns: int, seconds: float, use_proxy: bool) -> float:
    out: list = []
    span = 60 * 1024 * 1024
    threads = []
    for i in range(conns):
        s = i * span
        threads.append(threading.Thread(target=fetch, args=(url, s, s + span - 1, seconds, out, use_proxy), daemon=True))
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=seconds + 20)
    wall = max(0.1, time.time() - t0)
    total = sum(g for g, _ in out)
    status = out[0][1] if out else "no-result"
    kbs = total / 1024 / wall
    ok = all(s == "ok" for _, s in out)
    flag = "" if ok else f"  [{status}]"
    print(f"  {name:<24} {kbs:>9.0f} kB/s   ({conns}连接, 取到 {total / 1e6:.1f} MB){flag}")
    return kbs if ok else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--conns", type=int, default=4)
    args = ap.parse_args()

    print(f"目标文件：{REL}")
    print(f"每通道 {args.conns} 连接 x {args.seconds:.0f}s\n")
    results: dict[str, float] = {}
    for name, url in SOURCES.items():
        use_proxy = "huggingface.co" in url  # 官方站需要走系统代理
        results[name] = test(name, url, args.conns, args.seconds, use_proxy)

    best = max(results, key=lambda k: results[k])
    print()
    if results[best] > 0:
        kbs = results[best]
        print(f"最快通道：{best}  {kbs:.0f} kB/s")
        print(f"按此速度下载 15.4 GB 需要约 {15.4e9 / 1024 / kbs / 3600:.1f} 小时")
    else:
        print("所有通道都失败了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
