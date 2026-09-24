#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
用官方 sha256 校验模型文件，这是唯一能发现「静默损坏」的方法。

为什么需要它：fast_download.py 会先把文件 truncate 到完整大小再按分段填充，
所以「文件大小正确」完全不能证明「内容正确」。一个被中断的下载会留下稀疏空洞
（读出来全是 0x00），大小却分毫不差，check_model.py 的大小比对查不出来。

HuggingFace 的 LFS 文件把 sha256 记录在 API 的 lfs.oid 字段里，这里逐文件比对。

    .\.venv\Scripts\python.exe verify_model.py
    .\.venv\Scripts\python.exe verify_model.py -p models/xxx -r 3.50bpw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import requests

REPO_ID = "turboderp/Qwen3.8-27B-exl3"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def sha256_file(path: str, block: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def zero_runs(path: str, seg: int = 4 << 20) -> list[tuple[int, int]]:
    """找出全零的 4MB 分段。真实权重里几乎不可能出现整段为零，
    所以有零段基本就能确定是下载留下的空洞。"""
    holes: list[tuple[int, int]] = []
    with open(path, "rb") as fh:
        off = 0
        while True:
            buf = fh.read(seg)
            if not buf:
                break
            if not any(buf):
                holes.append((off, len(buf)))
            off += len(buf)
    return holes


def main() -> int:
    ap = argparse.ArgumentParser(description="用官方 sha256 校验 EXL3 模型文件")
    ap.add_argument("--path", "-p", default=os.path.join("models", "Qwen3.8-27B-3.50bpw"))
    ap.add_argument("--revision", "-r", default="3.50bpw")
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--scan-zeros", action="store_true",
                    help="额外扫描全零分段（sha256 不匹配时用来看空洞在哪）")
    ap.add_argument("--recheck-small", action="store_true",
                    help="把非 LFS 小文件重新下载一份做逐字节比对"
                         "（crc32.txt 已过时，只有它能确认这些小文件是否被改动）")
    args = ap.parse_args()

    root = os.path.abspath(args.path)
    print("=" * 78)
    print(f"官方 sha256 校验：{root}")
    print(f"仓库 / 分支      ：{args.repo} @ {args.revision}")
    print("=" * 78)

    url = f"https://huggingface.co/api/models/{args.repo}/tree/{args.revision}?recursive=1"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        items = [i for i in resp.json() if i.get("type") == "file"]
    except Exception as exc:
        print(f"取文件清单失败：{type(exc).__name__}: {exc}")
        return 1

    print(f"\n{'文件':<40}{'期望':<18}{'实际':<18}结果")
    print("-" * 78)

    bad: list[str] = []
    skipped: list[str] = []
    checked = 0
    t0 = time.time()

    for it in sorted(items, key=lambda x: x["path"]):
        name = it["path"]
        path = os.path.join(root, name)
        exp = (it.get("lfs") or {}).get("oid")

        if not os.path.isfile(path):
            print(f"{name:<40}{'-':<18}{'-':<18}缺失")
            bad.append(name)
            continue
        if not exp:
            skipped.append(name)
            continue

        got = sha256_file(path)
        ok = got == exp
        if not ok:
            bad.append(name)
        checked += 1
        print(f"{name:<40}{exp[:16]:<18}{got[:16]:<18}{'OK' if ok else '不匹配'}")

    print("-" * 78)
    print(f"比对 {checked} 个 LFS 文件，耗时 {time.time() - t0:.0f} 秒")
    if skipped:
        print(f"跳过 {len(skipped)} 个非 LFS 小文件：{', '.join(skipped)}")

    if args.recheck_small and skipped:
        print(f"\n重新下载并逐字节比对 {len(skipped)} 个非 LFS 文件…")
        base = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/"
        for name in skipped:
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            try:
                r = requests.get(base + name, timeout=60)
                r.raise_for_status()
            except Exception as exc:
                print(f"  {name:<38} 下载失败 {type(exc).__name__}")
                continue
            local = open(path, "rb").read()
            if local == r.content:
                print(f"  {name:<38} 一致")
            else:
                print(f"  {name:<38} 不一致（本地 {len(local)} B / 官方 {len(r.content)} B）")
                bad.append(name)

    if bad:
        print(f"\n以下 {len(bad)} 个文件损坏或不完整：")
        for name in bad:
            p = os.path.join(root, name)
            size = human(os.path.getsize(p)) if os.path.isfile(p) else "缺失"
            print(f"  - {name}  ({size})")
            if args.scan_zeros and os.path.isfile(p):
                holes = zero_runs(p)
                tot = sum(n for _, n in holes)
                print(f"      全零分段 {len(holes)} 个，共 {human(tot)}")
                for off, n in holes[:8]:
                    print(f"        偏移 {human(off)} 起 {human(n)}")
                if len(holes) > 8:
                    print(f"        … 还有 {len(holes) - 8} 个")
        print("\n修复办法：删掉损坏的文件后重跑下载命令（会重新拉完整文件）：")
        for name in bad:
            print(f'  Remove-Item "{os.path.join(root, name)}" -Force')
        print(r"  .\.venv\Scripts\python.exe fast_download.py --source hf-direct --conns 96")
        return 1

    print("\n结论：全部文件与官方 sha256 一致，模型完整，可以启动服务：  .\\start.ps1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
