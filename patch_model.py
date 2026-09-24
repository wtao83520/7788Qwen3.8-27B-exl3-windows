#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
定位并修复模型文件里的「空洞」。

背景：fast_download.py 会先把文件 truncate 到完整大小，再按 4MB 分段填充。
如果下载中途被杀掉，正在写的那几个分段会留下「前半段是正确数据、后半段是
零」的空洞。这种损坏：
  - 文件大小完全正确（check_model.py 查不出来）
  - 整个 4MB 分段并不全零（粗粒度零扫描也查不出来）
  - 但 sha256 一定不匹配

这里用细粒度扫描找出零区段，只把这些区间重新下载并原地写回，
避免重下整个 8.5 GB。

    .\.venv\Scripts\python.exe patch_model.py --scan            # 只扫描
    .\.venv\Scripts\python.exe patch_model.py                   # 扫描 + 修复 + 复验
"""

from __future__ import annotations

import argparse
import hashlib
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


def find_zero_runs(path: str, block: int = 64 << 10, min_run: int = 64 << 10) -> list[list[int]]:
    """返回 [start, end) 的零区段列表，把相邻的零块合并起来。"""
    runs: list[list[int]] = []
    with open(path, "rb") as fh:
        off = 0
        while True:
            buf = fh.read(block)
            if not buf:
                break
            if not any(buf):
                if runs and runs[-1][1] == off:
                    runs[-1][1] = off + len(buf)
                else:
                    runs.append([off, off + len(buf)])
            off += len(buf)
    return [r for r in runs if r[1] - r[0] >= min_run]


def sha256_file(path: str, block: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_sha(repo: str, revision: str, name: str) -> str | None:
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=1"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    for it in resp.json():
        if it.get("path") == name:
            return (it.get("lfs") or {}).get("oid")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="定位并修复模型文件中的零空洞")
    ap.add_argument("--path", "-p", default=os.path.join("models", "Qwen3.8-27B-3.50bpw"))
    ap.add_argument("--file", "-f", default="model-00001-of-00002.safetensors")
    ap.add_argument("--revision", "-r", default="3.50bpw")
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--scan", action="store_true", help="只扫描，不下载修复")
    args = ap.parse_args()

    root = os.path.abspath(args.path)
    path = os.path.join(root, args.file)
    if not os.path.isfile(path):
        print(f"文件不存在：{path}")
        return 1
    size = os.path.getsize(path)

    print("=" * 78)
    print(f"目标文件：{args.file}  ({human(size)})")
    print("=" * 78)

    print("\n扫描零空洞（64 KB 粒度）…")
    t0 = time.time()
    runs = find_zero_runs(path)
    total = sum(e - s for s, e in runs)
    print(f"扫描完成，耗时 {time.time() - t0:.0f} 秒")
    if not runs:
        print("没找到零空洞。若 sha256 仍不匹配，说明损坏不是零空洞——请整文件重下。")
        return 1

    print(f"\n找到 {len(runs)} 处空洞，合计 {human(total)}：")
    for s, e in runs:
        print(f"  偏移 {human(s):>12} – {human(e):>12}   （{human(e - s)}）")

    if args.scan:
        print("\n（--scan 模式，未修改文件）")
        return 0

    # ---------------- 只重下这些区间 ----------------
    base = "https://huggingface.co/"
    url = f"{base}{args.repo}/resolve/{args.revision}/{args.file}"
    print(f"\n重新下载这 {len(runs)} 个区间并原地写回…")
    sess = requests.Session()
    ok = 0
    with open(path, "r+b") as fh:
        for i, (s, e) in enumerate(runs, 1):
            headers = {"Range": f"bytes={s}-{e - 1}"}
            done = False
            for attempt in range(6):
                try:
                    r = sess.get(url, headers=headers, timeout=60)
                    if r.status_code not in (200, 206):
                        raise RuntimeError(f"HTTP {r.status_code}")
                    data = r.content
                    if len(data) != e - s:
                        raise RuntimeError(f"长度不符 {len(data)}/{e - s}")
                    fh.seek(s)
                    fh.write(data)
                    fh.flush()
                    done = True
                    break
                except Exception as exc:
                    if attempt == 5:
                        print(f"  [{i}/{len(runs)}] 失败：{type(exc).__name__}: {exc}")
                    else:
                        time.sleep(min(10, attempt * 2))
            if done:
                ok += 1
                print(f"  [{i}/{len(runs)}] 已修复 {human(s)} – {human(e)}")
            else:
                print(f"  [{i}/{len(runs)}] 未能修复")
    print(f"\n修复 {ok}/{len(runs)} 个区间")

    # ---------------- 复验 ----------------
    print("\n复验 sha256…")
    exp = expected_sha(args.repo, args.revision, args.file)
    got = sha256_file(path)
    print(f"  期望 {exp}")
    print(f"  实际 {got}")
    if exp and got == exp:
        print("\n结论：sha256 一致，文件已修复完整。")
        return 0
    print("\n结论：仍不匹配，请删掉该文件后整文件重下：")
    print(f'  Remove-Item "{path}" -Force')
    print(r"  .\.venv\Scripts\python.exe fast_download.py --source hf-direct --conns 96")
    return 1


if __name__ == "__main__":
    sys.exit(main())
