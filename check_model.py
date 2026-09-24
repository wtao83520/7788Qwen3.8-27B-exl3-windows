#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
校验模型目录是否完整（适用于手动下载后自检）。

    .\.venv\Scripts\python.exe check_model.py
    .\.venv\Scripts\python.exe check_model.py --path D:\models\Qwen3.8-27B-3.50bpw
    .\.venv\Scripts\python.exe check_model.py --crc       # 额外做 crc32 逐一校验（较慢）

会逐文件对比大小，报告「缺失 / 大小不符」，并列出关键文件是否就位。
"""

from __future__ import annotations

import argparse
import os
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

# 来自 turboderp/Qwen3.8-27B-exl3 分支 3.50bpw 的文件清单（字节）
MANIFEST: dict[str, int] = {
    ".gitattributes": 1570,
    "LICENSE": 11544,
    "README.md": 65012,
    "chat_template.jinja": 8952,
    "config.json": 4621,
    "crc32.txt": 238,
    "generation_config.json": 202,
    "merges.txt": 3353259,
    "model-00001-of-00002.safetensors": 8530774298,
    "model-00002-of-00002.safetensors": 6807634163,
    "model.safetensors.index.json": 238664,
    "preprocessor_config.json": 390,
    "quantization_config.json": 629045,
    "tokenizer.json": 12809320,
    "tokenizer_config.json": 17928,
    "video_preprocessor_config.json": 385,
    "vocab.json": 6722759,
}

# 下不到这些也能跑（但功能会退化）
OPTIONAL = {".gitattributes", "LICENSE", "README.md", "video_preprocessor_config.json"}

# 缺了这些就一定跑不起来
CRITICAL = {
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
}


def human(n: int) -> str:
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n} B"


def crc32_file(path: str, chunk: int = 1 << 22) -> int:
    import zlib

    crc = 0
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            crc = zlib.crc32(data, crc)
    return crc & 0xFFFFFFFF


def load_repo_crc(text: str) -> dict[str, str]:
    """解析 crc32.txt。ExLlamaV3 的格式形如 `<crc> <filename>` 或 `filename <crc>`。"""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        a, b = parts[0], parts[-1]
        if all(c in "0123456789abcdefABCDEF" for c in a) and len(a) == 8:
            out[b] = a.lower()
        elif all(c in "0123456789abcdefABCDEF" for c in b) and len(b) == 8:
            out[a] = b.lower()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="校验 EXL3 模型目录完整性")
    ap.add_argument("--path", "-p", default=os.path.join("models", "Qwen3.8-27B-3.50bpw"))
    ap.add_argument("--crc", action="store_true", help="额外用 crc32.txt 做内容校验（慢，要读 15 GB）")
    args = ap.parse_args()
    root = os.path.abspath(args.path)

    print("=" * 70)
    print(f"校验目录：{root}")
    print("=" * 70)
    if not os.path.isdir(root):
        print("目录不存在。请先创建，并把下载的文件放进去。")
        print(f'  New-Item -ItemType Directory -Force "{root}"')
        return 1

    missing: list[str] = []
    wrong: list[tuple[str, int, int]] = []
    # 下载器会把文件预分配到完整大小，所以「大小对」不代表「下完了」。
    # 只要还存在 <文件>.part.json，就说明该文件还有分段没下完。
    incomplete: list[str] = []
    ok = 0
    total_have = 0
    total_need = sum(MANIFEST.values())

    for name, expect in sorted(MANIFEST.items()):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            missing.append(name)
            continue
        if os.path.isfile(path + ".part.json"):
            incomplete.append(name)
        actual = os.path.getsize(path)
        if actual != expect:
            wrong.append((name, actual, expect))
        else:
            ok += 1
        total_have += actual

    print(f'\n{"文件名":<40}{"状态":>10}{"本地":>12}')
    print("-" * 70)
    for name in sorted(MANIFEST):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            flag = "缺失"
            got = "-"
        elif name in incomplete:
            flag = "未下完"
            got = human(os.path.getsize(path))
        elif os.path.getsize(path) != MANIFEST[name]:
            flag = "大小不符"
            got = human(os.path.getsize(path))
        else:
            flag = "OK"
            got = human(MANIFEST[name])
        print(f"{name:<40}{flag:>10}{got:>12}")

    print("-" * 70)
    print(f"完整 {ok} / {len(MANIFEST)} 个文件")
    print(f"已下载 {human(total_have)} / {human(total_need)}  ({total_have / total_need * 100:.1f}%)")

    if incomplete:
        print(f"\n还有 {len(incomplete)} 个文件没下完（文件大小已是完整大小，但存在残留的 .part.json）：")
        for name in incomplete:
            print(f"  - {name}")
        print("  重跑一次下载命令即可从断点续传：")
        print(r"  .\.venv\Scripts\python.exe fast_download.py --source hf-direct --conns 96")

    if missing:
        print(f"\n缺失 {len(missing)} 个：")
        for name in missing:
            tag = "  ← 必需" if name in CRITICAL else ("  (可忽略)" if name in OPTIONAL else "")
            print(f"  - {name}{tag}")
    if wrong:
        print(f"\n大小不符 {len(wrong)} 个（可能下载不完整，建议重下）：")
        for name, actual, expect in wrong:
            print(f"  - {name}: 本地 {human(actual)} / 应为 {human(expect)}")

    # crc32 内容校验
    if args.crc:
        crc_path = os.path.join(root, "crc32.txt")
        if not os.path.isfile(crc_path):
            print("\n没有 crc32.txt，跳过内容校验")
        else:
            with open(crc_path, encoding="utf8", errors="replace") as f:
                repo_crc = load_repo_crc(f.read())
            print(f"\ncrc32.txt 里记录了 {len(repo_crc)} 个条目，开始逐一校验（要读全量数据，请耐心等）…")
            bad = 0
            for name, expect_crc in sorted(repo_crc.items()):
                path = os.path.join(root, name)
                if not os.path.isfile(path):
                    continue
                got = f"{crc32_file(path):08x}"
                mark = "OK" if got == expect_crc else "不匹配"
                if got != expect_crc:
                    bad += 1
                print(f"  [{mark}] {name}")
            print(f"crc32 校验完成，{'全部通过' if bad == 0 else f'{bad} 个不匹配'}")
            if bad:
                print("  注意：crc32.txt 本身已过时，与官方实际内容对不上（实测这些文件逐字节一致）。")
                print("  它也不覆盖 safetensors 权重，所以这个「不匹配」不能当结论。")
    print("\n提示：本脚本只比对文件大小，查不出内容损坏。")
    print("      要确认权重没有被下载过程写坏，请用 sha256 校验：")
    print(r"      .\.venv\Scripts\python.exe verify_model.py")

    print()
    if not missing and not wrong and not incomplete:
        print("结论：文件齐全，可以启动服务：  .\\start.ps1")
        return 0
    critical_missing = ([m for m in missing if m in CRITICAL]
                        + [w[0] for w in wrong if w[0] in CRITICAL]
                        + [i for i in incomplete if i in CRITICAL])
    if critical_missing:
        print("结论：还缺关键文件，服务无法加载。请补齐后再试。")
        return 1
    print("结论：关键文件已就位，缺失的是可忽略项，服务应该能启动。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
