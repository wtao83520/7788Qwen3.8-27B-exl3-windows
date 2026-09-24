#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
校验 README 第 8 节那段「config.yaml 关键项」摘要是否与 config.yaml 真实一致。

为什么需要：那段注释写着「与仓库里实际生效的 config.yaml 一致」，但实际上
改配置时很容易只改 config.yaml、忘了同步 README（本脚本第一次跑就抓到了
draft_model 写的是 null 而实际已启用 DFlash2）。文档与实物不符比没文档更坑人。

用法：
    .\.venv\Scripts\python.exe _bench\readme_sync_check.py
退出码 0 = 一致；1 = 有差异（会列出）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent.parent
README = BASE / "README.md"
CONFIG = BASE / "config.yaml"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass


def readme_block() -> dict:
    """从 README 里抠出第 8 节那个 yaml 代码块并解析。"""
    text = README.read_text(encoding="utf8")
    # 第 8 节标题之后、下一个二级标题之前
    m = re.search(r"^## 8\. .*?$(.*?)^## 9\.", text, re.DOTALL | re.MULTILINE)
    if not m:
        raise SystemExit("找不到 README 第 8 节")
    section = m.group(1)
    # 取第一个 ``` 代码块
    m2 = re.search(r"```(?:yaml)?\n(.*?)```", section, re.DOTALL)
    if not m2:
        raise SystemExit("第 8 节里没找到 yaml 代码块")
    return yaml.safe_load(m2.group(1)) or {}


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def main() -> int:
    doc = flatten(readme_block())
    real = flatten(yaml.safe_load(CONFIG.read_text(encoding="utf8")) or {})

    print("=" * 74)
    print("  README 第 8 节 vs config.yaml")
    print("=" * 74)

    bad = []
    for key, want in doc.items():
        # context 字段只在 README 里做说明，不是真实配置
        if key.endswith(".context"):
            continue
        if key not in real:
            bad.append((key, want, "<config.yaml 里没有这个键>"))
            continue
        got = real[key]
        if got != want:
            bad.append((key, want, got))

    if bad:
        print()
        for key, want, got in bad:
            print(f"  ❌ {key}")
            print(f"       README  = {want!r}")
            print(f"       config  = {got!r}")
        print()
        print(f"  共 {len(bad)} 处不一致")
    else:
        print()
        print(f"  ✅ 全部 {len(doc)} 项一致")

    # 顺带提示：README 摘要没覆盖到的键（不是错误，只是提醒）
    only_real = sorted(set(real) - set(doc))
    if only_real:
        print()
        print(f"  （config.yaml 里另有 {len(only_real)} 项未在 README 摘要中出现，属正常，"
              f"摘要是「关键项」不是全文）")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
