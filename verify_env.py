#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查运行环境：torch / exllamav3 / GPU。供 setup 脚本调用。"""

from __future__ import annotations

import sys


def _safe() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def main() -> int:
    _safe()
    ok = True
    try:
        import torch

        print("torch      :", torch.__version__)
        print("torch CUDA :", torch.version.cuda)
        print("CUDA 可用  :", torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print("GPU        :", torch.cuda.get_device_name(0))
            print("显存       :", round(props.total_memory / 1024**3, 1), "GB")
            print("算力       :", f"sm_{props.major}{props.minor}")
        else:
            print("警告：torch 看不到 CUDA 设备")
            ok = False
    except Exception as exc:
        print("torch 不可用：", exc)
        ok = False

    try:
        import exllamav3

        # 触发一次扩展 import，确认 CUDA 内核能加载
        from exllamav3 import Cache, Config, Model, Tokenizer  # noqa: F401

        print("exllamav3  : ok")
        try:
            from exllamav3.ext import exllamav3_ext  # noqa: F401

            print("CUDA 扩展  : ok")
        except Exception as exc:
            print("CUDA 扩展  : 加载失败 ->", exc)
            ok = False
    except Exception as exc:
        print("exllamav3 不可用：", exc)
        ok = False

    print("结果        :", "通过" if ok else "有问题")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
