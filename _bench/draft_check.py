#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""离线校验 DFlash2 草稿模型能否被本机 exllamav3 加载。

安全设计：
  * 只跑 CPU（device="cpu"），不碰显存 —— 推理服务可以照常在跑，不会被影响
  * 不启动服务、不读 config.yaml、不改任何配置
  * 只做「配置解析 → 建模块树 → 载权重」三步，验证张量键名与量化格式

    .\.venv\Scripts\python.exe _bench\draft_check.py models/DFlash2-EXL3-4.00bpw
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: draft_check.py <草稿模型目录>")
        return 2
    path = Path(sys.argv[1]).resolve()
    if not path.is_dir():
        print(f"目录不存在: {path}")
        return 2

    print("=" * 74)
    print(f"校验草稿模型: {path}")
    print("=" * 74)

    import exllamav3
    from exllamav3 import Config, Model
    from importlib.metadata import version

    print(f"exllamav3: {version('exllamav3')}")
    print()

    # ---------------- 1. 配置解析 ----------------
    print("[1/3] 解析 config.json …")
    t0 = time.time()
    try:
        config = Config.from_directory(str(path))
    except Exception:
        print("  ❌ 配置解析失败：")
        traceback.print_exc()
        return 1
    print(f"  ✅ {type(config).__name__}  arch_string={getattr(config, 'arch_string', '?')}  ({time.time()-t0:.2f}s)")
    for attr in ("block_size", "conv_kernel_size", "conv_group_size",
                 "selector_rank", "selector_top_k", "target_layer_ids",
                 "num_hidden_layers", "hidden_size", "vocab_size"):
        if hasattr(config, attr):
            print(f"     {attr:20} = {getattr(config, attr)}")
    print()

    # ---------------- 2. 建模块树 ----------------
    print("[2/3] 构造模块树 …")
    t0 = time.time()
    try:
        model = Model.from_config(config)
    except Exception:
        print("  ❌ 构造失败：")
        traceback.print_exc()
        return 1
    print(f"  ✅ 模块数 {len(model.modules)}  ({time.time()-t0:.2f}s)")
    caps = dict(model.caps)
    print("  caps:")
    for k in ("dflash_draft", "attach_target", "default_draft_size",
              "supports_tp", "uncalibrated_quantize"):
        print(f"     {k:22} = {caps.get(k)}")
    print()

    # ---------------- 3. 载权重（CPU） ----------------
    print("[3/3] 载入权重到 CPU（不占显存）…")
    t0 = time.time()
    try:
        model.load(device="cpu", progressbar=False)
    except Exception:
        print("  ❌ 载入失败：")
        traceback.print_exc()
        return 1
    dt = time.time() - t0
    print(f"  ✅ 载入成功，用时 {dt:.1f}s")

    # exllamav3 的 Model 不是 nn.Module，没有 named_parameters，
    # 官方的占用统计接口是 get_storage_info() → (平均 bpw, head bpw, 总位数)
    try:
        bpw, head_bpw, vram_bits = model.get_storage_info()
        print(f"  量化：平均 {bpw:.3f} bpw，输出层 {head_bpw:.2f} bpw")
        print(f"  权重占用（含 head，按 8 倍放缩统计）：{vram_bits / 8 / 1024**3:.2f} GiB")
    except Exception as exc:
        print(f"  （get_storage_info 不可用：{type(exc).__name__}: {exc}）")

    # 落盘体积作为交叉核对
    disk = sum(f.stat().st_size for f in path.rglob("*.safetensors"))
    print(f"  文件体积：{disk/1024**3:.2f} GiB")
    try:
        model.check_compat()
        print("  check_compat()：通过")
    except Exception as exc:
        print(f"  ❌ check_compat() 失败：{type(exc).__name__}: {exc}")
        return 1
    print()

    print("=" * 74)
    print("结论：这个草稿模型能被本机 exllamav3 直接加载 ✅")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
