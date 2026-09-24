#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""视觉输入端到端验证。

生成一张可控的图（3 个红圆点），问模型图里有几个，确认：
  1. 视觉塔真的加载了（/health 的 vision=True 只是配置层面的）
  2. 图像走完 preprocessor → 视觉塔 → 语言模型的全链路
  3. 答案正确（说明确实“看见”了，不是瞎猜）

    .\.venv\Scripts\python.exe _bench\vision_check.py
"""

from __future__ import annotations

import argparse
import base64
import io
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


def make_image(n_dots: int = 3, size: int = 336) -> bytes:
    """白底 + n 个红圆点。尺寸取小一点：空闲显存只有 ~1.5 GiB。"""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    r = size // 10
    for i in range(n_dots):
        cx = int(size * (i + 1) / (n_dots + 1))
        cy = size // 2
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description="视觉输入端到端验证")
    ap.add_argument("--dots", type=int, default=3, help="图上画几个红圆点（默认 3）")
    ap.add_argument("--question", default=None,
                    help="自定义问题（默认问有几个红圆点）")
    args = ap.parse_args()

    cfg = yaml.safe_load((BASE / "config.yaml").read_text(encoding="utf-8"))
    port = int(cfg["server"]["port"])
    S = f"http://127.0.0.1:{port}"

    h = requests.get(f"{S}/health", timeout=30).json()
    print("=" * 74)
    print("视觉输入验证")
    print("=" * 74)
    print(f"  /health vision = {h.get('vision')}   draft_kind = {h.get('draft_kind')}")
    print(f"  空闲显存 = {(h.get('gpu') or {}).get('vram_free_gb')} GiB")
    print()

    N = args.dots
    question = args.question or "这张图里有几个红色的圆点？只回答数字，不要解释。"
    png = make_image(N)
    b64 = base64.b64encode(png).decode()
    print(f"  已生成 {N} 个红圆点的测试图（{len(png)} 字节 PNG）")
    print()

    payload = {
        "model": "qwen",
        "stream": False,
        "enable_thinking": False,
        "max_tokens": 128,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text",
                 "text": question},
            ],
        }],
    }

    print("  发送图片请求…")
    t0 = time.time()
    try:
        r = requests.post(f"{S}/v1/chat/completions", json=payload, timeout=900)
    except Exception as exc:
        print(f"  ✗ 请求异常：{type(exc).__name__}: {exc}")
        return 1
    dt = time.time() - t0

    if r.status_code != 200:
        print(f"  ✗ HTTP {r.status_code}: {r.text[:400]}")
        return 1

    body = r.json()
    text = (body["choices"][0]["message"].get("content") or "").strip()
    u = body.get("usage") or {}
    print(f"  耗时 {dt:.1f}s   prompt {u.get('prompt_tokens')} token   "
          f"生成 {u.get('completion_tokens')} token")
    print(f"  回答：{text!r}")
    print()

    # 图片会展开成大量 token（影像 token），这是视觉链路真的走了的证据
    pt = u.get("prompt_tokens") or 0
    ok_answer = str(N) in text
    ok_expand = pt > 50

    print(f"  [{'PASS' if ok_expand else 'FAIL'}] 影像被展开成 token（prompt {pt} > 50）")
    print(f"  [{'PASS' if ok_answer else 'FAIL'}] 答案里出现「{N}」")
    print()
    if ok_expand and ok_answer:
        print("结论：视觉塔确实生效，图片能正常输入并正确识别 ✅")
        return 0
    if ok_expand and not ok_answer:
        print("结论：视觉链路通了（影像 token 有展开），但答案不对。")
        print("      可能只是识别能力问题，也可能是视觉塔权重有问题，值得再试一张更清楚的图。")
        return 1
    print("结论：prompt token 没有明显膨胀，图片可能根本没被处理 ❌")
    return 1


if __name__ == "__main__":
    sys.exit(main())
