#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
显存增长探测：连续发多个请求，每次记录空闲显存。

背景：跑完一轮工具调用测试后，空闲显存从 1.51 GiB 掉到 0.77 GiB，
重启就恢复。需要判断这是
    (a) PyTorch 缓存分配器一次性预留住了（正常，会到平台期），还是
    (b) 每请求泄漏（危险，迟早 OOM）。
判断依据：看空闲显存是「跑几次后就不再降」还是「一直线性下降」。

用法：
    .\.venv\Scripts\python.exe _bench\vram_probe.py [请求数]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests
import yaml

BASE = Path(__file__).resolve().parent.parent

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "按正则搜索",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "isRegexp": {"type": "boolean"},
                    "includePattern": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    }
]

# 刻意让每个请求的「形状」都不一样：prompt 长短、思考开关、工具/闲聊，
# 这样才能把可变形状的激活内存都逼出来。
CASES = [
    ("短闲聊", [{"role": "user", "content": "说个数字"}], False, False),
    ("短闲聊流式", [{"role": "user", "content": "说个数字"}], False, True),
    ("工具流式", [{"role": "user", "content": "在 backend 里正则搜 llm-setting，调用工具。"}], True, True),
    ("工具非流式", [{"role": "user", "content": "在 backend 里正则搜 llm-setting，调用工具。"}], True, False),
    ("思考+工具", [{"role": "user", "content": "在 backend 里正则搜 llm-setting，调用工具。"}], True, True),
    ("长正文", [{"role": "user", "content": "用 300 字介绍量化感知训练。"}], False, True),
]


def port() -> int:
    try:
        cfg = yaml.safe_load((BASE / "config.yaml").read_text(encoding="utf8")) or {}
        return int((cfg.get("server") or {}).get("port") or 2345)
    except Exception:
        return 2345


def free_gb() -> float | None:
    try:
        h = requests.get(f"http://127.0.0.1:{port()}/health", timeout=20).json()
        return (h.get("gpu") or {}).get("vram_free_gb")
    except Exception:
        return None


def send(messages, use_tools: bool, stream: bool, thinking: bool, max_tokens: int = 256) -> None:
    payload = {
        "model": "qwen",
        "messages": messages,
        "max_tokens": max_tokens,
        "enable_thinking": thinking,
        "stream": stream,
    }
    if use_tools:
        payload["tools"] = TOOLS
    r = requests.post(f"http://127.0.0.1:{port()}/v1/chat/completions",
                      json=payload, stream=stream, timeout=900)
    if stream:
        for _ in r.iter_lines(decode_unicode=True):
            pass
    else:
        r.json()


def main() -> int:
    # 生成上限会直接影响激活内存的形状 —— 短请求（256）和长请求（4096）测出的
    # 地板可能差很多，所以做成可调。
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("rounds", nargs="?", type=int, default=3, help="跑几轮（默认 3）")
    ap.add_argument("--max-tokens", type=int, default=256, help="每次生成的 token 上限")
    args = ap.parse_args()

    rounds = args.rounds
    mt = args.max_tokens
    base = free_gb()
    print(f"起始空闲显存: {base} GiB（max_tokens={mt}）")
    print()

    samples: list[float] = []
    for i in range(rounds):
        for label, msgs, use_tools, stream in CASES:
            try:
                send(msgs, use_tools, stream, thinking=("思考" in label), max_tokens=mt)
            except Exception as exc:
                print(f"  第{i+1}轮 {label} 失败: {exc}")
                continue
            f = free_gb()
            if f is not None:
                samples.append(f)
            print(f"  第{i+1}轮 {label:<12} 流式={str(stream):<5} 工具={str(use_tools):<5} "
                  f"→ 空闲 {f} GiB")
        print()

    if not samples or base is None:
        print("没拿到有效数据")
        return 1

    print("=" * 66)
    print(f"  起始   : {base} GiB")
    print(f"  最低   : {min(samples)} GiB")
    print(f"  最终   : {samples[-1]} GiB")
    print(f"  共 {len(samples)} 次采样")
    print("=" * 66)

    # 简单判据：看后半段的斜率。如果每轮仍在明显下降 → 疑似泄漏。
    half = len(samples) // 2
    first_avg = sum(samples[:half]) / max(1, half)
    last_avg = sum(samples[half:]) / max(1, len(samples) - half)
    drop = first_avg - last_avg
    print(f"  前半平均 {first_avg:.2f} GiB → 后半平均 {last_avg:.2f} GiB"
          f"（差 {drop:+.2f} GiB）")
    if drop > 0.3:
        print("  >>> ⚠️ 后半段仍在明显下降，像是有增长；建议加大轮数再确认")
    else:
        print("  >>> ✅ 已进入平台期（掉到某个水平就不动了）")
        print("      说明是 PyTorch 缓存分配器一次性预留了可变形状的激活内存，")
        print("      不是每请求泄漏；重启服务即可归还给驱动。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
