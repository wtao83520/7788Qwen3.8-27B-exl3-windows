#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
诊断工具调用：对比「流式」和「非流式」两条路径。

背景：VS Code Copilot 一律用流式请求。如果只有非流式那条路会解析 <tool_call>，
那 Copilot 里就会看到原始 XML 文本当成正文 —— 本脚本就是来确认这一点的。

用法：
    .\.venv\Scripts\python.exe _bench\tool_diag.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
import yaml

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"],
            },
        },
    }
]

PROMPT = "北京现在天气怎么样？请调用工具查询。"


def port() -> int:
    try:
        cfg = yaml.safe_load((BASE / "config.yaml").read_text(encoding="utf8")) or {}
        return int((cfg.get("server") or {}).get("port") or 2345)
    except Exception:
        return 2345


def run(stream: bool) -> None:
    url = f"http://127.0.0.1:{port()}/v1/chat/completions"
    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 512,
        "enable_thinking": False,
        "tools": TOOLS,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}

    mode = "流式" if stream else "非流式"
    print("=" * 72)
    print(f"  {mode}（stream={stream}）")
    print("=" * 72)

    resp = requests.post(url, json=payload, stream=stream, timeout=600)
    if resp.status_code != 200:
        print("  HTTP", resp.status_code, resp.text[:400])
        return

    if not stream:
        b = resp.json()
        ch = b["choices"][0]
        msg = ch["message"]
        print("  finish_reason :", ch["finish_reason"])
        print("  content       :", repr((msg.get("content") or "")[:300]))
        tc = msg.get("tool_calls")
        print("  tool_calls    :", json.dumps(tc, ensure_ascii=False) if tc else "（无）")
        print()
        print("  >>> 结论:", "✅ 解析出 tool_calls" if tc else "❌ 没有 tool_calls")
        return

    # 流式：把 chunk 收下来看看究竟发了什么
    content = ""
    reasoning = ""
    tool_calls: dict[int, dict] = {}
    finish = None
    n_chunks = 0
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        n_chunks += 1
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if choices[0].get("finish_reason"):
            finish = choices[0]["finish_reason"]
        if delta.get("content"):
            content += delta["content"]
        if delta.get("reasoning_content"):
            reasoning += delta["reasoning_content"]
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            slot = tool_calls.setdefault(idx, {"id": tc.get("id"), "name": "", "args": ""})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] += fn["name"]
            if fn.get("arguments"):
                slot["args"] += fn["arguments"]

    print(f"  chunk 数      : {n_chunks}")
    print(f"  finish_reason : {finish}")
    print(f"  content       : {repr(content[:300])}")
    if reasoning:
        print(f"  reasoning     : {repr(reasoning[:120])}")
    if tool_calls:
        print("  tool_calls    :")
        for i in sorted(tool_calls):
            s = tool_calls[i]
            print(f"    [{i}] id={s['id']} name={s['name']!r} args={s['args']!r}")
    else:
        print("  tool_calls    : （无）")

    print()
    if tool_calls:
        print("  >>> 结论: ✅ 流式也解析出了 tool_calls")
    elif "<tool_call>" in content:
        print("  >>> 结论: ❌ 原始 <tool_call> 被当成正文发出去了（这就是 Copilot 里看到的现象）")
    else:
        print("  >>> 结论: ⚠️ 既没有 tool_calls，也没有原始 XML（模型没按格式生成？）")


def main() -> int:
    run(stream=False)
    print()
    run(stream=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
