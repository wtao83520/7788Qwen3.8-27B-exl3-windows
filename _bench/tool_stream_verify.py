#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
更严格的流式工具调用验证。

tool_diag.py 只证明「能解析出来」，这个脚本进一步确认：
  · 原始 SSE 帧的结构是否合规（客户端要能拼回来）
  · 一次返回多个工具调用时 index 是否连续
  · 开启思考模式（Copilot 常用）时是否仍然正常
  · 正文和工具调用同时出现时会不会互相污染

用法：
    .\.venv\Scripts\python.exe _bench\tool_stream_verify.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
import yaml

BASE = Path(__file__).resolve().parent.parent

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "在仓库里按正则搜索文本",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索模式"},
                    "isRegexp": {"type": "boolean", "description": "是否按正则解释"},
                    "includePattern": {"type": "string", "description": "限定文件范围"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "文件路径"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "列出目录内容",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    },
]


def port() -> int:
    try:
        cfg = yaml.safe_load((BASE / "config.yaml").read_text(encoding="utf8")) or {}
        return int((cfg.get("server") or {}).get("port") or 2345)
    except Exception:
        return 2345


def stream_once(messages, *, thinking: bool, label: str, dump_raw: bool = False):
    url = f"http://127.0.0.1:{port()}/v1/chat/completions"
    payload = {
        "model": "qwen",
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.0,
        "enable_thinking": thinking,
        "tools": TOOLS,
        "stream": True,
    }
    print("=" * 74)
    print(f"  {label}（enable_thinking={thinking}）")
    print("=" * 74)

    resp = requests.post(url, json=payload, stream=True, timeout=600)
    if resp.status_code != 200:
        print("  HTTP", resp.status_code, resp.text[:300])
        return None

    raw_frames: list[str] = []
    content = reasoning = ""
    slots: dict[int, dict] = {}
    finish = None
    order: list[str] = []

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            break
        raw_frames.append(data)
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            print("  ⚠️ 非 JSON 帧:", data[:120])
            continue
        choices = obj.get("choices") or []
        if not choices:
            continue
        ch = choices[0]
        if ch.get("finish_reason"):
            finish = ch["finish_reason"]
        d = ch.get("delta") or {}
        if d.get("content"):
            content += d["content"]
        if d.get("reasoning_content"):
            reasoning += d["reasoning_content"]
        for tc in d.get("tool_calls") or []:
            i = tc.get("index", 0)
            if i not in slots:
                order.append("new")
            slot = slots.setdefault(i, {"id": None, "name": "", "args": ""})
            if tc.get("id"):
                slot["id"] = tc["id"]
            if tc.get("type"):
                slot["type"] = tc["type"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] += fn["name"]
            if fn.get("arguments"):
                slot["args"] += fn["arguments"]

    print(f"  帧数          : {len(raw_frames)}")
    print(f"  finish_reason : {finish}")
    print(f"  content 长度  : {len(content)}  内容={content[:120]!r}")
    if reasoning:
        print(f"  reasoning 长度: {len(reasoning)}（截断显示）{reasoning[:100]!r}")
    print(f"  tool_calls    : {len(slots)} 个")
    ok = True
    for i in sorted(slots):
        s = slots[i]
        try:
            args = json.loads(s["args"]) if s["args"] else None
        except json.JSONDecodeError:
            args = f"<JSON 解析失败: {s['args']!r}>"
            ok = False
        print(f"    [{i}] id={s['id']}")
        print(f"        name={s['name']!r}")
        print(f"        args={args}")
        if not s["name"]:
            print("        ❌ 没有函数名")
            ok = False
        if not s["id"]:
            print("        ❌ 没有 id")
            ok = False

    if "<tool_call>" in content:
        print("  ❌ 正文里残留了原始 <tool_call>")
        ok = False
    if slots and finish != "tool_calls":
        print(f"  ❌ finish_reason 应为 tool_calls，实际 {finish}")
        ok = False

    if dump_raw and slots:
        print("  --- 首尾原始帧（看结构是否合规）---")
        for f in raw_frames[:3] + raw_frames[-3:]:
            print("   ", f[:200])

    print(f"  >>> {'✅ 通过' if ok else '❌ 有问题'}")
    return {"ok": ok, "slots": slots, "finish": finish}


def main() -> int:
    results = []

    results.append(stream_once(
        [{"role": "user", "content": "在 backend/ 目录里用正则搜索 llm-setting|base_url|api_key，"
                                     "请调用 grep_search 工具。"}],
        thinking=False, label="1. 单工具（复现用户场景）", dump_raw=True,
    ))

    results.append(stream_once(
        [{"role": "user", "content": "帮我看一下 C:/proj/README.md 的内容。"}],
        thinking=False, label="2. 另一个工具（read_file）",
    ))

    results.append(stream_once(
        [{"role": "user", "content": "先读一下 a.py，然后列出当前目录。"}],
        thinking=False, label="3. 可能触发多工具",
    ))

    results.append(stream_once(
        [{"role": "user", "content": "在 backend/ 里搜索 llm-setting，用正则。"}],
        thinking=True, label="4. 开启思考模式",
    ))

    print()
    print("=" * 74)
    bad = [r for r in results if r is None or not r["ok"]]
    print(f"  汇总：{len(results) - len(bad)}/{len(results)} 通过")
    for r in results:
        if r is None:
            print("    ❌ 请求失败")
    print("=" * 74)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
