#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""针对性验收：
  - 思考模式的 reasoning_content 是否正确分离
  - enable_thinking=false 是否真的关掉思考
  - max_tokens 能不能顶到接近 262144
  - 长 prompt 的 prefill + 生成吞吐

    .\.venv\Scripts\python.exe check_features.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import requests
import yaml


def _service_port() -> int:
    """端口从 config.yaml 读（server.port 可改），写死会在改端口后直接连不上。"""
    try:
        cfg = yaml.safe_load(Path(__file__).resolve().parent.joinpath("config.yaml").read_text(encoding="utf8")) or {}
        return int((cfg.get("server") or {}).get("port") or 8000)
    except Exception:
        return 8000


BASE = f"http://127.0.0.1:{_service_port()}"


def chat(**kw):
    payload = {
        "model": "Qwen3.8-27B-3.50bpw",
        "messages": [{"role": "user", "content": kw.pop("prompt")}],
        "stream": False,
    }
    payload.update(kw)
    t0 = time.time()
    r = requests.post(f"{BASE}/v1/chat/completions", json=payload, timeout=900)
    dt = time.time() - t0
    if r.status_code != 200:
        return None, dt, r.status_code, r.text[:300]
    return r.json(), dt, r.status_code, ""


print("=" * 74)
print("1. 思考模式：reasoning_content 是否分离出来")
print("=" * 74)
body, dt, code, err = chat(
    prompt="一个农夫要带狼、羊、白菜过河，船一次只能带一样。农夫不在时狼会吃羊，羊会吃白菜。给出步骤。",
    max_tokens=2048,
    enable_thinking=True,
    reasoning_effort="xhigh",
)
if body is None:
    print(f"  失败 HTTP {code}: {err}")
else:
    msg = body["choices"][0]["message"]
    reason = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    print(f"  finish_reason     : {body['choices'][0]['finish_reason']}")
    print(f"  reasoning_content : {len(reason)} 字符")
    print(f"  content           : {len(content)} 字符")
    print(f"  usage             : {body['usage']}")
    print(f"  耗时              : {dt:.1f}s")
    if reason:
        head = reason[:180].replace("\n", " ")
        print(f"  思考开头          : {head}…")
    else:
        print("  !! 没有 reasoning_content，思考模式可能没生效")
    if content:
        tail = content[:180].replace("\n", " ")
        print(f"  回答开头          : {tail}…")

print()
print("=" * 74)
print("2. enable_thinking=false 应该没有 reasoning_content")
print("=" * 74)
body2, dt2, code2, err2 = chat(prompt="1+1 等于几？只回答数字。", max_tokens=256, enable_thinking=False)
if body2 is None:
    print(f"  失败 HTTP {code2}: {err2}")
else:
    msg2 = body2["choices"][0]["message"]
    print(f"  reasoning_content : {len(msg2.get('reasoning_content') or '')} 字符")
    print(f"  content           : {(msg2.get('content') or '').strip()[:80]!r}")
    print(f"  usage             : {body2['usage']}")

print()
print("=" * 74)
print("3. max_tokens 能不能设到很大（上下文上限 262144）")
print("=" * 74)
for want in (4096, 65536, 262144, 999999):
    body3, dt3, code3, err3 = chat(prompt="从 1 开始一直数下去，不要停。", max_tokens=want, enable_thinking=False)
    if body3 is None:
        note = f"HTTP {code3}: {err3.splitlines()[0][:90]}"
    else:
        note = f"接受，实际生成 {body3['usage']['completion_tokens']} token"
    print(f"  max_tokens={want:<8} -> {note}")

print()
print("=" * 74)
print("4. 长上下文 prefill 吞吐（约 8K token 的输入）")
print("=" * 74)
filler = ("这是一段用于填充上下文的无关文字。" * 12 + "\n") * 60
prompt = filler + "\n请用一句话总结上面这段文字在讲什么。"
est = len(prompt) // 2
print(f"  构造输入约 {est} token …")
body4, dt4, code4, err4 = chat(prompt=prompt, max_tokens=128, enable_thinking=False)
if body4 is None:
    print(f"  失败 HTTP {code4}: {err4}")
else:
    u = body4["usage"]
    print(f"  prompt_tokens     : {u['prompt_tokens']}")
    print(f"  completion_tokens : {u['completion_tokens']}")
    print(f"  耗时              : {dt4:.2f}s  → prefill 约 {u['prompt_tokens'] / dt4:.0f} tok/s")
    print(f"  回答              : {(body4['choices'][0]['message'].get('content') or '').strip()[:120]!r}")

print()
print("=" * 74)
print("5. 上下文超限应该被明确拒绝（而不是崩掉）")
print("=" * 74)
body5, dt5, code5, err5 = chat(prompt="hi", max_tokens=262144 + 100)
print(f"  max_tokens 超上限 -> HTTP {code5}: {err5.splitlines()[0][:120] if err5 else json.dumps(body5)[:120]}")
