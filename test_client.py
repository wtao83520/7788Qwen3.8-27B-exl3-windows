#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理服务的冒烟测试脚本。

    python test_client.py                      # 跑全部基础测试
    python test_client.py --image a.jpg        # 额外测图片输入
    python test_client.py --tools              # 额外测工具调用
    python test_client.py --base-url http://127.0.0.1:2345 --api-key sk-xxx

默认端口从 config.yaml 的 server.port 读，不走 JSON/写死。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import requests
try:
    import yaml
except ImportError:  # 没装 PyYAML 也不应该影响冒烟测试
    yaml = None

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass


def _default_base_url() -> str:
    """端口从 config.yaml 的 server.port 读，不写死。"""
    port = 2345
    if yaml is not None:
        try:
            cfg = yaml.safe_load(
                Path(__file__).resolve().parent.joinpath("config.yaml").read_text(encoding="utf8")
            ) or {}
            port = int((cfg.get("server") or {}).get("port") or port)
        except Exception:
            pass
    return f"http://127.0.0.1:{port}"


def show(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=_default_base_url())
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--prompt", default="用三句话解释什么是量化感知训练。")
    ap.add_argument("--image", default=None, help="要测试的图片路径")
    ap.add_argument("--tools", action="store_true", help="测试工具调用")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    def url(path: str) -> str:
        return f"{base}{path}"

    # ------------------------------------------------------------ health
    show("1. GET /health")
    try:
        resp = requests.get(url("/health"), timeout=30)
        print(resp.status_code, json.dumps(resp.json(), ensure_ascii=False, indent=2))
        if resp.status_code != 200:
            print("服务未就绪，后续测试可能失败。")
    except requests.RequestException as exc:
        print("连接失败：", exc)
        print("请确认服务已启动（.\\start.ps1）")
        return 1

    # ------------------------------------------------------------ models
    show("2. GET /v1/models")
    resp = requests.get(url("/v1/models"), headers=headers, timeout=30)
    print(resp.status_code, json.dumps(resp.json(), ensure_ascii=False, indent=2))
    model_id = resp.json()["data"][0]["id"]

    # ------------------------------------------------------------ 非流式
    show("3. POST /v1/chat/completions（非流式）")
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "你是一个简洁的中文技术助手。"},
            {"role": "user", "content": args.prompt},
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": args.max_tokens,
        "enable_thinking": False,
    }
    t0 = time.time()
    resp = requests.post(url("/v1/chat/completions"), headers=headers, data=json.dumps(payload), timeout=1800)
    elapsed = time.time() - t0
    if resp.status_code != 200:
        print("失败：", resp.status_code, resp.text[:800])
    else:
        body = resp.json()
        choice = body["choices"][0]["message"]
        print("content        :", choice.get("content"))
        if choice.get("reasoning_content"):
            print("reasoning(预览):", choice["reasoning_content"][:200], "…")
        print("finish_reason  :", body["choices"][0]["finish_reason"])
        print("usage          :", body["usage"])
        ct = body["usage"]["completion_tokens"]
        if ct and elapsed:
            print(f"耗时           : {elapsed:.2f}s  ({ct / elapsed:.1f} tok/s)")

    # ------------------------------------------------------------ 流式
    show("4. POST /v1/chat/completions（流式）")
    payload["stream"] = True
    payload["messages"][1]["content"] = "从 1 数到 10，每个数字用逗号分隔。"
    payload["max_tokens"] = 128
    t0 = time.time()
    first_token_at = None
    chunks = 0
    text = ""
    reasoning = ""
    usage = None
    with requests.post(url("/v1/chat/completions"), headers=headers, data=json.dumps(payload), stream=True, timeout=1800) as resp:
        if resp.status_code != 200:
            print("失败：", resp.status_code, resp.text[:800])
        else:
            print("流式输出： ", end="", flush=True)
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        if first_token_at is None:
                            first_token_at = time.time() - t0
                        text += piece
                        print(piece, end="", flush=True)
                        chunks += 1
                    rp = delta.get("reasoning_content")
                    if rp:
                        if first_token_at is None:
                            first_token_at = time.time() - t0
                        reasoning += rp
            print()
    elapsed = time.time() - t0
    print(f"\n首 token 延迟  : {first_token_at:.2f}s" if first_token_at else "\n没有收到内容")
    print(f"总耗时         : {elapsed:.2f}s，文本块 {chunks} 个")
    if reasoning:
        print(f"reasoning 长度 : {len(reasoning)} 字符")
    if usage:
        ct = usage["completion_tokens"]
        print("usage          :", usage)
        if ct and elapsed:
            print(f"平均速度       : {ct / elapsed:.1f} tok/s")

    # ------------------------------------------------------------ logprobs
    show("5. logprobs 测试")
    payload.pop("stream", None)
    payload["messages"][1]["content"] = "1+1 等于几？只回答数字。"
    payload["max_tokens"] = 16
    payload["logprobs"] = True
    payload["top_logprobs"] = 3
    resp = requests.post(url("/v1/chat/completions"), headers=headers, data=json.dumps(payload), timeout=600)
    if resp.status_code != 200:
        print("失败：", resp.status_code, resp.text[:500])
    else:
        lp = resp.json()["choices"][0].get("logprobs")
        if not lp:
            print("未返回 logprobs")
        else:
            for item in lp["content"][:6]:
                tops = ", ".join(f"{t['token']!r}:{t['logprob']:.2f}" for t in (item.get("top_logprobs") or [])[:3])
                print(f"  token={item['token']!r:<12} logprob={item['logprob']:.3f}   top: {tops}")

    # ------------------------------------------------------------ 图片
    if args.image:
        show("6. 图片输入测试")
        with open(args.image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "详细描述这张图片的内容。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
            "max_tokens": 256,
            "enable_thinking": False,
        }
        resp = requests.post(url("/v1/chat/completions"), headers=headers, data=json.dumps(payload), timeout=1800)
        if resp.status_code != 200:
            print("失败：", resp.status_code, resp.text[:800])
        else:
            print(resp.json()["choices"][0]["message"].get("content"))

    # ------------------------------------------------------------ 工具调用
    if args.tools:
        show("7. 工具调用测试")
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "北京现在天气怎么样？"}],
            "max_tokens": 512,
            "enable_thinking": False,
            "tools": [
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
            ],
        }
        resp = requests.post(url("/v1/chat/completions"), headers=headers, data=json.dumps(payload), timeout=1800)
        if resp.status_code != 200:
            print("失败：", resp.status_code, resp.text[:800])
        else:
            body = resp.json()
            print("content    :", body["choices"][0]["message"].get("content"))
            print("tool_calls :", json.dumps(body["choices"][0]["message"].get("tool_calls"), ensure_ascii=False, indent=2))
            print("finish     :", body["choices"][0]["finish_reason"])

    show("完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
