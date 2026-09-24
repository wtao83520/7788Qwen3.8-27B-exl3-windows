#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
离线自检：不加载模型、不需要 GPU，只验证服务里最容易出错的纯逻辑部分。

    .\.venv\Scripts\python.exe selftest.py [--template-dir _tmpl]

检查内容：
  1. server.py 能否正常导入（语法 / 依赖）
  2. 真实的 Qwen3.8 chat_template.jinja 能否被正确渲染（含思考模式、工具、图片）
  3. ThinkSplitter 在流式分块输入下的 reasoning/content 切分是否正确
  4. <tool_call> 解析
  5. 采样参数合并逻辑（默认值 / 请求覆盖 / 校验）
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows 控制台默认是 GBK，避免个别字符导致 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template-dir", default="_tmpl", help="含 chat_template.jinja 的目录")
    args = ap.parse_args()

    print("=" * 74)
    print("1) 导入 server.py")
    print("=" * 74)
    import server as srv

    check("import server", True)

    tdir = os.path.abspath(args.template_dir)
    has_template = os.path.isfile(os.path.join(tdir, "chat_template.jinja"))
    check("找到 chat_template.jinja", has_template, tdir)
    if not has_template:
        print("\n跳过模板测试（没找到模板文件）")
        return 1 if FAILED else 0

    cfg = copy.deepcopy(srv.DEFAULT_CONFIG)
    cfg["model"]["path"] = tdir
    engine = srv.Engine(cfg)
    engine._load_template()
    check("加载对话模板", engine.template is not None)

    # ---------------------------------------------------------------- 2. 渲染
    print()
    print("=" * 74)
    print("2) 对话模板渲染")
    print("=" * 74)

    msgs = [{"role": "user", "content": "你好"}]
    prompt, reasoning = engine.render(msgs, {"enable_thinking": True, "reasoning_effort": "xhigh"})
    check("思考模式开启 → prompt 以 <think> 结尾", prompt.rstrip().endswith("<think>"), repr(prompt[-40:]))
    check("思考模式 → started_in_reasoning=True", reasoning is True)

    prompt, reasoning = engine.render(msgs, {"enable_thinking": False})
    check("思考模式关闭 → started_in_reasoning=False", reasoning is False)
    check("思考模式关闭 → 预填空的 think 块", "<think>\n\n</think>" in prompt, repr(prompt[-40:]))

    sys_msgs = [{"role": "system", "content": "你是助手"}, {"role": "user", "content": "hi"}]
    prompt, _ = engine.render(sys_msgs, {"enable_thinking": False})
    check("system 消息被写入", "你是助手" in prompt)

    think_msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "答案", "reasoning_content": "推理过程"},
        {"role": "user", "content": "再来"},
    ]
    prompt, _ = engine.render(think_msgs, {"enable_thinking": True, "preserve_thinking": True})
    check("历史保留 reasoning（preserve_thinking=true）", "推理过程" in prompt)

    prompt, _ = engine.render(think_msgs, {"enable_thinking": True, "preserve_thinking": False})
    check("历史丢弃 reasoning（preserve_thinking=false）", "推理过程" not in prompt)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询天气",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }
    ]
    prompt, _ = engine.render(
        [{"role": "user", "content": "北京天气"}],
        {"enable_thinking": False, "tools": tools},
    )
    check("工具定义被注入", "get_weather" in prompt and "# Tools" in prompt)

    img_msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "描述图片"},
                {"type": "image_url", "image_url": {"url": "http://example.com/a.png"}},
            ],
        }
    ]
    prompt, _ = engine.render(img_msgs, {"enable_thinking": False})
    check("图片内容块生成视觉占位", "<|vision_start|>" in prompt and "<|image_pad|>" in prompt)

    bad_effort = False
    try:
        engine.render(msgs, {"enable_thinking": True, "reasoning_effort": "超高"})
    except Exception:
        bad_effort = True
    check("非法 reasoning_effort 会被模板拒绝（服务会转成 400）", bad_effort)

    # ---------------------------------------------------------------- 3. 切分
    print()
    print("=" * 74)
    print("3) ThinkSplitter 流式切分")
    print("=" * 74)

    full = "让我想想……\n1+1=2</think>\n\n答案是 2。"
    pieces = [full[i:i + 3] for i in range(0, len(full), 3)]  # 故意切成 3 字符一块
    sp = srv.ThinkSplitter(True, True)
    got: list[tuple[str, str]] = []
    for p in pieces:
        got += sp.feed(p)
    got += sp.flush()
    reasoning_text = "".join(t for k, t in got if k == "reasoning")
    content_text = "".join(t for k, t in got if k == "content")
    check("推理段落正确", reasoning_text.strip() == "让我想想……\n1+1=2", repr(reasoning_text))
    check("正文段落正确", content_text.strip() == "答案是 2。", repr(content_text))
    check("正文开头没有残留换行", not content_text.startswith("\n"), repr(content_text))
    check(
        "不丢字（分词后与原文一致）",
        (reasoning_text + " " + content_text).split() == full.replace("</think>", " ").split(),
        repr((reasoning_text + " " + content_text).split()),
    )

    sp = srv.ThinkSplitter(True, False)
    got = []
    for ch in ["普通", "输出", "，", "没有思考"]:
        got += sp.feed(ch)
    got += sp.flush()
    check("非思考模式全部进 content", "".join(t for k, t in got if k == "content") == "普通输出，没有思考")

    sp = srv.ThinkSplitter(False, True)
    got = sp.feed("<think>abc</think>def") + sp.flush()
    check("split_reasoning=false 时不做切分", len(got) == 1 and got[0][0] == "content")

    # ---------------------------------------------------------------- 4. 工具
    print()
    print("=" * 74)
    print("4) <tool_call> 解析")
    print("=" * 74)

    text = (
        "好的，我来查一下。\n"
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\n北京\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    rest, calls = srv.parse_tool_calls(text)
    check("解析出 1 个调用", len(calls) == 1, json.dumps(calls, ensure_ascii=False))
    if calls:
        check("函数名正确", calls[0]["function"]["name"] == "get_weather")
        args = json.loads(calls[0]["function"]["arguments"])
        check("参数正确", args.get("city") == "北京")
        check("正文里移除了 tool_call 块", "<tool_call>" not in rest)
    rest, calls = srv.parse_tool_calls("普通回答，没有工具")
    check("无工具时原样返回", calls == [] and rest == "普通回答，没有工具")

    # ---------------------------------------------------------------- 5. 消息
    print()
    print("=" * 74)
    print("5) OpenAI 消息 → 模板消息")
    print("=" * 74)

    req = srv.ChatCompletionRequest(
        messages=[
            {"role": "user", "content": "天气"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "上海"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "晴，25 度"},
        ]
    )
    out = srv.build_chat_messages(req)
    check("工具调用 arguments 由字符串转成对象", isinstance(out[1]["tool_calls"][0]["function"]["arguments"], dict))
    check("tool 角色的 tool_call_id 保留", out[2].get("tool_call_id") == "call_1")
    check("纯文本 content 保留", out[0]["content"] == "天气")

    # ---------------------------------------------------------------- 6. 参数
    print()
    print("=" * 74)
    print("6) 采样参数合并")
    print("=" * 74)

    class FakeEngine:
        dcfg = srv.DEFAULT_CONFIG["defaults"]

    fe = FakeEngine()
    params, stops = srv.resolve_params(fe, srv.ChatCompletionRequest(messages=[], **{}))
    # max_tokens 不写死数字：它随 config.yaml 调整，断言只验证「默认值被原样带出来」
    check(
        "默认值生效",
        params["temperature"] == 0.6
        and params["top_k"] == 20
        and params["max_tokens"] == srv.DEFAULT_CONFIG["defaults"]["max_tokens"],
    )

    params, stops = srv.resolve_params(
        fe,
        srv.ChatCompletionRequest(
            messages=[], temperature=0.1, top_k=50, top_p=0.5, repetition_penalty=1.1, stop=["###"], max_completion_tokens=99
        ),
    )
    check("请求覆盖默认值", params["temperature"] == 0.1 and params["top_k"] == 50 and params["repetition_penalty"] == 1.1)
    check("max_completion_tokens 生效", params["max_tokens"] == 99)
    check("stop 转成列表", stops == ["###"])

    params, _ = srv.resolve_params(fe, srv.ChatCompletionRequest(messages=[], logit_bias={"100": -5.0, "abc": 1.0}))
    # resolve_params 原样透传，非数字键（以及越界 id）在 Engine.build_sampler 里过滤
    check("logit_bias 原样透传", params["logit_bias"] == {"100": -5.0, "abc": 1.0})
    check("未传 logit_bias 时为 None", srv.resolve_params(fe, srv.ChatCompletionRequest(messages=[]))[0]["logit_bias"] is None)

    try:
        srv.resolve_params(fe, srv.ChatCompletionRequest(messages=[], max_tokens=0))
        check("max_tokens=0 被拒绝", False)
    except Exception:
        check("max_tokens=0 被拒绝", True)

    # ---------------------------------------------------------------- 结果
    print()
    print("=" * 74)
    if FAILED:
        print(f"有 {len(FAILED)} 项失败：")
        for f in FAILED:
            print("  -", f)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
