#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
`reasoning_effort`（思考深度）到底是怎么生效的。

结论先行：它不是 token 预算，而是**往 system 位置塞一句提示词**，靠文字引导模型
自己控制思维链长度。本脚本用来证明这一点，并逐字打印每个档位实际注入的内容。

不需要 GPU（只渲染模板，不加载权重）。

用法：
    .\.venv\Scripts\python.exe _bench\effort_check.py            # 只看模板注入
    .\.venv\Scripts\python.exe _bench\effort_check.py --live     # 再真发请求量思维链长度
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

EFFORTS = ("xhigh", "medium", "low")


def make_engine():
    """用一个只含模板的临时 Engine —— 和 selftest.py 一样的做法，不加载权重。"""
    import server as srv

    cfg = copy.deepcopy(srv.DEFAULT_CONFIG)
    cfg["model"]["path"] = str(BASE / "_tmpl")
    eng = srv.Engine(cfg)
    eng._load_template()
    return eng


def extract_system(prompt: str) -> list[str]:
    """取出 prompt 里所有 system 段的内容。"""
    return [m.strip() for m in re.findall(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", prompt, re.DOTALL)]


def show_template_injection() -> dict[str, str]:
    print("=" * 78)
    print("  1) 模板层面：每个档位往 prompt 里注入了什么")
    print("=" * 78)

    eng = make_engine()
    msgs = [{"role": "user", "content": "1+1 等于几"}]
    injected: dict[str, str] = {}

    for effort in EFFORTS:
        prompt, started = eng.render(
            msgs, {"enable_thinking": True, "reasoning_effort": effort}
        )
        systems = extract_system(prompt)
        # 用户消息没写 system，所以这里出现的 system 段就是「纯注入」的
        text = "\n".join(systems)
        injected[effort] = text
        print(f"\n  ── reasoning_effort = {effort} ──")
        if text:
            print("  注入的 system 内容：")
            for line in text.splitlines():
                print("    │ " + line)
        else:
            print("  ⚠️ 什么也没注入（模板里 medium 没有对应分支）")
        print(f"  prompt 长度 {len(prompt)} 字符，started_in_reasoning={started}")

    print()
    print("  ── 对比 ──")
    print(f"  xhigh  注入 {len(injected['xhigh'])} 字符")
    print(f"  medium 注入 {len(injected['medium'])} 字符")
    print(f"  low    注入 {len(injected['low'])} 字符")
    same_med_low = injected["medium"] == injected["low"]
    print(f"  medium 与 low 是否相同：{same_med_low}")
    print()
    print("  ★ 关键点：三者差别只是**一句自然语言提示**，没有任何数值化的 token 预算。")
    print("     所以 reasoning_effort 是「软引导」，模型完全可以不听话 ——")
    print("     真正能限住思考长度的只有 max_tokens / 上下文上限。")

    print()
    print("=" * 78)
    print("  2) 边界情况")
    print("=" * 78)

    # 关掉思考时，effort 应该完全不起作用
    prompt_off, _ = eng.render(msgs, {"enable_thinking": False, "reasoning_effort": "xhigh"})
    prompt_off2, _ = eng.render(msgs, {"enable_thinking": False, "reasoning_effort": "low"})
    ok = ("Reasoning effort" not in prompt_off) and (prompt_off == prompt_off2)
    print(f"  [{'PASS' if ok else 'FAIL'}] enable_thinking=false 时 effort 被忽略（两者 prompt 完全相同）")

    # 不传 effort 时默认 xhigh
    prompt_def, _ = eng.render(msgs, {"enable_thinking": True})
    same_as_xhigh = extract_system(prompt_def) == extract_system(
        eng.render(msgs, {"enable_thinking": True, "reasoning_effort": "xhigh"})[0]
    )
    print(f"  [{'PASS' if same_as_xhigh else 'FAIL'}] 不传 effort 时默认就是 xhigh")

    # 用户自己写了 system 消息时，注入的内容拼在用户 system 前面
    sysmsgs = [{"role": "system", "content": "你是助手"}, {"role": "user", "content": "hi"}]
    p, _ = eng.render(sysmsgs, {"enable_thinking": True, "reasoning_effort": "low"})
    sysseg = extract_system(p)
    ok2 = bool(sysseg) and ("你是助手" in sysseg[0]) and ("Reasoning effort" in sysseg[0])
    print(f"  [{'PASS' if ok2 else 'FAIL'}] 用户自带 system 时，注入内容拼在其前面（同一段）")

    # 有工具时，注入进 "tools 系统提示" 里，而不是单独一段
    tools = [{"type": "function", "function": {"name": "f", "description": "d",
              "parameters": {"type": "object", "properties": {}}}}]
    pt, _ = eng.render([{"role": "user", "content": "hi"}],
                       {"enable_thinking": True, "reasoning_effort": "xhigh", "tools": tools})
    seg = extract_system(pt)
    ok3 = bool(seg) and ("Reasoning effort" in seg[0]) and ("# Tools" in seg[0])
    print(f"  [{'PASS' if ok3 else 'FAIL'}] 带 tools 时注入进同一段 tools 提示（不是两段 system）")

    # tools + 关思考：注入不该出现。
    # 原理：reasoning_instructions 只在 enable_thinking 那个 if 块里被赋值，
    # 而 tools 分支里是 `{%- if reasoning_instructions %}` 守卫的，所以这里应该是空。
    pt_off, _ = eng.render([{"role": "user", "content": "hi"}],
                           {"enable_thinking": False, "reasoning_effort": "xhigh", "tools": tools})
    seg_off = extract_system(pt_off)
    ok4 = ("Reasoning effort" not in "\n".join(seg_off)) and ("# Tools" in "\n".join(seg_off))
    print(f"  [{'PASS' if ok4 else 'FAIL'}] 带 tools + enable_thinking=false 时不注入 effort"
          f"（tools 分支有 reasoning_instructions 守卫）")

    # 非法值
    bad = False
    try:
        eng.render(msgs, {"enable_thinking": True, "reasoning_effort": "high"})
    except Exception as exc:
        bad = True
        print(f"  [PASS] 'high' 被拒：{str(exc)[:90]}")
    if not bad:
        print("  [FAIL] 'high' 竟然被接受了")
    print("       ↑ 注意：'high' 不是合法值！合法值只有 xhigh / medium / low")

    return injected


def live_measure() -> None:
    print()
    print("=" * 78)
    print("  3) 实测：同一问题、同一 seed，三个档位各跑一次，量思维链长度")
    print("=" * 78)

    import requests
    import yaml

    try:
        cfg = yaml.safe_load((BASE / "config.yaml").read_text(encoding="utf8")) or {}
        port = int((cfg.get("server") or {}).get("port") or 2345)
    except Exception:
        port = 2345

    # 挑一个「需要动脑但不算太难」的问题：太简单看不出档位差异
    question = ("一个披萨切成 8 块，我吃了 3 块，剩下的分给 5 个人，"
                "每人能分到多少块？请说明计算过程。")

    print(f"  问题：{question[:60]}…")
    print()
    rows = []
    for effort in EFFORTS:
        payload = {
            "model": "qwen",
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 4096,
            "temperature": 0.0,
            "seed": 1234,
            "enable_thinking": True,
            "reasoning_effort": effort,
            "stream": False,
        }
        try:
            r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                              json=payload, timeout=900)
            b = r.json()
        except Exception as exc:
            print(f"  {effort:<7} 请求失败：{exc}")
            continue
        if r.status_code != 200:
            print(f"  {effort:<7} HTTP {r.status_code}: {str(b)[:200]}")
            continue
        msg = b["choices"][0]["message"]
        cot = msg.get("reasoning_content") or ""
        ans = msg.get("content") or ""
        usage = b.get("usage") or {}
        rows.append((effort, len(cot), len(ans), usage.get("completion_tokens"), len(cot) / 2))
        print(f"  {effort:<7} 思维链 {len(cot):>6} 字符 | 正文 {len(ans):>5} 字符 | "
              f"计费 token {usage.get('completion_tokens')}")

    if len(rows) > 1:
        print()
        counts = [r[1] for r in rows]
        spread = (max(counts) - min(counts)) / max(1, max(counts))
        print(f"  思维链长度范围：{min(counts)} ~ {max(counts)} 字符"
              f"（相差 {spread * 100:.0f}%）")
        if spread < 0.15:
            print("  >>> 差异很小：说明这个问题是「模型心里已有定论」的，")
            print("      提示词推不动它 —— 这正是软引导的典型表现。")
            print("      换个真正需要权衡的问题（方案选型、有取舍的设计决策）差异会更明显。")
        else:
            print("  >>> 三个档位确实产生了不同的思考长度，说明这层软引导是有效的。")


def main() -> int:
    show_template_injection()
    if "--live" in sys.argv:
        live_measure()
    else:
        print()
        print("  （加 --live 可以真发请求，量一下三个档位的思维链长度差多少）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
