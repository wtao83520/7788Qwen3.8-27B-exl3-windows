#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""DFlash2 草稿模型对比评测。

回答一个问题：把内置 MTP 换成 DFlash2 草稿，值不值？

设计上要能**分离变量**，所以四档都测：
    cq4_mtp      当前配置（KV 4bit + MTP）        ← 基线
    cq3_mtp      KV 3bit + MTP                    ← 单独看降 KV 精度的影响
    cq4_dflash   KV 4bit + DFlash2                ← 单独看换草稿的影响（可能偏紧）
    cq3_dflash   KV 3bit + DFlash2                ← 候选「长期保留」配置

指标两个：
  * 解码 tok/s —— 用户实际感受到的速度。用**差分法**测（见 bench_kv.bench_speed），
    单次测量会把首 token 延迟折进生成时间，短生成严重低估。
  * **接受长度** —— 每次验证平均产出多少 token。这个才是草稿模型本身的成绩，
    不受草稿自身计算开销影响。从 /health 的累计计数器换算：
        验证轮数 ≈ 生成 token 数 - 接受草稿数
        接受长度 = 生成 token 数 / 验证轮数
    （每轮除了接受的草稿，还会多出 1 个目标模型自己采样的 token，所以有这个关系。）

安全：生成的配置一律写到 _bench/ 下（复用 bench_kv.make_config 的路径守卫），
结束时恢复 config.yaml 原配置并重启。

    .\.venv\Scripts\python.exe _bench\bench_draft.py
    .\.venv\Scripts\python.exe _bench\bench_draft.py --only cq3_dflash
"""

from __future__ import annotations

import argparse
import json
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

import bench_kv as bk

BENCH_DIR = BASE / "_bench"
OUT = BENCH_DIR / "draft_results.json"
DFLASH = "models/DFlash2-EXL3-4.00bpw"
MAX_SEQ = 262144

# (名字, cache_quant, draft_model, 说明)
MATRIX = [
    ("cq4_mtp",     4, None,    "基线：KV 4bit + 内置 MTP"),
    ("cq3_mtp",     3, None,    "KV 3bit + MTP（单独看降精度的代价）"),
    ("cq4_dflash",  4, DFLASH,  "KV 4bit + DFlash2（单独看换草稿的收益）"),
    ("cq3_dflash",  3, DFLASH,  "KV 3bit + DFlash2（候选长期配置）"),
]


def health() -> dict:
    return requests.get(f"{bk.SERVICE}/health", timeout=30).json()


def draft_counters() -> dict:
    h = health()
    return {
        "accepted": h.get("draft_accepted_tokens") or 0,
        "rejected": h.get("draft_rejected_tokens") or 0,
        "requests": h.get("draft_requests") or 0,
        "kind": h.get("draft_kind"),
        "window": h.get("draft_tokens_per_window"),
    }


def measure_al(prompt: str, num_requests: int = 3, max_tokens: int = 768) -> dict:
    """用某个 prompt 测接受长度（/health 计数器差值）+ 该场景下的实际吞吐。

    ⚠️ prompt 的选择对接受长度影响极大：「从 1 数下去」这种几乎确定的文本，
    草稿模型能近乎全中（实测接受率 0.988、接受长度 7.92/8），是**最好情况**。
    所以这里测两类，避免拿最好情况当结论。

    另外记录 wall time，算出「验证轮/秒」——它排除了每轮产出多少 token 的影响，
    能单独看草稿本身带来的每轮开销。真实吞吐也应该看这里，而不是只看计数 prompt。
    """
    before = draft_counters()
    total_tokens = 0
    t0 = time.time()
    for _ in range(num_requests):
        r = bk.chat(prompt, max_tokens=max_tokens)
        total_tokens += int(r["usage"]["completion_tokens"])
    wall = time.time() - t0
    after = draft_counters()

    acc = after["accepted"] - before["accepted"]
    rej = after["rejected"] - before["rejected"]
    rounds = total_tokens - acc          # 每轮除了接受的草稿，还多 1 个目标采样 token
    al = (total_tokens / rounds) if (rounds > 0 and total_tokens > 0) else float("nan")
    return {
        "accept_len": round(al, 2) if al == al else None,
        "draft_window": after["window"],
        "gen_tokens": total_tokens,
        "accepted_draft": acc,
        "rejected_draft": rej,
        "verify_rounds": rounds,
        "accept_rate": round(acc / (acc + rej), 3) if (acc + rej) > 0 else None,
        "al_ceiling": (after["window"] + 1) if after["window"] else None,
        "wall_s": round(wall, 2),
        # 直接实测吞吐（含 TTFT 与 prefill，prompt 短所以影响小）
        "tps_direct": round(total_tokens / wall, 1) if wall > 0 else None,
        # 验证轮/秒：不受「每轮产出几个 token」影响，看的是每轮开销
        "rounds_per_s": round(rounds / wall, 1) if wall > 0 else None,
    }


def diff_tps(prompt: str, n_small: int, n_big: int) -> dict:
    """差分法测解码吞吐：把首 token 延迟和 prefill 抵消掉。

        (nB - nA) / (wallB - wallA)

    单次测量会把 TTFT 折进生成时间，短生成会严重低估解码速度。
    """
    a = bk.chat(prompt, max_tokens=n_small)
    b = bk.chat(prompt, max_tokens=n_big)
    na, nb = a["usage"]["completion_tokens"], b["usage"]["completion_tokens"]
    dt = b["_wall"] - a["_wall"]
    tps = (nb - na) / dt if dt > 1e-6 and nb > na else float("nan")
    return {
        "tps": round(tps, 1) if tps == tps else None,
        "na": na, "nb": nb, "dt": round(dt, 3),
    }


# 「难」prompt：长文写作，内容不可预测 —— 草稿模型没法靠套路命中，
# 更接近真实代码/写作场景的接受率。贪心解码保证可比性。
HARD_PROMPT = (
    "请详细论述中国从秦朝到清朝中央集权制度的演变过程，"
    "逐朝分析其官制、选官方式与地方行政的变化，并说明每次变化的原因与后果。"
    "要求内容具体、条理清楚，尽量写长。"
)


def measure_speed_and_al(num_requests: int = 3, max_tokens: int = 768) -> dict:
    """测真实的解码吞吐 + 接受长度。

    关键点：**两类 prompt 都要测**。「从 1 数下去」几乎完全确定，草稿模型能近乎
    全中，是最好情况；用它得到的 tok/s 会把 DFlash 类大窗口草稿吹得很高，但真实
    写作/代码场景命中率完全不同。所以主结论要看 hard 那一组。
    """
    grow = "从 1 开始一直数下去，不要停。"

    bk.chat("热身，忽略这句话。", max_tokens=32)          # 触发 CUDA 图预热

    out: dict = {"draft_kind": draft_counters()["kind"]}

    # ---- 好情况（几乎确定的文本）----
    try:
        easy = diff_tps(grow, 64, 512)
        out["easy_tps"] = easy["tps"]
        out["easy_tps_detail"] = easy
    except Exception as exc:
        out["easy_tps_error"] = f"{type(exc).__name__}: {exc}"

    # ---- 难情况（长文写作，不可预测）----
    try:
        hard = diff_tps(HARD_PROMPT, 128, 512)
        out["hard_tps"] = hard["tps"]
        out["hard_tps_detail"] = hard
    except Exception as exc:
        out["hard_tps_error"] = f"{type(exc).__name__}: {exc}"

    # ---- 接受长度 + 该场景实测吞吐 ----
    for tag, prompt in (("easy", grow), ("hard", HARD_PROMPT)):
        try:
            out.update({f"{tag}_{k}": v for k, v in measure_al(prompt, num_requests, max_tokens).items()})
        except Exception as exc:
            out[f"{tag}_error"] = f"{type(exc).__name__}: {exc}"

    return out


def run_one(name: str, cache_quant: int, draft_model, note: str) -> dict:
    print(f"\n{'=' * 78}")
    print(f"▶ {name}   cache_quant={cache_quant}   draft={draft_model or 'MTP(内置)'}")
    print(f"  {note}")
    print("=" * 78)

    rel = bk.make_config(name, cache_quant, MAX_SEQ, draft_model=draft_model)
    bk.stop_service()
    t0 = time.time()
    try:
        bk.panel(f"/api/start?config={rel}", method="POST", timeout=120)
        bk.wait_status("running")
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        print(f"  ✗ 不可用：{msg}")
        return {"name": name, "ok": False, "error": msg, "note": note,
                "cache_quant": cache_quant, "draft_model": draft_model}

    load_s = time.time() - t0
    h = health()
    used = bk.smi_used()
    free = (h.get("gpu") or {}).get("vram_free_gb")
    print(f"  加载 {load_s:.1f}s   整卡已用 {used} GiB   空闲 {free} GiB   "
          f"draft_kind={h.get('draft_kind')}  窗口={h.get('draft_tokens_per_window')}")

    result: dict = {
        "name": name, "ok": True, "note": note,
        "cache_quant": cache_quant,
        "k_bits": h.get("cache_k_bits"), "v_bits": h.get("cache_v_bits"),
        "draft_model": draft_model,
        "draft_kind": h.get("draft_kind"),
        "load_s": round(load_s, 1),
        "smi_used_gib": used,
        "vram_free_gib": free,
        "kv_theory_gib": round(bk.kv_gib(MAX_SEQ, cache_quant), 2),
    }

    try:
        m = measure_speed_and_al()
        result.update(m)
        print(f"  差分解码：简单 prompt {m.get('easy_tps')} tok/s   "
              f"困难 prompt {m.get('hard_tps')} tok/s")
        for tag, label in (("easy", "简单(计数)"), ("hard", "困难(长文)")):
            al, ar = m.get(f"{tag}_accept_len"), m.get(f"{tag}_accept_rate")
            if al is None:
                continue
            print(f"    {label:<12} 接受长度 {al:>5}/{m.get(tag + '_al_ceiling')}  "
                  f"接受率 {ar}  实测 {m.get(tag + '_tps_direct')} tok/s  "
                  f"{m.get(tag + '_rounds_per_s')} 轮/s")
    except Exception as exc:
        result["error_speed"] = f"{type(exc).__name__}: {exc}"
        print(f"  ✗ 测量失败：{exc}")

    # 长文检索：确认草稿不影响正确性（DFlash2 号称无损，值得验一下）
    try:
        mn = bk.bench_multi_needle()
        result["multi_needle"] = {"hit": mn["hit"], "placed": mn["placed"],
                                  "prompt_tokens": mn["prompt_tokens"]}
        flag = "✓" if mn["hit"] == mn["placed"] else "✗"
        print(f"  {flag} 多针 {mn['hit']}/{mn['placed']}  "
              f"(上下文 {mn['prompt_tokens']} token)")
    except Exception as exc:
        result["error_needle"] = f"{type(exc).__name__}: {exc}"
        print(f"  ✗ 多针失败：{exc}")

    result["smi_used_after"] = bk.smi_used()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="DFlash2 vs MTP 对比评测")
    ap.add_argument("--only", default=None, help="只跑指定条目，逗号分隔")
    ap.add_argument("--keep", action="store_true",
                    help="结束后不恢复原配置（留在最后一档上）")
    args = ap.parse_args()

    try:
        bk.panel("/api/status", timeout=20)
    except Exception as exc:
        print(f"控制面板不可用（{bk.PANEL}）：{exc}")
        print("请先运行： .\\start_panel.ps1 -Background -NoBrowser")
        return 1

    todo = MATRIX
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        todo = [m for m in MATRIX if m[0] in want]

    print("=" * 78)
    print("DFlash2 草稿模型 vs 内置 MTP")
    print(f"上下文 {MAX_SEQ}   草稿 {DFLASH}")
    print("=" * 78)

    results = []
    for name, q, dm, note in todo:
        r = run_one(name, q, dm, note)
        results.append(r)
        BENCH_DIR.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf8")

    # ---------------- 汇总 ----------------
    print("\n\n" + "=" * 112)
    print("汇总")
    print("=" * 112)
    print(f"{'配置':<13}{'KV':<6}{'草稿':<9}{'KV理论':>8}{'整卡已用':>10}"
          f"{'空闲':>7}{'易tok/s':>9}{'难tok/s':>9}{'轮/s':>7}"
          f"{'接受(易)':>9}{'接受(难)':>9}{'多针':>7}")
    print("-" * 112)
    base = next((r for r in results if r["name"] == "cq4_mtp" and r.get("ok")), None)
    for r in results:
        if not r.get("ok"):
            print(f"{r['name']:<13}启动失败：{r.get('error', '')[:50]}")
            continue
        nd = r.get("multi_needle")
        nd_s = f"{nd['hit']}/{nd['placed']}" if nd else "-"
        print(f"{r['name']:<13}{str(r['cache_quant'])+'bit':<6}"
              f"{str(r.get('draft_kind') or 'mtp'):<9}"
              f"{r['kv_theory_gib']:>7.1f}G{str(r.get('smi_used_gib')):>10}"
              f"{str(r.get('vram_free_gib')):>7}{str(r.get('easy_tps')):>9}"
              f"{str(r.get('hard_tps')):>9}{str(r.get('hard_rounds_per_s')):>7}"
              f"{str(r.get('easy_accept_len')):>9}{str(r.get('hard_accept_len')):>9}"
              f"{nd_s:>7}")
    print("-" * 112)

    if base:
        print(f"\n相对基线 cq4_mtp（易 {base.get('easy_tps')} / 难 {base.get('hard_tps')} tok/s，"
              f"难接受 {base.get('hard_accept_len')}）：")
        for r in results:
            if r["name"] == "cq4_mtp" or not r.get("ok"):
                continue
            bt, rt = base.get("hard_tps"), r.get("hard_tps")
            if bt and rt:
                delta = f"{rt - bt:+7.1f}  {(rt - bt) / bt * 100:+6.1f}%"
            else:
                delta = "     n/a"
            nd = r.get("multi_needle") or {}
            nd_s = f"{nd.get('hit')}/{nd.get('placed')}" if nd else "-"
            print(f"  {r['name']:<13} 难场景 {str(rt):>6} tok/s  {delta}   "
                  f"空闲 {str(r.get('vram_free_gib')):>5} GiB   "
                  f"难接受 {str(r.get('hard_accept_len')):>5}   多针 {nd_s}")

    print(f"\n结果已存：{OUT}")

    if args.keep:
        print("（--keep：服务留在当前配置上，未恢复）")
        return 0

    print("\n恢复原配置并重启…")
    bk.stop_service()
    bk.panel("/api/start", method="POST", timeout=120)
    bk.wait_status("running")
    h = health()
    print(f"已恢复：draft_kind={h.get('draft_kind')}  cache_quant={h.get('cache_quant')}  "
          f"空闲 {h.get('gpu', {}).get('vram_free_gb')} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
