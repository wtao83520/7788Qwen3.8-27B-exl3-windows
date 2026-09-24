#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""DFlash2 草稿模型二选一：4.00bpw 还是 5.0bpw？

关键前提（决定了这个问题怎么答）：
    DFlash2 的验证是**精确匹配才接受**（见 architecture/dflash2.py 的
    sample_from_state 注释 "the target verifier still samples normally and
    accepts only exact matches"）。这是标准的无损投机解码：**草稿模型的精度
    不影响输出质量**，只影响接受率（也就是速度）。

    所以「哪个最佳」等价于「哪个速度/显存更划算」，不需要比质量。

两类 prompt 都测，因为接受长度对 prompt 极度敏感：
    easy = 「从 1 数下去」几乎确定，草稿几乎全中（好看但没代表性）
    hard = 长文写作，不可预测（接近真实使用）

    .\.venv\Scripts\python.exe _bench\draft_ab.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "_bench"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

import requests

import bench_kv as bk
import bench_draft as bd

OUT = BASE / "_bench" / "draft_ab.json"
CQ = 4          # 用户确认就用 4
MAX_SEQ = 262144

CANDS = [
    ("dflash_4bpw", "models/DFlash2-EXL3-4.00bpw", "4.00bpw / 1.08 GiB 磁盘"),
    ("dflash_5bpw", "models/DFlash2-EXL3-5.0bpw",  "5.00bpw / 1.37 GiB 磁盘"),
]


def run(name: str, draft: str, note: str) -> dict:
    print(f"\n{'=' * 76}")
    print(f"▶ {name}   {draft}   ({note})")
    print("=" * 76)

    rel = bk.make_config(name, CQ, MAX_SEQ, draft_model=draft)
    bk.stop_service()
    t0 = time.time()
    try:
        bk.panel(f"/api/start?config={rel}", method="POST", timeout=120)
        bk.wait_status("running")
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        print(f"  ✗ 不可用：{msg}")
        return {"name": name, "draft": draft, "ok": False, "error": msg}
    load_s = time.time() - t0

    h = requests.get(f"{bk.SERVICE}/health", timeout=30).json()
    free = (h.get("gpu") or {}).get("vram_free_gb")
    used = bk.smi_used()
    print(f"  加载 {load_s:.1f}s   整卡 {used} GiB   空闲 {free} GiB   "
          f"draft_kind={h.get('draft_kind')} 窗口={h.get('draft_tokens_per_window')}")

    r: dict = {"name": name, "draft": draft, "ok": True, "note": note,
               "load_s": round(load_s, 1), "smi_used_gib": used,
               "vram_free_gib": free,
               "draft_kind": h.get("draft_kind"),
               "draft_window": h.get("draft_tokens_per_window")}

    bk.chat("热身，忽略这句话。", max_tokens=32)   # CUDA 图预热

    # 接受长度：好 / 难两类 prompt 各跑 3 次
    for tag, prompt in (("easy", "从 1 开始一直数下去，不要停。"), ("hard", bd.HARD_PROMPT)):
        try:
            m = bd.measure_al(prompt, num_requests=3, max_tokens=768)
            r.update({f"{tag}_{k}": v for k, v in m.items()})
            print(f"  {tag:>5}: 接受长度 {m['accept_len']}/{m['al_ceiling']}  "
                  f"接受率 {m['accept_rate']}  实测 {m['tps_direct']} tok/s  "
                  f"{m['rounds_per_s']} 轮/s")
        except Exception as exc:
            r[f"{tag}_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  ✗ {tag} 测量失败：{exc}")

    # 热身后的差分吞吐（难 prompt）
    try:
        d = bd.diff_tps(bd.HARD_PROMPT, 128, 512)
        r["hard_diff_tps"] = d["tps"]
        print(f"  差分吞吐（难 prompt）: {d['tps']} tok/s")
    except Exception as exc:
        r["hard_diff_error"] = f"{type(exc).__name__}: {exc}"

    r["smi_used_after"] = bk.smi_used()
    return r


def main() -> int:
    try:
        bk.panel("/api/status", timeout=20)
    except Exception as exc:
        print(f"控制面板不可用：{exc}")
        return 1

    print("=" * 76)
    print("DFlash2 4.00bpw vs 5.0bpw（cache_quant 4 / 262144）")
    print("草稿精度不影响输出质量（精确匹配才接受），所以只比速度与显存")
    print("=" * 76)

    results = []
    for name, draft, note in CANDS:
        r = run(name, draft, note)
        results.append(r)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf8")

    print("\n\n" + "=" * 84)
    print("汇总")
    print("=" * 84)
    print(f"{'草稿':<14}{'空闲GiB':>9}{'接受(易)':>10}{'接受(难)':>10}"
          f"{'轮/s(难)':>10}{'tok/s(难)':>11}{'加载s':>8}")
    print("-" * 84)
    for r in results:
        if not r.get("ok"):
            print(f"{r['name']:<14}启动失败：{r.get('error', '')[:50]}")
            continue
        print(f"{r['name']:<14}{str(r.get('vram_free_gib')):>9}"
              f"{str(r.get('easy_accept_len')):>10}{str(r.get('hard_accept_len')):>10}"
              f"{str(r.get('hard_rounds_per_s')):>10}{str(r.get('hard_tps_direct')):>11}"
              f"{str(r.get('load_s')):>8}")
    print("-" * 84)

    ok = [r for r in results if r.get("ok")]
    if len(ok) == 2:
        a, b = ok
        for key, label in (("hard_accept_len", "难场景接受长度"),
                           ("hard_rounds_per_s", "难场景验证轮/秒"),
                           ("hard_tps_direct", "难场景吞吐 tok/s")):
            va, vb = a.get(key), b.get(key)
            if va is None or vb is None:
                continue
            better = a["name"] if va > vb else b["name"]
            print(f"  {label:<20} {a['name']} {va:>7}  vs  {b['name']} {vb:>7}"
                  f"   → {better} 更好 ({abs(va - vb):.2f} 差)")
        print(f"  空闲显存              {a['name']} {a.get('vram_free_gib'):>7} GiB"
              f"  vs  {b['name']} {b.get('vram_free_gib'):>7} GiB")

    print(f"\n结果已存：{OUT}")

    print("\n恢复 config.yaml 的原始配置（dflash 4.00bpw）并重启…")
    bk.stop_service()
    bk.panel("/api/start", method="POST", timeout=120)
    bk.wait_status("running")
    h = requests.get(f"{bk.SERVICE}/health", timeout=30).json()
    print(f"已恢复：draft_kind={h.get('draft_kind')}  "
          f"cache_quant={h.get('cache_quant')}  "
          f"空闲 {(h.get('gpu') or {}).get('vram_free_gb')} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
