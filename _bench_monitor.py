#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""看一眼 bench_kv.py 跑到哪儿了（避免在 PowerShell 里写嵌套引号）。"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "_bench" / "results.json"

if not RESULTS.is_file():
    print("还没有结果文件（第一条还没跑完）")
    raise SystemExit(0)

rows = json.loads(RESULTS.read_text(encoding="utf8"))
print(f"{'配置':<9}{'状态':<8}{'KV量化':<9}{'上下文':>9}{'整卡GiB':>9}"
      f"{'解码':>7}{'prefill':>9}{'找针':>7}{'空闲':>7}")
print("-" * 78)
for r in rows:
    if not r.get("ok"):
        print(f"{r['name']:<9}{'失败':<8}{'':<9}{r.get('max_seq_len', 0):>9,}"
              f"{'':>9}{'':>7}{'':>9}{'':>7}  {r.get('error', '')[:40]}")
        continue
    q = "fp16" if r["cache_quant"] >= 16 else (
        f"{r['k_bits']}/{r['v_bits']}" if r.get("k_bits") else f"{r['cache_quant']}bit")
    nd = r.get("needle")
    nd_s = f"{nd['hit']}/{nd['total']}" if nd else "-"
    free = r.get("vram_free_gib")
    print(f"{r['name']:<9}{'ok':<8}{q:<9}{r['max_seq_len']:>9,}"
          f"{str(r.get('smi_used_gib')):>9}{str(r.get('decode_tps')):>7}"
          f"{str(r.get('prefill_tps')):>9}{nd_s:>7}{str(free):>7}")
    for key in ("speed_error", "needle_error"):
        if r.get(key):
            print(f"          {key}: {r[key][:70]}")
print()
print(f"共 {len(rows)} 条；文件：{RESULTS}")
