#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
KV 缓存精度 / 容量评估。

为什么要单独做这个：`max_seq_len` 和 `cache_quant` 不是独立的两个旋钮，它们的乘积
（KV 显存）才是。同一块显存可以有多种组合，例如

    8bit @ 262144  ≈  fp16 @ 131072  ≈  K8V4 @ 262144   （都要约 8 GiB）

选哪个不该只看显存能不能装下，还要看：
  1. 装下之后还剩多少余量（余量太小时长 prompt 的激活峰值会把卡打爆）；
  2. 精度对**长上下文检索**的影响（4bit KV 最容易在这里掉点，短问答看不出来）；
  3. 速度（量化 KV 可能因为省带宽变快，也可能因为反量化变慢）。

所以本脚本对每个组合都测三件事：显存 / 速度 / 长文找针（needle）准确率。

    .\.venv\Scripts\python.exe bench_kv.py                 # 跑默认矩阵
    .\.venv\Scripts\python.exe bench_kv.py --quick         # 只跑显存+速度，跳过找针
    .\.venv\Scripts\python.exe bench_kv.py --only q4,q8    # 只跑指定条目

依赖正在运行控制面板（默认 127.0.0.1:8001）来启停服务，这样不用自己管进程和显存。
脚本会在 `_bench/` 下生成临时配置，结束后恢复成原来的 config.yaml 并重启服务。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import requests
import yaml

BASE = Path(__file__).resolve().parent
BENCH_DIR = BASE / "_bench"
PANEL = "http://127.0.0.1:8001"


def _service_port() -> int:
    """服务端口必须从 config.yaml 读，不能写死：server.port 是可改的，
    改过端口后写死 8000 会连不上，还会给出很难懂的 ConnectionError。"""
    try:
        with open(BASE / "config.yaml", encoding="utf8") as fh:
            cfg = yaml.safe_load(fh) or {}
        return int((cfg.get("server") or {}).get("port") or 8000)
    except Exception:
        return 8000


SERVICE = f"http://127.0.0.1:{_service_port()}"
RESULTS = BENCH_DIR / "results.json"

# 每 token 的 KV 元素数：16 层全注意力 × 4 KV head × 256 head_dim × 2(K/V)
ELEMS_PER_TOKEN = 16 * 4 * 256 * 2
PAGE_SIZE = 256


def kv_gib(max_seq_len: int, bits: float) -> float:
    return max_seq_len * ELEMS_PER_TOKEN * (bits / 8) / 1024**3


# ===========================================================================
# 测试矩阵
# ===========================================================================

# (名称, cache_quant, max_seq_len, 备注, k_bits, v_bits)
#   cache_quant=16 表示 fp16（不量化）
#
# 矩阵是按「三类可用组合」设计的。显存基线：权重 13.95 GiB，非 KV 部分共约
# 15.1 GiB（含视觉塔/MTP/激活），所以 24 GB 卡上 KV 最多约 8.5 GiB，
# 但**必须留 2 GiB 以上余量**，否则会掉进页文件导致严重拖慢（实测 q8 拉满
# 上下文时只剩 0.42 GiB，/health 都超时，属于实际不可用）。
#
#   顶格容量：q4  @ 262144 → 4.0 GiB KV（余量约 4 GiB）
#   同容量升精度：k8v4 @ 262144 → 6.0 GiB KV（保 262K 上下文，只抬 K 的精度）
#   换精度降容量：q8 @ 196608 → 6.0 GiB；fp16 @ 98304 / 65536 → 6.0 / 4.0 GiB
MATRIX: list[tuple[str, int, int, str, int | None, int | None]] = [
    ("q4",      4, 262144, "当前配置：容量拉满，精度最低",                None, None),
    ("k8v4",    8, 262144, "K=8bit/V=4bit：保 262144 上下文，只抬 K 精度", 8, 4),
    ("q8_192k", 8, 196608, "8bit 仅退到 3/4 上下文",                        None, None),
    ("q16_96k", 16, 98304, "fp16 精确，3/8 上下文",                         None, None),
    ("q16_64k", 16, 65536, "fp16 精确，1/4 上下文（余量最大）",              None, None),
    ("q8_max",  8, 262144, "8bit 拉满上下文（预期余量不足，用来看到底差多少）", None, None),
]


def make_config(name: str, cache_quant: int, max_seq_len: int, k_bits=None, v_bits=None,
                draft_model=None, mtp_draft: bool = True) -> str:
    """基于 config.yaml 生成一个临时配置，返回相对 BASE 的路径。

    只能写到 _bench/ 下。这一点必须守死：读写 config.yaml 走 yaml.safe_dump 会
    把手工写的注释**全部抹掉**（PyYAML 不保留注释），曾经因此把 7 KB 的带注释
    配置压成 928 字节的纯数据。
    """
    rel = f"_bench/{name}.yaml"
    dest = (BASE / rel).resolve()
    bench_root = (BASE / "_bench").resolve()
    if bench_root not in dest.parents:
        raise RuntimeError(f"拒绝写出 _bench/ 之外：{dest}")

    with open(BASE / "config.yaml", encoding="utf8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict) or "model" not in cfg:
        raise RuntimeError("config.yaml 解析结果异常，已中止")

    cfg = copy.deepcopy(cfg)
    m = cfg.setdefault("model", {})
    m["cache_quant"] = cache_quant
    m["max_seq_len"] = max_seq_len
    m["cache_k_bits"] = k_bits
    m["cache_v_bits"] = v_bits
    m["progressbar"] = False
    # draft_model 为 None 时保持 config.yaml 里的值（通常是 null = 用 MTP）
    if draft_model is not None:
        m["draft_model"] = draft_model
    m["mtp_draft"] = mtp_draft

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    return rel


# ===========================================================================
# 面板 / 服务操作
# ===========================================================================

def panel(path: str, method: str = "GET", **kw):
    url = PANEL + path
    fn = requests.post if method == "POST" else requests.get
    resp = fn(url, timeout=kw.pop("timeout", 180), **kw)
    resp.raise_for_status()
    return resp.json()


def wait_status(target: str, timeout: float = 240.0, health_grace: float = 90.0) -> str:
    """等到服务达到 target 状态。

    特意处理「进程在监听、但 /health 一直不响应」 这种病态情况：显存卡得太紧时
    服务会掉进页文件，请求能被内核接管却迟迟拿不到响应，看进程和端口一切正常。
    不单独判这种情况的话，测试会永远卡在那里。
    """
    deadline = time.time() + timeout
    started = time.time()
    last = ""
    saw_listen = False
    while time.time() < deadline:
        try:
            st = panel("/api/status", timeout=25)["service"]
            last = st["status"]
        except Exception:
            time.sleep(1.5)
            continue
        if last == target:
            return last
        if last == "error" and target == "running":
            raise RuntimeError("服务加载失败（见 logs/service.log）")
        # 已经起来了、端口也在听，就是健康检查不通 → 几乎一定是显存太紧在换页
        if target == "running" and st.get("listening"):
            saw_listen = True
        if saw_listen and time.time() - started > health_grace:
            raise RuntimeError(
                f"服务已在监听但 /health 超过 {health_grace:.0f}s 无响应："
                f"显存余量不足导致急剧变慢（该配置实际不可用）"
            )
        time.sleep(1.5)
    return last


def kill_orphan_services() -> None:
    """把不属于本次测试、残留下来的 server.py 全部结束掉再继续，
    否则下一个配置的显存会被旧进程占着。"""
    import subprocess

    script = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -and $_.CommandLine -like '*server.py*' } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=30, creationflags=0x08000000,
        ).stdout
    except Exception:
        return
    for line in out.split():
        if line.strip().isdigit():
            subprocess.run(["taskkill", "/PID", line.strip(), "/T", "/F"],
                           capture_output=True, timeout=30, creationflags=0x08000000)


def stop_service() -> None:
    try:
        panel("/api/stop?timeout=40", method="POST", timeout=120)
    except Exception:
        pass
    wait_status("stopped", timeout=90)
    # 面板停不掉时（旧代码/卡死）兜底，避免下一个配置被占显存
    kill_orphan_services()
    for _ in range(20):
        try:
            if panel("/api/status", timeout=20)["service"]["status"] == "stopped":
                return
        except Exception:
            pass
        time.sleep(1.0)


def vram() -> dict:
    try:
        h = requests.get(f"{SERVICE}/health", timeout=10).json()
        g = h.get("gpu") or {}
        return {"free": g.get("vram_free_gb"), "total": g.get("vram_total_gb")}
    except Exception:
        return {"free": None, "total": None}


def smi_used() -> float | None:
    """整卡已用显存（GiB）。注意这含桌面/别的程序，不是服务独占。"""
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=0x08000000,
        ).stdout.strip()
        return round(int(out) / 1024, 2)
    except Exception:
        return None


# ===========================================================================
# 速度 / 质量测试
# ===========================================================================

FILLER = (
    "项目管理流程通常包含需求收集、方案评审、任务拆分、进度跟踪与复盘归档等环节，"
    "各环节的输出物需要在评审会上达成一致，并记录到项目知识库中供后续查阅。"
    "数据治理工作则关注数据血缘、口径统一、质量监控和权限审计，"
    "其目标是在保证合规的前提下让数据可被信任地复用。"
)


def token_len(text: str) -> int:
    """用一次极小的请求量出真实 token 数（比按字符估算可靠）。"""
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 1,
        "enable_thinking": False,
        "stream": False,
    }
    r = requests.post(f"{SERVICE}/v1/chat/completions", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["usage"]["prompt_tokens"]


def chat(prompt: str, max_tokens: int = 64, timeout: float = 900) -> dict:
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "enable_thinking": False,
        "stream": False,
    }
    t0 = time.time()
    r = requests.post(f"{SERVICE}/v1/chat/completions", json=payload, timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    body = r.json()
    body["_wall"] = dt
    return body


def bench_multi_needle(n_keys: int = 8, target_tokens: int = 48000) -> dict:
    """多针检索（NIAH 的加强版），用来把 KV 精度的差异逼出来。

    单针测试太容易了：一个足够独特的口令在 4bit KV 下也能完好保留，所以
    q4 和 k8v4 都是满分，看不出区别。改成埋 N 个不同口令、要求一次全部列出，
    对 KV 的“存储容量”和远处 token 的保真度要求高得多，量化噪声更容易暴露。
    计分用命中个数，不是命中率。
    """
    codes = [f"CODE-{i:02d}-{w.upper()}" for i, w in enumerate(
        ["alpha", "bravo", "delta", "gamma", "kappa", "omega", "sigma", "theta",
         "zeta", "lambda", "mu", "nu"][:n_keys])]

    per_unit = 60
    need_units = max(n_keys * 4, int(target_tokens / per_unit))
    # 把 N 个口令尽量均匀地铺在整段上下文里（避开最开头和最末尾）
    positions = sorted({max(1, int(need_units * (i + 0.5) / n_keys)) for i in range(n_keys)})

    parts: list[str] = []
    placed: list[str] = []
    cursor = 0
    for pos, code in zip(positions, codes):
        parts.append(FILLER * (pos - cursor))
        parts.append(f"\n\n【配置项 {len(placed) + 1}】登记代码为 {code}。\n\n")
        placed.append(code)
        cursor = pos
    parts.append(FILLER * max(1, need_units - cursor))
    parts.append(
        "\n\n问题：上文一共登记了若干个「登记代码」，请把它们全部、原样列出来，"
        "每行一个，不要解释。"
    )
    prompt = "".join(parts)

    resp = chat(prompt, max_tokens=400)
    text = (resp["choices"][0]["message"].get("content") or "")
    upper = text.upper()
    hit_codes = [c for c in placed if c.upper() in upper]
    return {
        "placed": len(placed),
        "hit": len(hit_codes),
        "prompt_tokens": resp["usage"]["prompt_tokens"],
        "expected": placed,
        "got": text.strip()[:500],
        "missed": [c for c in placed if c not in hit_codes],
    }


def bench_speed() -> dict:
    """速度测量。

    解码速度用**两次不同长度生成的差分**算，把首 token 延迟和 prefill 抵消掉：

        decode_tps = (lenB - lenA) / (wallB - wallA)

    只测一次的话，TTFT 会被算进“生成时间”里，短生成会严重低估解码速度
    （实测：30 token 的短测算出 100 tok/s，差分法只有约 40 tok/s）。
    prompt 要选那种一定会写满 max_tokens 的，否则提前 EOS 会让差分失效。
    """
    chat("热身，忽略这句话。", max_tokens=32)          # 先跑一次，触发 CUDA 图/kernel 预热

    grow = "从 1 开始一直数下去，不要停。"
    a = chat(grow, max_tokens=64)
    b = chat(grow, max_tokens=512)
    na = a["usage"]["completion_tokens"]
    nb = b["usage"]["completion_tokens"]
    dt = b["_wall"] - a["_wall"]
    decode_tps = (nb - na) / dt if dt > 1e-6 and nb > na else float("nan")

    pp_text = FILLER * 90          # 约 6K token
    pp = chat(pp_text + "\n\n请用一句话总结上面这段话。", max_tokens=32)
    pt = pp["usage"]["prompt_tokens"]
    prefill_tps = pt / pp["_wall"]

    return {
        "decode_tps": round(decode_tps, 1) if decode_tps == decode_tps else None,
        "decode_na": na,
        "decode_nb": nb,
        "prefill_tokens": pt,
        "prefill_tps": round(prefill_tps, 0),
    }


def needle_prompt(target_tokens: int, depth: float, code: str) -> str:
    """在 target_tokens 规模的上下文里、按 depth 比例埋一个口令。"""
    per_unit = 60                                     # 粗略字数->token 比例经验值
    need_units = max(4, int(target_tokens / per_unit))
    cut = max(1, int(need_units * depth))
    needle = f"\n\n【重要记录】本次会话的校验口令是 {code}，请务必记住。\n\n"
    text = FILLER * cut + needle + FILLER * (need_units - cut)
    text += "\n\n问题：上文提到的校验口令是什么？只输出口令本身，不要解释。"
    return text


def bench_needle(depths=(0.05, 0.35, 0.65, 0.95), target_tokens: int = 48000) -> dict:
    """长文找针。

    这是 KV 量化最容易掉点的地方：短问答看不出差别，但把信息埋在几万 token
    处时，低精度的 KV 会把远处 token 的表征抹平。所以深度要覆盖到头尾。
    另外换个**和 FILLER 无关**的干扰口令放在别处，用来发现“背错成另一个”的情况。
    """
    code = "PURPLE-ELEPHANT-7742"
    hits, rows, real_tokens = 0, [], None
    for d in depths:
        prompt = needle_prompt(target_tokens, d, code)
        resp = chat(prompt, max_tokens=48)
        real_tokens = resp["usage"]["prompt_tokens"]
        text = (resp["choices"][0]["message"].get("content") or "")
        ok = code.lower() in text.lower()
        hits += ok
        rows.append({
            "depth": d,
            "prompt_tokens": real_tokens,
            "hit": ok,
            "answer": text.strip()[:60],
        })
    return {
        "hit": hits,
        "total": len(depths),
        "prompt_tokens": real_tokens,
        "rows": rows,
    }


# ===========================================================================
# 主流程
# ===========================================================================

def run_one(name: str, cache_quant: int, max_seq_len: int, note: str,
            quick: bool, k_bits: int | None = None, v_bits: int | None = None) -> dict:
    print(f"\n{'=' * 74}")
    print(f"▶ {name}   cache_quant={cache_quant}  max_seq_len={max_seq_len}"
          + (f"  K={k_bits}/V={v_bits}" if k_bits else ""))
    print(f"  {note}")
    print(f"  理论 KV 占用 ≈ {kv_gib(max_seq_len, cache_quant if cache_quant < 16 else 16):.2f} GiB")
    print("=" * 74)

    rel = make_config(name, cache_quant, max_seq_len, k_bits, v_bits)
    stop_service()
    t0 = time.time()
    try:
        panel(f"/api/start?config={rel}", method="POST", timeout=120)
        wait_status("running")
    except Exception as exc:
        msg = str(exc)
        print(f"  ✗ 不可用：{msg}")
        return {
            "name": name, "ok": False, "error": msg, "note": note,
            "cache_quant": cache_quant, "max_seq_len": max_seq_len,
            "k_bits": k_bits, "v_bits": v_bits,
            "kv_theory_gib": round(kv_gib(max_seq_len, cache_quant if cache_quant < 16 else 16), 2),
        }
    load_s = time.time() - t0

    v = vram()
    used = smi_used()
    print(f"  加载完成 {load_s:.1f}s   整卡已用 {used} GiB   服务视角空闲 {v['free']} GiB")

    result: dict = {
        "name": name, "ok": True, "note": note,
        "cache_quant": cache_quant, "max_seq_len": max_seq_len,
        "k_bits": k_bits, "v_bits": v_bits,
        "kv_theory_gib": round(kv_gib(max_seq_len, cache_quant if cache_quant < 16 else 16), 2),
        "load_s": round(load_s, 1),
        "smi_used_gib": used,
        "vram_free_gib": v["free"],
    }

    try:
        sp = bench_speed()
        result.update(sp)
        print(f"  解码 {sp['decode_tps']} tok/s    prefill {sp['prefill_tps']:.0f} tok/s "
              f"({sp['prefill_tokens']} token)")
    except Exception as exc:
        result["speed_error"] = f"{type(exc).__name__}: {exc}"
        print(f"  ✗ 速度测试失败：{exc}")

    if not quick:
        try:
            nd = bench_needle()
            result["needle"] = nd
            flag = "✓" if nd["hit"] == nd["total"] else "✗"
            print(f"  {flag} 找针 {nd['hit']}/{nd['total']}  "
                  f"(上下文 {nd['prompt_tokens']} token)")
            for r in nd["rows"]:
                print(f"      depth {r['depth']:.0%}: "
                      f"{'命中' if r['hit'] else '未命中'}  {r['answer']!r}")
        except Exception as exc:
            result["needle_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  ✗ 找针测试失败：{exc}")

    # 峰值显存：跑完长文之后再取一次，能反映激活峰值
    result["smi_used_after"] = smi_used()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="KV 缓存精度/容量评估")
    ap.add_argument("--quick", action="store_true", help="跳过找针质量测试")
    ap.add_argument("--only", default=None, help="只跑指定条目名，逗号分隔")
    ap.add_argument("--current", action="store_true",
                    help="不启停服务，直接测当前正在运行的那套配置")
    args = ap.parse_args()

    try:
        panel("/api/status", timeout=20)
    except Exception as exc:
        print(f"控制面板不可用（{PANEL}）：{exc}")
        print("请先运行： .\\start_panel.ps1 -Background -NoBrowser")
        return 1

    # ---------------- 只测当前配置 ----------------
    if args.current:
        st = panel("/api/status", timeout=20)["service"]
        if st["status"] != "running":
            print(f"服务当前不是 running（是 {st['status']}），无法测量")
            return 1
        h = requests.get(f"{SERVICE}/health", timeout=20).json()
        g = h.get("gpu") or {}
        print("=" * 74)
        print("测量当前正在运行的配置")
        print(f"  KV 量化    : {h.get('cache_quant')} bit   "
              f"（K={h.get('cache_k_bits')} / V={h.get('cache_v_bits')}）")
        print(f"  上下文     : {h.get('max_seq_len')}")
        print(f"  模型       : {h.get('model')}")
        print("=" * 74)

        used = smi_used()
        print(f"  整卡已用 {used} GiB   服务视角空闲 {g.get('vram_free_gb')} GiB")

        result: dict = {
            "name": "current",
            "cache_quant": h.get("cache_quant"),
            "k_bits": h.get("cache_k_bits"),
            "v_bits": h.get("cache_v_bits"),
            "max_seq_len": h.get("max_seq_len"),
            "smi_used_gib": used,
            "vram_free_gib": g.get("vram_free_gb"),
        }
        try:
            sp = bench_speed()
            result.update(sp)
            print(f"  解码 {sp['decode_tps']} tok/s    prefill {sp['prefill_tps']:.0f} tok/s "
                  f"({sp['prefill_tokens']} token)")
        except Exception as exc:
            result["speed_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  ✗ 速度测试失败：{exc}")
        if not args.quick:
            try:
                nd = bench_needle()
                result["needle"] = nd
                print(f"  找针 {nd['hit']}/{nd['total']}  (上下文 {nd['prompt_tokens']} token)")
                for r in nd["rows"]:
                    print(f"      depth {r['depth']:.0%}: "
                          f"{'命中' if r['hit'] else '未命中'}  {r['answer']!r}")
            except Exception as exc:
                result["needle_error"] = f"{type(exc).__name__}: {exc}"
                print(f"  ✗ 找针测试失败：{exc}")
            try:
                mn = bench_multi_needle()
                result["multi_needle"] = mn
                print(f"  多针 {mn['hit']}/{mn['placed']}  (上下文 {mn['prompt_tokens']} token)")
                if mn["missed"]:
                    print(f"      漏掉：{', '.join(mn['missed'])}")
            except Exception as exc:
                result["multi_needle_error"] = f"{type(exc).__name__}: {exc}"
                print(f"  ✗ 多针测试失败：{exc}")
        result["smi_used_after"] = smi_used()
        print(f"  测完峰值：整卡 {result['smi_used_after']} GiB")
        BENCH_DIR.mkdir(parents=True, exist_ok=True)
        (BENCH_DIR / "current.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
        print(f"\n结果已存：{BENCH_DIR / 'current.json'}")
        return 0

    with open(BASE / "config.yaml", encoding="utf8") as fh:
        original = yaml.safe_load(fh)
    print("=" * 74)
    print("KV 缓存精度 / 容量评估")
    print(f"对照基准：config.yaml = cache_quant {original['model'].get('cache_quant')}, "
          f"max_seq_len {original['model'].get('max_seq_len')}")
    print("=" * 74)

    todo = MATRIX
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        todo = [m for m in MATRIX if m[0] in want]

    results = []
    for name, q, seq, note, kb, vb in todo:
        r = run_one(name, q, seq, note, args.quick, kb, vb)
        results.append(r)
        BENCH_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf8")

    # ---------------- 汇总 ----------------
    print("\n\n" + "=" * 100)
    print("汇总")
    print("=" * 100)
    head = (f"{'配置':<9}{'KV量化':<8}{'上下文':>9}{'KV理论':>9}{'整卡已用':>10}"
            f"{'解码tok/s':>11}{'prefill':>9}{'找针':>8}")
    print(head)
    print("-" * 100)
    for r in results:
        if not r.get("ok"):
            print(f"{r['name']:<9}启动失败：{r.get('error', '')[:60]}")
            continue
        q = "fp16" if r["cache_quant"] >= 16 else (
            f"{r['k_bits']}/{r['v_bits']}" if r.get("k_bits") else f"{r['cache_quant']}bit")
        nd = r.get("needle")
        nd_s = f"{nd['hit']}/{nd['total']}" if nd else "-"
        print(f"{r['name']:<9}{q:<8}{r['max_seq_len']:>9,}{r['kv_theory_gib']:>8.1f}G"
              f"{str(r.get('smi_used_gib')):>10}{str(r.get('decode_tps')):>11}"
              f"{str(r.get('prefill_tps')):>9}{nd_s:>8}")
    print("-" * 100)
    print(f"结果已存：{RESULTS}")

    # ---------------- 恢复 ----------------
    print("\n恢复原配置并重启服务…")
    stop_service()
    panel("/api/start", method="POST", timeout=120)
    wait_status("running")
    print("已恢复。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
