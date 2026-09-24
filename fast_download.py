#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
多连接高速下载器：把 HuggingFace 分支的文件按需切成很多区间并行拉取。

为什么需要它：国内访问 HF 的单连接速度只有 ~40 kB/s，但服务端允许很多并发连接，
实测 96 连接可以把总速度提到 ~5.9 MB/s（16 小时 → 约 40 分钟）。连接数不是越多越好，
128 连接会被服务端拒连（ConnectionError），96 是实测甜点。

防重复启动：输出目录下会写一个 .download.lock（内容是 PID）。两个进程同时写同一个
safetensors 会把文件写坏，所以检测到已有活着的实例时会直接退出。

    # 自动探测最快的源并下载全部文件
    .\.venv\Scripts\python.exe fast_download.py

    # 指定源与连接数
    .\.venv\Scripts\python.exe fast_download.py --source hf-direct --conns 64
    .\.venv\Scripts\python.exe fast_download.py --source hf-mirror --conns 32
    .\.venv\Scripts\python.exe fast_download.py --probe-only     # 只测速不下载

    # 下载别的仓库（比如 DFlash2 草稿模型）：仓库名 + 分支
    .\.venv\Scripts\python.exe fast_download.py --repo <owner/name> --revision main \
        --out models/<name> --source hf-direct

支持断点续传：进度记录在 <目标文件>.part.json 里，中断后重跑同一条命令即可。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

import requests

REPO_ID = "turboderp/Qwen3.8-27B-exl3"
DEFAULT_REVISION = "3.50bpw"

# 各源的 URL 前缀。{} 里是 <repo>/resolve/<rev>/<file>
SOURCES = {
    "hf-direct": ("https://huggingface.co/", True),      # 走系统代理
    "hf-mirror": ("https://hf-mirror.com/", False),      # 国内镜像
    "aifasthub": ("https://aifasthub.com/", False),
}

CHUNK = 1 << 18  # 256 KiB


# ===========================================================================
# 工具
# ===========================================================================

def human(n: float) -> str:
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n:.0f} B"


class Progress:
    """全局进度统计（线程安全）。"""

    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.last_print = 0.0

    def add(self, n: int) -> None:
        with self.lock:
            self.done += n

    def maybe_print(self, force: bool = False) -> None:
        now = time.time()
        with self.lock:
            if not force and now - self.last_print < 2.0:
                return
            self.last_print = now
            done, total = self.done, self.total
        elapsed = max(1e-6, now - self.t0)
        speed = done / elapsed
        pct = done / total * 100 if total else 0
        eta = (total - done) / speed if speed > 0 else 0
        bar_len = 28
        filled = int(bar_len * pct / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(
            f"\r  [{bar}] {pct:5.1f}%  {human(done)}/{human(total)}  "
            f"{speed / 1024 / 1024:.2f} MB/s  剩余 {eta / 3600:.2f} h   ",
            end="",
            flush=True,
        )


# ===========================================================================
# 文件列表
# ===========================================================================

def list_files(source: str, revision: str, repo: str = REPO_ID, timeout: float = 30.0) -> list[dict]:
    """列出分支下所有文件（含大小）。"""
    base, use_proxy = SOURCES[source]
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true"
    proxies = None if use_proxy else {"http": None, "https": None}
    r = requests.get(url, timeout=timeout, proxies=proxies)
    r.raise_for_status()
    return [{"path": f["path"], "size": f["size"]} for f in r.json() if f.get("type") == "file"]


# ===========================================================================
# 单文件多连接下载
# ===========================================================================

class FileDownloader:
    def __init__(self, url: str, dest: str, size: int, conns: int, use_proxy: bool,
                 progress: Progress, seg_mb: int = 4):
        self.url = url
        self.dest = dest
        self.size = size
        self.conns = max(1, conns)
        self.use_proxy = use_proxy
        self.progress = progress
        self.seg_size = max(1 << 20, seg_mb << 20)
        self.state_path = dest + ".part.json"
        self.state = self._load_state()
        self.lock = threading.Lock()

    # ---------------- 断点状态 ----------------

    def _load_state(self) -> dict:
        if os.path.isfile(self.state_path) and os.path.isfile(self.dest):
            try:
                with open(self.state_path, encoding="utf8") as f:
                    st = json.load(f)
                if st.get("size") == self.size and st.get("url") == self.url:
                    return st
            except Exception:
                pass
        return {"url": self.url, "size": self.size, "done": []}

    def _save_state(self) -> None:
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf8") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.state_path)

    def _segments(self) -> list[tuple[int, int, int]]:
        segs = []
        idx = 0
        start = 0
        while start < self.size:
            end = min(start + self.seg_size, self.size) - 1
            segs.append((idx, start, end))
            idx += 1
            start = end + 1
        return segs

    # ---------------- 下载一段 ----------------

    def _download_segment(self, session: requests.Session, idx: int, start: int, end: int) -> None:
        for attempt in range(1, 6):
            try:
                proxies = None if self.use_proxy else {"http": None, "https": None}
                with session.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}"},
                    stream=True,
                    timeout=(20, 60),
                    proxies=proxies,
                ) as r:
                    if r.status_code not in (200, 206):
                        raise RuntimeError(f"HTTP {r.status_code}")
                    # 以 r+b 打开，按偏移写入；每个线程独立句柄，避免互相影响文件指针
                    with open(self.dest, "r+b") as fh:
                        fh.seek(start)
                        got = 0
                        for chunk in r.iter_content(CHUNK):
                            if not chunk:
                                continue
                            fh.write(chunk)
                            got += len(chunk)
                            self.progress.add(len(chunk))
                    if got != (end - start + 1):
                        # 少写了，回退这段重试：把已写部分计入进度但标记未完成
                        raise RuntimeError(f"区间不完整 {got}/{end - start + 1}")
                return
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(10, attempt * 2))

    # ---------------- 主流程 ----------------

    def run(self) -> None:
        segs = self._segments()
        done = set(self.state.get("done", []))
        todo = [s for s in segs if s[0] not in done]

        # 预分配文件，保证 seek 写入有效
        os.makedirs(os.path.dirname(os.path.abspath(self.dest)), exist_ok=True)
        if not os.path.isfile(self.dest):
            with open(self.dest, "wb") as f:
                f.truncate(self.size)
        else:
            cur = os.path.getsize(self.dest)
            if cur != self.size:
                with open(self.dest, "r+b") as f:
                    f.truncate(self.size)

        if not todo:
            return

        with ThreadPoolExecutor(max_workers=self.conns) as ex:
            futures = {}
            for idx, start, end in todo:
                sess = requests.Session()
                fut = ex.submit(self._download_segment, sess, idx, start, end)
                futures[fut] = idx
            for fut in as_completed(futures):
                idx = futures[fut]
                fut.result()  # 有异常直接抛
                with self.lock:
                    self.state["done"].append(idx)
                    self.state["done"].sort()
                    self._save_state()


# ===========================================================================
# 测速
# ===========================================================================

def probe(sources: list[str], revision: str, conns: int, seconds: float,
          rel: str = "model-00001-of-00002.safetensors", repo: str = REPO_ID) -> dict[str, float]:
    """对每个源做多连接测速，返回 name -> kB/s。

    rel 要传一个该仓库真实存在的大文件（调用方先用 list_files 拿最大的那个）。
    """
    results: dict[str, float] = {}
    for name in sources:
        base, use_proxy = SOURCES[name]
        url = f"{base}{repo}/resolve/{revision}/{rel}"
        got_total = 0
        lock = threading.Lock()
        stop_at = time.time() + seconds

        def worker(i: int) -> None:
            nonlocal got_total
            proxies = None if use_proxy else {"http": None, "https": None}
            span = 64 << 20
            try:
                with requests.get(
                    url,
                    headers={"Range": f"bytes={i * span}-{i * span + span - 1}"},
                    stream=True,
                    timeout=(15, 45),
                    proxies=proxies,
                ) as r:
                    if r.status_code >= 400:
                        return
                    for chunk in r.iter_content(CHUNK):
                        if time.time() > stop_at:
                            return
                        with lock:
                            got_total += len(chunk)
            except Exception:
                return

        with ThreadPoolExecutor(max_workers=min(conns, 64)) as ex:
            list(ex.map(worker, range(min(conns, 64))))
        wall = seconds
        kbs = got_total / 1024 / wall
        results[name] = kbs
        print(f"  {name:<12} {kbs:>9.0f} kB/s   ({conns} 连接, 取到 {got_total / 1e6:.1f} MB)")
    return results


# ===========================================================================
# 入口
# ===========================================================================

def _pid_alive(pid: int) -> bool:
    """判断进程是否还活着。注意：Windows 上绝不能用 os.kill(pid, 0) 探测——
    那会直接给目标进程发 TerminateProcess，等于把它杀掉。"""
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.GetExitCodeProcess.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if k32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return code.value == STILL_ACTIVE
        return False
    finally:
        k32.CloseHandle(handle)


def _acquire_lock(out_dir: str) -> str | None:
    """同一输出目录只允许一个下载器实例，避免两个进程写同一个文件把它写坏。
    返回锁文件路径；若已有活着的实例在跑则返回 None。"""
    lock = os.path.join(out_dir, ".download.lock")
    if os.path.isfile(lock):
        try:
            old = int(open(lock, encoding="utf8").read().strip() or "0")
        except Exception:
            old = 0
        if old and old != os.getpid() and _pid_alive(old):
            return None
    with open(lock, "w", encoding="utf8") as fh:
        fh.write(str(os.getpid()))
    return lock


def main() -> int:
    # 输出重定向到文件时也要能实时看到进度
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="多连接高速下载 HuggingFace 模型分支")
    ap.add_argument("--repo", default=REPO_ID, help=f"仓库名 owner/name（默认 {REPO_ID}）")
    ap.add_argument("--revision", "-r", default=None, help=f"分支名（默认：主模型 {DEFAULT_REVISION}，其他仓库为 main）")
    ap.add_argument("--out", "-o", default=None, help="输出目录（默认 models/<仓库名末段>）")
    ap.add_argument("--source", "-s", default="auto", choices=["auto", *SOURCES], help="下载源（默认自动测速选择）")
    ap.add_argument("--conns", "-j", type=int, default=64, help="总并发连接数（默认 64）")
    ap.add_argument("--probe-only", action="store_true", help="只测速，不下载")
    ap.add_argument("--probe-seconds", type=float, default=12.0, help="每个源的测速时长")
    args = ap.parse_args()

    # 分支默认值：下载主模型时维持原来的 3.50bpw，下别的仓库时用 main
    if args.revision is None:
        args.revision = DEFAULT_REVISION if args.repo == REPO_ID else "main"

    default_name = os.path.basename(args.repo.rstrip("/")) or "model"
    out_dir = os.path.abspath(args.out or os.path.join("models", default_name))
    os.makedirs(out_dir, exist_ok=True)

    lock = _acquire_lock(out_dir)
    if lock is None:
        print("已有下载器实例在运行（见 " + os.path.join(out_dir, ".download.lock") + "），")
        print("为避免两个进程写坏同一个文件，本次退出。等它跑完或先结束它再重试。")
        return 1

    print("=" * 74)
    print(f"仓库   : {args.repo}")
    print(f"分支   : {args.revision}")
    print(f"输出到 : {out_dir}")
    print(f"本进程 : PID {os.getpid()}")
    print("=" * 74)

    # ---------------- 测速选源 ----------------
    if args.probe_only or args.source == "auto":
        # 测速要指向一个真实存在的大文件：不能假定分片名固定，
        # 不同仓库的文件名不一样（草稿模型就只有一个 model.safetensors）。
        try:
            probe_files = list_files("hf-direct", args.revision, args.repo)
            probe_rel = max(probe_files, key=lambda f: f["size"])["path"]
        except Exception:
            probe_rel = "model-00001-of-00002.safetensors"  # 退回到主模型的老名字
        print(f"\n测速中（每源 {args.probe_seconds:.0f}s，{args.conns} 连接，探测文件 {probe_rel}）…")
        cands = list(SOURCES)
        results = probe(cands, args.revision, args.conns, args.probe_seconds,
                        rel=probe_rel, repo=args.repo)
        usable = {k: v for k, v in results.items() if v > 50}
        if not usable:
            print("\n所有源都不可用（速度 < 50 kB/s）")
            return 1
        best = max(usable, key=lambda k: usable[k])
        kbs = usable[best]
        print(f"\n最快：{best}  {kbs:.0f} kB/s")
        if args.probe_only:
            return 0
        args.source = best
    else:
        print(f"\n使用指定源：{args.source}")

    # ---------------- 取文件列表 ----------------
    try:
        files = list_files(args.source, args.revision, args.repo)
    except Exception as exc:
        print(f"获取文件列表失败：{type(exc).__name__}: {exc}")
        return 1
    total = sum(f["size"] for f in files)
    print(f"共 {len(files)} 个文件，合计 {human(total)}\n")

    base, use_proxy = SOURCES[args.source]
    os.makedirs(out_dir, exist_ok=True)

    # 已存在的完整文件先跳过。注意：下载器会预分配文件到完整大小，所以光看大小
    # 会把「预分配好但没下完」的文件误判成已完成——必须同时确认没有残留的 .part.json。
    todo: list[dict] = []
    already = 0
    for f in sorted(files, key=lambda x: -x["size"]):
        dest = os.path.join(out_dir, f["path"])
        if (os.path.isfile(dest) and os.path.getsize(dest) == f["size"]
                and not os.path.isfile(dest + ".part.json")):
            already += f["size"]
            continue
        todo.append(f)
    if already:
        print(f"已就绪 {human(already)}（跳过）")

    remain = sum(f["size"] for f in todo)
    if remain == 0:
        print("\n所有文件都已完成")
        return 0

    progress = Progress(remain)
    print(f"待下载 {len(todo)} 个文件 / {human(remain)}\n")

    # 文件是顺序下载的，所以把全部连接数都给当前文件（小文件瞬间完成）
    per_file = max(1, args.conns)
    try:
        for f in todo:
            dest = os.path.join(out_dir, f["path"])
            url = f"{base}{args.repo}/resolve/{args.revision}/{f['path']}"
            conns = per_file if f["size"] > (64 << 20) else 4
            dl = FileDownloader(url, dest, f["size"], conns, use_proxy, progress)
            done_before = len(dl.state.get("done", []))
            segs_total = len(dl._segments())
            if f["size"] > (4 << 20):
                print(f"\n下载 {f['path']}  ({human(f['size'])}, 分 {segs_total} 段, "
                      f"已完 {done_before} 段, {conns} 连接)")
            dl.run()
            progress.maybe_print(force=True)
    except KeyboardInterrupt:
        print("\n\n已中断。重新运行同一条命令会从断点继续。")
        return 130
    except Exception as exc:
        print(f"\n\n下载出错：{type(exc).__name__}: {exc}")
        print("重新运行同一条命令会从断点继续。")
        return 1

    print("\n\n下载完成，开始校验…")
    ok = True
    for f in sorted(files, key=lambda x: x["path"]):
        dest = os.path.join(out_dir, f["path"])
        if not os.path.isfile(dest):
            print(f"  缺失 {f['path']}")
            ok = False
        elif os.path.getsize(dest) != f["size"]:
            print(f"  大小不符 {f['path']}: {os.path.getsize(dest)} != {f['size']}")
            ok = False
        else:
            # 完成后清掉分段状态文件
            st = dest + ".part.json"
            if os.path.isfile(st):
                os.remove(st)
    if ok and lock and os.path.isfile(lock):
        os.remove(lock)
    print("校验通过，可以启动服务：  .\\start.ps1" if ok else "校验未通过，请重跑本命令续传")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
