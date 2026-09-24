#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""pm2 托管用的启动包装。

为什么需要这一层（Windows 特有问题）：
    `.venv\Scripts\python.exe` 是**转发器**（255 KB，而 base 解释器只有 105 KB）。
    它自己会再拉起 base 解释器并把参数原样传过去，所以一次启动实际上是
    **两个进程**：

        pm2  ──▶  转发器 python.exe (PID A)  ──▶  base python.exe (PID B)
                  不占显存                        真正占端口 + 21.5 GiB 显存

    pm2 只认得 A 的 PID。它停进程时如果只杀 A，**B 会变成孤儿继续占着显存**，
    下次启动必然 CUDA OOM。控制面板之所以没这个问题，是因为它用
    `taskkill /PID <A> /T /F` 杀整棵树；pm2 的停止路径不保证这样。

这个包装脚本负责三件事：

  1. **孤儿看门狗**：B 每隔几秒检查自己的父进程 A 是否还活着。
     A 一旦消失（pm2 杀了它），B 立刻自己退出，把显存还回去。
     这是让 pm2 的 stop 能真正释放显存的关键。
  2. **日志**：把 stdout/stderr 复制一份到 `logs/service.log`，和控制面板读的是
     同一份文件，这样面板的日志页在 pm2 托管下照样能用。
  3. **信号处理**：收到 CTRL_C / TERM 时走 uvicorn 的优雅退出
     （`SERVER.should_exit = True`），跑完 lifespan 释放显存，而不是硬死。

    pm2 start ecosystem.config.js
"""

from __future__ import annotations

import ctypes
import json
import os
import runpy
import sys
import threading
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 日志：先建好重定向，再让 server.py 去 logging.basicConfig（否则会绑到旧 stdout）
# ---------------------------------------------------------------------------

class _Tee:
    """把写入同时送到原 stdout 和日志文件。

    必须实现 __getattr__ 转发：uvicorn 的 formatter 会调 sys.stdout.isatty()，
    只转发 write/flush 会抛 ValueError: Unable to configure formatter 'default'。
    """

    def __init__(self, original, handle):
        self._original = original
        self._handle = handle

    def write(self, data):
        if isinstance(data, str):
            try:
                self._handle.write(data.encode("utf-8", "replace"))
                self._handle.flush()
            except Exception:
                pass
        return self._original.write(data)

    def flush(self):
        try:
            self._handle.flush()
        except Exception:
            pass
        return self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


_LOG_HANDLE = None


def _setup_logging() -> None:
    """把 stdout/stderr 复制到 logs/service.log（受 QWEN38_LOG_TO_FILE 控制）。"""
    global _LOG_HANDLE

    # 控制台编码：被重定向到文件时 Windows 默认用系统 ANSI(GBK)，
    # 面板按 UTF-8 读会全是乱码。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if os.environ.get("QWEN38_LOG_TO_FILE") != "1":
        return

    log_dir = BASE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # 二进制模式：子进程/本进程直接写字节，编码由我们自己控制
    _LOG_HANDLE = open(log_dir / "service.log", "wb")
    sys.stdout = _Tee(sys.stdout, _LOG_HANDLE)
    sys.stderr = _Tee(sys.stderr, _LOG_HANDLE)


# ---------------------------------------------------------------------------
# 启动参数：由控制面板写盘，这样 pm2 模式下也能切配置
# ---------------------------------------------------------------------------

def _service_args() -> list[str]:
    """读取控制面板写下的启动参数（--config / --no-vision / --max-seq-len ...）。

    pm2 托管的命令行是 ecosystem.config.js 里固定的，没法在 start 时临时追加参数，
    所以面板把「本次想用的参数」写到 _state/service_args.json，由这里读出来传给
    server.py。这样 pm2 模式下切换配置文件依然有效。
    """
    path = BASE / "_state" / "service_args.json"
    if not path.is_file():
        return []
    try:
        args = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[run_server] 参数文件读取失败，按默认配置启动：{exc}",
              file=sys.stderr, flush=True)
        return []
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        print("[run_server] 参数文件格式不对，按默认配置启动", file=sys.stderr, flush=True)
        return []
    return args


# ---------------------------------------------------------------------------
# 孤儿看门狗
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    """这个 PID 对应的进程还活着吗。"""
    if pid <= 0:
        return False
    k32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if k32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return code.value == STILL_ACTIVE
        return False
    finally:
        k32.CloseHandle(handle)


def _watch_parent(parent_pid: int, stop_event: threading.Event) -> None:
    """父进程（venv 转发器）消失就自杀，避免变成孤儿继续占显存。

    这是整个包装存在的核心理由：pm2 只跟踪转发器的 PID，杀它不会连带杀掉
    我们（真正的解释器）。不自己退出的话，21.5 GiB 显存会一直卡着。
    """
    while not stop_event.wait(2.0):
        if not _pid_alive(parent_pid):
            # 用 os._exit 而不是 sys.exit：此时 pm2 已经在等我们死了，
            # 走正常退出流程（可能卡在 uvicorn 收尾）反而会拖到 kill_timeout。
            # 显存由操作系统回收，进程一消失就还回去了。
            print(
                f"[run_server] 父进程 {parent_pid} 已退出，本进程随之退出以释放显存",
                file=sys.stderr, flush=True,
            )
            os._exit(0)


# ---------------------------------------------------------------------------
# 信号：尽量走优雅退出
# ---------------------------------------------------------------------------

def _install_signals() -> None:
    import signal

    def handler(signum, frame):  # noqa: ARG001
        # server 模块里的全局 SERVER 就是 uvicorn 的 Server 实例，
        # 置 should_exit 会跑完整 lifespan → engine.close() 释放显存
        try:
            srv = sys.modules.get("server")
            server_obj = getattr(srv, "SERVER", None) if srv else None
            if server_obj is not None:
                print(f"[run_server] 收到信号 {signum}，走优雅退出", file=sys.stderr, flush=True)
                server_obj.should_exit = True
                return
        except Exception:
            pass
        # 还没起来就收到信号：直接退出
        os._exit(0)

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, handler)
        except (OSError, ValueError):
            pass


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> int:
    parent_pid = os.getppid()

    _setup_logging()

    stop_event = threading.Event()
    watcher = threading.Thread(
        target=_watch_parent, args=(parent_pid, stop_event),
        name="orphan-watchdog", daemon=True,
    )
    watcher.start()

    print(
        f"[run_server] 启动：本进程 PID {os.getpid()}，父进程（pm2 跟踪的转发器）PID {parent_pid}",
        file=sys.stderr, flush=True,
    )

    # 把控制面板写下的参数接到 sys.argv 上，server.py 的 argparse 会照常解析。
    # 注意要在 run_path 之前改：runpy 会用 sys.argv[0] 作为 __main__ 的名字。
    extra_args = _service_args()
    # 保留 pm2/命令行额外传进来的参数（如果有），面板写的优先级更高（放后面）
    passthrough = [a for a in sys.argv[1:]]
    sys.argv = [str(BASE / "server.py"), *passthrough, *extra_args]
    if extra_args:
        print(f"[run_server] 参数（来自控制面板）：{' '.join(extra_args)}",
              file=sys.stderr, flush=True)

    # 把 server.py 当成主脚本跑（和 `python server.py` 行为完全一致），
    # 但在**当前进程**里执行，所以只有一个解释器在占显存。
    # 注意：这里不能改成 import server + 调 main，因为 server.py 里有
    # `if __name__ == "__main__"` 的入口约定，run_path 更省事也更少歧义。
    try:
        runpy.run_path(str(BASE / "server.py"), run_name="__main__")
    finally:
        stop_event.set()

    return 0


if __name__ == "__main__":
    _install_signals()
    sys.exit(main())
