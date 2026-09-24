#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
推理服务的控制面板：在网页上启动 / 停止 / 重启 server.py。

为什么需要单独一个进程，而不是把按钮做进 server.py：
    服务停掉之后它自己也没了，自然没法再把自己启动起来。所以这里是一个独立的
    小控制台（默认 8001 端口），它负责拉起 / 关掉推理服务。
    推理服务的端口从 config.yaml 的 server.port 读（本仓库当前是 2345），不写死。

用法：
    .\start_panel.ps1                       # 推荐，会自己开浏览器
    .\start_panel.ps1 -Lan                  # 同时允许局域网访问（需要令牌）
    .\.venv\Scripts\python.exe control_panel.py
    .\.venv\Scripts\python.exe control_panel.py --panel-port 8081

安全说明（为什么面板默认是 127.0.0.1，开 -Lan 之后又靠什么兜底）：
    面板能启停进程、读日志，属于管理面。管理面和服务面**必须分开**，原因见
    README「面板为什么不能和推理服务用同一个端口」一节 —— 简单说：
      1) 面板要能把服务拉起来，所以它必须比服务活得久；合成一个进程的话
         「停止服务」会把面板自己也干掉，就再也启不回来了（鸡生蛋问题）；
      2) 服务的 host 是 0.0.0.0（局域网都在用 OpenAI 接口），管理端点和
         推理端点放在同一个监听端口上，等于把「谁能关掉我的服务」开放给整个局域网；
      3) 模型 OOM / CUDA 崩掉会把同进程的面板一起带走，恰恰在最需要看日志的时候
         面板没了，也就没法用 UI 重新拉起来。

    所以是「两个进程、两个端口」（面板 8001 / 服务 2345）。

    默认只监听 127.0.0.1；-Lan（即 --host 0.0.0.0）会把面板暴露到局域网，
    默认**不需要令牌**（当前就是这么用的）。如果以后想加一道门，不用改代码，
    下面两种任一方式即可启用：
      · 启动时 --token <自定的值>（或环境变量 QWEN38_PANEL_TOKEN）；
      · 把令牌写进 _state/panel_token.txt（一行纯文本，面板启动时自动读）。
    一旦有了令牌：本机（127.0.0.1 / ::1）仍然免令牌，局域网机器则必须带令牌，
    否则只能看首页和 /api/status，不能启停服务。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import requests
import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

BASE = Path(__file__).resolve().parent
WEB_DIR = BASE / "web"
LOG_DIR = BASE / "logs"
STATE_FILE = LOG_DIR / "panel_state.json"
# 服务的 stdout 和 stderr 合并到一个文件：分开写的话两路日志的相对时间顺序就丢了，
# 页面上看起来会很跳。合并后是真实的先后来后到。
SERVICE_LOG = LOG_DIR / "service.log"
PANEL_LOG = LOG_DIR / "panel.log"
DEFAULT_CONFIG = "config.yaml"
STATE_DEFAULT_PORT = 8000
DEFAULT_PANEL_PORT = 8001

# pm2 托管（可选）。ecosystem.config.js 里的应用名要与此一致。
# 托管之后启停必须走 pm2：直接杀进程的话 pm2 会立刻把它拉起来（autorestart）。
PM2_APP = "qwen38-server"
PM2_CONFIG = "ecosystem.config.js"
# pm2 托管的服务，命令行参数由面板写在这里，run_server.py 启动时读。
# 这样在 pm2 模式下「切换配置 / --no-vision」依然有效（pm2 的命令是固定的，
# 没法在 start 时临时改参数）。
STATE_DIR = BASE / "_state"
SERVICE_ARGS_FILE = STATE_DIR / "service_args.json"

# 面板自己的监听地址/端口/令牌。本来这两个是 main() 里的局部变量，但因为
# /api/status 要把「局域网该怎么访问」回给前端，所以提到模块级。
PANEL_HOST = "127.0.0.1"
PANEL_PORT = DEFAULT_PANEL_PORT
PANEL_TOKEN = ""
PANEL_TOKEN_FILE = STATE_DIR / "panel_token.txt"

# 回环地址：来自这些地址的请求视为本机操作，免令牌。
# ::ffff:127.0.0.1 是 IPv6 映射写法，Windows 上双栈监听时很常见，必须一起认。
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
# 这些 GET 即使从局域网来也不要求令牌：首页是静态 HTML、/api/status 是仪表盘
# 渲染所必需的只读数据。注意 /api/status 只在**本机**请求时才会把令牌回显出去，
# 否则 LAN 上任何人读一次 status 就能拿到令牌，令牌就白设了。
TOKEN_FREE_GET = {"/", "/favicon.ico", "/api/status"}

# 图标 / manifest 这类静态资源。用白名单显式列出，而不是 StaticFiles 挂载整个 web/：
#   · 挂目录会把 index.html 也变成 /index.html 可访问（同一份内容两个 URL）；
#   · 白名单能保证只有这几个文件会出去，不会因为 web/ 下多丢了个文件而意外暴露。
STATIC_FILES: dict[str, tuple[str, str]] = {
    "/favicon.ico":          ("favicon.ico",          "image/x-icon"),
    "/favicon.svg":          ("icon.svg",             "image/svg+xml"),
    "/icon.svg":             ("icon.svg",             "image/svg+xml"),
    "/icon-192.png":         ("icon-192.png",         "image/png"),
    "/icon-512.png":         ("icon-512.png",         "image/png"),
    "/apple-touch-icon.png": ("apple-touch-icon.png", "image/png"),
    "/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json"),
}
# 图标不影响安全（是 GET、且不含任何本机信息），但必须免令牌：
# 否则设了令牌之后，局域网浏览器取 favicon 会拿到 401，标签页就没图标了。
TOKEN_FREE_GET |= set(STATIC_FILES)

# Windows 上启动子进程时不要弹黑框、也不要 Ctrl+C 波及到它
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200

_lock = threading.Lock()
_pm2_exe_cache: str | None = None


# ===========================================================================
# Windows 进程工具
# ===========================================================================

def pid_alive(pid: int) -> bool:
    """进程是否还活着。

    千万不要用 os.kill(pid, 0) 在 Windows 上探活：那里没有"信号 0"的概念，
    os.kill 会直接走 TerminateProcess，等于把目标进程杀掉。
    """
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


def _run(cmd: list[str], timeout: float = 20.0, encoding: str | None = "utf-8") -> subprocess.CompletedProcess:
    """跑一个子进程并把输出按指定编码解码。

    注意 taskkill / nvidia-smi 这类 Windows 自带工具按系统 ANSI（简中是 GBK）
    输出中文，用 utf-8 解码会得到一堆 U+FFFD。按需求传 encoding="gbk" 更靠谱。
    """
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding=encoding,
        errors="replace",
        timeout=timeout,
        creationflags=CREATE_NO_WINDOW,
    )


_PS_PREFIX = (
    "$OutputEncoding = [Console]::OutputEncoding = "
    "[System.Text.UTF8Encoding]::new($false); "
)


def _ps_json(script: str, timeout: float = 20.0) -> Any:
    """跑一段 PowerShell 并把输出当 JSON 解析。

    cmdline 是二进制进程信息，只能通过 WMI 拿；这里用 PowerShell 而不是 wmic
    （wmic 在新版 Windows 上已被移除）。
    """
    proc = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS_PREFIX + script],
                timeout=timeout)
    text = (proc.stdout or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Get-CimInstance 单条结果不会包成数组，补一下再看
        try:
            return json.loads("[" + text + "]")
        except json.JSONDecodeError:
            return None


def scan_service_processes() -> list[dict[str, Any]]:
    """找出所有属于本项目的 server.py 进程（通常有两个，见下面的说明）。

    venv 里的 Scripts\\python.exe 在 Windows 上是个"转发器"，它自己会再拉起
    base 解释器并把参数原样传过去。所以一次启动会看到两个进程：
      - 父：venv 的 python.exe（转发器，不占显存）
      - 子：base 的 python.exe（真正跑服务、占显存、占端口）
    只杀其中一个会留下另一个，所以下面统一按"整棵树"处理。
    """
    needle = str(BASE).replace("\\", "\\\\")
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -and $_.CommandLine -like '*server.py*' } | "
        "Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | "
        "ConvertTo-Json -Compress -Depth 3"
    )
    data = _ps_json(script)
    if data is None:
        return []
    if isinstance(data, dict):
        data = [data]
    out: list[dict[str, Any]] = []
    for item in data:
        cmd = str(item.get("CommandLine") or "")
        if "server.py" not in cmd:
            continue
        # 只认本项目的实例，避免误伤别的目录下的同名脚本
        if str(BASE) not in cmd and needle not in cmd:
            continue
        out.append({
            "pid": int(item["ProcessId"]),
            "ppid": int(item.get("ParentProcessId") or 0),
            "cmdline": cmd,
        })
    return out


def kill_tree(pid: int) -> tuple[bool, str]:
    """杀掉进程及其所有子进程。"""
    try:
        proc = _run(["taskkill", "/PID", str(pid), "/T", "/F"], timeout=25, encoding="gbk")
        ok = proc.returncode == 0
        return ok, (proc.stdout or proc.stderr or "").strip()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.4) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


# ===========================================================================
# 访问控制（仅当面板监听 0.0.0.0 时起作用）
# ===========================================================================

def client_is_local(request: Request) -> bool:
    """请求是否来自本机。

    ★ 「本机」不能简单写成 request.client.host == "127.0.0.1"：
      以 --host 0.0.0.0 双栈监听时，本机浏览器（走 localhost）常常被上报成
      ::ffff:127.0.0.1 或 ::1，漏判的话本机自己反而被要求输令牌。
    """
    host = (request.client.host if request.client else "") or ""
    if host in LOOPBACK_HOSTS:
        return True
    if host.startswith("::ffff:"):          # IPv4-mapped IPv6
        return host[7:] in LOOPBACK_HOSTS
    return False


def token_ok(request: Request) -> bool:
    """令牌校验。没设令牌（回环监听模式）直接放行。

    支持两种携带方式：
      · 请求头 X-Panel-Token（前端 fetch 用这个，不会漏进地址栏历史）
      · 查询串 ?token=…（EventSource 没法自定义请求头，日志 SSE 只能用它）
    """
    if not PANEL_TOKEN:
        return True
    got = request.headers.get("x-panel-token") or request.query_params.get("token") or ""
    # 用常数时间比较，避免按字符逐位试探出令牌
    return secrets.compare_digest(got, PANEL_TOKEN)


def load_panel_token(explicit: str = "") -> str:
    """取面板令牌。返回空串 = 不做鉴权（当前默认）。

    ★ 故意**不**自动生成：局域网访问是默认用法，如果启动就自动造个令牌并要求
      携带，那么每换一次面板实例（改代码后 -Force、重启机器）都得重新拿令牌，
      反而把「局域网能直接用」这件事弄麻烦了。所以令牌是纯按需：
        命令行 --token  >  环境变量 QWEN38_PANEL_TOKEN  >  _state/panel_token.txt
      三者都没有就返回空串，请求全部放行。

    留着文件这个入口是为了以后想开鉴权时不用改代码也不用改启动脚本，
    手写一行令牌进去、重启面板即可。
    """
    if explicit:
        return explicit
    try:
        saved = PANEL_TOKEN_FILE.read_text(encoding="utf8").strip()
        if saved:
            return saved
    except Exception:
        pass
    return ""


def lan_ipv4_addresses() -> list[str]:
    """本机所有非回环 IPv4，用于拼给用户看的局域网地址。

    只用 socket.gethostbyname_ex 会只返回一张网卡（多网卡机器上往往是 WLAN），
    所以再拿「UDP connect 到公网」的副作用补一个默认出口地址。
    两种办法都不发送任何数据包，只是让内核选路由。
    """
    found: list[str] = []
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if not ip.startswith("127.") and ip not in found:
                found.append(ip)
    except Exception:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.2)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127.") and ip not in found:
                found.insert(0, ip)
    except Exception:
        pass
    return found


def panel_urls() -> dict[str, Any]:
    """面板的可访问地址。前端底部会把它显示出来，省得用户去 ipconfig 里翻。"""
    local = f"http://127.0.0.1:{PANEL_PORT}/"
    exposed = PANEL_HOST in ("0.0.0.0", "::")
    lan = [f"http://{ip}:{PANEL_PORT}/" for ip in lan_ipv4_addresses()] if exposed else []
    return {"local": local, "lan": lan, "exposed": exposed}


def service_base_urls(port: int, api_keys: list[Any] | None = None) -> dict[str, Any]:
    """推理服务（OpenAI 兼容）的 base_url 候选，给前端一键复制用。

    为什么必须由服务端算、不能让前端用 location.hostname 拼：
      · 面板端口（8001）和推理服务端口（2345）不是同一个，前端拿 location 拼必错；
      · 服务绑的是 0.0.0.0，局域网该用哪个 IP（本机有两张网卡：以太网 + WLAN）
        只有服务端自己知道；
      · 用户实际要填给 LM Studio / 客户端的是 **base_url**（带 /v1，不带具体端点），
        这里直接给到位，避免拼错路径。
    """
    local = f"http://127.0.0.1:{port}/v1"
    lan = [f"http://{ip}:{port}/v1" for ip in lan_ipv4_addresses()]
    return {
        "local": local,
        "lan": lan,
        "all": [local, *lan],
        "port": port,
        # 有 key 时客户端就得带 Authorization: Bearer <key>；config 里默认是空的。
        "auth_required": bool(api_keys),
        "models": f"http://127.0.0.1:{port}/v1/models",
        "ends": {
            "对话": "/v1/chat/completions",
            "补全": "/v1/completions",
            "模型列表": "/v1/models",
            "健康检查": "/health",
        },
    }


# ===========================================================================
# 配置 / 状态
# ===========================================================================

def read_config(name: str = DEFAULT_CONFIG) -> dict[str, Any]:
    path = BASE / name
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def load_state() -> dict[str, Any]:
    if not STATE_FILE.is_file():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf8"))
    except Exception:
        return {}


def save_state(state: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf8")


def clear_state() -> None:
    try:
        STATE_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def gpu_snapshot() -> dict[str, Any] | None:
    """用 nvidia-smi 取显存信息。取不到就返回 None（没有 NVIDIA 卡也算正常）。"""
    try:
        proc = _run([
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], timeout=10)
    except Exception:
        return None
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return None
    parts = [p.strip() for p in line[0].split(",")]
    if len(parts) < 5:
        return None
    try:
        total, used, free, util = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    except ValueError:
        return None

    apps: list[dict[str, Any]] = []
    try:
        proc2 = _run([
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ], timeout=10)
        for row in (proc2.stdout or "").strip().splitlines():
            bits = [b.strip() for b in row.split(",")]
            if len(bits) >= 2:
                apps.append({"pid": bits[0], "name": os.path.basename(bits[1]),
                             "mem": bits[2] if len(bits) > 2 else ""})
    except Exception:
        pass

    return {
        "name": parts[0],
        "total_gb": round(total / 1024, 2),
        "used_gb": round(used / 1024, 2),
        "free_gb": round(free / 1024, 2),
        "util": util,
        "apps": apps,
    }


def probe_health(port: int, timeout: float = 2.0) -> dict[str, Any]:
    """问一下推理服务自己的 /health。"""
    try:
        resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=timeout)
    except requests.RequestException as exc:
        return {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        body = resp.json()
    except ValueError:
        body = {"raw": resp.text[:500]}
    return {"reachable": True, "status_code": resp.status_code, "body": body}


def read_service_args() -> list[str]:
    """读面板写下的启动参数（pm2 模式下用）。"""
    try:
        data = json.loads(SERVICE_ARGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [a for a in data if isinstance(a, str)] if isinstance(data, list) else []


def active_config_name(state: dict[str, Any] | None = None) -> str:
    """当前生效的配置文件。pm2 模式下从参数文件里取，否则看面板状态。"""
    args = read_service_args()
    if "--config" in args:
        idx = args.index("--config")
        if idx + 1 < len(args):
            return args[idx + 1]
    state = state if state is not None else load_state()
    return state.get("config") or DEFAULT_CONFIG


def current_status() -> dict[str, Any]:
    state = load_state()
    # 端口要按「当前实际生效的配置」来读：pm2 模式下配置文件来自参数文件，
    # 直接拿 DEFAULT_CONFIG 会在切过配置后连错端口。
    cfg = read_config(active_config_name(state))
    port = int((cfg.get("server") or {}).get("port") or STATE_DEFAULT_PORT)

    procs = scan_service_processes()
    recorded = int(state.get("pid") or 0)
    if recorded and not pid_alive(recorded) and not procs:
        clear_state()
        state = {}
        recorded = 0

    health = probe_health(port) if procs or port_open(port) else {"reachable": False}
    listening = port_open(port)

    # 判定状态
    if not procs and not listening:
        if recorded:
            clear_state()
        status = "stopped"
    elif health.get("reachable") and health.get("status_code") == 200:
        status = "running"
    elif health.get("reachable") and health.get("status_code") == 503:
        body = health.get("body") or {}
        status = "error" if body.get("status") == "error" else "loading"
    else:
        status = "starting"

    uploaded = state.get("started_at")
    uptime = None
    if status in ("running", "loading", "starting", "error") and uploaded:
        uptime = round(time.time() - float(uploaded), 1)

    mc = cfg.get("model") or {}
    dc = cfg.get("defaults") or {}
    lc = cfg.get("limits") or {}
    app = pm2_app()
    # pm2 托管时面板没有 started_at（不是它启动的），用 pm2 自己的启动时间兜底，
    # 否则「运行时长」永远显示 —。
    if uptime is None and app is not None and status in ("running", "loading", "starting"):
        pm_uptime = (app.get("pm2_env") or {}).get("pm_uptime")
        if pm_uptime:
            try:
                uptime = round(time.time() - float(pm_uptime) / 1000.0, 1)
            except (TypeError, ValueError):
                uptime = None
    return {
        "service": {
            "status": status,
            "listening": listening,
            "port": port,
            # 谁在管这个进程："pm2" 或 "panel"。启停行为不一样，UI 上要区分
            "supervisor": "pm2" if app is not None else "panel",
            "pm2": pm2_info(app),
            "pids": [p["pid"] for p in procs],
            "roots": find_roots(procs),
            "recorded_pid": recorded,
            "uptime_s": uptime,
            "started_at": uploaded,
            "health": health if health.get("reachable") else None,
            "health_error": health.get("error"),
            "config_file": active_config_name(state),
            "extra_args": state.get("extra_args") or [],
            # OpenAI 兼容接入地址（前端一键复制用）
            "urls": service_base_urls(port, (cfg.get("server") or {}).get("api_keys")),
        },
        "config": {
            "path": mc.get("path"),
            "name": mc.get("name"),
            "max_seq_len": mc.get("max_seq_len"),
            "cache_quant": mc.get("cache_quant"),
            "max_batch_size": mc.get("max_batch_size"),
            "max_chunk_size": mc.get("max_chunk_size"),
            "max_history": mc.get("max_history"),
            "load_vision": mc.get("load_vision"),
            "mtp_draft": mc.get("mtp_draft"),
            "draft_model": mc.get("draft_model"),
            "tensor_parallel": mc.get("tensor_parallel"),
            "max_tokens": dc.get("max_tokens"),
            "max_tokens_limit": dc.get("max_tokens_limit"),
            "enable_thinking": (cfg.get("chat_template") or {}).get("vars", {}).get("enable_thinking")
            if isinstance((cfg.get("chat_template") or {}).get("vars"), dict) else None,
            "concurrency": lc.get("max_concurrent_requests"),
            "allow_shutdown": (cfg.get("server") or {}).get("allow_shutdown", True),
        },
        "gpu": gpu_snapshot(),
        "panel": {
            "pid": os.getpid(),
            "python": sys.executable,
            "base": str(BASE),
            "log_dir": str(LOG_DIR),
        },
    }


def find_roots(procs: list[dict[str, Any]]) -> list[int]:
    """取出这批进程里最上层的那些（父进程不在集合内的），用于整棵树一起杀。"""
    if not procs:
        return []
    pids = {p["pid"] for p in procs}
    roots = [p["pid"] for p in procs if p["ppid"] not in pids]
    return roots or [p["pid"] for p in procs]


# ===========================================================================
# 启停
# ===========================================================================

def build_command(cfg_name: str, extra: list[str]) -> list[str]:
    venv_py = BASE / ".venv" / "Scripts" / "python.exe"
    if not venv_py.is_file():
        raise HTTPException(
            status_code=400,
            detail="还没创建虚拟环境：请先运行 setup.ps1 或 _bootstrap.ps1",
        )
    return [str(venv_py), "-u", "server.py", "--config", cfg_name, *extra]


# ===========================================================================
# pm2 托管
# ===========================================================================
#
# 为什么要让面板认识 pm2：
#   服务的生命周期只能有一个主人。如果 pm2 用 autorestart 托管着服务，
#   面板却还在自己 spawn / 自己杀，就会出两种事故：
#     1) 面板「停止」→ 进程退出 → pm2 立刻把它拉起来 → 按钮看起来没反应
#     2) 面板「启动」→ 又起一个进程 → 两个进程抢同一个端口和 21.5 GiB 显存 → OOM
#   所以：只要 pm2 里注册了这个应用，面板的启停一律转发给 pm2。

def pm2_exe() -> str | None:
    """pm2 的可执行文件路径。Windows 上是 pm2.cmd，直接调 "pm2" 会找不到。"""
    global _pm2_exe_cache
    if _pm2_exe_cache is not None:
        return _pm2_exe_cache or None
    for name in ("pm2.cmd", "pm2.exe", "pm2"):
        found = shutil.which(name)
        if found:
            _pm2_exe_cache = found
            return found
    _pm2_exe_cache = ""
    return None


def pm2_run(args: list[str], timeout: float = 120.0) -> tuple[bool, str]:
    exe = pm2_exe()
    if not exe:
        return False, "未安装 pm2"
    try:
        proc = _run([exe, *args], timeout=timeout)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    text = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return proc.returncode == 0, text


def pm2_app() -> dict[str, Any] | None:
    """pm2 里这个应用的信息；没注册就返回 None。"""
    exe = pm2_exe()
    if not exe:
        return None
    try:
        proc = _run([exe, "jlist"], timeout=30)
    except Exception:
        return None
    text = (proc.stdout or "").strip()
    if not text:
        return None
    try:
        apps = json.loads(text)
    except json.JSONDecodeError:
        return None
    for app in apps if isinstance(apps, list) else []:
        if app.get("name") == PM2_APP:
            return app
    return None


def pm2_info(app: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """把 pm2 的应用信息整理成面板要展示的样子。"""
    app = app if app is not None else pm2_app()
    if app is None:
        return None
    env = app.get("pm2_env") or {}
    return {
        "app": PM2_APP,
        "status": env.get("status"),
        "pid": app.get("pid") or 0,
        "restarts": env.get("restart_time"),
        "unstable_restarts": env.get("unstable_restarts"),
        "uptime_ms": env.get("pm_uptime"),
        "script": env.get("pm_exec_path"),
    }


def write_service_args(cfg_name: str, extra: list[str]) -> None:
    """把本次启动的参数落盘，供 run_server.py 在 pm2 模式下读取。"""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    args = ["--config", cfg_name, *extra]
    SERVICE_ARGS_FILE.write_text(json.dumps(args, ensure_ascii=False, indent=2), encoding="utf-8")


def pm2_start(cfg_name: str, extra: list[str], wait: float = 0.0) -> dict[str, Any]:
    """通过 pm2 启动/重启服务。"""
    app = pm2_app()
    if app is None:
        raise HTTPException(status_code=500, detail=f"pm2 里没有 {PM2_APP}，请先 pm2 start {PM2_CONFIG}")

    write_service_args(cfg_name, extra)

    env = app.get("pm2_env") or {}
    status = env.get("status")
    if status == "online":
        # 已经在跑：改配置要重启才生效
        ok, out = pm2_run(["restart", PM2_APP, "--update-env"], timeout=240)
        action = "restarted"
    else:
        ok, out = pm2_run(["start", PM2_APP], timeout=240)
        action = "started"
    if not ok:
        raise HTTPException(status_code=500, detail=f"pm2 {action} 失败：{out[:300]}")

    port = int((read_config(cfg_name).get("server") or {}).get("port") or STATE_DEFAULT_PORT)
    result: dict[str, Any] = {
        "status": action, "supervisor": "pm2", "app": PM2_APP,
        "port": port, "config": cfg_name, "extra_args": extra,
        "output": out[-400:],
    }

    if wait:
        deadline = time.time() + wait
        while time.time() < deadline:
            if port_open(port):
                break
            time.sleep(0.3)
    return result


def pm2_stop(graceful_note: str = "") -> dict[str, Any]:
    """通过 pm2 停止服务，并确认进程与显存真的释放了。

    ⚠️ 这里必须**额外确认**：Windows 上 .venv\\Scripts\\python.exe 是转发器，
    它会再拉起真正的解释器（占显存的那个）。pm2 只跟踪转发器的 PID。
    如果只杀掉转发器，真解释器会变成孤儿继续占着 21.5 GiB 显存，
    下一次启动必然 OOM。所以停完要扫一遍进程，有残留就补一刀。
    """
    if pm2_app() is None:
        raise HTTPException(status_code=500, detail=f"pm2 里没有 {PM2_APP}")

    ok, out = pm2_run(["stop", PM2_APP], timeout=180)
    steps = [f"pm2 stop {PM2_APP}：{'成功' if ok else '失败'}", graceful_note]
    if not ok:
        return {"status": "failed", "detail": out[:300], "steps": steps, "supervisor": "pm2"}

    port = int((read_config(DEFAULT_CONFIG).get("server") or {}).get("port") or STATE_DEFAULT_PORT)

    # 等转发器 + 真解释器都退出（run_server.py 的孤儿看门狗最多 2s 响应）
    deadline = time.time() + 40
    while time.time() < deadline:
        if not scan_service_processes() and not port_open(port):
            clear_state()
            return {"status": "stopped", "graceful": True, "steps": steps, "supervisor": "pm2"}
        time.sleep(0.5)

    # 还有残留 → 补一刀整棵树
    leftover = scan_service_processes()
    steps.append(f"pm2 stop 后仍有残留进程 {[p['pid'] for p in leftover]}，补杀整棵树")
    for root in find_roots(leftover):
        ok2, msg = kill_tree(root)
        steps.append(f"结束 PID {root}: {'成功' if ok2 else '失败'} {msg[:100]}")

    deadline = time.time() + 15
    while time.time() < deadline:
        if not scan_service_processes() and not port_open(port):
            clear_state()
            return {"status": "stopped", "graceful": False, "steps": steps, "supervisor": "pm2"}
        time.sleep(0.4)

    still = scan_service_processes()
    return {
        "status": "failed",
        "detail": f"仍有进程未退出：{[p['pid'] for p in still]}",
        "steps": steps,
        "supervisor": "pm2",
    }


def child_env() -> dict[str, str]:
    """给子进程一套环境变量：强制 UTF-8 输出。

    Windows 上被子进程继承 stdout/stderr 句柄时，Python 默认用系统 ANSI（GBK）
    写日志，而面板是按 UTF-8 读的，不强制就全是乱码。
    PYTHONIOENCODING 在解释器启动最早期生效，比在代码里 reconfigure 更可靠。
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8:replace"
    env["PYTHONUTF8"] = "1"
    return env


def start_service(cfg_name: str, extra: list[str], wait: float = 0.0) -> dict[str, Any]:
    with _lock:
        # pm2 托管时一律交给 pm2：自己再 spawn 一个会变成两个进程抢端口和显存
        if pm2_app() is not None:
            return pm2_start(cfg_name, extra, wait)

        procs = scan_service_processes()
        if procs:
            raise HTTPException(status_code=409, detail=f"服务已经在运行（PID {find_roots(procs)}）")

        cfg = read_config(cfg_name)
        port = int((cfg.get("server") or {}).get("port") or STATE_DEFAULT_PORT)
        if port_open(port):
            raise HTTPException(
                status_code=409,
                detail=f"端口 {port} 已被占用（可能是手工启动的服务，或别的程序）。"
                       f"请先停止它，或改 config.yaml 里的 server.port",
            )

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        cmd = build_command(cfg_name, extra)

        # 每次启动重新写日志，免得旧内容混进来。
        # 用二进制模式打开、不包 text 层：子进程是直接往文件描述符写字节的，
        # 父进程这边的编码设置影不到它，所以真正的编码由子进程的 PYTHONIOENCODING 决定。
        log_fh = open(SERVICE_LOG, "wb")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(BASE),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
                close_fds=True,
                env=child_env(),
            )
        except Exception as exc:
            log_fh.close()
            raise HTTPException(status_code=500, detail=f"启动失败：{type(exc).__name__}: {exc}")
        finally:
            # 句柄交给子进程了，父进程这边关掉不影响它写入
            log_fh.close()

        save_state({
            "pid": proc.pid,
            "started_at": time.time(),
            "config": cfg_name,
            "extra_args": extra,
            "port": port,
            "cmd": cmd,
        })

    if wait:
        deadline = time.time() + wait
        while time.time() < deadline:
            if port_open(port):
                break
            if not pid_alive(proc.pid):
                break
            time.sleep(0.3)

    return {"status": "started", "pid": proc.pid, "port": port, "cmd": cmd}


def stop_service(graceful: bool = True, timeout: float = 30.0, force: bool = False) -> dict[str, Any]:
    with _lock:
        # pm2 托管时不能走优雅退出：进程一退 pm2 的 autorestart 会立刻把它拉起来。
        # 要真的停下就必须让 pm2 知道「这是有意的停止」。
        if pm2_app() is not None:
            note = ""
            if graceful and not force:
                note = "（pm2 托管：改用 pm2 stop，/admin/shutdown 会被 autorestart 抵消）"
            return pm2_stop(note)

        state = load_state()
        cfg = read_config(state.get("config") or DEFAULT_CONFIG)
        port = int((cfg.get("server") or {}).get("port") or STATE_DEFAULT_PORT)
        procs = scan_service_processes()
        roots = find_roots(procs)

        if not procs and not port_open(port):
            clear_state()
            return {"status": "already_stopped"}

        steps: list[str] = []

        # 1) 优先优雅退出：走 /admin/shutdown，让 uvicorn 正常跑完 lifespan，
        #    engine.close() 会释放显存。硬杀进程虽然也能收显存，但不会有机会清理。
        if graceful and not force:
            try:
                resp = requests.post(f"http://127.0.0.1:{port}/admin/shutdown", timeout=5)
                if resp.status_code == 200:
                    steps.append("已发送优雅退出请求")
                elif resp.status_code == 403:
                    steps.append(f"服务拒绝关机（HTTP 403）：{resp.text[:160]}")
                else:
                    steps.append(f"优雅退出返回 HTTP {resp.status_code}")
            except requests.RequestException as exc:
                steps.append(f"优雅退出请求失败（{type(exc).__name__}），将改为强制结束")

        deadline = time.time() + timeout
        while time.time() < deadline:
            if not scan_service_processes() and not port_open(port):
                clear_state()
                return {"status": "stopped", "graceful": graceful and not force, "steps": steps}
            time.sleep(0.4)

        # 2) 还活着就强杀整棵树
        steps.append("优雅退出超时，执行强制结束（taskkill /T /F）")
        for root in roots:
            ok, msg = kill_tree(root)
            steps.append(f"结束 PID {root}: {'成功' if ok else '失败'} {msg[:120]}")

        deadline = time.time() + 15
        while time.time() < deadline:
            if not scan_service_processes() and not port_open(port):
                clear_state()
                return {"status": "stopped", "graceful": False, "steps": steps}
            time.sleep(0.4)

        still = scan_service_processes()
        return {
            "status": "failed",
            "detail": f"仍有进程未退出：{[p['pid'] for p in still]}",
            "steps": steps,
        }


# ===========================================================================
# 日志
# ===========================================================================

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _read_tail(path: Path, lines: int) -> list[str]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf8", errors="replace") as fh:
            data = fh.readlines()
    except Exception:
        return []
    return [_ANSI_RE.sub("", ln.rstrip("\n")) for ln in data[-lines:]]


def read_logs(tail: int = 200, source: str = "service") -> dict[str, Any]:
    path = PANEL_LOG if source == "panel" else SERVICE_LOG
    lines = _read_tail(path, tail)
    return {"path": str(path), "exists": path.is_file(), "lines": lines}


def stream_logs(poll: float = 0.5, source: str = "service") -> Iterator[str]:
    """把日志文件的新增内容用 SSE 推给前端。

    只跟踪一个文件，所以进程重启（文件被重写、大小变小）时直接把偏移归零，
    前端会看到从头开始的日志，不会漏也不会重复。
    """
    path = PANEL_LOG if source == "panel" else SERVICE_LOG
    try:
        offset = path.stat().st_size
    except OSError:
        offset = 0

    yield "retry: 2000\n\n"
    idle = 0.0
    while True:
        wrote = False
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        if size is not None:
            if size < offset:
                offset = 0
                yield "data: " + json.dumps({"src": "meta", "line": f"—— 日志已重置（{path.name}）——"},
                                            ensure_ascii=False) + "\n\n"
            if size > offset:
                try:
                    with path.open("r", encoding="utf8", errors="replace") as fh:
                        fh.seek(offset)
                        chunk = fh.read()
                        offset = fh.tell()
                except Exception:
                    chunk = ""
                for line in chunk.splitlines():
                    line = _ANSI_RE.sub("", line)
                    if line.strip():
                        payload = json.dumps({"src": "log", "line": line}, ensure_ascii=False)
                        yield f"data: {payload}\n\n"
                        wrote = True
        if not wrote:
            idle += poll
            if idle >= 15.0:              # 心跳，避免中间层把长连接掉
                idle = 0.0
                yield ": ping\n\n"
        else:
            idle = 0.0
        time.sleep(poll)


# ===========================================================================
# Web 应用
# ===========================================================================

app = FastAPI(title="Qwen3.8-27B 控制面板", version="1.0.0")


@app.middleware("http")
async def guard_lan_access(request: Request, call_next):
    """局域网请求的令牌门禁。

    规则（回环监听时整段逻辑等于没有，本机体验不变）：
      · 来自 127.0.0.1 / ::1 / localhost 的 → 直接放行；
      · 其他来源：GET 首页、/favicon.ico、/api/status 放行（仪表盘得能打开），
        其余（所有 POST，以及 /api/logs、/api/logs/stream）必须带令牌。

    为什么用中间件而不是逐个端点加依赖：端点太多且以后会加，漏一个就等于
    把启停服务的权限送出去了；这里默认拒绝、白名单很小，忘了写新端点只会
    变成「新端点也要令牌」，不会变成「新端点裸奔」。
    """
    if client_is_local(request):
        return await call_next(request)
    path = request.url.path
    if request.method == "GET" and path in TOKEN_FREE_GET:
        return await call_next(request)
    if token_ok(request):
        return await call_next(request)
    return JSONResponse(
        {
            "detail": (
                "需要面板访问令牌。本机访问不需要，局域网访问请在地址后加 ?token=… "
                "（或在请求头带 X-Panel-Token）。令牌在服务端 _state/panel_token.txt "
                "里，或看面板首页底部、启动横幅。"
            ),
            "need_token": True,
        },
        status_code=401,
    )


@app.get("/")
async def index() -> FileResponse:
    page = WEB_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=500, detail=f"缺少页面文件：{page}")
    return FileResponse(page, media_type="text/html; charset=utf-8",
                        headers={"Cache-Control": "no-store"})


def _make_static_route(filename: str, media_type: str):
    """生成一个只负责发 web/<filename> 的处理函数。

    图标文件用 .venv 里的 Pillow 由 make_icon.py 生成（Pillow 读不了 SVG，
    所以 SVG 和 PNG 是同一套几何形状的两份实现，改配色要两处一起改）。
    页面引用的文件缺失时**不能只报 404 就算了**：这种静默降级会让人以为是
    浏览器缓存问题查半天，所以直接在 detail 里把完整路径写出来。
    """
    async def _serve() -> FileResponse:
        path = WEB_DIR / filename
        if not path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"缺少静态文件：{path}（跑 .\\.venv\\Scripts\\python.exe make_icon.py 生成图标）",
            )
        # ★ 用 no-cache（= 每次都带 ETag 回来问一下，命中就 304），不要用长 max-age。
        #   这些文件是 make_icon.py 生成的产物，改完配色重跑脚本后如果浏览器
        #   还攥着一小时的旧副本，会让人以为「改了没生效」而白白排查半天。
        #   ★ 这个坑我自己先踩了一次：调 SVG 时被缓存卡住，看着像是文件没改对。
        #   FileResponse 自带 ETag/Last-Modified，所以 revalidate 只花一个 304。
        return FileResponse(path, media_type=media_type,
                            headers={"Cache-Control": "no-cache"})
    return _serve


for _route, (_fname, _mt) in STATIC_FILES.items():
    # 名字按路由算：同一份 icon.svg 挂在 /favicon.svg 和 /icon.svg 两个路径上，
    # 用文件名当路由名会重名（Starlette 会报已有同名路由）。
    app.get(_route, name="static_" + _route.strip("/").replace(".", "_"))(
        _make_static_route(_fname, _mt)
    )


@app.get("/api/status")
async def api_status(request: Request) -> JSONResponse:
    import asyncio

    data = await asyncio.to_thread(current_status)
    local = client_is_local(request)
    urls = panel_urls()
    data["panel"].update({
        "host": PANEL_HOST,
        "port": PANEL_PORT,
        "exposed": urls["exposed"],
        "urls": urls,
        "client": request.client.host if request.client else None,
        "is_local": local,
        # 只有本机请求才回显令牌：否则局域网任何人读一次 status 就能拿到令牌，
        # 门禁形同虚设。局域网客户端只告诉它「需要令牌」以及怎么拿。
        "token": PANEL_TOKEN if (local and PANEL_TOKEN) else None,
        "token_required": bool(PANEL_TOKEN) and not local,
        "token_file": str(PANEL_TOKEN_FILE) if local and PANEL_TOKEN else None,
    })
    return JSONResponse(data)


@app.post("/api/start")
async def api_start(
    config: str = Query(DEFAULT_CONFIG),
    no_vision: bool = Query(False),
    max_seq_len: int = Query(0),
    port: int = Query(0),
) -> JSONResponse:
    import asyncio

    extra: list[str] = []
    if no_vision:
        extra.append("--no-vision")
    if max_seq_len > 0:
        extra += ["--max-seq-len", str(max_seq_len)]
    if port > 0:
        extra += ["--port", str(port)]
    result = await asyncio.to_thread(start_service, config, extra)
    return JSONResponse(result)


@app.post("/api/stop")
async def api_stop(
    graceful: bool = Query(True),
    force: bool = Query(False),
    timeout: float = Query(30.0),
) -> JSONResponse:
    import asyncio

    result = await asyncio.to_thread(stop_service, graceful, timeout, force)
    return JSONResponse(result)


@app.post("/api/restart")
async def api_restart(
    config: str = Query(DEFAULT_CONFIG),
    no_vision: bool = Query(False),
) -> JSONResponse:
    import asyncio

    extra = ["--no-vision"] if no_vision else []

    def _do() -> dict[str, Any]:
        stopped = stop_service(graceful=True, timeout=30.0)
        if stopped.get("status") == "failed":
            return {"status": "failed", "stage": "stop", **stopped}
        time.sleep(1.0)          # 等端口彻底释放
        started = start_service(config, extra)
        return {"status": "restarted", "stop": stopped, "start": started}

    result = await asyncio.to_thread(_do)
    return JSONResponse(result)


@app.get("/api/logs")
async def api_logs(tail: int = Query(200, ge=1, le=5000),
                   source: str = Query("service", pattern="^(service|panel)$")) -> JSONResponse:
    import asyncio

    return JSONResponse(await asyncio.to_thread(read_logs, tail, source))


@app.get("/api/logs/stream")
async def api_logs_stream(source: str = Query("service", pattern="^(service|panel)$")):
    return StreamingResponse(
        stream_logs(source=source),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.post("/api/smoke")
async def api_smoke(prompt: str = Query("用一句话解释什么是量化。")) -> JSONResponse:
    """发一个很小的请求，确认推理链路真的通了（不只是进程活着）。"""
    import asyncio

    def _do() -> dict[str, Any]:
        cfg = read_config(load_state().get("config") or DEFAULT_CONFIG)
        port = int((cfg.get("server") or {}).get("port") or STATE_DEFAULT_PORT)
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "enable_thinking": False,
            "stream": False,
        }
        t0 = time.time()
        try:
            resp = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json=payload, timeout=180)
        except requests.RequestException as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        dt = max(time.time() - t0, 1e-6)
        if resp.status_code != 200:
            return {"ok": False, "status_code": resp.status_code, "error": resp.text[:400]}
        body = resp.json()
        usage = body.get("usage") or {}
        done = int(usage.get("completion_tokens") or 0)
        return {
            "ok": True,
            "seconds": round(dt, 2),
            "tokens": done,
            "tok_per_s": round(done / dt, 1),
            "finish_reason": (body.get("choices") or [{}])[0].get("finish_reason"),
            "text": ((body.get("choices") or [{}])[0].get("message") or {}).get("content", "")[:600],
            "usage": usage,
        }

    return JSONResponse(await asyncio.to_thread(_do))


@app.post("/api/verify-env")
async def api_verify_env() -> JSONResponse:
    """跑一遍 verify_env.py，把输出原样返回，方便在页面上排查环境问题。"""
    import asyncio

    def _do() -> dict[str, Any]:
        venv_py = BASE / ".venv" / "Scripts" / "python.exe"
        if not venv_py.is_file():
            return {"ok": False, "output": "找不到虚拟环境，请先运行 setup.ps1"}
        try:
            proc = _run([str(venv_py), "verify_env.py"], timeout=180)
        except Exception as exc:
            return {"ok": False, "output": f"{type(exc).__name__}: {exc}"}
        out = ((proc.stdout or "") + (proc.stderr or "")).strip()
        return {"ok": proc.returncode == 0, "output": out[-4000:]}

    return JSONResponse(await asyncio.to_thread(_do))


@app.post("/api/reveal-logs")
async def api_reveal_logs(request: Request) -> JSONResponse:
    """在资源管理器里打开日志目录。"""
    import asyncio

    if request.client and request.client.host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(status_code=403, detail="只允许本机操作")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.Popen(["explorer", str(LOG_DIR)], creationflags=CREATE_NO_WINDOW)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)})
    return JSONResponse({"ok": True})


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3.8-27B 推理服务控制面板")
    ap.add_argument("--panel-port", type=int, default=DEFAULT_PANEL_PORT, help="面板监听端口（默认 8001）")
    ap.add_argument("--host", default=os.environ.get("QWEN38_PANEL_HOST", "127.0.0.1"),
                    help="面板监听地址。127.0.0.1=只本机（默认）；0.0.0.0=允许局域网访问（需要令牌）")
    ap.add_argument("--lan", action="store_true",
                    help="等价于 --host 0.0.0.0，允许局域网访问局域网机器需持令牌")
    ap.add_argument("--token", default=os.environ.get("QWEN38_PANEL_TOKEN", ""),
                    help="可选。面板访问令牌；不填则不鉴权（默认）。也可写在 _state/panel_token.txt")
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = ap.parse_args()

    global PANEL_HOST, PANEL_PORT, PANEL_TOKEN
    PANEL_HOST = "0.0.0.0" if args.lan else args.host
    PANEL_PORT = args.panel_port
    exposed = PANEL_HOST in ("0.0.0.0", "::")
    # 只有对外暴露时令牌才有意义（绑在 127.0.0.1 上本来就出不去）。
    # 而且默认就是空的 —— 局域网访问不要求令牌，见模块头部说明。
    PANEL_TOKEN = load_panel_token(args.token) if exposed else ""

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = open(PANEL_LOG, "a", encoding="utf8", errors="replace")

    class _Tee:
        """把自己 stdout/stderr 同时写进 panel.log。

        这样无论面板是前台跑还是被 start_panel.ps1 甩到后台，页面上「切换面板日志」
        看到的都是同一份完整记录，不用去猜标准流被重定向到哪儿了。

        必须实现 __getattr__ 转发：uvicorn 的日志格式化器会调用
        sys.stdout.isatty()，只转发 write/flush 的话它会直接抛
        ValueError: Unable to configure formatter 'default' 导致面板起不来。
        """
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                try:
                    st.write(s)
                    st.flush()
                except Exception:
                    pass

        def flush(self):
            for st in self.streams:
                try:
                    st.flush()
                except Exception:
                    pass

        def __getattr__(self, name):
            # 只兜底没显式定义的属性（isatty / fileno / encoding / buffer …）
            return getattr(self.streams[0], name)

    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    import uvicorn

    url = f"http://{args.host}:{args.panel_port}/"
    urls = panel_urls()
    print("=" * 66)
    print(f"控制面板：{url}")
    print(f"推理服务端口：{read_config().get('server', {}).get('port', STATE_DEFAULT_PORT)}")
    print(f"日志目录：{LOG_DIR}")
    if exposed:
        print("-" * 66)
        lan = urls["lan"] or ["（没探测到局域网地址，用 ipconfig 自己看一下）"]
        if PANEL_TOKEN:
            print("局域网访问已开启（已启用令牌）：")
            for u in lan:
                print(f"  {u}?token={PANEL_TOKEN}")
            print(f"  令牌：{PANEL_TOKEN}")
            print("  （本机 127.0.0.1 免令牌；想关掉鉴权就删除 _state/panel_token.txt）")
        else:
            print("局域网访问已开启（未启用令牌，局域网内任何机器都能启停服务）：")
            for u in lan:
                print(f"  {u}")
            print("  想加令牌：在 _state/panel_token.txt 写一行值后重启面板，或用 --token")
        print("  提示：Windows 防火墙还得放行，见 README「面板放到局域网」一节")
    print("=" * 66)

    if args.open:
        def _open() -> None:
            time.sleep(1.2)
            try:
                import webbrowser
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    # 必须用 PANEL_HOST 而不是 args.host：--lan 只把 PANEL_HOST 改成了 0.0.0.0，
    # args.host 仍是默认的 127.0.0.1。写错的话「开了局域网」其实只监听本机，
    # 而令牌已经生成 —— 外面连不上、本地一切正常，这种错很难查。
    uvicorn.run(app, host=PANEL_HOST, port=args.panel_port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
