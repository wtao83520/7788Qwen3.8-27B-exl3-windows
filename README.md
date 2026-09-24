# Qwen3.8-27B (EXL3 3.50bpw) 本地 OpenAI 兼容推理服务

在单张 **RTX 4090 24GB** 上跑 `turboderp/Qwen3.8-27B-exl3` 的 **3.50bpw** 量化模型，
对外提供 **OpenAI 格式** 的 HTTP API（`/v1/chat/completions`、`/v1/completions`、`/v1/models`）。

底层：**ExLlamaV3** + EXL3 量化格式，启用 **MTP 投机解码**（模型自带 mtp 层，无需额外草稿模型）。

---

## 1. 可行性结论

| 检查项 | 结果 |
|---|---|
| 硬件 | RTX 4090 24GB + 驱动 616.56 ✅ |
| 运行时 | ExLlamaV3 官方预编译 Windows wheel（cu132 / torch 2.13 / cp313）✅ |
| 模型体积 | `3.50bpw` 分支约 **15.4 GB**（含 BF16 视觉塔）✅ |
| 剩余显存 | 24 − 15.4 − 约 1.0（激活/CUDA 上下文）≈ **7.5 GB 给 KV 缓存** ✅ |

> ⚠️ **网络实测结论（2026-09-23，本机）— 关键是"多连接"而非"换镜像"**
>
> 单连接时所有通道都只有几十 kB/s，**但服务端允许大量并发连接**，加连接数能线性提速：
>
> | 通道 | 1 连接 | 32 连接 | 96 连接 | 说明 |
> |---|---|---|---|---|
> | **huggingface.co（走系统代理）** | ~40 kB/s | ~1.7 MB/s | **~5 MB/s** | ✅ 模型下载用它 |
> | hf-mirror.com | ~440 kB/s | ~1.0 MB/s | — | 单连接最快，但总量封顶低 |
> | aifasthub.com | ~130 kB/s | ~1.1 MB/s | 0.77 MB/s | 64 连接后反而下降 |
> | modelscope.cn | 1.28 MB/s | — | — | 只镜像热门仓库，**没有本模型的 exl3 分支** |
>
> 已确认的结论：
> - 系统里有本地代理 `http://127.0.0.1:7890`，**Python 会自动使用，`curl.exe` 不会**。
>   所以 `curl` 直连 huggingface.co 返回 HTTP 000 是正常的，不代表网络不通。
> - EXL3 量化在**分支**上，而 ModelScope 只自动同步主分支 → 国内镜像都没有这份权重。
> - `turboderp/Qwen3.8-27B-exl3` 这类仓库的文件**未被国内 CDN 缓存**，走镜像反而更慢。
>
> 因此本项目的分工：
> | 下载内容 | 通道 | 实测速度 | 耗时 |
> |---|---|---|---|
> | torch（1.9 GB） | 上科大 mirror.sjtu.edu.cn | 10 MB/s | ~3 分钟 |
> | exllamav3（412 MB） | ghfast.top 代理 GitHub | 1.2 MB/s | ~6 分钟 |
> | **模型（15.4 GB）** | **HF 直连 + 96 连接** | **5 MB/s** | **~1 小时** |
>
> 全部支持断点续传，中断后重跑同一条命令即可。

### 为什么不选 4.00bpw

`SC_4.00bpw_H5`（16.7 GB）留约 6.3 GB 给缓存、`3.50bpw`（15.4 GB）留约 7.5 GB。
两者质量差距很小，但 3.50bpw 在 **长上下文** 上更从容，也贴合社区在 3090/4090 上
「3.5bpw + 长上下文」的实测配置，所以这里默认 **`3.50bpw`**。

> 想要更高保真度可换 `SC_4.00bpw_H5`：改 `config.yaml` 的 `model.path`，
> 或 `python download_model.py --mirror --revision SC_4.00bpw_H5`。

---

## 2. 显存预算与参数建议

KV 缓存开销：该模型 64 层里只有 **16 层是全注意力**（其余 48 层是线性注意力），
每 token 约 **64 KiB**（fp16）。缓存位数越低越省：

| `max_seq_len` | fp16 (16) | 8 bit | 4 bit |
|---|---|---|---|
| 32 768 | 2.0 GB | 1.0 GB | 0.5 GB |
| 65 536 | 4.0 GB | 2.0 GB | 1.0 GB |
| 131 072 | 8.0 GB | 4.0 GB | 2.0 GB |
| 262 144 | 16.0 GB ❌ | 8.0 GB ❌ | 4.0 GB ✅ |

> 262144 + 8bit 实测是**真的不可用**，不是「勉强能跑」：能加载完，但整卡占到
> 23.57 / 23.99 GiB，只剩 0.42 GiB，服务严重依赖页面文件，连 `/health` 都超时。
> 判据：余量低于 2 GiB 就危险。实测过程见 `bench_kv.py`，结论写在 `config.yaml`
> 的 `cache_k_bits` 注释里。

> **`max_seq_len` 是缓存总预算，并发请求共享**（ExLlamaV3 动态生成器的分页缓存语义），
> 不是「每个请求各有一份」。
>
> **本仓库的 `config.yaml` 默认按「单路并发」配置**（`max_batch_size: 1`、
> `limits.max_concurrent_requests: 1`、`cache_quant: 4`），把整个缓存都给一个请求，
> 可以直接顶到 262144：
>
> | 配置 | KV 缓存 | 权重 | 激活 | 合计 |
> |---|---|---|---|---|
> | 262144 + `cache_quant: 4` | 4.0 GiB | 14.3 GiB | ~1.5 GiB | **≈ 19.8 GiB** |
>
> ❗ **262144 下不要用 `cache_quant: 8`**：8 GiB + 15.8 GiB ≈ 23.8 GiB，会 OOM。
>
> 要改回多路并发，必须同时调小 `max_seq_len` 并调大 `max_batch_size`、
> `limits.max_concurrent_requests`。

### 4090 + 3.50bpw 的几档预设

| 场景 | `max_seq_len` | `cache_quant` | 显存占用 | `max_batch_size` / 并发 |
|---|---|---|---|---|
| 日常对话（多路并发） | 32768 | 8 | ~17.5 GB | 4 / 8 |
| 长文档 / 代码库 | 131072 | 8 | ~20 GB | 1 / 1 |
| **极限 262K（当前默认，单路）** | **262144** | **4** | **~19.8 GiB** | **1 / 1** |

> 上表仅为参考，`config.yaml` 里已是最后一行的单路 262K 配置。

另外：线性注意力层的**循环状态放在系统内存**里（ExLlamaV3 `recurrent_cache_size` 默认 4 GB），
跑长上下文时建议系统内存 **≥ 32 GB**。

---

## 3. 目录结构

```
qwen3.8-27b-exl3/
├── setup.ps1            # 一键：建 venv + 装 torch + 装 exllamav3 + 装依赖
├── _bootstrap.ps1       # 全自动（含模型下载、可断点续传、日志落盘），后台跑首选
├── start.ps1            # 启动服务
├── start_panel.ps1      # 启动控制面板（网页上启停服务）
├── ecosystem.config.js  # ★ pm2 常驻托管配置（开机自启 + 崩溃自动拉起）
├── server.py            # OpenAI 兼容服务（FastAPI + ExLlamaV3）
├── scripts/
│   └── run_server.py    # ★ pm2 用的启动包装（孤儿看门狗 + 日志 + 优雅退出）
├── control_panel.py     # ★ 控制面板后端：负责启停 server.py、看日志、冒烟测试
├── make_icon.py         # 生成面板图标（favicon.ico / PNG / apple-touch）；改了配色要重跑
├── web/
│   ├── index.html       # ★ 控制面板页面（无外部依赖，离线也能用）
│   ├── icon.svg         # 图标源文件（手写，浏览器优先用它）
│   ├── favicon.ico      # ┐
│   ├── icon-192.png     # │ 由 make_icon.py 从同一套几何形状生成
│   ├── icon-512.png     # │（Pillow 读不了 SVG，所以两边各画一次）
│   ├── apple-touch-icon.png  # ┘
│   └── manifest.webmanifest  # 手机「添加到主屏幕」用
├── config.yaml          # 全部可调参数
├── fast_download.py     # ★ 多连接高速下载模型（推荐），自动测速选源
├── download_model.py    # 按分支下载（huggingface_hub，单连接，备用）
├── probe_sources.py     # 对比各下载通道速度
├── speed_test.py        # 测试单/多连接吞吐差异
├── check_model.py       # 校验模型目录完整性（逐文件比对大小，快但查不出内容损坏）
├── verify_model.py      # ★ 用官方 sha256 校验权重（唯一能发现静默损坏的方法）
├── patch_model.py       # 只重下损坏的字节区间并原地写回，不用重下整个分片
├── verify_env.py        # 环境自检（torch / exllamav3 / CUDA 扩展 / GPU）
├── selftest.py          # 逻辑自检（模板渲染、思维链切分、工具解析），不需要 GPU
├── test_client.py       # 端到端冒烟测试（流式/非流式/logprobs/图片/工具调用）
├── check_features.py    # 针对性验收（思考模式分离、max_tokens 边界、长上下文吞吐）
├── requirements.txt
├── .venv/               # 虚拟环境
├── _wheels/             # 下载的 wheel（可复用，可删）
├── _tmpl/               # 真实 chat_template.jinja，供 selftest.py 使用
├── logs/                # 服务 / 面板日志（service.log、panel.log、pm2-*.log）
├── _bench/              # KV 量化测评（bench_kv.py / draft_check.py 等）
├── _state/              # 面板/pm2 运行时状态（启动参数，可删）
└── models/              # 模型权重
    ├── Qwen3.8-27B-3.50bpw/        # 主模型（EXL3 3.50bpw）
    ├── DFlash2-EXL3-4.00bpw/       # 可选草稿（见第 13 节），不用可删
    └── DFlash2-EXL3-5.0bpw/        # 可选草稿（5bit，更准一点也更大）
```

---

## 4. 快速开始

```powershell
cd c:\AI\qwen3.8-27b-exl3

# 方式 A：全自动（推荐）
powershell -ExecutionPolicy Bypass -File .\_bootstrap.ps1
#   内部依次：上科大镜像下 torch → ghfast 代理下 exllamav3 → 装依赖 → fast_download.py 下模型

# 方式 B：分步手动
.\setup.ps1                                          # 建环境（torch + exllamav3 + 依赖）
.\.venv\Scripts\python.exe verify_env.py             # 环境自检，应输出「结果 : 通过」
.\.venv\Scripts\python.exe fast_download.py          # 多连接高速下模型（自动测速选源）
.\start.ps1                                          # 启动服务
```

### 下载模型（重点）

用 `fast_download.py`，**不要**用单连接的 `hf download`：

```powershell
.\\.venv\\Scripts\\python.exe fast_download.py --probe-only        # 先看各通道速度
.\\.venv\\Scripts\\python.exe fast_download.py                     # 自动选最快的源
.\\.venv\\Scripts\\python.exe fast_download.py -s hf-direct -j 96  # 指定源 + 连接数
.\\.venv\\Scripts\\python.exe fast_download.py -r SC_4.00bpw_H5    # 换分支
```

> **连接数比换镜像重要得多**：1 连接 ~40 kB/s，96 连接能到 ~5 MB/s（125 倍）。
> 分段进度记在 `<文件>.part.json`，中断后重跑同一条命令自动续传。

`_bootstrap.ps1` 的参数：

```powershell
.\_bootstrap.ps1 -OnlyModel                 # 只下模型（环境已就绪时）
.\_bootstrap.ps1 -SkipTorch                 # 跳过 torch
.\_bootstrap.ps1 -Revision SC_4.00bpw_H5    # 换分支
.\_bootstrap.ps1 -Cuda cu128 -TorchVersion 2.9.0   # 换 wheel 组合
Get-Content bootstrap.log -Tail 20 -Wait    # 另开一个窗口看进度
```

> 后台运行时请保持该终端 / VS Code 开着。中途关掉也没关系：
> 三个下载全部支持断点续传，重跑同一条命令即从断点继续。

下载完成后**一定要校验**，别省略这一步：

```powershell
.\\.venv\\Scripts\\python.exe verify_model.py            # 用官方 sha256 逐个文件比对
.\\.venv\\Scripts\\python.exe check_model.py --crc       # 附带仓库自带的 crc32 对照
```

> **为什么必须用 sha256**：下载器会先把文件 `truncate` 到完整大小再分段填充，
> 所以「大小对」完全不能证明「内容对」。被中断的下载会留下**稀疏空洞**
> （读出来是 0），大小却分毫不差，`check_model.py` 那种按大小比对的检查查不出来，
> 启动时也不报错，只会输出垃圾。
>
> 另外注意：仓库自带的 `crc32.txt` **已经过时**（会报 3 个文件不匹配，但重下后
> 逐字节完全一致），而且它**不覆盖 safetensors 权重**，不要拿它下结论。

如果 `verify_model.py` 报了损坏：

```powershell
.\\.venv\\Scripts\\python.exe patch_model.py --scan       # 先看空洞在哪
.\\.venv\\Scripts\\python.exe patch_model.py              # 只重下这些区间，秒级修复
```

实测案例：分片 1 因下载被打断留下 **10 处空洞共 13.5 MB**，且**没有任何一个
整 4MB 分片是全零的**（每个分段前半段写对了、只有尾部是零），所以粗粒度零扫描
也查不出来。`patch_model.py` 用 64 KB 粒度定位并原地修补，比重下 8.5 GB 快得多。

### 预计下载耗时（本机实测带宽）

| 项目 | 体积 | 单连接 | 优化后 |
|---|---|---|---|
| torch wheel | 1.9 GB | 阿里云 ~380 kB/s（约 1.3 h） | **上科大镜像 10 MB/s（3 分钟）** |
| exllamav3 wheel | 412 MB | GitHub 直连 ~35 kB/s（约 3.3 h） | **ghfast.top 代理 1.2 MB/s（6 分钟）** |
| 模型 3.50bpw | 14.3 GB | hf-mirror ~440 kB/s（约 10 h） | **HF 直连 + 96 连接 6–8 MB/s（约 35 分钟）** |

> 网络曾经是主要瓶颈，但根因不是带宽不足而是**单连接被限速**：
> 1 连接 ~40 kB/s，96 连接能到 6–8 MB/s。所有下载都支持断点续传。

### 控制面板（在网页上启停服务）

不想每次开终端敲命令的话，用控制面板：

```powershell
.\start_panel.ps1                    # 启动并自动开浏览器（仅本机）
.\start_panel.ps1 -Lan               # 允许局域网访问
.\start_panel.ps1 -NoBrowser         # 不开浏览器
.\start_panel.ps1 -Background        # 甩到后台，关掉终端也不退出
.\start_panel.ps1 -Background -Force # 先停掉旧面板再启动（改过 control_panel.py 后用）
.\start_panel.ps1 -PanelPort 8081    # 换面板端口（默认 8001）
```

打开 `http://127.0.0.1:8001/`，页面上有：

| 功能 | 说明 |
|---|---|
| **启动 / 停止 / 重启** | 三个按钮直接管服务；停止走优雅退出（约 2 秒，会释放显存） |
| **强制停止** | 优雅退出卡住时的保险，直接 `taskkill /T /F` |
| **API 接入地址** | 列出可直接填给客户端（LM Studio / 脚本）的 `base_url`，**每行都有「复制」**，还有「复制全部」 |
| **冒烟测试** | 真发一个请求，显示 tok/s，确认推理链路通了 |
| **环境自检** | 在页面上跑 `verify_env.py`，不用切终端 |
| **实时日志** | 跟着日志流滚动，可按级别上色；默认过滤掉面板自己的健康检查探测 |
| **生效配置** | 把 `config.yaml` 里真正生效的关键项列出来，不用翻文件 |
| **显存 / GPU** | 实时占用条，以及是哪些进程占着 GPU |
| **图标** | 标签页 / 书签 / 任务栏都有图标，手机可「添加到主屏幕」（见下） |

### API 接入地址（一键复制）

面板顶部有一块「API 接入地址」，直接把要填进客户端的 `base_url` 列出来：

```
本机    http://127.0.0.1:2345/v1            [复制]
局域网  http://192.168.31.67:2345/v1        [复制]
```

- 已经是 **`base_url`**（带 `/v1`，不带具体端点），填进 LM Studio 的
  「Server Base URL」或 OpenAI SDK 的 `base_url` 就能直接用，不用自己拼。
- 下面会同时列出四个端点（`/v1/chat/completions`、`/v1/completions`、
  `/v1/models`、`/health`）和当前是否要鉴权。
- 局域网地址是本机网卡的真实 IP，由服务端探测（多网卡时会优先给默认路由那张），
  不是拿浏览器地址栏拼的 —— 面板在 8001、服务在 2345，靠前端拼必错。
- 服务停着的时候地址仍然显示，只是置灰：地址没变，只是连不上。

复制功能有个实现上的坑：`navigator.clipboard` **只在安全上下文里存在**
（https 或 localhost）。局域网用户从 `http://192.168.31.67:8001` 打开时它是
`undefined`，所以代码里留了一条 `document.execCommand('copy')` 的退路，
两种环境都能复制。

### 面板放到局域网（`-Lan`）

```powershell
.\start_panel.ps1 -Background -Lan
```

加 `-Lan` 后面板监听 `0.0.0.0:8001`，局域网机器直接开
`http://192.168.31.67:8001/` 就能用（当前**不需要令牌**），本机地址依然是
`http://127.0.0.1:8001/`，两种都行。启动时会打印可用的局域网地址。

> ⚠️ 这等于把「启停推理服务」的权限开放给了能连上这个端口的机器。
> 家里 / 实验室的内网一般没问题，**不要在不受信任的网络上这么开**。

以后想加一道门**不用改代码**，任选一种后重启面板：

```powershell
# 方式 1：命令行 / 环境变量指定
.\.venv\Scripts\python.exe control_panel.py --lan --token 你的令牌

# 方式 2：写一行到 _state/panel_token.txt，面板启动时自动读
"你的令牌" | Out-File -Encoding utf8 _state\panel_token.txt
.\start_panel.ps1 -Background -Lan -Force      # 重启生效
```

（`-Lan` 是给 `start_panel.ps1` 用的；想连令牌一起指定就直接跑
`control_panel.py --lan --token …`，或设环境变量 `QWEN38_PANEL_TOKEN`。）

一旦令牌非空：**本机仍然免令牌**（体验不变），局域网机器则必须带令牌，
否则只能看首页和 `/api/status`，启停类接口返回 401。前端会自动弹框让你输令牌，
输一次存在 `localStorage`，之后每次请求自动带上。

#### 局域网连不上时先看防火墙

`-Lan` 只解决「监听哪个网卡」，Windows 防火墙是另一道。当前网卡配置是
Private（会被放行），但换网络后可能变成 Public 而被拦。以管理员身份跑一次：

```powershell
New-NetFirewallRule -DisplayName 'Qwen38 控制面板' -Direction Inbound `
  -Protocol TCP -LocalPort 8001 -Action Allow -Profile Private
```

排查顺序：`Get-NetTCPConnection -LocalPort 8001 -State Listen` 应看到
`0.0.0.0`（只看到 `127.0.0.1` = 没加 `-Lan`）→ 本机 `curl` 局域网 IP →
别的机器。推理服务（2345）用的是同一个原理，但它的监听地址由
`config.yaml` 的 `server.host: 0.0.0.0` 决定。

### 图标（favicon / 手机主屏）

面板带一套完整图标，标签页、书签、任务栏固定、手机主屏都能用：

| 文件 | 规格 | 用途 |
|---|---|---|
| `web/icon.svg` | 矢量 | 主图标，现代浏览器优先用它 |
| `web/favicon.ico` | 16/32/48/64 | 旧书签、任务栏固定 |
| `web/icon-192.png` `icon-512.png` | 192/512 | Android / PWA manifest |
| `web/apple-touch-icon.png` | 180（不透明） | iOS「添加到主屏幕」 |

图标是一个风格化的 **Q**（圆环 + 尾巴）—— 环用面板的强调色 `#58a6ff`，
右上角一个绿点对应面板的「运行中」指示灯，整体在 16px 下也能认出来。
同一份图标也显示在页面标题左边。

配色/比例改了之后要同时在两处改并重新生成：

```powershell
# ① 改 web/icon.svg（手写，浏览器用的就是这份）
# ② 改 make_icon.py 顶部的设计常量（必须与 SVG 一致）
.\.venv\Scripts\python.exe make_icon.py --preview   # --preview 额外输出放大对比图
```

> 为什么 SVG 和 PNG 是两套实现？Pillow 读不了 SVG，而装 `cairosvg` 只是为了
> 构建图标又太重。宁可两处各画一次，也不为图标引入一个运行时依赖。
> 代价是：**改配色要两处一起改**。

> 改完图标如果浏览器还显示旧的，把 `web/index.html` 里图标链接上的 `?v=2`
> 往上加一位（图标会被缓存，这个参数用来强制刷新）。

### 面板为什么不能和推理服务用同一个端口

看着确实更简洁（一个端口搞定），但合不了，三个原因都是结构性的：

1. **鸡生蛋** —— 面板的核心职责是「在服务没跑的时候把它拉起来」。它必须比服务
   活得久，所以只能是个独立进程。合成一个进程后，点「停止服务」会把面板自己
   一起干掉，页面也一起没了，就再也没法用自己的按钮把服务启回来。
2. **安全** —— 推理服务的 `host` 是 `0.0.0.0`（局域网的 LM Studio 都在调它）。
   管理端点和推理端点放同一个监听端口上，等于把「谁能关掉我的服务」向整个
   局域网开放。现在面板默认只听 `127.0.0.1`，要用 `-Lan` 才显式开。
3. **隔离** —— 模型 OOM、CUDA 崩掉会把整个进程带走。如果面板同进程，就恰恰在
   最需要看日志 / 重新拉起的时候没了；分开的话服务崩了面板还在，能直接拉起来
   （README 第 14 节的 pm2 方案也是同一个思路）。

所以是**两个进程、两个端口**：面板 8001（管理面）、服务 2345（服务面）。
两边的监听地址都可以单独调：面板 `-Lan`、服务 `config.yaml` 的 `server.host`。

---

## 5. 离线自检（不需要模型、不需要 GPU）

在 15 GB 下载完之前，就能先把最容易出错的部分验证掉：

```powershell
.\.venv\Scripts\python.exe selftest.py
```

它会用 `_tmpl/chat_template.jinja`（从模型仓库取的真实模板）检查：

- 思考模式开 / 关时 prompt 结尾是否正确（`<think>` 与预填空 think 块）
- `reasoning_effort` / `preserve_thinking` 是否生效，非法取值是否被拒
- 工具定义是否注入系统提示、图片内容块是否生成视觉占位
- `ThinkSplitter` 在流式 3 字符分块下是否把推理 / 正文切干净且不丢字
- `<tool_call>` 解析、OpenAI 消息（含 `tool_calls` / `tool_call_id`）转换
- 采样参数合并、`max_tokens=0` 等非法输入拦截

全部输出 `[PASS]` 且最后打印「全部通过」即正常。

---

## 6. 调用示例

> 不确定 `base_url` 该填什么（尤其换成局域网 IP 之后）？直接看控制面板
> 顶部的「**API 接入地址**」卡片，每行一个现成地址，点「复制」即可。

### curl

```bash
curl http://127.0.0.1:2345/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B-3.50bpw",
    "messages": [{"role": "user", "content": "你好，介绍一下你自己"}],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 1024,
    "enable_thinking": false
  }'
```

> 端口以 `config.yaml` 的 `server.port` 为准（本仓库当前是 **2345**）。
> 下面的例子也一样，换了端口要一起改。

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:2345/v1", api_key="not-needed")

stream = client.chat.completions.create(
    model="Qwen3.8-27B-3.50bpw",
    messages=[{"role": "user", "content": "写一个 Python 快速排序"}],
    temperature=0.6,
    top_p=0.95,
    top_k=20,              # 非标准参数，本服务支持
    extra_body={
        "enable_thinking": True,
        "reasoning_effort": "xhigh",   # 合法值只有 xhigh / medium / low
    },
    stream=True,
)
for chunk in stream:
    d = chunk.choices[0].delta
    if getattr(d, "reasoning_content", None):
        print(d.reasoning_content, end="")   # 思维链
    if d.content:
        print(d.content, end="")             # 正式回答
```

### 图片输入（多模态）

```python
import base64
b64 = base64.b64encode(open("cat.png", "rb").read()).decode()
resp = client.chat.completions.create(
    model="Qwen3.8-27B-3.50bpw",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "描述这张图片"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}],
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

也支持 `http(s)://` 图片地址和本地文件路径。

**视觉塔是包含在权重里的，开箱可用**（`load_vision: true`，实测已验证）：

| 项 | 值 |
|---|---|
| 架构声明 | `Qwen3_5ForConditionalGeneration`，`language_model_only: false` |
| 权重位置 | `model-00002-of-00002.safetensors` 里的 **333 个 `model.visual.*`**（BF16 未量化） |
| 视觉塔规模 | 27 层 / hidden 1152 / out_hidden 5120 / patch 16 |
| `/health` | `vision: true` |

> ⚠️ 视觉张量**不出现在** `quantization_config.json` 的 `tensor_storage` 里 ——
> 那张表只列**被量化**的张量（707 项全是 `model.language_model.*` + `lm_head`）。
> 想确认视觉权重在不在，要看 safetensors 头部，不能看量化清单。

自检命令（会生成一张可控图片并检答案，`--dots N` 换数量）：

```powershell
.\.venv\Scripts\python.exe _bench\vision_check.py --dots 5
```

实测 N=1/3/5/6/7 全部答对（0.2–0.5 s）。不需要图片可以设 `load_vision: false` 省约 0.7 GB。

> 注意显存：开 DFlash2 后空闲只剩约 1.5 GiB，图片输入需要额外的视觉激活显存。
> 若图片请求报错，先看 `logs/service.log` 是不是 CUDA OOM。

---

## 7. 支持的参数

### 采样参数（请求级，覆盖 `config.yaml` 的 `defaults`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `temperature` | 0.6 | 0 表示贪心解码 |
| `top_p` | 0.95 | 核采样 |
| `top_k` | 20 | 0 = 关闭 |
| `min_p` | 0.0 | 最小概率阈值 |
| `repetition_penalty` | 1.0 | 1.0 = 关闭 |
| `presence_penalty` | 0.0 | |
| `frequency_penalty` | 0.0 | |
| `temperature_last` | false | 把温度放在 top_p/top_k 之后应用 |
| `logit_bias` | — | `{"token_id": bias}`，非数字键忽略 |
| `dry_multiplier` | 0.0 | DRY 重复抑制，>0 生效 |
| `seed` | — | 固定随机种子（不保证完全确定性） |

### 输出控制

| 参数 | 说明 |
|---|---|
| `max_tokens` / `max_completion_tokens` | 生成长度上限；**请求里传了就以它为准**，没传才用 `defaults.max_tokens` |
| `stop` | string 或 string 列表 |
| `stream` | SSE 流式 |
| `stream_options.include_usage` | 流式末尾返回 `usage`（默认 true） |
| `logprobs` / `top_logprobs` | 返回每个 token 的 log 概率与 top-k 候选 |
| `n` | 只支持 1 |

### 上下文长度：四个参数，各管一段

「上下文长度」和「生成长度」是两件事，容易搞混：

| 参数 | 位置 | 管什么 |
|---|---|---|
| `model.max_seq_len` | `config.yaml` | **总上下文窗口**：`prompt + 生成 + 预留 ≤ 它`，本机 262144 |
| `defaults.max_tokens` | `config.yaml` | **只在请求没传 `max_tokens` 时生效**的默认值（32768） |
| `defaults.max_tokens_limit` | `config.yaml` | 请求 `max_tokens` 的硬天花板，`null` = 不限制 |
| `max_tokens` / `max_completion_tokens` | **请求体** | 单次生成长度上限；**传了就以它为准，配置默认值完全不起作用** |

**结论：服务端不会限制你生成多长。** `max_tokens_limit: null`，所以请求里写
`max_tokens: 262144`（甚至 `999999`）都行，服务会自动收窄到 `max_seq_len - prompt - 预留`：

```text
可生成长度 = 262144 - prompt_tokens - 6
```

预留的 6 = `RESERVED_TOKENS(2)` + MTP 草稿 4（见第 12 节 ③）。
所以塞进 30K 的代码库上下文，还能生成约 232K token。

> **生成长度不够时，先怀疑客户端自己传了个小的 `max_tokens`**（不少 IDE 插件默认
> 4096/8192，或者按它以为的模型上下文自行推算），而不是服务端的默认值。
> 服务日志里每个请求都会打一行，括号里就是来源：
>
> ```
> 本请求：prompt 57 + 生成上限 32768（配置默认）+ 预留 6 = 占满 32831 / 上下文 262144
> 本请求：prompt 57 + 生成上限 64（请求指定）+ 预留 6 = 占满 127 / 上下文 262144
> 本请求：prompt 57 + 生成上限 262081（请求指定，再被上下文上限收窄）+ 预留 6 = 占满 262144 / 上下文 262144
> ```
>
> 可能出现的来源标注：`配置默认`、`请求指定`、`…，被 max_tokens_limit=32768 截断`、
> `…，再被上下文上限收窄`。
>
> 客户端一般需要手动把「上下文长度」设成 262144——模型名 `Qwen3.8-27B-3.50bpw`
> 不在客户端的已知模型表里，它可能退回到很保守的默认值。

### Qwen3.8 特有（顶层字段或 `extra_body`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `enable_thinking` | true | 关闭后不输出思维链，响应更快 |
| `reasoning_effort` | xhigh | `xhigh` / `medium` / `low` —— **软引导**，不是 token 预算，见下 |
| `preserve_thinking` | true | 多轮对话时是否在历史里保留 `<think>` 内容 |
| `tools` / `tool_choice` | — | 会把工具定义注入系统提示，并从输出解析 `<tool_call>` 成 `tool_calls`（**流式 / 非流式都支持**） |

### 工具调用（Function Calling）

模型输出的是自己模板里的 XML 格式，服务端把它转成 OpenAI 的 `tool_calls`：

```
模型原始输出：                          客户端收到：
<tool_call>                             message.tool_calls = [{
<function=grep_search>                     "id": "call_...",
<parameter=query>llm-setting</parameter>   "type": "function",
<parameter=isRegexp>True</parameter>       "function": {
</function>                                   "name": "grep_search",
</tool_call>                                   "arguments": "{\"query\": \"llm-setting\", \"isRegexp\": true}"
                                           }
                                         }]
                                         finish_reason = "tool_calls"
```

几个要点：

- **流式和非流式都解析**。流式下按 OpenAI 协议分帧发
  （先 `id` + `name`、再 `arguments`），最后 `finish_reason: "tool_calls"`。
- **参数类型按 tool schema 转换**。模型参数是纯文本，类型信息只存在于 schema 里。
  服务端会按声明的类型转：`boolean` → 真布尔、`integer`/`number` → 数字、
  `string` → 保持字符串。没声明类型时退回启发式（JSON 字面量，
  再加一层大小写不敏感的 `true`/`false`）。
  > 这层转换不是可选项：模型经常写成 `True`（首字母大写），而 JSON 只认小写
  > `true`。不转的话 `isRegexp` 会变成**字符串** `"True"`，
  > 按 schema 校验的客户端会直接拒掉整个工具调用。
- **被 `max_tokens` 截断也能救回来**。少了 `</function>`/`</tool_call>` 时用宽松
  解析尽量取出已收全的参数，而不是把整段 XML 当正文返回。
- **正文和工具调用可以共存**：`<tool_call>` 之外的文字照常进 `content`，
  标签本身不会泄到正文（连半个 `<tool` 都不会提前发出去）。
- 开启思考模式时，`<tool_call>` 约定出现在正文段；推理段进 `reasoning_content`。

#### 在 VS Code Copilot 里用这个模型

Copilot 只支持 **OpenAI 兼容的自定义模型（BYOK）**，且**一律走流式** ——
所以“非流式能用工具、流式不能用”这种问题在 Copilot 里会表现为工具完全不可用。

在 VS Code 里（`Manage Models…` → `OpenAI` → 添加自定义模型）填：

| 项 | 值 |
|---|---|
| Base URL | `http://192.168.31.67:2345/v1`（本机可用 `http://127.0.0.1:2345/v1`）|
| API Key | 随便填一个非空值（`config.yaml` 的 `api_keys` 为空时不校验，但客户端通常不接受空串）|
| Model / ID | 照实际填；服务端不校验 `model` 字段 |

> 上面的 Base URL 直接到控制面板顶部「**API 接入地址**」卡片里点「复制」就行。

已知限制（不是 bug，是 Copilot 侧的预期）：

- **Agent 模式有额外要求**：Copilot 的 Agent 模式对工具的「指挥」能力要求较高，
  本模型属于社区量化版，不能保证在复杂多轮 agent 循环里表现稳定。
  单次工具调用（模型决定调哪个函数、参数是什么）是可靠的。
- Copilot 不会把整个工作区塞进 prompt，它会自己检索后把相关片段给你；
  所以**不需要**为了 Copilot 把 `max_seq_len` 调得更大。
- 图片输入在 Copilot 里不可用（Copilot 的 BYOK 不传图片）；
  要试视觉能力请用 README 第 6 节的图片示例。

### 思考深度（`reasoning_effort`）

**支持，但它不是「思考 token 预算」，而是往 system 位置塞一句自然语言提示。**

模型模板里只有两个分支（见 `chat_template.jinja` 第 47 行起）：

| 取值 | 实际注入的 system 内容 | 效果 |
|---|---|---|
| `xhigh`（默认） | *"Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."* | 显式要求仔细推敲、验证假设、考虑备选方案 |
| `medium` | **什么都不注入** | 等同「不说」，由模型自己决定 |
| `low` | *"Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."* | 要求简短思考、直奔结论 |

几个必须知道的性质：

- ★ **没有数值化的预算**。三者差别就是上面那两句话，所以是「软引导」——
  模型完全可以不听话。（对比 OpenAI 的 `reasoning_effort` 是真实控制推理规模，
  本模型只是提示词。）
- ★ **`medium` 是空操作**，不是「中等强度」。它是模板里唯一没写分支的档位，
  所以 `medium` 比 `xhigh` 和 `low` 都更「放任」。
- ★ **真正能硬限思考长度的只有 `max_tokens` / 上下文上限**（超了 `finish_reason: length`）。
- 合法值只有这三个，**`high` 不是合法值**，传了会 400。
- `enable_thinking: false` 时 `reasoning_effort` 被完全忽略（不注入任何东西）。
- 自己带了 system 消息时，注入内容拼在它**前面**（同一段）；
  带 `tools` 时拼在同一段 tools 提示里，而不是另起一段 system。

实测（同一问题、`temperature=0`、`seed=1234`，用 `_bench/effort_check.py --live`）：

| 档位 | 思维链 | 正文 | completion tokens |
|---|---|---|---|
| `xhigh` | **306 字符** | 141 | 173 |
| `medium` | 130 字符 | 163 | 202 |
| `low` | 142 字符 | 120 | 186 |

`xhigh` 明显更长（+58% 对齐 medium/low），但 **`medium` 与 `low` 基本无差别** ——
对这种「模型心里已有定论」的简单问题，不加约束本来就写不长，所以两句提示
拉不开差距。换成真正需要权衡的问题（方案选型、有取舍的设计决策）差异才会明显。

实践建议：

- 想要**尽量少想、快出结果** → `low`（`medium` 起不到这个作用）。
- 想要**最充分的推理**（写代码、排查问题）→ 保持默认 `xhigh`，
  并把 `max_tokens` 给够（见第 8 节，`xhigh` 的思维链动辄上万 token）。
- 想**彻底关掉思考** → `enable_thinking: false`，比调 `reasoning_effort` 干净得多。

### 行为说明

- **思考链拆分**：开启思考时，`<think>…</think>` 的内容放在 `message.reasoning_content`，
  正式回答放在 `message.content`（流式下按 `delta.reasoning_content` / `delta.content` 分别推送）。
  由 `chat_template.split_reasoning` 控制，设为 false 则全部塞进 `content`。
- **`finish_reason`**：达到 `max_tokens` 为 `length`，命中工具为 `tool_calls`，其余为 `stop`。
- **鉴权**：`server.api_keys` 为空则不校验；填了之后请求需带 `Authorization: Bearer <key>`。

---

## 8. `config.yaml` 关键项

下面是与仓库里实际生效的 `config.yaml` 一致的摘要（单路并发 + 262K 上下文）：

```yaml
server:
  host: 0.0.0.0
  port: 2345               # 改成别的端口后，上面 curl / SDK 例子里的端口也要改
  api_keys: []            # 填了就要求 Authorization: Bearer <key>
  allow_shutdown: true    # 允许控制面板调 /admin/shutdown 优雅退出（仅限本机回环）
  cors_origins: ["*"]

model:
  path: models/Qwen3.8-27B-3.50bpw   # 模型目录
  device: cuda:0
  max_seq_len: 262144       # 缓存 token 总预算（并发共享），上限就是 262144
  cache_quant: 4            # 16(不量化) / 8 / 6 / 4；262144 下必须用 4
  cache_k_bits: null        # 可单独指定 K 的位数，null 表示跟随 cache_quant
  cache_v_bits: null
  max_batch_size: 1         # 并发批大小（单路并发＝把整套缓存给一个请求）
  max_chunk_size: 2048      # prefill 分块大小
  max_history: null         # 递归层历史预留；null = 自动跟随草稿长度（必须留够，见第 12 节）
  load_vision: true         # 不需要图片就设 false，省约 0.7 GB
  mtp_draft: true           # MTP 投机解码（加速，默认开）
  draft_model: null         # 外部草稿模型目录（DFlash2），设置后优先于 MTP；见第 13 节
  tensor_parallel: false    # 单卡保持 false；双卡拼显存可开

defaults:                   # 采样默认值，请求里可覆盖
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  max_tokens: 32768         # 默认生成长度；思考模式(xhigh)不建议低于 32768
  max_tokens_limit: null    # null = 不限制，交给 max_seq_len - prompt - 预留 自动收窄

chat_template:
  vars:
    enable_thinking: true
    reasoning_effort: xhigh
    preserve_thinking: true

limits:
  max_concurrent_requests: 1   # 想开并发请同时调大 max_batch_size 并调小 max_seq_len
```

命令行可临时覆盖：`.\start.ps1 -Port 8080 -MaxSeqLen 65536 -NoVision`

---

## 9. 环境变量

| 变量 | 作用 |
|---|---|
| `QWEN38_CONFIG` | 配置文件路径 |
| `QWEN38_HOST` / `QWEN38_PORT` | 监听地址 / 端口 |
| `QWEN38_MODEL_PATH` | 模型目录 |
| `QWEN38_MAX_SEQ_LEN` | 上下文预算 |
| `QWEN38_CACHE_QUANT` | KV 量化位数 |
| `QWEN38_API_KEYS` | `key1,key2` 逗号分隔 |

---

## 10. 常见问题

**Q：`huggingface.co` 连不上 / 下载卡住**
A：分三种情况：
① 用 `curl.exe` 直连 huggingface.co 会失败（curl 不读系统代理），**这是正常的**，
   本项目只用 Python 下 HF 文件，Python 会自动走系统代理。
② 下载慢：**真正的原因是单连接被限速**，不是带宽不够。用 `fast_download.py` 加连接数
   （`-j 96`），实测能从 40 kB/s 提到 5 MB/s。
③ 想确认各通道快慢：`python probe_sources.py`。

**Q：国内镜像（ModelScope / hf-mirror）下有更快吗**
A：对**这个模型没有**。EXL3 量化放在 git 分支上，而 ModelScope 只自动同步主分支，
所以国内镜像都没有这份权重；hf-mirror 又没有缓存它，反而比直连慢。
（ModelScope 上的 `unsloth/Qwen3.8-27B-GGUF` 等热仓库倒是很快，
 但那需要改用 llama.cpp 后端，不是 EXL3。）

**Q：`setup.ps1` 报找不到 wheel**
A：CUDA / torch / Python 三者要组合存在。到
`https://github.com/turboderp-org/exllamav3/releases` 看资产名，例如
`exllamav3-1.5.1+cu132.torch2.13.0-cp313-cp313-win_amd64.whl` 对应
`-Cuda cu132 -TorchVersion 2.13.0` + Python 3.13。换组合重跑即可。

**Q：控制面板打开是空白的 / 按钮点了没反应**
A：按顺序排查：
① 面板进程还在不在：`curl http://127.0.0.1:8001/api/status`，
   连不上就重新 `.\start_panel.ps1`；
② 看 `logs\panel.log`，面板启动失败的原因都会写在那里；
③ 面板只监听 127.0.0.1，用 `localhost` 或别的机器 IP 访问都会连不上，请用
   `http://127.0.0.1:8001/`。

**Q：面板里点“停止服务”失败 / 卡住**
A：正常情况下停止走 `/admin/shutdown`，约 2 秒完成并释放显存。如果卡住，多半是：

- 配置里把 `server.allow_shutdown` 设成了 `false` → 面板拿不到优雅退出入口，
  会自动退化为强制结束（30 秒超时后 taskkill）；
- 服务是用**旧版代码**启动的（改 `server.py` 之前启的），没有那个接口 →
  同样会退化为强制结束，重启一次面板启动的服务即可；
- 实在不行点页面上的**强制停止**，它直接 `taskkill /T /F`，不走优雅流程。

**Q：面板会不会把服务一起带走？**
A：不会。面板和推理服务是两个独立进程，关掉面板（或 `Ctrl+C`）不影响正在跑的
服务；重新打开面板也能自动接管（它靠扫进程发现服务，不依赖状态文件）。

**Q：启动时报显存不足（CUDA out of memory）**
A：先看是不是**别的程序占着显存**。用 `nvidia-smi` 检查：

```powershell
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

常见占用者：**LM Studio**（`llama-server.exe`）、其它 llama.cpp 实例、
Ollama、ComfyUI、正在跑的游戏。这些必须先自己关掉 / 在 LM Studio 里 Unload Model，
否则 24GB 卡没地方放 14.3 GiB 权重。

确认显存已释放后，再按顺序调参：① 降 `max_seq_len`；② 降 `cache_quant`（8 → 4）；
③ `load_vision: false`；④ `mtp_draft: false`。

**Q：第一次请求特别慢**
A：正常。首次会做 CUDA 图/kernel 编译与显存预热，之后短上下文稳定在 **170 tok/s**
左右（开 MTP）。

**Q：想要更高吞吐**
A：`mtp_draft: true`（默认已开）+ 保持 `max_batch_size ≥ 4`；
ExLlamaV3 的 `max_q_size`、`num_draft_tokens` 属于进阶调参，本服务未暴露。

**Q：怎么知道 KV 缓存用了多少显存**
A：`curl http://127.0.0.1:2345/health` 会返回 GPU 显存余量，以及实际生效的 KV 位数
（`cache_k_bits` / `cache_v_bits`，非对称量化时会和 `cache_quant` 不一致）和
当前默认生成长度（`max_tokens`）。

**Q：下载中断了怎么办**
A：直接重跑同一条命令即可。`_bootstrap.ps1` 用 `curl -C -` 断点续传，
`download_model.py` 依赖 `huggingface_hub` 的续传。已下好的部分不会重下。

**Q：`start.ps1` / `setup.ps1` 报一堆「缺少表达式」之类语法错误**
A：这两个脚本含中文，必须是 **UTF-8 with BOM** 编码（PowerShell 5.1 会按系统 ANSI
即 GBK 读取无 BOM 的文件，导致乱码并连带解析失败）。仓库里的文件已带 BOM；
如果你手动编辑后保存成了「UTF-8 无 BOM」，重新存成「UTF-8 with BOM」即可，
或者干脆用 `python server.py` 直接启动绕开脚本。

**Q：想省一半下载时间**
A：换更小的分支，例如 `-r SC_2.00bpw_H3`（10.2 GB）或 `-r 3.00bpw`（13.8 GB），
改 `config.yaml` 的 `model.path` 指向新目录即可。

**Q：能不能不用这个仓库**
A：只要不是 NVIDIA 卡、或想用 Ollama / LM Studio / vLLM，就改用 GGUF
（`unsloth/Qwen3.8-27B-GGUF`、`ggml-org/Qwen3.8-27B-GGUF`）。
EXL3 格式**只能**用 ExLlamaV3 / TabbyAPI，vLLM 与 llama.cpp 都不支持。

---

## 11. 性能预期（4090 24GB / 3.50bpw）

| 场景 | 预期 | 实测 |
|---|---|---|
| 短上下文（< 8K）生成 | 约 150–180 tok/s | **170 tok/s**（开 MTP 投机解码） |
| 首 token 延迟 | 短 prompt 约 0.2–0.5 s | **0.14–0.15 s** |
| 8K prompt prefill | — | 约 2000 tok/s（无缓存）；前缀命中时 ~11000 tok/s |
| 显存占用 | — | 权重 14.3 + KV 4.0 + 激活 ≈ 19.1 GiB（余 4.0 GiB） |

> 解码速度要用**差分法**测：同一个 prompt 分别生成 64 和 512 token，再算
> `(nB - nA) / (tB - tA)`。单次测量会把首 token 延迟折进生成时间，30 token 的短测
> 会低估到 100 tok/s 左右。`bench_kv.py` 用的就是这个方法。
>
> KV 量化档位的完整实测对比（K4/V4 vs K8/V4 vs K8/V8）见 `config.yaml` 里
> `cache_k_bits` 那一段注释。

实测数据以你机器为准，可在 `test_client.py` 的输出里看到 tok/s。

---

## 12. 与 ExLlamaV3 对接时踩过的坑

这几处都不是本项目的 bug，而是 ExLlamaV3 的接口约定，但不对齐就会直接崩或输出垃圾。
如果你要改 `server.py`，注意别把它们改回去：

**① KV 缓存必须传 `max_history`，否则开投机解码必崩**

这个模型 64 层里 48 层是线性注意力（Gated Delta Net）。验证草稿 token 那一步会把
`recurrent_history` 打开，递归层就需要 `[slots, max_history + 1, ...]` 的状态张量。
留 0 会分配成 `[slots, 1, ...]`，运行时抛：

```
RuntimeError: recurrent_state must be [num_slots, max_history + 1, num_v_heads, k_head_dim, v_head_dim]
```

正确做法是跟随草稿模型的 `default_draft_size`（`qwen3_5_mtp` 声明为 4），
`config.yaml` 里的 `max_history: null` 即自动。代价约 0.7 GiB 显存。

**② 停止条件必须显式传入 EOS token id**

ExLlamaV3 的 `Job.stop_tokens` 默认是**空集合**，它不会自己去查 `config.eos_token_id_list`。
不传的后果是模型一直生成到 `max_tokens`，输出里滚出大段
`<|im_start|><|im_end|><|im_start|>...` 垃圾。另外这个模型的 EOS 有讲究：

- `config.json` 的 `text_config.eos_token_id` 是 `248044`（`<|endoftext|>`，基座）
- 对话真正的 EOS 是 `248046`（`<|im_end|>`），只在 `generation_config.json` 里声明
- 正确的列表要等 `Tokenizer` 构造完才被补全（它会合并 `tokenizer_config.json`
  和 `generation_config.json`），所以要从 `config.eos_token_id_list` 取，**不能自己拼**

**③ `max_tokens` 不能顶到 `max_seq_len`**

Job 内部按 `max_new_tokens + 1 + num_draft_tokens` 算页数，另外前缀 token 再占一格。
所以 `prompt + max_tokens` 正好等于 262144 时会失败：

```
AssertionError: Job requires 1025 pages (only 1024 available)
```

本服务会自动把 `max_tokens` 收窄到 `max_seq_len - prompt - (2 + num_draft_tokens)` 并打警告，
所以请求里可以随便写大值（`262144`、`999999` 都行）。

这里的 `num_draft_tokens` 必须和 exllamav3 自己算的一致 —— `Generator` 没显式传值时
用 `draft_model.caps["default_draft_size"]`：MTP 是 4，**DFlash2 是 `block_size - 1 = 7`**。
之前服务对非 MTP 的草稿写死 0，结果是少预留 6 格，开 DFlash2 后请求把上下文顶满就报
页数不足（已修）。启动时现在会对一次账，不一致会明确告警。

**④ 注意 `engine.generator` 是包装类**

`AsyncGenerator` 才是对外用的，真正的 `Generator` 在它的 `.generator` 属性里。
直接 `engine.generator.num_draft_tokens` 会静默拿到默认值 0（踩过这个坑，
导致上面的页数预算算少）。

**⑤ `Tokenizer.decode()` 返回值是 `str | list[str]`**

形状 `(1, n)` 的输入返回的是**列表**，当字符串用会抛
`AttributeError: 'list' object has no attribute 'encode'`。

**⑥ EOS 那一帧的 logprobs 藏在 `held` 里**

逐帧的 `token_ids` / `token_probs` 在 EOS 帧会被塞进 `r["held"]` 子字典。
只读顶层会一无所获（表现为「未返回 logprobs」）；只取最后一帧则只会剩最后一个 token。
本服务会把所有帧收集起来沿序列维拼接。

---

## 13. 可选：DFlash2 草稿模型（比 MTP 更快的投机解码）

本模型自带 MTP 草稿层（`mtp_draft: true`）。另外还有一个更强的选择：**DFlash2**，
一种 block-diffusion 草稿，一次预测整个 8 token 的块（7 草稿 + 1 锚点）。

官方参考数据（H200 / SGLang，7 token 草稿对 7 token MTP）：

| 任务 | MTP 接受长度 | DFlash2 接受长度 | MTP tok/s | DFlash2 tok/s |
|---|---|---|---|---|
| GSM8K | 5.02 | **5.46** | 178.5 | **236.1** |
| HumanEval | 3.91 | **4.39** | 151.9 | **214.6** |
| MT-Bench | 3.74 | **4.10** | 134.9 | **184.0** |

一个第三方在 24GB RTX 3090 + ExLlamaV3 上测得：MTP 接受长度 4.12 / 116.3 tok/s，
DFlash2 5.66 / **162.9 tok/s**（+40%）。解码是无损的：贪心输出与目标模型完全一致。

### 哪些能用（本机 exllamav3 1.5.1 实测）

| 仓库 | 大小 | 结果 |
|---|---|---|
| `igor255/Qwen3.8-27B-DFlash2-EXL3-4.00bpw` | 1.08 GiB | **可用 ✅**（平均 4.01 bpw） |
| `Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw` | 1.37 GiB | **可用 ✅**（平均 5.41 bpw） |
| `z-lab/Qwen3.8-27B-DFlash2`（官方 bf16） | 3.58 GiB | 可用，但显存占用 3.6 GiB |
| `r0b0tlab/...-EXL3-4.00bpw` | 1.2 GiB | **不可用 ❌**（需它的 fork） |

离线校验工具：`.\_bench\draft_check.py <目录>`（只跑 CPU，不占显存、不影响正在跑的服务）。

⚠️ **`r0b0tlab` 那份不能用**，它把 selector 码本存成了
`candidate_selector.*_codebook.weight`，而上游 exllamav3 要的是**不带 `.weight`** 的
裸名字（`DFlash2Config` 没有声明 `get_tensor_name_fixes`），会直接报
`Required tensor candidate_selector.predecessor_codebook not found`。
它要求换成 `r0b0tlab/exllamav3` 的 `community` 分支。

### 显存代价（重要）

| 项 | 占用 |
|---|---|
| 草稿权重（4.00bpw） | ≈ 0.9 GiB |
| GDN 递归层验证历史 | ≈ 1.22 GiB（不开草稿时只占 ~0.15 GiB） |
| **合计** | **≈ +2.0 GiB** |

那个 1.22 GiB 很有迷惑性：验证草稿块时递归层要 `[slots, max_history + 1, ...]`，
本服务会自动取 `max_history = default_draft_size = 7`，也就是 48 层 × 8 行 fp32。

所以在 24GB 卡上，开草稿前要把当前余量减掉 2 GiB：

| 配置 | KV | 预计余量 | 评价 |
|---|---|---|---|
| 262144 + `cache_quant: 4` | 4.0 GiB | ≈ 2.9 GiB | 偏紧 |
| 262144 + `cache_quant: 3` | 3.0 GiB | ≈ 3.9 GiB | 比较稳 |

（3bit 是支持的，`cache/quant.py` 里断言 `2 <= k_bits <= 8`。上面那个 3090 的例子里
用的就是 `cq3`。）

### 怎么用

```yaml
model:
  draft_model: models/DFlash2-EXL3-4.00bpw   # 目录存在就在启动时自动接管，优先于 MTP
```

启动日志会扥出 `draft_kind = dflash`（`/health` 里的 `draft_kind` 字段也是），
此时草稿窗口 = 7，服务的页数预留也会跟着变成 `2 + 7 = 9`。

---

## 14. 用 pm2 常驻托管（开机自启 + 崩溃自动拉起）

`ecosystem.config.js` 已配好，直接用：

```powershell
pm2 start ecosystem.config.js   # 启动并托管
pm2 save                        # 记住进程列表（开机自启靠它）
pm2 logs qwen38-server          # 看日志
pm2 restart qwen38-server       # 重启（重新加载权重）
pm2 stop qwen38-server          # 停止（停止后不会自动重启）
```

开机自启：`pm2-windows-startup` 已装且注册表钩子已就位，`pm2 save` 之后就生效。

### ⚠️ 托管之后启停必须走 pm2

**不要再用 `taskkill` 或向 `/admin/shutdown` 发请求去关它** —— 进程一退出，
`autorestart` 会立刻把它拉起来。要真的停就 `pm2 stop`。

控制面板已经适配：检测到 pm2 托管时，「停止」会转发成 `pm2 stop`，
「启动/重启」转发成 `pm2 start` / `pm2 restart --update-env`，
页面的「托管方式」卡片会显示 `pm2 · online`。面板的 `restart` 还会把
`--config` / `--no-vision` 等参数写进 `_state/service_args.json`，
由 `scripts/run_server.py` 读取 —— 这样 pm2 模式下切换配置依然有效。

### 为什么需要 `scripts/run_server.py` 这层包装

Windows 上 `.venv\Scripts\python.exe` 是**转发器**（255 KB，而 base 解释器只有
105 KB）。它自己会再拉起 base 解释器，所以一次启动实际是**两个进程**：

```
pm2 ──▶ 转发器 python.exe ──▶ base python.exe
          不占显存              真正占端口 + 21.5 GiB 显存
```

pm2 只跟踪转发器的 PID，**停进程时不保证连带杀掉子进程**。真解释器一旦变成
孤儿，21.5 GiB 显存就一直卡着，下次启动必然 CUDA OOM。

包装脚本解决三件事：

| 机制 | 作用 |
|---|---|
| **孤儿看门狗** | 每 2s 检查父进程（转发器）是否还在；父进程一消失就自己退出，把显存还回去 |
| **日志** | 把 stdout/stderr 复制到 `logs/service.log`，和控制面板读同一份 |
| **信号处理** | 收到 INT/TERM/BREAK 时置 `SERVER.should_exit`，走完整 lifespan 释放显存 |

### 实测验证（2026-09，RTX 4090）

| 测试 | 结果 |
|---|---|
| `pm2 stop` 是否真释放显存 | ✅ **2s 后进程全清，21.58 → 0.49 GB**（释放 21.1 GB） |
| 崩溃后自动恢复 | ✅ taskkill 全部进程 → **23s 后自动恢复**，`restarts 0→1`，`unstable` 仍为 0 |
| 面板停止转发 | ✅ `{status: stopped, supervisor: pm2}` |
| 面板启动/切配置转发 | ✅ `--no-vision` 生效（`vision=false`，省约 0.8 GB） |
| 服务独立于面板 | ✅ 杀掉面板后服务照常返回 200 |

### 恢复策略为何是这组数字

`autorestart` + 退避重启，但参数不能乱调 —— pm2 判定「重启超限」的源码逻辑
（`lib/God.js:411-423`）是：

```js
if (now - created_at < min_uptime * max_restarts) {
    if (now - pm_uptime < min_uptime) unstable_restarts += 1;
}
if (unstable_restarts >= max_restarts) → ERRORED（不再重启）
```

真正决定宽容度的是**乘积** `min_uptime * max_restarts`（一个以「应用创建时刻」
为起点的时间窗）。这里取 `30s * 10 = 300s`：

- `min_uptime` 必须大于加载耗时（实测 7–15 s），否则「加载中就崩」不会被判为失败
- 窗口内允许 10 次失败尝试，是为了让**密集切换配置**（每次切换都是一次重启，
  模型要 10 s 左右加载完）不至于把名额烧光
- 早先设成 `90s * 5`，窗口 450 s 却只有 5 个名额，密集切配置会误判

万一真进了 `errored`：`pm2 start qwen38-server` 即可恢复（面板的「启动」走的就是这条路）。

### 与其它 pm2 应用共存

本机 pm2 里还有 `wx-msg`、`qwen3-tts-api` 等应用，互不影响；
`pm2 save` 会把它们一起记下来。

---

## 15. 许可证与致谢

**本项目代码：MIT，见 [`LICENSE`](LICENSE)。** 可自由使用 / 修改 / 再分发。

第三方组件的归属是分开的：

| 组件 | 归属 | 说明 |
|---|---|---|
| 权重 `turboderp/Qwen3.8-27B-exl3` | 见模型目录下的 `LICENSE` | **不由本仓库分发**（`models/` 已在 `.gitignore` 中） |
| DFlash2 草稿模型（可选） | 见对应模型仓库 | 同上，需要时用 `fast_download.py` 自行下载 |
| 推理后端 ExLlamaV3 | MIT | https://github.com/turboderp-org/exllamav3 |
| `_tmpl/chat_template.jinja` | 随模型权重发布 | 从模型仓库取出的一份副本，离线自检要用（`selftest.py`） |

> 注意：`_tmpl/chat_template.jinja` 是为了让 `selftest.py` 能**不下载模型**就验证
> 模板渲染逻辑而内联的一份上游模板，版权归上游。若上游许可有额外限制，
> 删掉 `_tmpl/` 即可（只影响离线自检，不影响推理 —— 推理用的是模型目录里的那份）。

