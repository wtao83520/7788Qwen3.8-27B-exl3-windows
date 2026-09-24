<#
    内部引导脚本：显式下载 wheel（可断点续传）、安装、校验，然后下载模型。
    所有输出同时写入 bootstrap.log，便于后台运行时查看进度。

    手动执行：
        powershell -ExecutionPolicy Bypass -File .\_bootstrap.ps1
    单独重跑某一步：
        .\_bootstrap.ps1 -SkipTorch
        .\_bootstrap.ps1 -OnlyModel
#>
[CmdletBinding()]
param(
    [switch]$SkipTorch,
    [switch]$SkipExllama,
    [switch]$SkipModel,
    [switch]$OnlyModel,
    [string]$TorchVersion = "2.13.0",
    [string]$Cuda = "cu132",
    [string]$ExllamaVersion = "1.5.1",
    [string]$Revision = "3.50bpw",
    [string]$Log = "bootstrap.log"
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$wheelDir = Join-Path $PSScriptRoot "_wheels"
New-Item -ItemType Directory -Force $wheelDir | Out-Null
$PyPI = "https://pypi.tuna.tsinghua.edu.cn/simple"

function Log {
    param([string]$Message)
    $line = "$(Get-Date -Format 'HH:mm:ss')  $Message"
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding utf8
}

function Fail {
    param([string]$Message)
    Log "失败：$Message"
    exit 1
}

if ($OnlyModel) {
    $SkipTorch = $true
    $SkipExllama = $true
}

$cp = (& $venvPy -c "import sys; print('cp' + str(sys.version_info.major) + str(sys.version_info.minor))").Trim()
if (-not $cp) { Fail "无法在 .venv 里运行 Python" }
Log "Python 版本标签：$cp"

# ------------------------------------------------------------------ torch
if (-not $SkipTorch) {
    $name = "torch-$TorchVersion+$Cuda-$cp-$cp-win_amd64.whl"
    $file = Join-Path $wheelDir $name
    # 按实测速度排序的镜像列表（2026-09-23 实测，本机）：
    #   上科大 mirror.sjtu.edu.cn  ~7 MB/s      ← 最快
    #   阿里云 mirrors.aliyun.com  ~0.38 MB/s
    #   PyTorch 官方 CDN           ~0.28 MB/s
    $torchMirrors = @(
        "https://mirror.sjtu.edu.cn/pytorch-wheels/$Cuda/$name",
        "https://mirrors.aliyun.com/pytorch-wheels/$Cuda/$name",
        "https://download.pytorch.org/whl/$Cuda/$name"
    )
    $size = 0
    foreach ($url in $torchMirrors) {
        $have = 0
        if (Test-Path $file) { $have = [int]((Get-Item $file).Length / 1MB) }
        Log "下载 torch（当前已有 $have MB，断点续传）：$url"
        # -C - 从断点续传；镜像不支持续传时会整体重下，故失败再换源
        & curl.exe -L -C - --retry 3 --retry-delay 3 -s -o $file $url
        $size = 0
        if (Test-Path $file) { $size = [int]((Get-Item $file).Length / 1MB) }
        Log "  该源下载后体积：$size MB"
        if ($size -ge 1500) { break }
        Log "  该源未拿到完整文件，换下一个源"
        # 换源前删掉半截文件，避免不同源的字节范围拼接出错
        if (Test-Path $file) { Remove-Item $file -Force }
    }
    if ($size -lt 1500) { Fail "torch wheel 体积异常（$size MB），所有镜像都未拿到完整文件" }

    Log "安装 torch（--no-deps，依赖单独装）…"
    & $venvPy -m pip install --no-deps $file -i $PyPI --progress-bar off
    if ($LASTEXITCODE -ne 0) { Fail "torch 安装失败" }
    # torch 自身的依赖
    & $venvPy -m pip install filelock typing-extensions setuptools sympy networkx jinja2 fsspec -i $PyPI --progress-bar off
}

# ------------------------------------------------------------------ exllamav3
if (-not $SkipExllama) {
    $name = "exllamav3-$ExllamaVersion+$Cuda.torch$TorchVersion-$cp-$cp-win_amd64.whl"
    $file = Join-Path $wheelDir $name
    $encoded = [uri]::EscapeDataString($name)
    $orig = "https://github.com/turboderp-org/exllamav3/releases/download/v$ExllamaVersion/$encoded"
    # GitHub Releases 直连在国内很慢（实测 ~35 kB/s），优先走加速代理
    #   实测：ghfast.top ~1.16 MB/s / gh-proxy.com ~150 kB/s / 直连 ~35 kB/s
    $ghUrls = @(
        "https://ghfast.top/$orig",
        "https://gh-proxy.com/$orig",
        $orig
    )
    $size = 0
    foreach ($url in $ghUrls) {
        $have = 0
        if (Test-Path $file) { $have = [int]((Get-Item $file).Length / 1MB) }
        Log "下载 exllamav3（已有 $have MB）：$url"
        & curl.exe -L -C - --retry 3 --retry-delay 3 -s -o $file $url
        $size = 0
        if (Test-Path $file) { $size = [int]((Get-Item $file).Length / 1MB) }
        Log "  该源下载后体积：$size MB"
        if ($size -ge 300) { break }
        Log "  该源未拿到完整文件，换下一个源"
        if (Test-Path $file) { Remove-Item $file -Force }
    }
    if ($size -lt 300) {
        Fail "体积异常（$size MB）。请到 https://github.com/turboderp-org/exllamav3/releases 确认 $Cuda + torch$TorchVersion + $cp 是否有对应资产"
    }
    Log "安装 exllamav3 …"
    & $venvPy -m pip install --no-deps $file --progress-bar off
    if ($LASTEXITCODE -ne 0) { Fail "exllamav3 安装失败" }

    # exllamav3 声明的运行时依赖（漏装会在 import 阶段报 ModuleNotFoundError）
    # torch/tokenizers/numpy/rich/typing_extensions/safetensors/ninja/pillow/pyyaml/
    # marisa_trie/pydantic/llguidance/triton-windows
    Log "安装 exllamav3 的依赖 …"
    & $venvPy -m pip install numpy safetensors tokenizers rich marisa_trie llguidance pydantic `
        pillow pyyaml ninja -i $PyPI --progress-bar off
    if ($LASTEXITCODE -ne 0) { Log "部分依赖安装失败，继续尝试校验" }
    # triton-windows 体积较大且在部分环境不可用，失败不致命
    & $venvPy -m pip install triton-windows -i $PyPI --progress-bar off
    if ($LASTEXITCODE -ne 0) { Log "triton-windows 安装失败（通常不影响基本推理）" }
}

# ------------------------------------------------------------------ 校验
if ((-not $SkipTorch) -or (-not $SkipExllama)) {
    Log "校验运行环境 …"
    & $venvPy verify_env.py *>> $Log
    if ($LASTEXITCODE -ne 0) { Log "环境校验有问题，请查看上面的输出" }
}

# ------------------------------------------------------------------ 模型
if (-not $SkipModel) {
    Log "开始下载模型（分支 $Revision，约 15.4 GB，可中断续传）…"
    & $venvPy download_model.py --mirror --revision $Revision *>> $Log
    if ($LASTEXITCODE -ne 0) { Fail "模型下载失败" }
    Log "模型下载完成"
}

Log "引导流程结束"
