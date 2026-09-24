<#
    一键搭建运行环境：
      1. 创建/复用 .venv 虚拟环境
      2. 安装匹配的 PyTorch（默认走阿里云镜像，速度更快）
      3. 安装 ExLlamaV3 官方预编译 wheel（无需本地编译 CUDA）
      4. 安装服务依赖（fastapi / uvicorn / pyyaml / huggingface_hub ...）

    用法示例：
      .\setup.ps1                          # 默认：Python 3.13 + torch 2.13.0 + CUDA 13.2
      .\setup.ps1 -TorchVersion 2.9.0 -Cuda cu128
      .\setup.ps1 -UseOfficialIndex        # 用 PyTorch 官方源（国内可能很慢）
      .\setup.ps1 -Force                   # 删除已有 .venv 重新来
#>
[CmdletBinding()]
param(
    [string]$TorchVersion = "2.13.0",
    [ValidateSet("cu128", "cu132")][string]$Cuda = "cu132",
    [string]$ExllamaVersion = "1.5.1",
    [string]$PythonExe = "",
    [switch]$UseOfficialIndex,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$PyPI = "https://pypi.tuna.tsinghua.edu.cn/simple"
$AliyunTorch = "https://mirrors.aliyun.com/pytorch-wheels"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg) { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }

# ---------------------------------------------------------------- 1. Python
Write-Step "检查 Python 解释器"
if ($PythonExe) {
    $py = $PythonExe
} elseif (Test-Path "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python313\python.exe") {
    $py = "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python313\python.exe"
} else {
    $py = "python"
}
& $py -c "import sys; print(sys.version)" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "找不到可用的 Python，请用 -PythonExe 指定路径" }
Write-Ok "使用 $py"

# ---------------------------------------------------------------- 2. venv
if ($Force -and (Test-Path ".venv")) {
    Write-Step "删除旧虚拟环境"
    Remove-Item ".venv" -Recurse -Force
}
if (-not (Test-Path ".venv")) {
    Write-Step "创建虚拟环境 .venv"
    & $py -m venv .venv
} else {
    Write-Step "复用已有虚拟环境 .venv"
}
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { throw "虚拟环境创建失败：$venvPy 不存在" }

& $venvPy -m pip install --upgrade pip --quiet
$cpTag = (& $venvPy -c "import sys; print('cp' + str(sys.version_info.major) + str(sys.version_info.minor))").Trim()
$pyVer = (& $venvPy -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
Write-Ok "Python $pyVer ($cpTag)"

# ---------------------------------------------------------------- 3. torch
Write-Step "安装 PyTorch $TorchVersion+$Cuda"
$wheel = "torch-$TorchVersion%2B$Cuda-$cpTag-$cpTag-win_amd64.whl"
if ($UseOfficialIndex -or $Cuda -eq "cu128" -and $TorchVersion -in @("2.11.0")) {
    # 阿里云镜像只镜像了部分组合，回退到官方源
    $torchUrl = "https://download.pytorch.org/whl/$Cuda/$wheel"
} else {
    $torchUrl = "$AliyunTorch/$Cuda/torch-$TorchVersion+$Cuda-$cpTag-$cpTag-win_amd64.whl"
}
Write-Ok $torchUrl
& $venvPy -m pip install $torchUrl -i $PyPI --progress-bar off
if ($LASTEXITCODE -ne 0) {
    Write-Warn "镜像下载失败，改用 PyTorch 官方源重试（国内可能较慢）…"
    & $venvPy -m pip install "torch==$TorchVersion" --index-url "https://download.pytorch.org/whl/$Cuda"
}
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch 安装失败。请确认 $Cuda 下有 torch $TorchVersion 的 $cpTag win_amd64 版本（见 https://github.com/turboderp-org/exllamav3/releases 的资产名）"
}
& $venvPy -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| 可用:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "PyTorch 导入失败" }

# ---------------------------------------------------------------- 4. exllamav3
Write-Step "安装 ExLlamaV3 $ExllamaVersion 预编译 wheel"
$exlUrl = "https://github.com/turboderp-org/exllamav3/releases/download/v$ExllamaVersion/" +
          "exllamav3-$ExllamaVersion%2B$Cuda.torch$TorchVersion-$cpTag-$cpTag-win_amd64.whl"
Write-Ok $exlUrl
& $venvPy -m pip install $exlUrl --progress-bar off
if ($LASTEXITCODE -ne 0) {
    Write-Warn "该组合的预编译 wheel 不可用。请到 https://github.com/turboderp-org/exllamav3/releases 查看可用资产，"
    Write-Warn "然后用 -TorchVersion / -Cuda 参数重新运行（例如 -TorchVersion 2.9.0 -Cuda cu128）。"
    throw "ExLlamaV3 安装失败"
}

# ---------------------------------------------------------------- 5. 依赖
Write-Step "安装服务依赖"
& $venvPy -m pip install -r requirements.txt -i $PyPI --progress-bar off
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败" }

Write-Step "校验安装"
& $venvPy -c @"
import torch, exllamav3
print('torch      :', torch.__version__)
print('exllamav3  :', getattr(exllamav3, '__version__', 'ok'))
print('CUDA 可用  :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU        :', torch.cuda.get_device_name(0))
    print('显存       :', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB')
"@

Write-Host ""
Write-Host "环境就绪。下一步：" -ForegroundColor Green
Write-Host "  1) 下载模型：  .\.venv\Scripts\python.exe download_model.py --mirror --revision 3.50bpw"
Write-Host "  2) 按需改配置：config.yaml（模型路径 / 上下文长度 / KV 量化位数 / 默认采样参数）"
Write-Host "  3) 启动服务：  .\start.ps1"
