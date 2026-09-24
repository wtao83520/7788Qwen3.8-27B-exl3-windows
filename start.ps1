<#
    启动 OpenAI 兼容推理服务。

    用法：
      .\start.ps1                                  # 用 config.yaml
      .\start.ps1 -Port 8080 -MaxSeqLen 65536      # 临时覆盖
      .\start.ps1 -NoVision                        # 不加载视觉塔，省显存
      .\start.ps1 -- --log-level debug             # 其余参数原样传给 server.py
#>
[CmdletBinding()]
param(
    [string]$Config = "config.yaml",
    [int]$Port = 0,
    [string]$ModelPath = "",
    [int]$MaxSeqLen = 0,
    [switch]$NoVision,
    [string]$LogLevel = "INFO",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Extra
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "还没创建虚拟环境，请先运行：  .\setup.ps1" -ForegroundColor Red
    exit 1
}

$argsList = @("server.py", "--config", $Config, "--log-level", $LogLevel)
if ($Port -gt 0) { $argsList += @("--port", "$Port") }
if ($ModelPath) { $argsList += @("--model-path", $ModelPath) }
if ($MaxSeqLen -gt 0) { $argsList += @("--max-seq-len", "$MaxSeqLen") }
if ($NoVision) { $argsList += "--no-vision" }
if ($Extra) { $argsList += $Extra }

Write-Host "执行： $venvPy $($argsList -join ' ')" -ForegroundColor Cyan
Write-Host "（首次启动要把 15 GB 权重读进显存，通常 30~90 秒；看到 “服务就绪” 后再发请求）" -ForegroundColor DarkGray
Write-Host ""

& $venvPy @argsList
