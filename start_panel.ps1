<#
    启动控制面板（网页上启停推理服务）。

    用法：
      .\start_panel.ps1                       # 启动并自动打开浏览器（仅本机）
      .\start_panel.ps1 -Lan                  # 允许局域网访问（需令牌）
      .\start_panel.ps1 -PanelPort 8081       # 换面板端口
      .\start_panel.ps1 -NoBrowser            # 不开浏览器
      .\start_panel.ps1 -Background           # 后台运行（关掉终端也不退出）
      .\start_panel.ps1 -Background -Force    # 先停掉旧面板再启动（改完代码后用）

    默认只监听 127.0.0.1。加 -Lan 后监听 0.0.0.0，局域网机器可以直接访问，
    默认不需要令牌（注意：任何能连上这个端口的机器都能启停推理服务）。
    以后想加令牌也不用改代码：在 _state/panel_token.txt 写一行值再重启面板。

    面板与推理服务是两个进程两个端口（面板 8001 / 服务看 config.yaml 的
    server.port，当前 2345）—— 为什么不合并，见 README
    「面板为什么不能和推理服务用同一个端口」一节。

    注意：以前这个脚本在后台模式下**无条件**报「已启动」，即使进程因为 8001 被占
    （WinError 10048）而立刻退出 —— 看起来启动成功，其实面板根本没起来。
    现在会先探测端口、启动后再验证真的能连上。
#>
[CmdletBinding()]
param(
    [int]$PanelPort = 8001,
    [switch]$NoBrowser,
    [switch]$Background,
    [switch]$Force,
    [switch]$Lan
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "还没创建虚拟环境，请先运行：  .\setup.ps1" -ForegroundColor Red
    exit 1
}

# 快速探测端口是否已被占用（不用 Test-NetConnection，它很慢）
function Test-PanelPort([int]$Port) {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $client.Connect('127.0.0.1', $Port)
        $client.Close()
        return $true
    } catch {
        return $false
    }
}

function Get-PanelPid([int]$Port) {
    try {
        $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -EA SilentlyContinue |
            Select-Object -First 1
        if ($conn) { return $conn.OwningProcess }
    } catch { }
    return 0
}

# 端口已被占 —— 不要默默再起一个（会因 WinError 10048 直接死掉）
if (Test-PanelPort $PanelPort) {
    $oldPid = Get-PanelPid $PanelPort
    if ($Force) {
        Write-Host "端口 $PanelPort 已被 PID $oldPid 占用，-Force 已指定，先停掉它…" -ForegroundColor Yellow
        # 面板是 venv 转发器 + 真解释器两个进程，要整棵树停。
        #
        # ⚠️ 这里必须临时关掉 $ErrorActionPreference = "Stop"：
        #    taskkill 把「没有找到进程」之类的信息写到 **stderr**，而 PowerShell 5.1
        #    会把原生命令的 stderr 当成 ErrorRecord —— 配合 Stop 策略就变成终止错误，
        #    直接中断整个脚本（表现为「先杀了旧面板，然后自己也没起来」）。
        #    先杀掉根进程（/T 会连带子进程），子进程的 PID 随后就不存在了，
        #    所以后续几次 taskkill 一定会报「没有找到进程」，属于正常情况。
        $savedEap = $ErrorActionPreference
        $ErrorActionPreference = "SilentlyContinue"
        try {
            foreach ($t in (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                            Where-Object { $_.CommandLine -like "*control_panel.py*" } |
                            Select-Object -ExpandProperty ProcessId)) {
                taskkill /PID $t /T /F 2>&1 | Out-Null
            }
        } finally {
            $ErrorActionPreference = $savedEap
        }
        Start-Sleep -Seconds 2
        if (Test-PanelPort $PanelPort) {
            Write-Host "旧面板仍未退出，请手工处理后再试。" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "面板已经在运行（端口 $PanelPort，PID $oldPid）。" -ForegroundColor Yellow
        Write-Host "  打开： http://127.0.0.1:$PanelPort/" -ForegroundColor Cyan
        Write-Host "  要换掉它（比如刚改过 control_panel.py）就用： .\start_panel.ps1 -Background -Force" -ForegroundColor DarkGray
        exit 0
    }
}

$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$pyArgs = @("control_panel.py", "--panel-port", "$PanelPort")
if ($Lan) { $pyArgs += @("--host", "0.0.0.0") }
if (-not $NoBrowser) { $pyArgs += "--open" }

$bindHost = if ($Lan) { "0.0.0.0" } else { "127.0.0.1" }
$url = "http://127.0.0.1:$PanelPort/"

# 局域网模式下先把防火墙问题提出来。加规则要管理员权限，
# 这里不自己提权（会弹 UAC，而且远程场景下根本点不到），只把命令给出来。
if ($Lan) {
    Write-Host "已开启局域网监听（$bindHost`:$PanelPort）。" -ForegroundColor Yellow
    Write-Host "  不需要令牌 —— 能连上这个端口的机器都能启停推理服务。" -ForegroundColor DarkGray
    Write-Host "  以后想加令牌：在 _state/panel_token.txt 写一行值，重启面板即可。" -ForegroundColor DarkGray
    $fwCmd = "New-NetFirewallRule -DisplayName 'Qwen38 控制面板' -Direction Inbound " +
             "-Protocol TCP -LocalPort $PanelPort -Action Allow -Profile Private"
    Write-Host "  如果局域网还是连不上，以管理员身份运行一次：" -ForegroundColor DarkGray
    Write-Host "    $fwCmd" -ForegroundColor DarkGray
    Write-Host ""
}

if ($Background) {
    $outLog = Join-Path $logDir "panel.out"
    $errLog = Join-Path $logDir "panel.err"
    $proc = Start-Process -FilePath $venvPy -ArgumentList $pyArgs `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog

    # 关键：不要拿到 PID 就宣布成功。进程可能在绑定端口时立刻死掉
    # （WinError 10048 = 端口被占）。必须等端口真的能连上才算起来。
    $ok = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        if ($proc.HasExited) { break }
        if (Test-PanelPort $PanelPort) { $ok = $true; break }
    }

    if (-not $ok) {
        Write-Host "面板启动失败（PID $($proc.Id)）。" -ForegroundColor Red
        if ($proc.HasExited) {
            Write-Host "  进程已退出，退出码 $($proc.ExitCode)" -ForegroundColor Red
        } else {
            Write-Host "  进程还在，但端口 $PanelPort 一直连不上" -ForegroundColor Red
        }
        foreach ($f in @($errLog, $outLog)) {
            if (Test-Path $f) {
                $tail = Get-Content $f -Tail 15 -EA SilentlyContinue
                if ($tail) {
                    Write-Host "  --- $([IO.Path]::GetFileName($f)) 末尾 ---" -ForegroundColor DarkGray
                    $tail | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
                }
            }
        }
        Write-Host "  也可以用  .\start_panel.ps1 -Background -Force  让脚本先清掉旧面板" -ForegroundColor DarkGray
        exit 1
    }

    Write-Host "控制面板已在后台启动（PID $($proc.Id)）" -ForegroundColor Green
    Write-Host "  地址： $url" -ForegroundColor Cyan
    Write-Host "  日志： $(Join-Path $logDir 'panel.log')   （面板页面里也能直接看）"
    Write-Host "  停止： Stop-Process -Id $($proc.Id) -Force"
} else {
    Write-Host "控制面板地址： $url" -ForegroundColor Cyan
    Write-Host "（按 Ctrl+C 结束面板；面板关掉不会影响正在运行的推理服务）" -ForegroundColor DarkGray
    Write-Host ""
    & $venvPy @pyArgs
}
