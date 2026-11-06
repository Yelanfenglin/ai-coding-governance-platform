<#
.SYNOPSIS
  StockPilot 之外的本仓发布/回滚脚本：按镜像 tag 做快照与回退。

.DESCRIPTION
  本栈由 compose.yaml 编排，自建镜像只有两个：
    embedded-ai-coding-agent-runtime
    embedded-ai-coding-rag-api
  postgres 使用上游固定 tag，不参与快照。

  snapshot  —— 把当前 :latest 打上版本 tag（发布前做，用于回退）
  rollback  —— 把指定版本 tag 重新指向 :latest，并强制重建容器
  status    —— 打印 compose 服务状态
  release   —— 构建 -> 快照 -> 启动

.EXAMPLE
  .\scripts\compose-release.ps1 -Action snapshot -Version v1
  .\scripts\compose-release.ps1 -Action rollback -Version v1
#>
[CmdletBinding()]
param(
    [ValidateSet('snapshot', 'rollback', 'status', 'release')]
    [string]$Action = 'status',

    [string]$Version = 'v1',

    [string]$ComposeFile
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

$images = @(
    'embedded-ai-coding-agent-runtime',
    'embedded-ai-coding-rag-api'
)

function Invoke-Compose {
    param([string[]]$ComposeArgs)
    # docker 把进度信息写到 stderr；$ErrorActionPreference='Stop' 会把它当成
    # 终止性错误。这里临时放宽，只按退出码判定成败。
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        if ($ComposeFile) { & docker compose -f $ComposeFile @ComposeArgs }
        else { & docker compose @ComposeArgs }
        $code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) { throw "docker compose 执行失败(exit=$code): $($ComposeArgs -join ' ')" }
}

function Get-ImageId {
    param([string]$Name)
    $id = & docker images -q "${Name}:latest"
    if (-not $id) { return $null }
    return ($id | Select-Object -First 1)
}

try {
    switch ($Action) {

        'status' {
            Invoke-Compose @('ps', '--format', '{{.Service}}\t{{.Status}}')
        }

        'snapshot' {
            if ([string]::IsNullOrWhiteSpace($Version)) { throw 'snapshot 需要 -Version' }
            foreach ($image in $images) {
                $id = Get-ImageId $image
                if (-not $id) { throw "找不到镜像 ${image}:latest，请先 build" }
                & docker tag "${image}:latest" "${image}:${Version}"
                if ($LASTEXITCODE -ne 0) { throw "打 tag 失败: ${image}:${Version}" }
                Write-Host "[snapshot] ${image}:latest -> ${image}:${Version}"
            }
            Write-Host "[snapshot] 完成，版本 $Version"
        }

        'rollback' {
            if ([string]::IsNullOrWhiteSpace($Version)) { throw 'rollback 需要 -Version' }
            foreach ($image in $images) {
                $target = & docker images -q "${image}:${Version}"
                if (-not $target) { throw "找不到回退目标 ${image}:${Version}，先执行 snapshot" }
                & docker tag "${image}:${Version}" "${image}:latest"
                if ($LASTEXITCODE -ne 0) { throw "回退打 tag 失败: ${image}" }
                Write-Host "[rollback] ${image}:${Version} -> :latest"
            }
            # --no-build：回退到已存在的镜像，绝不触发重新构建
            Invoke-Compose @('up', '-d', '--force-recreate', '--no-build')
            Write-Host "[rollback] 已按 $Version 重建容器"
        }

        'release' {
            if ([string]::IsNullOrWhiteSpace($Version)) { throw 'release 需要 -Version' }
            Invoke-Compose @('build')
            foreach ($image in $images) {
                $id = Get-ImageId $image
                if (-not $id) { throw "构建后仍找不到 ${image}:latest" }
                & docker tag "${image}:latest" "${image}:${Version}"
                Write-Host "[release] ${image}:latest -> ${image}:${Version}"
            }
            Invoke-Compose @('up', '-d', '--no-build')
            Write-Host "[release] 已发布 $Version"
        }
    }
}
finally {
    Pop-Location
}
