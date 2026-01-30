# Consciousness Exploration Runner
# Wrapper script for cron job invocation

param(
    [string]$Model = "gpt-oss:20b",
    [string]$Focus = "",
    [switch]$UpdateSummary,
    [switch]$NotifyDiscord
)

$ErrorActionPreference = "Stop"

# Navigate to CORE backend
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Split-Path -Parent $ScriptDir

Push-Location $BackendDir

try {
    # Build arguments
    $args = @()
    if ($Model) { $args += "--model", $Model }
    if ($Focus) { $args += "--focus", $Focus }
    if ($UpdateSummary) { $args += "--update-summary" }
    if ($NotifyDiscord) { $args += "--notify-discord" }
    
    # Run with uv
    Write-Host "Starting consciousness exploration..."
    & uv run python scripts/consciousness_exploration.py @args
    
    $exitCode = $LASTEXITCODE
    Write-Host "Exploration completed with exit code: $exitCode"
    exit $exitCode
}
finally {
    Pop-Location
}
