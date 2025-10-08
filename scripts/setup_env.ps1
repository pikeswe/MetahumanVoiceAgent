Param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path $root
$venvPath = Join-Path $repoRoot ".venv"

if (-Not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment at $venvPath"
    & $Python -m venv $venvPath
} else {
    Write-Host "Virtual environment already exists at $venvPath"
}

$venvPython = Join-Path $venvPath "Scripts/python.exe"
Write-Host "Installing Python dependencies..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $repoRoot "requirements.txt")

Write-Host "Environment ready. Next steps:"
Write-Host " 1) .\\rt_llm\\build_llamacpp.ps1"
Write-Host " 2) Copy your GGUF model into models\\llm"
Write-Host " 3) Install neutts-air resources into models\\tts"
Write-Host " 4) Use scripts\\start_agent.ps1 to launch the orchestrator"
