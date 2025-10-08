Param(
    [string]$ModelPath = "models\\llm\\model.gguf",
    [string]$LlamaBinary = $null,
    [string]$TtsModelDir = "models\\tts",
    [int]$Port = 17860,
    [switch]$Mock
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path $scriptRoot
$venvPython = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found. Run scripts\\setup_env.ps1 first."
}

$arguments = @(
    (Join-Path $repoRoot "server\\agent_orchestrator.py"),
    "--host", "127.0.0.1",
    "--port", $Port,
    "--ws-host", "127.0.0.1",
    "--ws-port", $Port,
    "--model-path", $ModelPath,
    "--tts-model-dir", $TtsModelDir
)
if ($LlamaBinary) {
    $arguments += @("--llama-binary", $LlamaBinary)
}
if ($Mock) {
    $arguments += "--mock"
}

Write-Host "Starting MetahumanVoiceAgent orchestrator..."
& $venvPython $arguments
