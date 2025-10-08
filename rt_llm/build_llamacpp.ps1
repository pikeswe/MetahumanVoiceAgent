Param(
    [string]$RepoUrl = "https://github.com/ggerganov/llama.cpp.git",
    [string]$Branch = "master"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$binDir = Join-Path $root "bin/llama"
$srcDir = Join-Path $root "llama.cpp"

if (-not (Test-Path $binDir)) {
    New-Item -ItemType Directory -Path $binDir | Out-Null
}

if (Test-Path $srcDir) {
    Write-Host "Updating llama.cpp repository..."
    git -C $srcDir fetch --all
    git -C $srcDir checkout $Branch
    git -C $srcDir pull
} else {
    Write-Host "Cloning llama.cpp..."
    git clone --branch $Branch $RepoUrl $srcDir
}

Write-Host "Configuring CMake build..."
$buildDir = Join-Path $srcDir "build"
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

Push-Location $buildDir
cmake .. -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build . --config Release
Pop-Location

Write-Host "Copying binaries to $binDir"
Get-ChildItem -Path $buildDir -Recurse -Filter "*.exe" | ForEach-Object {
    Copy-Item $_.FullName -Destination $binDir -Force
}
Get-ChildItem -Path $buildDir -Recurse -Filter "*.dll" | ForEach-Object {
    Copy-Item $_.FullName -Destination $binDir -Force
}

Write-Host "llama.cpp build complete. Binaries in $binDir"
