@echo off
setlocal enabledelayedexpansion

if not exist .venv (
    echo [run] Creating virtual environment...
    py -3 -m venv .venv
)

call .venv\Scripts\activate.bat

python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt

if exist .env (
    for /f "usebackq tokens=*" %%A in (`type .env ^| findstr /r "^HOST="`) do set %%A
    for /f "usebackq tokens=*" %%A in (`type .env ^| findstr /r "^PORT="`) do set %%A
    for /f "usebackq tokens=*" %%A in (`type .env ^| findstr /r "^LOG_LEVEL="`) do set %%A
) else (
    copy .env.example .env >nul
)

if "%HOST%"=="" set HOST=127.0.0.1
if "%PORT%"=="" set PORT=7860
if "%LOG_LEVEL%"=="" set LOG_LEVEL=info

python tools\verify_env.py

for /f "tokens=1" %%A in ('where ollama 2^>nul') do set OLLAMA_BIN=%%A
if "%OLLAMA_BIN%"=="" (
    echo [warn] Ollama CLI not found in PATH. Install from https://ollama.com/download and run "ollama serve".
) else (
    echo [run] Using Ollama at %OLLAMA_BIN%
)

echo [run] Starting server on %HOST%:%PORT% ...
uvicorn src.app:app --host %HOST% --port %PORT% --log-level %LOG_LEVEL%
