# Metahuman Voice Agent

A minimal Windows-first voice agent built around local models. The pipeline streams audio from the browser to Whisper for speech-to-text, forwards transcripts to Ollama, and speaks responses through Piper (or optionally NeuTTS-Air) straight into a VB-CABLE output.

## Features

- **CPU-first** with optional CUDA acceleration for Whisper and NeuTTS.
- **Streaming pipeline:** microphone → VAD → faster-whisper → Ollama SSE → Piper/NeuTTS → VB-CABLE.
- **FastAPI backend** with a compact HTML/JS front-end (push-to-talk, chat history, runtime settings).
- **Configurable** via `.env`; runtime changes persist automatically.
- **Diagnostics** script for quick environment validation.

## Prerequisites

1. **Python** 3.10 or newer (64-bit).
2. **VB-CABLE** Virtual Audio Device installed and set up. Download from [VB-Audio](https://vb-audio.com/Cable/).
3. **Ollama** installed and running locally ([download](https://ollama.com/download)).
4. **Piper** CLI installed and in `PATH`. Get Windows binaries from [rhasspy/piper](https://github.com/rhasspy/piper/releases).
5. (Optional) **NeuTTS-Air** Python package and models if you plan to use that backend.

## Installation (Windows 10/11)

1. Clone this repository and open PowerShell or Command Prompt inside `voice-agent/`.
2. Copy `.env.example` to `.env` and adjust as needed:
   ```powershell
   copy .env.example .env
   ```
3. Download a Piper voice model (for example `en_US-amy-medium.onnx`). Place the `.onnx` (and `.json` metadata if provided) in `models/piper/` or update `PIPER_VOICE_PATH` in `.env`.
   ```powershell
   python tools\download_models.py
   ```
4. Ensure Ollama is serving and the desired model is available:
   ```powershell
   ollama serve
   ollama pull llama3.1:8b
   ```

## Running the agent

Launch the agent with the one-click batch script:

```powershell
run.bat
```

The script will:
- Create/activate a virtual environment.
- Install Python dependencies.
- Print environment diagnostics (audio devices, Piper availability, Ollama reachability).
- Start FastAPI with Uvicorn on `http://127.0.0.1:7860` (configurable in `.env`).

Navigate to the URL in your browser. Hold the push-to-talk button (or spacebar) to send audio. Partial transcripts appear immediately; once speech ends, the transcript is finalized, passed to the LLM, and read out through VB-CABLE.

## Configuration

Key options (all in `.env`):

- **LLM**
  - `OLLAMA_HOST`: Ollama server URL.
  - `OLLAMA_MODEL`: Model tag (e.g., `llama3.1:8b`).
- **Speech-to-Text**
  - `WHISPER_MODEL`: faster-whisper model name (`small.en`, `medium.en`, etc.).
  - `WHISPER_DEVICE`: `auto`, `cpu`, or `cuda`.
  - `VAD_AGGRESSIVENESS`: 0–3 (higher = more aggressive speech detection).
- **TTS**
  - `TTS_BACKEND`: `piper` (default) or `neutts`.
  - `PIPER_VOICE_PATH`: Path to the Piper `.onnx` voice file.
  - `PIPER_SAMPLE_RATE`: Voice sample rate (16k, 22.05k, or 24k typical).
  - NeuTTS settings (`NEUTTS_DEVICE`, `NEUTTS_REF_*`) if enabled.
- **Audio Output**
  - `AUDIO_OUTPUT_DEVICE_NAME`: Partial match for the desired playback device (default searches for “VB-CABLE”).
  - `AUDIO_SAMPLE_RATE`: Playback sample rate (resampling is handled automatically).
- **Server/UI**
  - `HOST`, `PORT`, `LOG_LEVEL`.
  - `SPEAK_PARTIALS`: `true` to speak LLM tokens as they stream.

Settings changed in the web UI persist back to `.env` instantly.

## Optional NeuTTS-Air backend

Set `TTS_BACKEND=neutts` to switch from Piper. The backend loads lazily; if imports or model loading fail, the agent automatically falls back to Piper and logs the reason. Provide optional reference audio/text via `NEUTTS_REF_AUDIO` and `NEUTTS_REF_TEXT` to steer the voice.

## Troubleshooting

- **VB-CABLE not detected**: Ensure the device name contains “VB-CABLE” (case-insensitive). Use the UI device picker or edit `AUDIO_OUTPUT_DEVICE_NAME` to match exactly.
- **No audio output**: Confirm Piper is installed and accessible. `tools\verify_env.py` prints the detected Piper binary and voice path.
- **Whisper too slow**: Switch to a smaller model (e.g., `base.en` or `small.en`) or enable CUDA by installing GPU drivers and libraries, then set `WHISPER_DEVICE=auto`.
- **Ollama unreachable**: Start `ollama serve` and confirm the host/port matches `.env`. Pull the requested model once before launching the agent.
- **NeuTTS errors**: Leave `TTS_BACKEND=piper` or ensure NeuTTS-Air dependencies are installed; failures fall back automatically with a log message.

## Known performance notes

- CPU-only systems work out of the box; expect higher latency on first token. Consider `WHISPER_MODEL=base.en` for lower-end CPUs.
- GPU acceleration (CUDA) dramatically reduces Whisper and NeuTTS latency when available. The app auto-detects GPUs when `WHISPER_DEVICE=auto` / `NEUTTS_DEVICE=auto`.
- Piper voices at 22.05–24 kHz provide the best quality/latency trade-off. Audio is resampled to the configured playback rate before being sent to VB-CABLE.

## Diagnostics

Run the environment verifier anytime:

```powershell
python tools\verify_env.py
```

It prints Python/platform details, detected audio devices, Piper status, Whisper config, and Ollama reachability.

## Repository layout

```
voice-agent/
  README.md
  .env.example
  requirements.txt
  run.bat
  src/
    app.py
    audio_io.py
    config.py
    llm_stream.py
    mic_ws.py
    pipeline.py
    stt_stream.py
    vad.py
    tts/
      __init__.py
      piper_tts.py
      neutts_tts.py
    web/
      index.html
      main.js
      style.css
  tools/
    verify_env.py
    download_models.py
  models/
    piper/
      (place Piper voices here)
```

## License

MIT License. See `LICENSE` for details.
