# Local Voice Agent (Ollama + NeuTTS Air)

This project provides a Windows-only, fully local voice assistant that connects a microphone, the Ollama LLM runtime, and NeuTTS Air voice cloning. The CLI supports both text chat and hands-free listen/respond flows while streaming synthesized audio directly to the VB-CABLE virtual device.

## Features

- Purely local stack: Ollama for generation, faster-whisper for transcription, and NeuTTS Air for neural TTS.
- Sentence-aware chunking so playback begins within a couple of seconds.
- Automatic routing of playback to the **VB-CABLE "CABLE Input"** device.
- Configurable reference voice (audio + transcript) with cached embeddings for fast synthesis.
- Graceful handling of phonemizer/espeak misconfiguration and optional NeuTTS watermark support.

## Quick start (Windows)

Assuming the virtual environment lives at `C:\voiceagent\.venv` and NeuTTS Air is installed in editable mode:

```powershell
cd C:\voiceagent
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py chat --prompt "Say hello in one sentence." --model llama3.1
```

To run the live microphone loop:

```powershell
cd C:\voiceagent
.\.venv\Scripts\python.exe app.py listen --model llama3.1
```

### Runtime notes

- The CLI reads defaults from `config.yaml`. Override any field with flags such as `--model`, `--ref-audio`, or `--ref-text`.
- NeuTTS Air automatically encodes the configured reference at startup. The first run may download the backbone and codec weights; subsequent runs reuse the cache.
- The playback device is detected by a substring match on `"CABLE Input"`. Ensure VB-CABLE is installed and enabled.

## Configuration

`config.yaml` ships with sensible defaults:

```yaml
ollama_model: "llama3.1"
ref_audio: "assets/voice/reference.wav"
ref_text: "assets/voice/reference.txt"
tts:
  backbone_repo: "neuphonic/neutts-air"
  backbone_device: "cpu"
  codec_repo: "neuphonic/neucodec"
  codec_device: "cpu"
```

Override settings per run:

```powershell
.\.venv\Scripts\python.exe app.py chat --model llama3.1:70b --ref-audio C:\path\to\voice.wav --ref-text C:\path\to\transcript.txt
```

## Troubleshooting

### VB-CABLE device not found

The app enumerates output devices and looks for the substring `"CABLE Input"`. If the device is missing, the CLI prints the full device list and exits. Open the **Sounds** control panel, enable the VB-CABLE output, and retry.

### Ollama connection errors

Ensure Ollama is running locally (`ollama serve`) and that the requested model (e.g., `llama3.1`) has been pulled. The CLI probes `http://localhost:11434/api/tags` during startup and exits with a clear error if the service is unreachable.

### NeuTTS Air model downloads

On the first invocation, NeuTTS Air may download backbone and codec weights. The CLI logs a single `Downloading models once…` message; wait for the download to finish. Subsequent runs skip the download.

### Phonemizer / eSpeak NG issues

If the phonemizer or eSpeak NG library cannot be loaded, chunks are skipped and a log message hints at the required environment variables:

- `PHONEMIZER_ESPEAK_LIBRARY`
- `PHONEMIZER_ESPEAK_PATH`

Verify that both point to the correct eSpeak NG installation (e.g., `C:\Program Files\eSpeak NG`).

## Directory structure

```
app.py              # CLI entry point (chat + listen modes)
config.yaml         # Default runtime configuration
requirements.txt    # Python dependencies for the CLI stack
assets/voice/       # Reference audio + transcript used to clone the voice
```

## License

See [LICENSE](LICENSE) for details.
