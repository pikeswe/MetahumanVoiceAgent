# MetahumanVoiceAgent

MetaHumanVoiceAgent is a Windows 11, fully offline speech agent that streams locally generated speech and emotions from a CPU LLM and TTS stack into Unreal Engine 5.6's MetaHuman Animator using the built-in audio-driven lipsync.

> **Tested on:** Windows 11 Pro, Intel i7-14700KF, 32 GB RAM, no discrete GPU requirement.

## Features

- Offline llama.cpp-based LLM with streaming JSON speech chunks.
- Real-time **neutts-air** native streaming API integration (lookahead/lookback with interpolation) producing low-latency PCM16 frames.
- Optional legacy Kani TTS fallback plus a deterministic mock tone generator for testing.
- WebSocket bridge that feeds MetaHuman Animator with PCM frames, start/end markers, and live emotion curves.
- Configurable control plane via [`config/default_config.json`](config/default_config.json) for HTTP/WebSocket ports and TTS behaviour.
- Desktop control panel (Tkinter) for starting/stopping servers, switching TTS backend, reloading models, and sending prompts interactively.
- Unreal Engine 5.6+ runtime plugin (`LiveAudioEmotionSource`) exposing procedural audio and anim curves ready for Sequencer/Blueprints.
- Emotion smoothing and curve mapping for Neutral, Happy, Sad, Angry, and Surprised, plus rate/intensity prosody values.

## Architecture Overview

```
┌────────────┐    stream_response     ┌───────────────┐    PCM + curves    ┌─────────────────────────┐
│  LLMEngine │ ─────────────────────▶ │ Agent Orchestr│ ─────────────────▶ │ VoiceStreamServer (WS) │
└────────────┘                        │   (FastAPI)   │                    └────────────┬──────────┘
        ▲                              │    │ │ │      │                                 │
        │                              │    ▼ ▼ ▼      │                                 │
        │                       ┌──────────────────┐   │                                 │
        │                       │ neutts/kani/mock │   │                                 │
        └───────── controls ─── │   TTS adapters   │ ◀─┘                                 ▼
                                └──────────────────┘                          Unreal MetaHuman plugin
```

## Repository Layout

```
rt_llm/                  # llama.cpp build script and LLM streaming wrapper
rt_tts/                  # neutts streaming wrapper, Kani fallback, mock TTS
server/                  # WebSocket voice server and orchestrator (FastAPI)
interface/               # Tkinter desktop control panel
unreal_plugin/           # UE 5.6 runtime plugin (C++)
utils/                   # Shared audio and text helpers
scripts/                 # Windows PowerShell automation
models/                  # (gitignored) drop-in LLM/TTS assets
```

## Installation (Windows 11, offline)

1. **Clone the repository** and open a Developer PowerShell prompt in the repo root.
2. **Create the Python environment**:
   ```powershell
   .\scripts\setup_env.ps1
   ```
3. **Build llama.cpp** (Release x64) and copy binaries to `rt_llm\bin\llama`:
   ```powershell
   .\rt_llm\build_llamacpp.ps1
   ```
4. **Add your LLM model**: copy a small GGUF instruct model (4B–7B) into `models\llm\YourModel.gguf`.
5. **Install neutts-air (streaming build)**:
   - Obtain the latest **neutts-air** wheel that exposes `stream_synthesize`.
   - Install it into the virtual environment: `.\.venv\Scripts\pip.exe install <path-to-neutts-air-wheel>`.
   - Place the neutts-air voice model files inside `models\tts\` (e.g., `models\tts\config.json`, `models\tts\weights.onnx`).
   - *(Optional)* Install your Kani TTS package if you want the legacy fallback backend.
6. **Copy the Unreal plugin** into your UE 5.6+ project:
   - `YourUEProject/Plugins/LiveAudioEmotionSource` → copy the entire `unreal_plugin/LiveAudioEmotionSource` folder.
   - Regenerate project files and build the plugin when opening the project.

## Configuration

Runtime defaults live in [`config/default_config.json`](config/default_config.json). Key options:

```json
"orchestrator": {
  "api_host": "127.0.0.1",
  "api_port": 8000,
  "ws_host": "127.0.0.1",
  "ws_port": 17860,
  "sample_rate": 24000,
  "chunk_ms": 20
},
"tts": {
  "backend": "neutts",
  "chunk_ms": 20,
  "lookahead": 2,
  "lookback": 2,
  "interpolate": true
}
```

Override any value on the CLI:

```powershell
.\.venv\Scripts\python.exe -m server.agent_orchestrator --config config\default_config.json --tts-backend neutts --tts-lookahead 4
```

The orchestrator exposes a FastAPI server on `http://<api_host>:<api_port>` and a WebSocket audio stream on `ws://<ws_host>:<ws_port>/voice`.

## Control Panel UI

Launch the Tkinter interface (starts/stops the orchestrator, shows status, and provides a chat console):

```powershell
.\.venv\Scripts\python.exe interface\app.py
```

The panel displays LLM/TTS status, active backend, WebSocket port, and connected client count. Use the prompt box to send test text, switch TTS backends with the dropdown + **Apply**, and toggle the emotion/rate/intensity override sliders for debugging prosody.

## Running from the command line

Start directly from PowerShell if you prefer scripts:

```powershell
.\.venv\Scripts\python.exe -m server.agent_orchestrator --config config\default_config.json --tts-backend neutts --tts-lookahead 2 --tts-lookback 2
```

Use the mock TTS if neither neutts nor Kani is available:

```powershell
.\.venv\Scripts\python.exe -m server.agent_orchestrator --config config\default_config.json --mock
```

Send a test prompt against the HTTP API:

```powershell
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"prompt\":\"Say a warm greeting in one sentence.\"}"
```

Switch backends on the fly:

```powershell
curl -X POST http://127.0.0.1:8000/backend -H "Content-Type: application/json" -d "{\"backend\":\"kani\"}"
```

## Unreal Engine Setup (MetaHuman Animator)

1. Enable the **MetaHuman**, **MetaHuman Animator**, **Audio**, and **WebSockets** plugins.
2. Copy `LiveAudioEmotionSource` into your project’s `Plugins` folder and rebuild.
3. In your level Blueprint (or GameInstance):
   - Call `ConnectVoiceServer("ws://127.0.0.1:17860/voice")` on begin play.
   - Obtain the procedural sound wave via `GetProceduralSoundWave` and assign it to an `AudioComponent` (Sound set to *Override*).
4. Place a **LiveAudioEmotionCurveComponent** on your MetaHuman actor (or any actor driving animation). This exposes:
   - `Emotion_Neutral`, `Emotion_Happy`, `Emotion_Sad`, `Emotion_Angry`, `Emotion_Surprised`
   - `Prosody_Rate` (0.3 slow, 0.6 normal, 0.9 fast)
   - `Prosody_Intensity` (0.3 calm, 0.6 normal, 0.9 excited)
5. In MetaHuman Animator:
   - Configure the Audio-to-Face input to use the same procedural sound wave so visemes update in real time.
   - Map the emotion/prosody curves to your Control Rig or Animation Blueprint.
6. To trigger speech from Blueprints, call `SendTextToAgent("Your line here")`.
7. Press **Play**: the MetaHuman should play speech, audio-driven lipsync, and blend the emotion curves.

## Latency Tuning

- Default chunks are 20 ms (50 fps). The neutts-air streaming API uses 2-frame lookahead/lookback with interpolation for smoother transitions.
- Reduce `"chunk_ms"` in the config for lower latency. Adjust lookahead/lookback accordingly to maintain quality.
- If the network is unstable, increase `chunk_ms` to 40–60 while leaving the WebSocket buffer size untouched.
- Ensure the Unreal audio device is configured for the orchestrator sample rate (defaults to 24 kHz) to avoid resampling.
- Switching to the mock backend is instant and useful for pipeline debugging when TTS engines are unavailable.

## Troubleshooting

- **Firewall prompts**: allow both the orchestrator HTTP port (default 8000) and UE editor WebSocket port (default 17860) through Windows Defender.
- **No lipsync**: confirm MetaHuman Animator listens to the procedural sound wave, not a microphone or file asset.
- **Flat emotion curves**: verify the curve names in Sequencer/Blueprint match exactly; adjust the LLM smoothing in `rt_llm/llm_engine.py` if needed.
- **neutts-air import errors**: reinstall the wheel inside `.venv` and confirm model files are present in `models\tts`.

## Testing

Run the lightweight test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```

## License

MIT License © 2024 MetahumanVoiceAgent Contributors
