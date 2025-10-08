# MetahumanVoiceAgent

MetaHumanVoiceAgent is a Windows 11, fully offline speech agent that streams locally generated speech and emotions from a CPU LLM and TTS stack into Unreal Engine 5.6's MetaHuman Animator using the built-in audio-driven lipsync.

> **Tested on:** Windows 11 Pro, Intel i7-14700KF, 32 GB RAM, no discrete GPU requirement.

## Features

- Offline llama.cpp-based LLM with streaming JSON speech chunks.
- Real-time neutts-air CPU TTS with per-chunk prosody control and PCM streaming.
- WebSocket bridge that feeds MetaHuman Animator with 20 ms PCM frames plus live emotion curves.
- Unreal Engine 5.6+ runtime plugin (`LiveAudioEmotionSource`) exposing procedural audio and anim curves ready for Sequencer/Blueprints.
- One-click PowerShell setup scripts for Python environment, llama.cpp build, and orchestrator launch.
- Emotion smoothing and curve mapping for Neutral, Happy, Sad, Angry, and Surprised, plus rate/intensity prosody values.

## Repository Layout

```
rt_llm/                  # llama.cpp build script and LLM streaming wrapper
rt_tts/                  # neutts-air streaming wrapper and mock TTS
server/                  # WebSocket voice server and orchestrator (FastAPI)
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
5. **Install neutts-air**:
   - Obtain the latest **neutts-air** CPU package (wheel or installer).
   - Install it into the virtual environment: `.\.venv\Scripts\pip.exe install <path-to-neutts-air-wheel>`.
   - Place the neutts-air voice model files inside `models\tts\` (e.g., `models\tts\config.json`, `models\tts\weights.onnx`).
6. **Copy the Unreal plugin** into your UE 5.6+ project:
   - `YourUEProject/Plugins/LiveAudioEmotionSource` → copy the entire `unreal_plugin/LiveAudioEmotionSource` folder.
   - Regenerate project files and build the plugin when opening the project.

## Running the Agent

Start the orchestrator (which internally spins up the WebSocket audio service):

```powershell
.\.venv\Scripts\python.exe server\agent_orchestrator.py --sr 22050 --chunk-ms 20 --model-path models\llm\YourModel.gguf
```

Or use the helper script (adds `--mock` to use the tone generator):

```powershell
.\scripts\start_agent.ps1 -ModelPath "models\llm\YourModel.gguf" -TtsModelDir "models\tts" [-Mock]
```

Send a test prompt:

```powershell
curl -X POST http://127.0.0.1:17860/ask -H "Content-Type: application/json" -d "{\"prompt\":\"Say a warm greeting in one sentence.\"}"
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

- Default chunks are 20 ms (50 fps). Buffer 2–3 frames ahead for stability.
- For a choppy connection, increase `--chunk-ms` to 40–60 while keeping sample rate 22050 Hz mono.
- Ensure the Unreal audio device is configured for 22050 Hz to avoid resampling.

## Troubleshooting

- **Firewall prompts**: allow both the orchestrator (HTTP 17860) and UE editor (WebSocket) through Windows Defender.
- **No lipsync**: confirm MetaHuman Animator listens to the procedural sound wave, not a microphone or file asset.
- **Flat emotion curves**: verify the curve names in Sequencer/Blueprint match exactly; adjust the LLM smoothing in `rt_llm/llm_engine.py` if needed.
- **neutts-air import errors**: reinstall the wheel inside `.venv` and confirm model files are present in `models\tts`.

## Roadmap

- Integrate local speech-to-text (e.g., Whisper.cpp) for full duplex conversations.
- Export MetaHuman viseme curves alongside audio for recording workflows.
- Add additional CPU TTS backends behind the `synth_stream` interface.

## Testing

Run the lightweight utility tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```

## License

MIT License © 2024 MetahumanVoiceAgent Contributors
