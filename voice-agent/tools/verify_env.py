from __future__ import annotations

import asyncio
import platform
import shutil
from pathlib import Path

import requests

from src import audio_io
from src.config import AppConfig


async def check_ollama(config: AppConfig) -> str:
    try:
        resp = requests.get(f"{config.llm.host.rstrip('/')}/api/tags", timeout=3)
        if resp.status_code == 200:
            return "reachable"
        return f"status {resp.status_code}"
    except Exception as exc:  # pragma: no cover
        return f"unreachable ({exc})"


def main() -> None:
    config = AppConfig.load()
    print("[verify] Python", platform.python_version())
    print("[verify] Platform", platform.platform())
    device_hint = config.audio.output_device_name or "(auto)"
    devices = audio_io.list_output_devices()
    match = audio_io.find_output_device(config.audio.output_device_name)
    if match:
        print(f"[verify] Audio device: {match.name} (id={match.id})")
    else:
        print(f"[verify] Audio device not found for hint '{device_hint}'")
    print(f"[verify] Total playback devices: {len(devices)}")
    voice_path = Path(config.piper.voice_path)
    print(f"[verify] Piper voice exists: {'yes' if voice_path.exists() else 'no'} ({voice_path})")
    print(f"[verify] Piper binary: {shutil.which('piper') or shutil.which('piper.exe') or 'missing'}")
    print(f"[verify] Whisper model: {config.stt.model} ({config.stt.device})")
    ollama_status = asyncio.run(check_ollama(config))
    print(f"[verify] Ollama host {config.llm.host}: {ollama_status}")


if __name__ == "__main__":
    main()
