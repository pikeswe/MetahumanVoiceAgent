from __future__ import annotations

import logging
from typing import AsyncGenerator, Optional

from ..audio_io import AudioPlayer
from ..config import AppConfig
from .piper_tts import PiperTTS

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional backend
    from .neutts_tts import NeuTTSTTS  # type: ignore
except Exception:  # pragma: no cover
    NeuTTSTTS = None  # type: ignore


class TTSManager:
    def __init__(self, config: AppConfig, player: AudioPlayer) -> None:
        self.config = config
        self.player = player
        self.backend = self._load_backend(config)
        self.last_text: Optional[str] = None

    def _load_backend(self, config: AppConfig):
        if config.tts_backend == "neutts" and NeuTTSTTS:
            try:
                return NeuTTSTTS(
                    device=config.neutts.device,
                    ref_audio=config.neutts.ref_audio,
                    ref_text=config.neutts.ref_text,
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.error("NeuTTS backend failed (%s); falling back to Piper", exc)
        piper = PiperTTS(config.piper.voice_path, config.piper.sample_rate)
        if not piper.is_ready():
            LOGGER.warning("Piper not ready. Ensure the binary and voice exist at %s", config.piper.voice_path)
        return piper

    async def speak(self, text: str) -> None:
        if not text.strip():
            return
        LOGGER.debug("Speaking chunk (%d chars)", len(text))
        stream = self.backend.stream(text)
        await self.player.play_stream(stream, getattr(self.backend, "output_sample_rate", self.config.audio.sample_rate))
        self.last_text = text

    async def speak_buffered(self, text_iter: AsyncGenerator[str, None]) -> None:
        buffer = ""
        async for token in text_iter:
            buffer += token
            if _should_flush(buffer):
                await self.speak(buffer)
                buffer = ""
        if buffer:
            await self.speak(buffer)

    async def replay_last(self) -> None:
        if self.last_text:
            await self.speak(self.last_text)


def _should_flush(text: str) -> bool:
    if len(text) > 80:
        return True
    if text.endswith(('.', '!', '?', '\n')):
        return True
    return False
