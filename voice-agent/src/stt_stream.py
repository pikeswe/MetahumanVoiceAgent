from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, List

import numpy as np

from .config import STTConfig

LOGGER = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
except Exception as exc:  # pragma: no cover - optional heavy dependency
    WhisperModel = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class TranscriptEvent:
    type: str  # "partial" or "final"
    text: str


class WhisperStreamer:
    def __init__(self, config: STTConfig) -> None:
        if WhisperModel is None:
            raise RuntimeError(f"faster-whisper is not available: {_IMPORT_ERROR}")
        self.config = config
        device = _resolve_device(config.device)
        compute_type = "float16" if device == "cuda" else "int8"
        LOGGER.info("Loading Whisper model %s on %s (%s)", config.model, device, compute_type)
        self.model = WhisperModel(config.model, device=device, compute_type=compute_type)

    async def transcribe(
        self,
        audio_pcm: bytes,
        *,
        yield_partials: bool = True,
        yield_final: bool = True,
    ) -> AsyncGenerator[TranscriptEvent, None]:
        loop = asyncio.get_running_loop()
        audio = np.frombuffer(audio_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        text = await loop.run_in_executor(None, self._transcribe_sync, audio)
        if not text:
            return
        words = text.strip().split()
        partial_text = ""
        if yield_partials:
            for word in words:
                partial_text = (partial_text + " " + word).strip()
                yield TranscriptEvent(type="partial", text=partial_text)
        if yield_final:
            yield TranscriptEvent(type="final", text=text.strip())

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            beam_size=1,
            temperature=0.0,
            vad_filter=False,
        )
        texts: List[str] = []
        for segment in segments:
            texts.append(segment.text.strip())
        return " ".join(texts).strip()


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except Exception:  # pragma: no cover
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
