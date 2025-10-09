"""Optional wrapper for the legacy Kani TTS backend."""
from __future__ import annotations

import importlib
from typing import Generator

from utils import audio_utils


class KaniUnavailable(RuntimeError):
    """Raised when Kani TTS backend cannot be used."""


def _load_engine():
    module = importlib.import_module("kani_tts")
    if hasattr(module, "stream"):
        return module.stream
    if hasattr(module, "synthesize_stream"):
        return module.synthesize_stream
    raise AttributeError("kani_tts module missing a streaming entry point")


def synth_stream(
    text: str,
    *,
    sr: int,
    chunk_ms: int,
    emotion: str,
    rate: str,
    intensity: str,
    **_: object,
) -> Generator[bytes, None, None]:
    try:
        stream_fn = _load_engine()
    except Exception as exc:  # pragma: no cover - requires kani install
        raise KaniUnavailable("Kani TTS backend is unavailable") from exc

    for frame in stream_fn(text=text, emotion=emotion, rate=rate, intensity=intensity, sample_rate=sr, chunk_ms=chunk_ms):
        yield audio_utils.ensure_pcm16(frame)


__all__ = ["synth_stream", "KaniUnavailable"]
