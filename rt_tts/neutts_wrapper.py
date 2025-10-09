"""Streaming wrapper around neutts-air with PCM16 output."""
from __future__ import annotations

import importlib
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional

import numpy as np

from utils import audio_utils

logger = logging.getLogger(__name__)


@dataclass
class NeuTTSConfig:
    chunk_ms: int = 20
    lookahead: int = 2
    lookback: int = 2
    interpolate: bool = True
    model_dir: Optional[Path] = None


class NeuTTSStreamingAdapter:
    """Thin wrapper around the neutts-air streaming API."""

    def __init__(self, config: Optional[NeuTTSConfig] = None) -> None:
        self.config = config or NeuTTSConfig()
        self._module = None

    def _import_module(self):
        if self._module is not None:
            return self._module
        errors = []
        for name in ("neuttsair", "neutts_air"):
            try:
                self._module = importlib.import_module(name)
                logger.info("Loaded %s for streaming TTS", name)
                return self._module
            except ImportError as exc:  # pragma: no cover - depends on install
                errors.append(str(exc))
        raise ImportError(
            "neutts-air package not found. Install the neutts-air wheel before selecting the Neutts backend."
            f" Tried imports errors: {'; '.join(errors)}"
        )

    def stream(
        self,
        text: str,
        sr: int,
        emotion: str,
        rate: str,
        intensity: str,
        *,
        chunk_ms: Optional[int] = None,
        lookahead: Optional[int] = None,
        lookback: Optional[int] = None,
        interpolate: Optional[bool] = None,
    ) -> Iterable[bytes]:
        module = self._import_module()
        stream_fn = getattr(module, "stream_synthesize", None)
        if stream_fn is None:
            raise RuntimeError("Installed neutts-air package does not expose stream_synthesize().")

        effective_chunk_ms = chunk_ms if chunk_ms is not None else self.config.chunk_ms
        chunk_size = max(1, int(sr * (effective_chunk_ms / 1000.0)))

        config = NeuTTSConfig(
            chunk_ms=effective_chunk_ms,
            lookahead=self.config.lookahead if lookahead is None else lookahead,
            lookback=self.config.lookback if lookback is None else lookback,
            interpolate=self.config.interpolate if interpolate is None else interpolate,
            model_dir=self.config.model_dir,
        )

        kwargs = {
            "text": text,
            "sample_rate": sr,
            "chunk_size": chunk_size,
            "lookahead": config.lookahead,
            "lookback": config.lookback,
            "interpolate": config.interpolate,
            "emotion": emotion,
            "rate": rate,
            "intensity": intensity,
        }
        if config.model_dir:
            kwargs["model_dir"] = str(config.model_dir)

        signature = inspect.signature(stream_fn)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
        logger.debug("Streaming with kwargs: %s", filtered_kwargs)

        for payload in stream_fn(**filtered_kwargs):
            if isinstance(payload, tuple) and len(payload) == 2:
                frame, native_sr = payload
            else:
                frame = payload
                native_sr = sr
            samples = np.asarray(frame, dtype=np.float32)
            if samples.ndim > 1:
                samples = np.mean(samples, axis=1)
            buffer = audio_utils.AudioBuffer(samples=samples, sample_rate=int(native_sr)).ensure_mono()
            if buffer.sample_rate != sr:
                buffer = audio_utils.resample_audio(buffer, sr)
            pcm = buffer.to_pcm16()
            yield audio_utils.numpy_to_pcm16_bytes(pcm)


_adapter: Optional[NeuTTSStreamingAdapter] = None


def load_adapter(config: Optional[NeuTTSConfig] = None) -> NeuTTSStreamingAdapter:
    global _adapter
    _adapter = NeuTTSStreamingAdapter(config)
    return _adapter


def get_adapter() -> NeuTTSStreamingAdapter:
    if _adapter is None:
        load_adapter()
    return _adapter  # type: ignore[return-value]


def synth_stream(
    text: str,
    *,
    sr: int,
    chunk_ms: int,
    emotion: str,
    rate: str,
    intensity: str,
    lookahead: int,
    lookback: int,
    interpolate: bool,
) -> Generator[bytes, None, None]:
    adapter = get_adapter()
    yield from adapter.stream(
        text,
        sr=sr,
        emotion=emotion,
        rate=rate,
        intensity=intensity,
        chunk_ms=chunk_ms,
        lookahead=lookahead,
        lookback=lookback,
        interpolate=interpolate,
    )


__all__ = ["NeuTTSConfig", "NeuTTSStreamingAdapter", "load_adapter", "get_adapter", "synth_stream"]
