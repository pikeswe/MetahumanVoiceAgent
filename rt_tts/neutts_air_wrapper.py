"""Wrapper around the CPU-only neutts-air TTS synthesiser."""
from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Generator, Iterable, Optional

import numpy as np

from utils import audio_utils, mapping


class NeuTTSAdapter:
    def __init__(self, model_dir: Optional[str] = None) -> None:
        self.model_dir = Path(model_dir or Path.cwd() / "models" / "tts")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._engine = None

    def _lazy_engine(self):
        if self._engine is None:
            try:
                module = importlib.import_module("neutts_air")
            except ImportError as exc:  # pragma: no cover - depends on user install
                raise ImportError(
                    "neutts-air is not installed. Install the wheel supplied by the project instructions."
                ) from exc
            if hasattr(module, "NeuTTS"):
                cls = getattr(module, "NeuTTS")
                if hasattr(cls, "from_pretrained"):
                    self._engine = cls.from_pretrained(str(self.model_dir))
                else:
                    self._engine = cls(str(self.model_dir))
            elif hasattr(module, "load_model"):
                self._engine = module.load_model(str(self.model_dir))
            else:  # pragma: no cover
                raise RuntimeError(
                    "Unsupported neutts-air version. Expected NeuTTS or load_model entry point."
                )
        return self._engine

    def synthesize(self, text: str, emotion: str, rate: str, intensity: str) -> tuple[np.ndarray, int]:
        engine = self._lazy_engine()
        kwargs = {"text": text, "emotion": emotion}
        if hasattr(engine, "infer"):
            audio, sr = engine.infer(**kwargs)
        elif hasattr(engine, "speak"):
            audio, sr = engine.speak(**kwargs)
        elif hasattr(engine, "tts"):
            result = engine.tts(text, emotion=emotion)
            if isinstance(result, tuple) and len(result) == 2:
                audio, sr = result
            else:
                audio = np.asarray(result, dtype=np.float32)
                sr = getattr(engine, "sample_rate", 22050)
        else:  # pragma: no cover
            raise RuntimeError("The neutts-air engine does not expose a known synthesis API.")
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)
        sr = int(sr)
        stretch, gain = mapping.tts_controls(rate, intensity)
        if not math.isclose(stretch, 1.0, rel_tol=1e-3):
            samples = audio_utils.time_stretch(samples, stretch)
        if not math.isclose(gain, 0.0, abs_tol=1e-3):
            samples = audio_utils.AudioBuffer(samples=samples, sample_rate=sr).apply_gain_db(gain).samples
        return samples.astype(np.float32), sr


_adapter: Optional[NeuTTSAdapter] = None


def _ensure_adapter() -> NeuTTSAdapter:
    global _adapter
    if _adapter is None:
        _adapter = NeuTTSAdapter()
    return _adapter


def synth_stream(
    text: str,
    sr: int = 22050,
    chunk_ms: int = 20,
    emotion: str = "neutral",
    rate: str = "normal",
    intensity: str = "normal",
) -> Generator[bytes, None, None]:
    adapter = _ensure_adapter()
    samples, native_sr = adapter.synthesize(text, emotion=emotion, rate=rate, intensity=intensity)
    buffer = audio_utils.AudioBuffer(samples=samples, sample_rate=native_sr).ensure_mono()
    if native_sr != sr:
        buffer = audio_utils.resample_audio(buffer, sr)
    pcm = buffer.to_pcm16()
    fade = int(sr * 0.005)
    buffer.samples = buffer.samples.astype(np.float32)
    buffer.apply_fade(fade)
    frame_size = int(sr * (chunk_ms / 1000.0))
    for frame in audio_utils.slice_frames(pcm, frame_size):
        yield audio_utils.numpy_to_pcm16_bytes(frame)


def load_adapter(model_dir: Optional[str] = None) -> NeuTTSAdapter:
    global _adapter
    _adapter = NeuTTSAdapter(model_dir=model_dir)
    return _adapter
