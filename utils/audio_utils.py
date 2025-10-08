"""Audio helper utilities for MetahumanVoiceAgent."""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

import numpy as np
import soundfile as sf

PCM16_MAX = 32767
PCM16_MIN = -32768


@dataclass
class AudioBuffer:
    """Container representing a mono audio buffer."""

    samples: np.ndarray
    sample_rate: int

    def ensure_mono(self) -> "AudioBuffer":
        if self.samples.ndim == 2:
            if self.samples.shape[1] == 1:
                self.samples = self.samples[:, 0]
            else:
                self.samples = np.mean(self.samples, axis=1)
        return self

    def to_pcm16(self) -> np.ndarray:
        clipped = np.clip(self.samples, -1.0, 1.0)
        return (clipped * PCM16_MAX).astype(np.int16)

    def apply_gain_db(self, gain_db: float) -> "AudioBuffer":
        factor = 10 ** (gain_db / 20.0)
        self.samples *= factor
        return self

    def apply_fade(self, fade_samples: int) -> "AudioBuffer":
        if fade_samples <= 0:
            return self
        fade_samples = min(fade_samples, len(self.samples) // 2)
        if fade_samples <= 0:
            return self
        fade_in = np.linspace(0.0, 1.0, fade_samples, endpoint=False)
        fade_out = np.linspace(1.0, 0.0, fade_samples, endpoint=False)
        self.samples[:fade_samples] *= fade_in
        self.samples[-fade_samples:] *= fade_out
        return self


def load_wav(path: str) -> AudioBuffer:
    samples, sr = sf.read(path, dtype="float32")
    return AudioBuffer(samples=np.asarray(samples, dtype=np.float32), sample_rate=sr).ensure_mono()


def save_wav(path: str, buffer: AudioBuffer) -> None:
    sf.write(path, buffer.samples, buffer.sample_rate)


def resample_audio(buffer: AudioBuffer, new_sr: int) -> AudioBuffer:
    if buffer.sample_rate == new_sr:
        return AudioBuffer(samples=np.copy(buffer.samples), sample_rate=new_sr)
    duration = len(buffer.samples) / float(buffer.sample_rate)
    new_length = int(round(duration * new_sr))
    x_old = np.linspace(0, duration, len(buffer.samples), endpoint=False)
    x_new = np.linspace(0, duration, new_length, endpoint=False)
    resampled = np.interp(x_new, x_old, buffer.samples)
    return AudioBuffer(samples=resampled.astype(np.float32), sample_rate=new_sr)


def pcm16_bytes_to_numpy(frames: bytes) -> np.ndarray:
    return np.frombuffer(frames, dtype=np.int16)


def numpy_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    return samples.astype(np.int16).tobytes()


def slice_frames(pcm: np.ndarray, frame_size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(pcm), frame_size):
        end = start + frame_size
        if end > len(pcm):
            pad = np.zeros(end - len(pcm), dtype=np.int16)
            chunk = np.concatenate([pcm[start:], pad])
        else:
            chunk = pcm[start:end]
        yield chunk


def chunk_stream(samples: np.ndarray, sample_rate: int, chunk_ms: int) -> Iterator[bytes]:
    frame_size = int(sample_rate * (chunk_ms / 1000.0))
    for frame in slice_frames(samples, frame_size):
        yield numpy_to_pcm16_bytes(frame)


def frames_per_second(chunk_ms: int) -> int:
    return int(round(1000 / chunk_ms))


def time_stretch(samples: np.ndarray, rate: float) -> np.ndarray:
    if np.isclose(rate, 1.0):
        return samples
    window = 256
    hop = window // 2
    rate = max(0.5, min(2.0, rate))
    phase = np.zeros(window)
    hanning = np.hanning(window)
    result = np.zeros(int(len(samples) / rate) + window)
    result_pos = 0
    for i in range(0, len(samples) - window, hop):
        chunk = samples[i : i + window]
        spectrum = np.fft.fft(hanning * chunk)
        phase += np.angle(spectrum)
        resynth = np.real(np.fft.ifft(np.abs(spectrum) * np.exp(1j * phase)))
        result[result_pos : result_pos + window] += hanning * resynth
        result_pos += int(hop / rate)
    return result[:result_pos].astype(np.float32)


def ensure_pcm16(frames: bytes) -> bytes:
    arr = pcm16_bytes_to_numpy(frames)
    arr = np.clip(arr, PCM16_MIN, PCM16_MAX)
    return arr.astype(np.int16).tobytes()


def join_pcm16(frames: Iterable[bytes]) -> bytes:
    return b"".join(frames)


def memory_wav_buffer(samples: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV")
    return buffer.getvalue()


def rms(samples: np.ndarray) -> float:
    return math.sqrt(float(np.mean(np.square(samples)))) if len(samples) else 0.0
