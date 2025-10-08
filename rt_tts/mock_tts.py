"""Mock TTS generator emitting tone/noise for integration testing."""
from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Generator

import numpy as np

from utils import audio_utils


def synth_stream(
    text: str,
    sr: int = 22050,
    chunk_ms: int = 20,
    emotion: str = "neutral",
    rate: str = "normal",
    intensity: str = "normal",
) -> Generator[bytes, None, None]:
    frequency = {
        "neutral": 220.0,
        "happy": 330.0,
        "sad": 200.0,
        "angry": 440.0,
        "surprised": 550.0,
    }.get(emotion, 220.0)
    duration = max(1.0, min(6.0, len(text.split()) * 0.3))
    total_samples = int(sr * duration)
    t = np.arange(total_samples) / sr
    tone = 0.2 * np.sin(2 * math.pi * frequency * t)
    noise = 0.02 * np.random.randn(total_samples)
    samples = tone + noise
    buffer = audio_utils.AudioBuffer(samples=samples.astype(np.float32), sample_rate=sr)
    pcm = buffer.to_pcm16()
    frame_size = int(sr * (chunk_ms / 1000.0))
    for frame in audio_utils.slice_frames(pcm, frame_size):
        yield audio_utils.numpy_to_pcm16_bytes(frame)
        time.sleep(chunk_ms / 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock TTS stream generator")
    parser.add_argument("text", help="Utterance to synthesise")
    parser.add_argument("--emotion", default="neutral")
    args = parser.parse_args()
    for frame in synth_stream(args.text, emotion=args.emotion):
        sys.stdout.buffer.write(frame)


if __name__ == "__main__":
    main()
