from __future__ import annotations

import collections
import logging
from typing import Deque, Generator, List, Tuple

import webrtcvad

LOGGER = logging.getLogger(__name__)


class VADStream:
    """Streaming voice activity detector producing speech segments."""

    def __init__(
        self,
        aggressiveness: int = 2,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        silence_ms: int = 600,
        partial_ms: int = 400,
    ) -> None:
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.partial_frames = max(1, partial_ms // frame_ms)
        self._frame_bytes = int(sample_rate * frame_ms / 1000) * 2
        self._buffer = bytearray()
        self._ring_buffer: Deque[Tuple[bytes, bool]] = collections.deque(maxlen=self.silence_frames)
        self._triggered = False
        self._voiced_frames: List[bytes] = []
        self._frames_since_partial = 0

    def process(self, pcm16: bytes) -> Generator[Tuple[str, bytes], None, None]:
        if not pcm16:
            return
        self._buffer.extend(pcm16)
        while len(self._buffer) >= self._frame_bytes:
            frame = bytes(self._buffer[: self._frame_bytes])
            del self._buffer[: self._frame_bytes]
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            if not self._triggered:
                self._ring_buffer.append((frame, is_speech))
                num_voiced = len([1 for _, speech in self._ring_buffer if speech])
                if num_voiced > 0.9 * self._ring_buffer.maxlen:
                    self._triggered = True
                    LOGGER.debug("VAD triggered")
                    while self._ring_buffer:
                        buffered_frame, _ = self._ring_buffer.popleft()
                        self._voiced_frames.append(buffered_frame)
                    self._frames_since_partial = 0
            else:
                self._voiced_frames.append(frame)
                self._frames_since_partial += 1
                self._ring_buffer.append((frame, is_speech))
                num_unvoiced = len([1 for _, speech in self._ring_buffer if not speech])
                if self._frames_since_partial >= self.partial_frames:
                    yield ("partial", b"".join(self._voiced_frames))
                    self._frames_since_partial = 0
                if num_unvoiced > 0.9 * self._ring_buffer.maxlen:
                    LOGGER.debug("VAD released")
                    yield ("final", b"".join(self._voiced_frames))
                    self._triggered = False
                    self._ring_buffer.clear()
                    self._voiced_frames.clear()
                    self._frames_since_partial = 0

    def flush(self) -> Generator[Tuple[str, bytes], None, None]:
        if self._voiced_frames:
            yield ("final", b"".join(self._voiced_frames))
            self._voiced_frames.clear()
        self._buffer.clear()
        self._ring_buffer.clear()
        self._triggered = False
        self._frames_since_partial = 0
