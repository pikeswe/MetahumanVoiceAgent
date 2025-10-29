from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover - optional dependency
    sd = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:
    from scipy.signal import resample_poly
except Exception:  # pragma: no cover - fallback
    resample_poly = None

LOGGER = logging.getLogger(__name__)

DEFAULT_DEVICE_NAME = "VB-CABLE"


@dataclass
class AudioDevice:
    id: int
    name: str
    is_default: bool


def list_output_devices() -> List[AudioDevice]:
    if sd is None:
        LOGGER.error("sounddevice import failed: %s", _IMPORT_ERROR)
        return []
    devices = sd.query_devices()
    default_idx = sd.default.device[1] if sd.default.device else None
    results: List[AudioDevice] = []
    for idx, info in enumerate(devices):
        if info["max_output_channels"] > 0:
            results.append(
                AudioDevice(
                    id=idx,
                    name=info["name"],
                    is_default=default_idx == idx,
                )
            )
    return results


def find_output_device(name_hint: str | None) -> Optional[AudioDevice]:
    devices = list_output_devices()
    if not devices:
        return None
    if name_hint:
        name_hint_lower = name_hint.lower()
        for device in devices:
            if name_hint_lower in device.name.lower():
                return device
    for device in devices:
        if DEFAULT_DEVICE_NAME.lower() in device.name.lower():
            LOGGER.info("Using fallback VB-CABLE match: %s", device.name)
            return device
    for device in devices:
        if device.is_default:
            LOGGER.warning("Using system default audio device: %s", device.name)
            return device
    return devices[0]


class AudioPlayer:
    def __init__(self, device: Optional[AudioDevice], sample_rate: int) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self._queue: "asyncio.Queue[np.ndarray]" = asyncio.Queue()
        self._stream: Optional[sd.OutputStream] = None if sd else None
        self._task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is required for playback")
        if self._stream is not None:
            return
        device_index = self.device.id if self.device else None
        LOGGER.info("Opening audio device %s @ %s Hz", self.device.name if self.device else "default", self.sample_rate)
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            device=device_index,
            blocksize=0,
        )
        self._stream.start()
        self._task = asyncio.create_task(self._drain_queue())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def _drain_queue(self) -> None:
        assert self._stream is not None
        while True:
            chunk = await self._queue.get()
            if chunk.size == 0:
                continue
            self._stream.write(chunk)

    async def play_stream(self, frames: Iterable[np.ndarray], input_sample_rate: int) -> None:
        await self.start()
        async for chunk in _async_iter(frames):
            chunk16 = ensure_int16(chunk)
            if input_sample_rate != self.sample_rate and resample_poly is not None:
                ratio = self.sample_rate / input_sample_rate
                chunk16 = _resample(chunk16, ratio)
            await self._queue.put(chunk16)

    async def play_bytes(self, pcm: bytes, input_sample_rate: int) -> None:
        data = np.frombuffer(pcm, dtype=np.int16)
        await self.play_stream(_iter_once(data), input_sample_rate)


async def _async_iter(frames: Iterable[np.ndarray]):
    if hasattr(frames, "__aiter__"):
        async for item in frames:  # type: ignore[attr-defined]
            yield item
    else:
        for item in frames:
            yield item
            await asyncio.sleep(0)


def ensure_int16(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int16:
        return data
    if data.dtype == np.float32 or data.dtype == np.float64:
        clipped = np.clip(data, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16)
    return data.astype(np.int16)


def _resample(chunk: np.ndarray, ratio: float) -> np.ndarray:
    if resample_poly is None:
        LOGGER.warning("scipy missing, cannot resample; playing at wrong speed")
        return chunk
    up = int(round(ratio * 1000))
    down = 1000
    resampled = resample_poly(chunk.astype(np.float32), up, down)
    return ensure_int16(resampled)


def _iter_once(data: np.ndarray):
    yield data
