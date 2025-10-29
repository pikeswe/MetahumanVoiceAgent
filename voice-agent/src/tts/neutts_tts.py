from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from neutts_air import NeuTTS
except Exception as exc:  # pragma: no cover
    NeuTTS = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class NeuTTSTTS:
    def __init__(self, device: str = "auto", ref_audio: Optional[str] = None, ref_text: Optional[str] = None) -> None:
        if NeuTTS is None:
            raise RuntimeError(f"NeuTTS-Air backend unavailable: {_IMPORT_ERROR}")
        LOGGER.info("Loading NeuTTS-Air backend (%s)", device)
        self.engine = NeuTTS(device=device, reference_audio=ref_audio, reference_text=ref_text)

    async def stream(self, text: str) -> AsyncGenerator[np.ndarray, None]:
        loop = asyncio.get_running_loop()
        chunks = await loop.run_in_executor(None, self._synthesize_sync, text)
        for chunk in chunks:
            yield chunk

    def _synthesize_sync(self, text: str):
        outputs = []
        audio = self.engine.synthesize_stream(text)  # type: ignore[attr-defined]
        for chunk in audio:
            outputs.append(np.array(chunk).astype(np.float32))
        return outputs

    @property
    def output_sample_rate(self) -> int:
        return getattr(self.engine, "sample_rate", 24000)
