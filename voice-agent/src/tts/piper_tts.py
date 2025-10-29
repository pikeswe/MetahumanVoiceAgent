from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import AsyncGenerator

import numpy as np

LOGGER = logging.getLogger(__name__)


class PiperTTS:
    def __init__(self, model_path: str, sample_rate: int = 22050) -> None:
        self.model_path = model_path
        self.sample_rate = sample_rate
        self._binary = shutil.which("piper") or shutil.which("piper.exe")
        if not self._binary:
            LOGGER.warning("Piper executable not found in PATH; streaming will fail until installed")

    def is_ready(self) -> bool:
        return bool(self._binary and Path(self.model_path).exists())

    async def stream(self, text: str) -> AsyncGenerator[np.ndarray, None]:
        if not self._binary:
            raise RuntimeError("Piper executable not found")
        if not Path(self.model_path).exists():
            raise RuntimeError(f"Piper voice missing: {self.model_path}")
        cmd = [self._binary, "--model", self.model_path, "--output-raw"]
        LOGGER.debug("Starting Piper process: %s", " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        text_bytes = (text.strip() + "\n").encode("utf-8")
        proc.stdin.write(text_bytes)
        await proc.stdin.drain()
        proc.stdin.close()
        try:
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                array = np.frombuffer(chunk, dtype=np.int16)
                if array.size == 0:
                    continue
                yield array
        finally:
            await proc.wait()
            stderr = await proc.stderr.read() if proc.stderr else b""
            if stderr:
                LOGGER.debug("Piper stderr: %s", stderr.decode("utf-8", errors="ignore"))

    @property
    def output_sample_rate(self) -> int:
        return self.sample_rate
