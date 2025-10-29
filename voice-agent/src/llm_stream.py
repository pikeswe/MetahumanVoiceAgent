from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, Optional

import aiohttp

from .config import LLMConfig

LOGGER = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=None)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def stream_completion(self, prompt: str) -> AsyncGenerator[str, None]:
        session = await self._get_session()
        url = f"{self.config.host.rstrip('/')}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,
        }
        LOGGER.info("Querying Ollama model %s", self.config.model)
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Ollama error {resp.status}: {text}")
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning("Failed to decode Ollama line: %s", line)
                    continue
                token = data.get("response")
                if token:
                    yield token
                if data.get("done"):
                    break

    async def check_health(self) -> bool:
        try:
            session = await self._get_session()
            url = f"{self.config.host.rstrip('/')}/api/tags"
            async with session.get(url) as resp:
                return resp.status == 200
        except aiohttp.ClientError as exc:
            LOGGER.error("Ollama health check failed: %s", exc)
            return False


async def stream_completion(config: LLMConfig, prompt: str) -> AsyncGenerator[str, None]:
    client = OllamaClient(config)
    try:
        async for token in client.stream_completion(prompt):
            yield token
    finally:
        await client.close()
