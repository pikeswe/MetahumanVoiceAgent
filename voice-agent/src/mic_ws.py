from __future__ import annotations

import logging

from fastapi import WebSocket, WebSocketDisconnect

from .pipeline import VoicePipeline

LOGGER = logging.getLogger(__name__)


CHUNK_SIZE = 3200  # 100ms of 16kHz PCM16


async def mic_handler(ws: WebSocket, pipeline: VoicePipeline) -> None:
    await ws.accept()
    LOGGER.info("Mic websocket connected")
    try:
        while True:
            data = await ws.receive_bytes()
            if not data:
                continue
            await pipeline.feed_audio(data)
    except WebSocketDisconnect:
        LOGGER.info("Mic websocket disconnected by client")
    finally:
        await pipeline.flush()
        LOGGER.info("Mic websocket disconnected")
