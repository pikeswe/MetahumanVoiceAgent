from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import audio_io
from .config import AppConfig
from .mic_ws import mic_handler
from .pipeline import VoicePipeline

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Metahuman Voice Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

WEB_DIR = Path(__file__).resolve().parent / "web"
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.on_event("startup")
async def startup_event() -> None:
    config = AppConfig.load()
    logging.getLogger().setLevel(getattr(logging, config.server.log_level.upper(), logging.INFO))
    app.state.config = config
    app.state.pipeline = VoicePipeline(config)
    LOGGER.info("Voice agent ready on %s:%s", config.server.host, config.server.port)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    pipeline: VoicePipeline = app.state.pipeline
    await pipeline.close()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/audio/devices")
async def list_devices() -> Dict[str, Any]:
    devices = audio_io.list_output_devices()
    return {
        "devices": [
            {
                "id": device.id,
                "name": device.name,
                "is_default": device.is_default,
            }
            for device in devices
        ]
    }


@app.post("/api/config")
async def update_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    config: AppConfig = app.state.config
    pipeline: VoicePipeline = app.state.pipeline
    config.update_from_payload(payload)
    config.save()
    await pipeline.close()
    app.state.pipeline = VoicePipeline(config)
    return {"status": "ok", "config": config.to_env()}


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    pipeline: VoicePipeline = app.state.pipeline
    devices = audio_io.list_output_devices()
    return {
        "status": "ok",
        "ollama_model": pipeline.config.llm.model,
        "tts_backend": pipeline.config.tts_backend,
        "device": pipeline.audio_device.name if pipeline.audio_device else None,
        "errors": pipeline.error_counts,
        "devices": len(devices),
        "speak_partials": pipeline.config.ui.speak_partials,
    }


@app.post("/api/repeat")
async def repeat_last() -> Dict[str, Any]:
    pipeline: VoicePipeline = app.state.pipeline
    await pipeline.repeat_last()
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    pipeline: VoicePipeline = app.state.pipeline
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Missing text")
    await pipeline.submit_text(str(text))
    return {"status": "ok"}


@app.websocket("/ws/mic")
async def ws_mic(ws: WebSocket) -> None:
    pipeline: VoicePipeline = app.state.pipeline
    await mic_handler(ws, pipeline)


@app.websocket("/ws/events")
async def ws_events(ws: WebSocket) -> None:
    pipeline: VoicePipeline = app.state.pipeline
    queue = pipeline.register_event_queue()
    await ws.accept()
    try:
        while True:
            event = await queue.get()
            await ws.send_json(event)
    except WebSocketDisconnect:
        LOGGER.info("Events websocket disconnected")
    finally:
        pipeline.unregister_event_queue(queue)
