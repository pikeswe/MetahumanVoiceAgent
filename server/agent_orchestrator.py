"""End-to-end orchestrator linking LLM streaming, TTS, and WebSocket output."""
from __future__ import annotations

import argparse
import asyncio
import threading
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from rt_llm.llm_engine import LLMEngine
from rt_tts import mock_tts
from rt_tts import neutts_air_wrapper
from server.voice_ws_server import StreamConfig, VoiceStreamServer
from utils import mapping

logger = logging.getLogger("agent_orchestrator")
logging.basicConfig(level=logging.INFO)


class PromptRequest(BaseModel):
    prompt: str


@dataclass
class OrchestratorConfig:
    ws_host: str = "127.0.0.1"
    ws_port: int = 17860
    sample_rate: int = 22050
    chunk_ms: int = 20
    model_path: Path = Path("models/llm/model.gguf")
    llama_binary: Optional[Path] = None
    tts_model_dir: Optional[Path] = None
    use_mock_tts: bool = False


class AgentOrchestrator:
    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self.voice_server = VoiceStreamServer(
            host=config.ws_host, port=config.ws_port, config=StreamConfig(sample_rate=config.sample_rate, chunk_ms=config.chunk_ms)
        )
        self.engine = LLMEngine(model_path=str(config.model_path), llama_binary=str(config.llama_binary) if config.llama_binary else None)
        if not config.use_mock_tts:
            neutts_air_wrapper.load_adapter(config.tts_model_dir)
        self._stream_lock = asyncio.Lock()

    async def ensure_voice_server(self) -> None:
        await self.voice_server.start()

    async def shutdown(self) -> None:
        await self.voice_server.stop()

    async def _async_tts_stream(
        self,
        text: str,
        emotion: str,
        rate: str,
        intensity: str,
    ) -> AsyncIterator[bytes]:
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def worker() -> None:
            try:
                generator = (
                    neutts_air_wrapper.synth_stream(
                        text,
                        sr=self.config.sample_rate,
                        chunk_ms=self.config.chunk_ms,
                        emotion=emotion,
                        rate=rate,
                        intensity=intensity,
                    )
                    if not self.config.use_mock_tts
                    else mock_tts.synth_stream(
                        text,
                        sr=self.config.sample_rate,
                        chunk_ms=self.config.chunk_ms,
                        emotion=emotion,
                        rate=rate,
                        intensity=intensity,
                    )
                )
                for frame in generator:
                    asyncio.run_coroutine_threadsafe(queue.put(frame), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            frame = await queue.get()
            if frame is sentinel:
                break
            yield frame
        thread.join()

    async def handle_prompt(self, prompt: str) -> Dict[str, Any]:
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt must not be empty")
        start = time.perf_counter()
        async with self._stream_lock:
            await self.voice_server.broadcast_emotion(
                {
                    "neutral": 1.0,
                    "happy": 0.0,
                    "sad": 0.0,
                    "angry": 0.0,
                    "surprised": 0.0,
                    "rate": mapping.RATE_VALUES["normal"],
                    "intensity": mapping.INTENSITY_VALUES["normal"],
                }
            )
            chunk_count = 0
            first_frame_time: Optional[float] = None
            for chunk in self.engine.stream_response(prompt):
                chunk_count += 1
                emotion_state = mapping.EmotionState(
                    emotion=chunk["emotion"], rate=chunk["speaking_rate"], intensity=chunk["intensity"]
                ).normalized()
                curve_payload = emotion_state.to_curve_payload()
                emotion_payload = {
                    "neutral": curve_payload["Emotion_Neutral"],
                    "happy": curve_payload["Emotion_Happy"],
                    "sad": curve_payload["Emotion_Sad"],
                    "angry": curve_payload["Emotion_Angry"],
                    "surprised": curve_payload["Emotion_Surprised"],
                    "rate": curve_payload["Prosody_Rate"],
                    "intensity": curve_payload["Prosody_Intensity"],
                }
                await self.voice_server.broadcast_emotion(emotion_payload)
                async for frame in self._async_tts_stream(
                    chunk["text"], chunk["emotion"], chunk["speaking_rate"], chunk["intensity"]
                ):
                    if first_frame_time is None:
                        first_frame_time = time.perf_counter()
                    await self.voice_server.broadcast_frame(frame)
                    await asyncio.sleep(self.config.chunk_ms / 1000.0)
            await self.voice_server.broadcast_end()
            latency_ms = (first_frame_time - start) * 1000.0 if first_frame_time else None
            logger.info("Completed prompt in %.2f ms with %d chunks", (time.perf_counter() - start) * 1000.0, chunk_count)
            return {"status": "ok", "chunks": chunk_count, "first_audio_ms": latency_ms}


def build_app(orchestrator: AgentOrchestrator) -> FastAPI:
    app = FastAPI(title="MetahumanVoiceAgent Orchestrator")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup() -> None:
        await orchestrator.ensure_voice_server()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await orchestrator.shutdown()

    @app.post("/ask")
    async def ask(request: PromptRequest) -> Dict[str, Any]:
        return await orchestrator.handle_prompt(request.prompt)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent orchestrator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17860)
    parser.add_argument("--ws-host", default="127.0.0.1")
    parser.add_argument("--ws-port", type=int, default=17860)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--chunk-ms", type=int, default=20)
    parser.add_argument("--model-path", default="models/llm/model.gguf")
    parser.add_argument("--llama-binary", default=None)
    parser.add_argument("--tts-model-dir", default=None)
    parser.add_argument("--mock", action="store_true", help="Use the mock tone generator instead of neutts-air")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OrchestratorConfig(
        ws_host=args.ws_host,
        ws_port=args.ws_port,
        sample_rate=args.sr,
        chunk_ms=args.chunk_ms,
        model_path=Path(args.model_path),
        llama_binary=Path(args.llama_binary) if args.llama_binary else None,
        tts_model_dir=Path(args.tts_model_dir) if args.tts_model_dir else None,
        use_mock_tts=args.mock,
    )
    orchestrator = AgentOrchestrator(config)
    app = build_app(orchestrator)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
