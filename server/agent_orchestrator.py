"""End-to-end orchestrator linking LLM streaming, TTS, and WebSocket output."""
from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterable, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rt_llm.llm_engine import LLMEngine
from rt_tts import mock_tts, neutts_wrapper
from server.voice_ws_server import StreamConfig, VoiceStreamServer
from utils import mapping
from utils.config import load_config

logger = logging.getLogger("agent_orchestrator")
logging.basicConfig(level=logging.INFO)


class PromptRequest(BaseModel):
    prompt: str
    emotion: Optional[str] = None
    speaking_rate: Optional[str] = None
    intensity: Optional[str] = None


class BackendRequest(BaseModel):
    backend: str


@dataclass
class OrchestratorConfig:
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    ws_host: str = "127.0.0.1"
    ws_port: int = 17860
    sample_rate: int = 22050
    chunk_ms: int = 20
    model_path: Path = Path("models/llm/model.gguf")
    llama_binary: Optional[Path] = None
    tts_model_dir: Optional[Path] = None
    tts_backend: str = "neutts"
    tts_lookahead: int = 2
    tts_lookback: int = 2
    tts_interpolate: bool = True
    use_mock_tts: bool = False


class AgentOrchestrator:
    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self.voice_server = VoiceStreamServer(
            host=config.ws_host,
            port=config.ws_port,
            config=StreamConfig(
                sample_rate=config.sample_rate,
                chunk_ms=config.chunk_ms,
                backend=config.tts_backend,
                lookahead=config.tts_lookahead,
                lookback=config.tts_lookback,
                interpolate=config.tts_interpolate,
            ),
        )
        self.engine = self._build_engine()
        self._tts_streamer: Callable[[str, str, str, str], Iterable[bytes]]
        self.active_tts_backend: str = "mock"
        self._configure_tts()
        self._stream_lock = asyncio.Lock()

    def _build_engine(self) -> LLMEngine:
        return LLMEngine(
            model_path=str(self.config.model_path),
            llama_binary=str(self.config.llama_binary) if self.config.llama_binary else None,
        )

    def _make_mock_stream(self) -> Callable[[str, str, str, str], Iterable[bytes]]:
        def _stream(text: str, emotion: str, rate: str, intensity: str) -> Iterable[bytes]:
            return mock_tts.synth_stream(
                text,
                sr=self.config.sample_rate,
                chunk_ms=self.config.chunk_ms,
                emotion=emotion,
                rate=rate,
                intensity=intensity,
            )

        return _stream

    def _make_neutts_stream(self) -> Callable[[str, str, str, str], Iterable[bytes]]:
        def _stream(text: str, emotion: str, rate: str, intensity: str) -> Iterable[bytes]:
            return neutts_wrapper.synth_stream(
                text,
                sr=self.config.sample_rate,
                chunk_ms=self.config.chunk_ms,
                emotion=emotion,
                rate=rate,
                intensity=intensity,
                lookahead=self.config.tts_lookahead,
                lookback=self.config.tts_lookback,
                interpolate=self.config.tts_interpolate,
            )

        return _stream

    def _make_kani_stream(self) -> Callable[[str, str, str, str], Iterable[bytes]]:
        from rt_tts import kani_wrapper

        def _stream(text: str, emotion: str, rate: str, intensity: str) -> Iterable[bytes]:
            return kani_wrapper.synth_stream(
                text,
                sr=self.config.sample_rate,
                chunk_ms=self.config.chunk_ms,
                emotion=emotion,
                rate=rate,
                intensity=intensity,
            )

        return _stream

    def _configure_tts(self) -> None:
        backend = (self.config.tts_backend or "").lower()
        streamer: Optional[Callable[[str, str, str, str], Iterable[bytes]]] = None
        active_backend = "mock"

        if self.config.use_mock_tts or backend == "mock":
            streamer = self._make_mock_stream()
            active_backend = "mock"
        else:
            if backend in {"neutts", "neutts-air", "neuttsair", ""}:
                adapter_config = neutts_wrapper.NeuTTSConfig(
                    chunk_ms=self.config.chunk_ms,
                    lookahead=self.config.tts_lookahead,
                    lookback=self.config.tts_lookback,
                    interpolate=self.config.tts_interpolate,
                    model_dir=self.config.tts_model_dir,
                )
                try:
                    neutts_wrapper.load_adapter(adapter_config)
                    streamer = self._make_neutts_stream()
                    active_backend = "neutts"
                except ImportError as exc:
                    logger.warning("Neutts backend unavailable: %s", exc)
            if streamer is None and backend in {"kani", "neutts", "neutts-air", "neuttsair", ""}:
                try:
                    streamer = self._make_kani_stream()
                    active_backend = "kani"
                except Exception as exc:  # pragma: no cover - requires kani installation
                    logger.warning("Kani backend unavailable: %s", exc)
        if streamer is None:
            streamer = self._make_mock_stream()
            active_backend = "mock"

        self._tts_streamer = streamer
        self.active_tts_backend = active_backend
        self.voice_server.config.backend = active_backend
        self.voice_server.config.chunk_ms = self.config.chunk_ms
        self.voice_server.config.lookahead = self.config.tts_lookahead
        self.voice_server.config.lookback = self.config.tts_lookback
        self.voice_server.config.interpolate = self.config.tts_interpolate

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
                for frame in self._tts_streamer(text, emotion, rate, intensity):
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

    async def handle_prompt(self, request: PromptRequest) -> Dict[str, Any]:
        if not request.prompt:
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
            for chunk in self.engine.stream_response(request.prompt):
                chunk_count += 1
                emotion_value = request.emotion or chunk["emotion"]
                rate_value = request.speaking_rate or chunk["speaking_rate"]
                intensity_value = request.intensity or chunk["intensity"]
                emotion_state = mapping.EmotionState(
                    emotion=emotion_value, rate=rate_value, intensity=intensity_value
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
                    chunk["text"], emotion_value, rate_value, intensity_value
                ):
                    if first_frame_time is None:
                        first_frame_time = time.perf_counter()
                    await self.voice_server.broadcast_frame(frame)
            await self.voice_server.broadcast_end()
            latency_ms = (first_frame_time - start) * 1000.0 if first_frame_time else None
            logger.info(
                "Completed prompt in %.2f ms with %d chunks via %s backend",
                (time.perf_counter() - start) * 1000.0,
                chunk_count,
                self.active_tts_backend,
            )
            return {
                "status": "ok",
                "chunks": chunk_count,
                "first_audio_ms": latency_ms,
                "backend": self.active_tts_backend,
            }

    async def get_status(self) -> Dict[str, Any]:
        return {
            "tts_backend": self.active_tts_backend,
            "ws_host": self.voice_server.host,
            "ws_port": self.voice_server.port,
            "sample_rate": self.config.sample_rate,
            "chunk_ms": self.config.chunk_ms,
            "clients": len(self.voice_server.clients),
        }

    async def reload_models(self) -> Dict[str, Any]:
        async with self._stream_lock:
            self.engine = self._build_engine()
            self._configure_tts()
            return {"status": "reloaded", "backend": self.active_tts_backend}

    async def switch_backend(self, backend: str) -> Dict[str, Any]:
        async with self._stream_lock:
            previous = self.active_tts_backend
            self.config.tts_backend = backend
            self.config.use_mock_tts = backend.lower() == "mock"
            self._configure_tts()
            return {"requested": backend, "active": self.active_tts_backend, "previous": previous}


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
        return await orchestrator.handle_prompt(request)

    @app.get("/status")
    async def status() -> Dict[str, Any]:
        return await orchestrator.get_status()

    @app.post("/reload")
    async def reload() -> Dict[str, Any]:
        return await orchestrator.reload_models()

    @app.post("/backend")
    async def set_backend(request: BackendRequest) -> Dict[str, Any]:
        return await orchestrator.switch_backend(request.backend)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent orchestrator")
    parser.add_argument("--config", default=str(Path("config") / "default_config.json"))
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--ws-host", default=None)
    parser.add_argument("--ws-port", type=int, default=None)
    parser.add_argument("--sr", type=int, default=None)
    parser.add_argument("--chunk-ms", type=int, default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--llama-binary", default=None)
    parser.add_argument("--tts-model-dir", default=None)
    parser.add_argument("--tts-backend", default=None)
    parser.add_argument("--tts-lookahead", type=int, default=None)
    parser.add_argument("--tts-lookback", type=int, default=None)
    parser.add_argument("--tts-interpolate", type=str, default=None)
    parser.add_argument("--mock", action="store_true", help="Use the mock tone generator instead of installed backends")
    return parser.parse_args()


def _str_to_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    if value.lower() in {"1", "true", "yes", "y"}:
        return True
    if value.lower() in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot interpret boolean from '{value}'")


def main() -> None:
    args = parse_args()
    config_data = load_config(args.config)
    orchestrator_section = config_data.get("orchestrator", {})
    tts_section = config_data.get("tts", {})

    api_host = args.host or orchestrator_section.get("api_host", "127.0.0.1")
    api_port = args.port or orchestrator_section.get("api_port", 8000)
    ws_host = args.ws_host or orchestrator_section.get("ws_host", "127.0.0.1")
    ws_port = args.ws_port or orchestrator_section.get("ws_port", 17860)
    sample_rate = args.sr or orchestrator_section.get("sample_rate", 22050)
    chunk_ms = args.chunk_ms or tts_section.get("chunk_ms", orchestrator_section.get("chunk_ms", 20))
    model_path = args.model_path or orchestrator_section.get("model_path", "models/llm/model.gguf")
    llama_binary = args.llama_binary or orchestrator_section.get("llama_binary")
    tts_model_dir = args.tts_model_dir or orchestrator_section.get("tts_model_dir")
    backend = args.tts_backend or tts_section.get("backend", "neutts")
    lookahead = args.tts_lookahead or tts_section.get("lookahead", 2)
    lookback = args.tts_lookback or tts_section.get("lookback", 2)
    interpolate_override = _str_to_bool(args.tts_interpolate) if args.tts_interpolate is not None else None
    interpolate = (
        interpolate_override
        if interpolate_override is not None
        else tts_section.get("interpolate", True)
    )

    config = OrchestratorConfig(
        api_host=api_host,
        api_port=api_port,
        ws_host=ws_host,
        ws_port=ws_port,
        sample_rate=sample_rate,
        chunk_ms=chunk_ms,
        model_path=Path(model_path),
        llama_binary=Path(llama_binary) if llama_binary else None,
        tts_model_dir=Path(tts_model_dir) if tts_model_dir else None,
        tts_backend=backend,
        tts_lookahead=lookahead,
        tts_lookback=lookback,
        tts_interpolate=bool(interpolate),
        use_mock_tts=args.mock or backend.lower() == "mock",
    )

    orchestrator = AgentOrchestrator(config)
    app = build_app(orchestrator)
    uvicorn.run(app, host=api_host, port=api_port)


if __name__ == "__main__":
    main()
