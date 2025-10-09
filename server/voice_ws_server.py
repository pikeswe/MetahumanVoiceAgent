"""WebSocket voice streaming server for Unreal Engine clients."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger("voice_ws_server")
logging.basicConfig(level=logging.INFO)


@dataclass
class StreamConfig:
    sample_rate: int = 22050
    channels: int = 1
    chunk_ms: int = 20
    backend: str = "mock"
    lookahead: int = 0
    lookback: int = 0
    interpolate: bool = False


class VoiceStreamServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 17860, config: Optional[StreamConfig] = None) -> None:
        self.host = host
        self.port = port
        self.config = config or StreamConfig()
        self.clients: Set[WebSocketServerProtocol] = set()
        self._lock = asyncio.Lock()
        self._server: Optional[asyncio.AbstractServer] = None

    async def _register(self, websocket: WebSocketServerProtocol) -> None:
        async with self._lock:
            self.clients.add(websocket)
            await websocket.send(
                json.dumps(
                    {
                        "type": "start",
                        "sample_rate": self.config.sample_rate,
                        "channels": self.config.channels,
                        "chunk_ms": self.config.chunk_ms,
                        "backend": self.config.backend,
                        "lookahead": self.config.lookahead,
                        "lookback": self.config.lookback,
                        "interpolate": self.config.interpolate,
                    }
                )
            )
            logger.info("Client connected (%d active)", len(self.clients))

    async def _unregister(self, websocket: WebSocketServerProtocol) -> None:
        async with self._lock:
            self.clients.discard(websocket)
            logger.info("Client disconnected (%d active)", len(self.clients))

    async def handler(self, websocket: WebSocketServerProtocol) -> None:
        if websocket.path not in {"/", "/voice"}:
            await websocket.close(code=1008, reason="Unsupported path")
            return
        await self._register(websocket)
        try:
            await websocket.wait_closed()
        finally:
            await self._unregister(websocket)

    async def start(self) -> None:
        logger.info("Starting voice WebSocket server on %s:%d", self.host, self.port)
        self._server = await websockets.serve(self.handler, self.host, self.port, ping_interval=20, ping_timeout=20)
        if self._server and self._server.sockets:
            socket = self._server.sockets[0]
            bound_port = socket.getsockname()[1]
            if bound_port != self.port:
                logger.info("Voice server bound to port %d", bound_port)
                self.port = bound_port

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        async with self._lock:
            for client in list(self.clients):
                await client.close()
            self.clients.clear()

    async def broadcast_frame(self, frame: bytes) -> None:
        async with self._lock:
            websockets_to_remove = []
            for client in list(self.clients):
                try:
                    await client.send(frame)
                except Exception:
                    websockets_to_remove.append(client)
            for client in websockets_to_remove:
                self.clients.discard(client)

    async def broadcast_emotion(self, payload: Dict[str, float]) -> None:
        message = json.dumps({"type": "emotion", **payload})
        async with self._lock:
            websockets_to_remove = []
            for client in list(self.clients):
                try:
                    await client.send(message)
                except Exception:
                    websockets_to_remove.append(client)
            for client in websockets_to_remove:
                self.clients.discard(client)

    async def broadcast_end(self) -> None:
        message = json.dumps({"type": "end"})
        async with self._lock:
            for client in list(self.clients):
                try:
                    await client.send(message)
                except Exception:
                    self.clients.discard(client)


async def _mock_stream(server: VoiceStreamServer, text: str, emotion: str, duration: float = 2.0) -> None:
    from rt_tts import mock_tts

    logger.info("Streaming mock audio: %s", text)
    for frame in mock_tts.synth_stream(text, emotion=emotion):
        await server.broadcast_frame(frame)
    await server.broadcast_emotion(
        {
            "neutral": 0.0,
            "happy": 1.0 if emotion == "happy" else 0.0,
            "sad": 1.0 if emotion == "sad" else 0.0,
            "angry": 1.0 if emotion == "angry" else 0.0,
            "surprised": 1.0 if emotion == "surprised" else 0.0,
            "rate": 0.6,
            "intensity": 0.6,
        }
    )
    await server.broadcast_end()


async def run_server(args: argparse.Namespace) -> None:
    config = StreamConfig(sample_rate=args.sr, channels=1, chunk_ms=args.chunk_ms)
    server = VoiceStreamServer(host=args.host, port=args.port, config=config)
    await server.start()
    if args.mock:
        await asyncio.sleep(0.5)
        await _mock_stream(server, "Mock tone", "happy")
    await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description="MetaHuman voice WebSocket server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17860)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--chunk-ms", type=int, default=20)
    parser.add_argument("--mock", action="store_true", help="Stream mock audio once clients connect")
    args = parser.parse_args()
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
