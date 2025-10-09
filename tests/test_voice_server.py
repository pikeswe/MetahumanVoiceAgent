import json

import pytest
websockets = pytest.importorskip("websockets")

from server.voice_ws_server import StreamConfig, VoiceStreamServer


@pytest.mark.asyncio
async def test_voice_server_sends_start_chunk_end():
    config = StreamConfig(sample_rate=8000, chunk_ms=20, backend="mock")
    server = VoiceStreamServer(host="127.0.0.1", port=0, config=config)
    await server.start()
    # Discover the ephemeral port used by the server.
    sockets = server._server.sockets  # type: ignore[attr-defined]
    port = sockets[0].getsockname()[1]
    server.port = port

    async with websockets.connect(f"ws://127.0.0.1:{port}") as websocket:
        start_message = await websocket.recv()
        start_payload = json.loads(start_message)
        assert start_payload["type"] == "start"
        frame = b"\x00\x01" * (config.sample_rate * config.chunk_ms // 1000)
        await server.broadcast_frame(frame)
        chunk_message = await websocket.recv()
        assert isinstance(chunk_message, bytes)
        assert chunk_message == frame
        await server.broadcast_end()
        end_message = await websocket.recv()
        assert json.loads(end_message)["type"] == "end"

    await server.stop()
