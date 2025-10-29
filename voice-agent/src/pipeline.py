from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Dict, Set

from .audio_io import AudioPlayer, find_output_device
from .config import AppConfig
from .llm_stream import OllamaClient
from .stt_stream import WhisperStreamer
from .tts import TTSManager
from .vad import VADStream

LOGGER = logging.getLogger(__name__)


class VoicePipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.event_queues: Set[asyncio.Queue[Dict[str, Any]]] = set()
        self.audio_queue: "asyncio.Queue[bytes]" = asyncio.Queue()
        self.vad = VADStream(aggressiveness=config.stt.vad_aggressiveness)
        self.whisper = WhisperStreamer(config.stt)
        self.ollama = OllamaClient(config.llm)
        self.audio_device = find_output_device(config.audio.output_device_name)
        if not self.audio_device:
            LOGGER.warning("No matching audio device found for '%s'", config.audio.output_device_name)
        self.audio_player = AudioPlayer(self.audio_device, config.audio.sample_rate)
        self.tts = TTSManager(config, self.audio_player)
        self._process_task = asyncio.create_task(self._run())
        self._llm_lock = asyncio.Lock()
        self.error_counts = {"stt": 0, "llm": 0, "tts": 0}
        self.last_response_text: str = ""

    def register_event_queue(self) -> asyncio.Queue[Dict[str, Any]]:
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.event_queues.add(queue)
        LOGGER.debug("Registered event listener (%d total)", len(self.event_queues))
        return queue

    def unregister_event_queue(self, queue: asyncio.Queue[Dict[str, Any]]) -> None:
        self.event_queues.discard(queue)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        for queue in list(self.event_queues):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                LOGGER.warning("Dropping event for slow listener")

    async def feed_audio(self, pcm_chunk: bytes) -> None:
        await self.audio_queue.put(pcm_chunk)

    async def flush(self) -> None:
        for event_type, payload in self.vad.flush():
            await self._handle_vad_event(event_type, payload)

    async def _run(self) -> None:
        while True:
            chunk = await self.audio_queue.get()
            for event_type, payload in self.vad.process(chunk):
                await self._handle_vad_event(event_type, payload)

    async def _handle_vad_event(self, event_type: str, audio: bytes) -> None:
        if event_type == "partial":
            await self._handle_partial(audio)
        elif event_type == "final":
            await self._handle_final(audio)

    async def _handle_partial(self, audio: bytes) -> None:
        try:
            async for event in self.whisper.transcribe(audio, yield_final=False):
                if event.type == "partial":
                    await self.broadcast({"type": "partial", "text": event.text})
                    break
        except Exception as exc:
            LOGGER.exception("Partial transcription failed: %s", exc)
            self.error_counts["stt"] += 1

    async def _handle_final(self, audio: bytes) -> None:
        try:
            final_text = ""
            async for event in self.whisper.transcribe(audio):
                if event.type == "partial":
                    await self.broadcast({"type": "partial", "text": event.text})
                elif event.type == "final":
                    final_text = event.text
                    await self.broadcast({"type": "final", "text": final_text})
            if final_text:
                await self._run_llm(final_text)
        except Exception as exc:
            LOGGER.exception("Final transcription failed: %s", exc)
            self.error_counts["stt"] += 1

    async def _run_llm(self, text: str) -> None:
        async with self._llm_lock:
            await self.broadcast({"type": "status", "message": "Generating response..."})
            self.last_response_text = ""
            try:
                buffer = ""
                full_text = ""
                async for token in self.ollama.stream_completion(text):
                    await self.broadcast({"type": "llm", "token": token})
                    full_text += token
                    if self.config.ui.speak_partials:
                        buffer += token
                        if _should_flush(buffer):
                            await self._speak_safe(buffer)
                            buffer = ""
                if self.config.ui.speak_partials and buffer:
                    await self._speak_safe(buffer)
                if not self.config.ui.speak_partials:
                    await self._speak_safe(full_text)
                self.last_response_text = full_text.strip()
            except Exception as exc:
                LOGGER.error("LLM generation failed: %s", exc)
                await self.broadcast({"type": "status", "message": f"LLM error: {exc}"})
                self.error_counts["llm"] += 1

    async def close(self) -> None:
        self._process_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._process_task
        await self.ollama.close()
        await self.audio_player.stop()

    async def _speak_safe(self, text: str) -> None:
        try:
            await self.tts.speak(text)
        except Exception as exc:
            LOGGER.error("TTS failed: %s", exc)
            self.error_counts["tts"] += 1

    async def repeat_last(self) -> None:
        if self.last_response_text:
            await self._speak_safe(self.last_response_text)

    async def submit_text(self, text: str) -> None:
        await self.broadcast({"type": "final", "text": text})
        await self._run_llm(text)


def _should_flush(text: str) -> bool:
    if len(text) > 80:
        return True
    if text.endswith((".", "!", "?", "\n")):
        return True
    return False
