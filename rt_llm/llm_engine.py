"""Streaming LLM interface built on top of llama.cpp binaries."""
from __future__ import annotations

import asyncio
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


from utils import mapping, text_chunker

CHUNK_MAX_WORDS = 15

EMOTION_KEYWORDS = {
    "happy": {"happy", "glad", "joy", "wonderful", "excited", "delight", "pleased"},
    "sad": {"sad", "sorry", "unfortunate", "regret", "down", "gloom"},
    "angry": {"angry", "furious", "annoyed", "mad", "irritated"},
    "surprised": {"surprised", "astonished", "wow", "unexpected", "shocked"},
}

INTENSITY_WORDS = {
    "excited": {"excited", "thrilled", "amazing", "incredible"},
    "calm": {"calm", "steady", "soft", "gentle"},
}

RATE_WORDS = {
    "fast": {"quick", "rapid", "hurry", "fast"},
    "slow": {"slow", "steady", "careful"},
}

DEFAULT_RESPONSE = "Hello! I'm online locally and ready to drive your MetaHuman."


@dataclass
class LLMChunk:
    text: str
    emotion: str
    speaking_rate: str
    intensity: str

    def to_payload(self) -> Dict[str, str]:
        return {
            "type": "speech_chunk",
            "text": self.text.strip(),
            "emotion": self.emotion,
            "speaking_rate": self.speaking_rate,
            "intensity": self.intensity,
        }


class EMAEmotion:
    def __init__(self, alpha: float = 0.6) -> None:
        self.alpha = alpha
        self.state = {emotion: 0.0 for emotion in mapping.EMOTION_ORDER}
        self.state["neutral"] = 1.0

    def update(self, detected: Dict[str, float]) -> str:
        total = sum(detected.values())
        if total <= 0:
            detected = {"neutral": 1.0}
            total = 1.0
        for key in mapping.EMOTION_ORDER:
            prior = self.state.get(key, 0.0)
            incoming = detected.get(key, 0.0) / total
            self.state[key] = (1 - self.alpha) * prior + self.alpha * incoming
        # normalise
        best = max(self.state.items(), key=lambda kv: kv[1])[0]
        return best


class LLMEngine:
    def __init__(
        self,
        model_path: str,
        llama_binary: Optional[str] = None,
        smoothing_alpha: float = 0.55,
    ) -> None:
        self.model_path = Path(model_path)
        self.llama_binary = Path(llama_binary) if llama_binary else self._auto_llama_binary()
        self.smoothing_alpha = smoothing_alpha
        self.ema = EMAEmotion(alpha=smoothing_alpha)

    def _auto_llama_binary(self) -> Optional[Path]:
        candidate_dir = Path(__file__).resolve().parent / "bin" / "llama"
        exe_name = "llama.exe" if os.name == "nt" else "main"
        for root in [candidate_dir, Path.cwd() / "rt_llm" / "bin" / "llama"]:
            executable = root / exe_name
            if executable.exists():
                return executable
        return None

    def _llama_available(self) -> bool:
        return self.llama_binary is not None and self.llama_binary.exists()

    def _launch_process(self, prompt: str) -> subprocess.Popen:
        if not self._llama_available():
            raise FileNotFoundError("llama.cpp binary not found. Run build_llamacpp.ps1 first.")
        args = [
            str(self.llama_binary),
            "-m",
            str(self.model_path),
            "--color",
            "--instruct",
            "--prompt",
            prompt,
        ]
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        return proc

    def _read_stdout_tokens(self, proc: subprocess.Popen) -> Iterator[str]:
        buffer = ""
        while True:
            chunk = proc.stdout.read(1)  # type: ignore[arg-type]
            if not chunk:
                break
            if chunk.isspace():
                if buffer:
                    yield buffer
                    buffer = ""
            else:
                buffer += chunk
        if buffer:
            yield buffer

    def _infer_emotion(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        scores = {emotion: 1e-3 for emotion in mapping.EMOTION_ORDER}
        for emotion, keywords in EMOTION_KEYWORDS.items():
            if any(word in text_lower for word in keywords):
                scores[emotion] += 1.0
        return scores

    def _infer_rate(self, text: str) -> str:
        text_lower = text.lower()
        for rate, keywords in RATE_WORDS.items():
            if any(word in text_lower for word in keywords):
                return rate
        return "normal"

    def _infer_intensity(self, text: str) -> str:
        text_lower = text.lower()
        for intensity, keywords in INTENSITY_WORDS.items():
            if any(word in text_lower for word in keywords):
                return intensity
        return "normal"

    def _chunks_from_text(self, text: str) -> Iterator[LLMChunk]:
        segments = text_chunker.chunk_text(text, max_words=CHUNK_MAX_WORDS)
        for segment in segments:
            detected = self._infer_emotion(segment)
            stable = self.ema.update(detected)
            rate = self._infer_rate(segment)
            intensity = self._infer_intensity(segment)
            yield LLMChunk(segment, stable, rate, intensity)

    def stream_response(self, prompt: str) -> Iterator[Dict[str, str]]:
        if not prompt.strip():
            return iter([])
        self.ema = EMAEmotion(alpha=self.smoothing_alpha)
        if self._llama_available() and self.model_path.exists():
            proc = self._launch_process(prompt)
            try:
                words = []
                for token in self._read_stdout_tokens(proc):
                    words.append(token)
                    if len(words) >= CHUNK_MAX_WORDS or token.endswith(tuple(".!?")):
                        text = " ".join(words)
                        words.clear()
                        for chunk in self._chunks_from_text(text):
                            yield chunk.to_payload()
                if words:
                    text = " ".join(words)
                    for chunk in self._chunks_from_text(text):
                        yield chunk.to_payload()
            finally:
                proc.terminate()
        else:
            # fallback deterministic text
            for chunk in self._chunks_from_text(DEFAULT_RESPONSE):
                yield chunk.to_payload()


async def stream_response_async(engine: LLMEngine, prompt: str) -> Iterator[Dict[str, str]]:
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop_sentinel = object()

    def worker() -> None:
        for payload in engine.stream_response(prompt):
            asyncio.run_coroutine_threadsafe(queue.put(payload), loop)
        asyncio.run_coroutine_threadsafe(queue.put(stop_sentinel), loop)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        item = await queue.get()
        if item is stop_sentinel:
            break
        yield item  # type: ignore[misc]
