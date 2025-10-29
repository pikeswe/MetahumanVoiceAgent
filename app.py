import argparse
import json
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Optional

import httpx
import librosa
import numpy as np
import sounddevice as sd
import yaml
from faster_whisper import WhisperModel

try:
    from neuttsair.neutts import NeuTTSAir
except ImportError as exc:  # pragma: no cover - environment specific
    raise SystemExit(
        "NeuTTS Air is not installed. Install it with 'pip install -e .' from the cloned repo."
    ) from exc


LOGGER = logging.getLogger("voice_agent")
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"
OLLAMA_URL = "http://localhost:11434"
STREAM_TIMEOUT = 120.0
SENTENCE_BREAK_CHARS = {".", "?", "!"}
CHUNK_TIMEOUT_SECONDS = 1.5
TARGET_SAMPLE_RATE = 24000
INPUT_SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 1e-3
SILENCE_DURATION_SECONDS = 0.8


@dataclass
class RuntimeConfig:
    ollama_model: str
    ref_audio: Optional[Path]
    ref_text: Optional[Path]
    tts_backbone_repo: str
    tts_backbone_device: str
    tts_codec_repo: str
    tts_codec_device: str


def load_config(path: Path) -> tuple[Dict[str, object], Path]:
    if not path.exists():
        raise SystemExit(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data, path.parent.resolve()


def build_runtime_config(
    cli_args: argparse.Namespace, config: Dict[str, object], base_dir: Path
) -> RuntimeConfig:
    ollama_model = cli_args.model or config.get("ollama_model")

    def resolve_path(value: Optional[str], *, cli_override: bool) -> Optional[Path]:
        if not value:
            return None
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        if cli_override:
            return (Path.cwd() / candidate).resolve()
        return (base_dir / candidate).resolve()

    ref_audio = (
        resolve_path(cli_args.ref_audio, cli_override=True)
        if cli_args.ref_audio
        else resolve_path(config.get("ref_audio"), cli_override=False)
    )
    ref_text = (
        resolve_path(cli_args.ref_text, cli_override=True)
        if cli_args.ref_text
        else resolve_path(config.get("ref_text"), cli_override=False)
    )
    tts_conf = config.get("tts", {})
    return RuntimeConfig(
        ollama_model=ollama_model,
        ref_audio=ref_audio,
        ref_text=ref_text,
        tts_backbone_repo=tts_conf.get("backbone_repo", "neuphonic/neutts-air"),
        tts_backbone_device=tts_conf.get("backbone_device", "cpu"),
        tts_codec_repo=tts_conf.get("codec_repo", "neuphonic/neucodec"),
        tts_codec_device=tts_conf.get("codec_device", "cpu"),
    )


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=httpx.Timeout(None, connect=10.0))

    def check_connection(self) -> None:
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
        except httpx.HTTPError as exc:
            raise SystemExit(
                "Unable to reach Ollama at http://localhost:11434. Is the service running?"
            ) from exc
        if response.status_code >= 400:
            raise SystemExit(
                f"Ollama returned HTTP {response.status_code}. Ensure the model is available."
            )

    def stream_generate(self, model: str, prompt: str) -> Generator[str, None, None]:
        payload = {"model": model, "prompt": prompt, "stream": True}
        try:
            with self._client.stream(
                "POST", f"{self.base_url}/api/generate", json=payload, timeout=STREAM_TIMEOUT
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="ignore")
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        LOGGER.debug("Discarding malformed chunk: %s", data)
                        continue
                    if payload.get("done"):
                        break
                    token = payload.get("response", "")
                    if token:
                        yield token
        except httpx.HTTPStatusError as exc:
            raise SystemExit(
                f"Ollama request failed with status {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise SystemExit("Lost connection to Ollama while streaming response.") from exc


class SentenceChunker:
    def __init__(self, timeout_seconds: float = CHUNK_TIMEOUT_SECONDS) -> None:
        self.timeout_seconds = timeout_seconds
        self.buffer: list[str] = []
        self.last_emit = time.time()

    def feed(self, token: str) -> Optional[str]:
        self.buffer.append(token)
        now = time.time()
        text = "".join(self.buffer)
        if any(token.strip().endswith(ch) for ch in SENTENCE_BREAK_CHARS) or (
            now - self.last_emit
        ) >= self.timeout_seconds:
            chunk = text.strip()
            if chunk:
                self.buffer.clear()
                self.last_emit = now
                return chunk
            self.buffer.clear()
            self.last_emit = now
        return None

    def flush(self) -> Optional[str]:
        if not self.buffer:
            return None
        chunk = "".join(self.buffer).strip()
        self.buffer.clear()
        return chunk or None


class AudioPlayer:
    def __init__(self, target_device_fragment: str = "CABLE Input") -> None:
        self.sample_rate = TARGET_SAMPLE_RATE
        self.stream = self._create_stream(target_device_fragment)
        self.stream.start()

    def _create_stream(self, name_fragment: str) -> sd.OutputStream:
        try:
            devices = sd.query_devices()
        except Exception as exc:
            raise SystemExit(f"Unable to query audio devices: {exc}") from exc
        chosen_index = None
        for index, device in enumerate(devices):
            if name_fragment.lower() in device["name"].lower() and device.get("max_output_channels", 0) > 0:
                chosen_index = index
                break
        if chosen_index is None:
            device_list = [device["name"] for device in devices if device.get("max_output_channels", 0) > 0]
            joined = "\n".join(device_list)
            raise SystemExit(
                "VB-CABLE output device not found. Available output devices:\n" + joined
            )
        LOGGER.info("Using audio output device #%d (%s)", chosen_index, devices[chosen_index]["name"])
        try:
            return sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                device=chosen_index,
                dtype="float32",
                blocksize=0,
            )
        except Exception as exc:
            raise SystemExit(f"Failed to open VB-CABLE output: {exc}") from exc

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        if audio is None or len(audio) == 0:
            return
        data = audio
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sample_rate != self.sample_rate:
            data = librosa.resample(data.astype(np.float32), orig_sr=sample_rate, target_sr=self.sample_rate)
            sample_rate = self.sample_rate
        data = np.asarray(data, dtype=np.float32)
        frames = data.reshape(-1, 1)
        self.stream.write(frames)

    def close(self) -> None:
        try:
            self.stream.stop()
        finally:
            self.stream.close()


class NeuTTSAirWrapper:
    def __init__(self, config: RuntimeConfig) -> None:
        self.model = NeuTTSAir(
            backbone_repo=config.tts_backbone_repo,
            codec_repo=config.tts_codec_repo,
            backbone_device=config.tts_backbone_device,
            codec_device=config.tts_codec_device,
        )
        if config.ref_audio is None or config.ref_text is None:
            raise SystemExit("Reference audio/text paths are not configured. Use config.yaml or CLI overrides.")
        self.ref_audio = config.ref_audio.resolve()
        self.ref_text_path = config.ref_text.resolve()
        self.ref_codes, self.ref_text = self._load_reference()
        self._download_notice_shown = False
        self._watermarker = getattr(self.model, "watermarker", None)
        if self._watermarker is None:
            LOGGER.info("NeuTTS Air watermarking disabled (module not present).")

    def _load_reference(self) -> tuple[object, str]:
        if self.ref_audio is None or not self.ref_audio.exists():
            raise SystemExit(f"Reference audio not found: {self.ref_audio}")
        if self.ref_text_path is None or not self.ref_text_path.exists():
            raise SystemExit(f"Reference text not found: {self.ref_text_path}")
        result = self.model.encode_reference(str(self.ref_audio), str(self.ref_text_path))
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1]
        with self.ref_text_path.open("r", encoding="utf-8") as handle:
            text = handle.read().strip()
        return result, text

    def infer(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        if not text:
            return None
        if not self._download_notice_shown:
            LOGGER.info("Downloading models once…")
            self._download_notice_shown = True
        try:
            audio = self.model.infer(text, self.ref_codes, self.ref_text)
        except Exception as exc:  # pragma: no cover - runtime specific failures
            message = str(exc).lower()
            if "phonemizer" in message or "espeak" in message:
                LOGGER.error(
                    "Phonemizer failed for chunk. Check PHONEMIZER_ESPEAK_LIBRARY and PHONEMIZER_ESPEAK_PATH."
                )
                return None
            raise
        sample_rate = TARGET_SAMPLE_RATE
        if isinstance(audio, tuple):
            data, maybe_sr = audio
            if isinstance(maybe_sr, int):
                sample_rate = maybe_sr
            else:
                data = audio[0]
            audio = data
        return np.asarray(audio), sample_rate


class TranscriptionLoop:
    def __init__(self, callback):
        self.callback = callback
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stop = threading.Event()

    def start_stream(self) -> sd.InputStream:
        return sd.InputStream(
            samplerate=INPUT_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )

    def _audio_callback(self, indata, frames, time_info, status):  # pragma: no cover - live audio
        if status:
            LOGGER.warning("Input stream status: %s", status)
        if self._stop.is_set():
            raise sd.CallbackStop
        self.queue.put(indata.copy())

    def run(self, model: WhisperModel) -> None:
        buffer: list[np.ndarray] = []
        silence_duration = 0.0
        speaking = False
        while not self._stop.is_set():
            try:
                data = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            energy = float(np.sqrt(np.mean(np.square(data))))
            frames = data.shape[0]
            block_duration = frames / INPUT_SAMPLE_RATE
            if energy > SILENCE_THRESHOLD:
                buffer.append(data)
                speaking = True
                silence_duration = 0.0
                continue
            if speaking:
                buffer.append(data)
                silence_duration += block_duration
                if silence_duration >= SILENCE_DURATION_SECONDS:
                    audio_chunk = np.concatenate(buffer, axis=0) if buffer else None
                    buffer.clear()
                    speaking = False
                    silence_duration = 0.0
                    if audio_chunk is not None:
                        text = self._transcribe(model, audio_chunk)
                        if text:
                            self.callback(text)
            else:
                buffer.clear()
                silence_duration = 0.0

        if buffer and speaking:
            audio_chunk = np.concatenate(buffer, axis=0)
            buffer.clear()
            text = self._transcribe(model, audio_chunk)
            if text:
                self.callback(text)

    def stop(self) -> None:
        self._stop.set()

    def _transcribe(self, model: WhisperModel, audio: np.ndarray) -> str:
        segments, _ = model.transcribe(
            audio.flatten(),
            language="en",
            beam_size=1,
            temperature=0.0,
            vad_filter=False,
        )
        texts = [segment.text.strip() for segment in segments if segment.text]
        transcription = " ".join(texts).strip()
        if transcription:
            LOGGER.info("User: %s", transcription)
        return transcription


def speak_stream(
    ollama: OllamaClient,
    tts: NeuTTSAirWrapper,
    player: AudioPlayer,
    model_name: str,
    prompt: str,
) -> str:
    chunker = SentenceChunker()
    full_response: list[str] = []
    for token in ollama.stream_generate(model_name, prompt):
        full_response.append(token)
        chunk = chunker.feed(token)
        if chunk:
            audio = tts.infer(chunk)
            if audio:
                player.play(*audio)
    leftover = chunker.flush()
    if leftover:
        audio = tts.infer(leftover)
        if audio:
            player.play(*audio)
    response_text = "".join(full_response).strip()
    if response_text:
        LOGGER.info("Assistant: %s", response_text)
    return response_text


def run_chat_mode(args: argparse.Namespace, config: RuntimeConfig) -> None:
    prompt = args.prompt
    if not prompt:
        LOGGER.info("Reading prompt from stdin (Ctrl+D to submit)...")
        prompt = sys.stdin.read().strip()
    if not prompt:
        LOGGER.warning("Empty prompt provided; nothing to do.")
        return
    ollama = OllamaClient()
    ollama.check_connection()
    tts = NeuTTSAirWrapper(config)
    player = AudioPlayer()
    try:
        speak_stream(ollama, tts, player, config.ollama_model, prompt)
    finally:
        player.close()


def run_listen_mode(args: argparse.Namespace, config: RuntimeConfig) -> None:
    ollama = OllamaClient()
    ollama.check_connection()
    tts = NeuTTSAirWrapper(config)
    player = AudioPlayer()
    whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

    def on_transcription(text: str) -> None:
        speak_stream(ollama, tts, player, config.ollama_model, text)

    loop = TranscriptionLoop(on_transcription)
    with loop.start_stream():
        listener_thread = threading.Thread(target=loop.run, args=(whisper_model,), daemon=True)
        listener_thread.start()
        LOGGER.info("Listening... Press Ctrl+C to stop.")
        try:
            while listener_thread.is_alive():
                listener_thread.join(timeout=0.2)
        except KeyboardInterrupt:
            LOGGER.info("Stopping listen mode...")
            loop.stop()
            listener_thread.join()
    player.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local voice agent powered by Ollama and NeuTTS Air")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument("--model", help="Override Ollama model name")
    parser.add_argument("--ref-audio", help="Path to reference audio for NeuTTS Air")
    parser.add_argument("--ref-text", help="Path to reference text for NeuTTS Air")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat_parser = subparsers.add_parser("chat", help="Send a text prompt to the LLM and speak the reply")
    chat_parser.add_argument("--prompt", help="Prompt text to send to the LLM")

    subparsers.add_parser("listen", help="Listen to the microphone and reply via voice")

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = parse_args()
    config_data, config_base = load_config(args.config)
    runtime_config = build_runtime_config(args, config_data, config_base)

    if not runtime_config.ollama_model:
        raise SystemExit("Ollama model not specified. Use config.yaml or --model.")
    if runtime_config.ref_audio is None or runtime_config.ref_text is None:
        raise SystemExit("Reference audio/text not specified. Update config.yaml or pass --ref-audio/--ref-text.")

    if args.command == "chat":
        run_chat_mode(args, runtime_config)
    elif args.command == "listen":
        run_listen_mode(args, runtime_config)
    else:  # pragma: no cover - argparse should prevent this
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
