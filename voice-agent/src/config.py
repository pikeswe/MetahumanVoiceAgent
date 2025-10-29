from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "llama3.1:8b"


@dataclass
class STTConfig:
    model: str = "small.en"
    device: str = "auto"
    vad_aggressiveness: int = 2


@dataclass
class PiperConfig:
    voice_path: str = "./models/piper/en_US-amy-medium.onnx"
    sample_rate: int = 22050


@dataclass
class NeuTTSConfig:
    device: str = "auto"
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None


@dataclass
class AudioConfig:
    output_device_name: str = "VB-CABLE"
    sample_rate: int = 24000


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    log_level: str = "INFO"


@dataclass
class UIConfig:
    speak_partials: bool = False


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts_backend: str = "piper"
    piper: PiperConfig = field(default_factory=PiperConfig)
    neutts: NeuTTSConfig = field(default_factory=NeuTTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    @classmethod
    def load(cls, dotenv_path: str = ".env") -> "AppConfig":
        load_dotenv(dotenv_path, override=False)
        config = cls()
        config.llm.host = os.getenv("OLLAMA_HOST", config.llm.host)
        config.llm.model = os.getenv("OLLAMA_MODEL", config.llm.model)

        config.stt.model = os.getenv("WHISPER_MODEL", config.stt.model)
        config.stt.device = os.getenv("WHISPER_DEVICE", config.stt.device)
        try:
            config.stt.vad_aggressiveness = int(os.getenv("VAD_AGGRESSIVENESS", config.stt.vad_aggressiveness))
        except ValueError:
            LOGGER.warning("Invalid VAD_AGGRESSIVENESS value; defaulting to %s", config.stt.vad_aggressiveness)

        config.tts_backend = os.getenv("TTS_BACKEND", config.tts_backend)
        config.piper.voice_path = os.getenv("PIPER_VOICE_PATH", config.piper.voice_path)
        try:
            config.piper.sample_rate = int(os.getenv("PIPER_SAMPLE_RATE", config.piper.sample_rate))
        except ValueError:
            LOGGER.warning("Invalid PIPER_SAMPLE_RATE; defaulting to %s", config.piper.sample_rate)

        config.neutts.device = os.getenv("NEUTTS_DEVICE", config.neutts.device)
        config.neutts.ref_audio = _empty_to_none(os.getenv("NEUTTS_REF_AUDIO"))
        config.neutts.ref_text = _empty_to_none(os.getenv("NEUTTS_REF_TEXT"))

        config.audio.output_device_name = os.getenv("AUDIO_OUTPUT_DEVICE_NAME", config.audio.output_device_name)
        try:
            config.audio.sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", config.audio.sample_rate))
        except ValueError:
            LOGGER.warning("Invalid AUDIO_SAMPLE_RATE; defaulting to %s", config.audio.sample_rate)

        config.server.host = os.getenv("HOST", config.server.host)
        try:
            config.server.port = int(os.getenv("PORT", config.server.port))
        except ValueError:
            LOGGER.warning("Invalid PORT; defaulting to %s", config.server.port)
        config.server.log_level = os.getenv("LOG_LEVEL", config.server.log_level)

        ui_flag = os.getenv("SPEAK_PARTIALS")
        if ui_flag is not None:
            config.ui.speak_partials = ui_flag.lower() in {"1", "true", "yes", "on"}

        return config

    def to_env(self) -> Dict[str, str]:
        return {
            "OLLAMA_HOST": self.llm.host,
            "OLLAMA_MODEL": self.llm.model,
            "WHISPER_MODEL": self.stt.model,
            "WHISPER_DEVICE": self.stt.device,
            "VAD_AGGRESSIVENESS": str(self.stt.vad_aggressiveness),
            "TTS_BACKEND": self.tts_backend,
            "PIPER_VOICE_PATH": self.piper.voice_path,
            "PIPER_SAMPLE_RATE": str(self.piper.sample_rate),
            "NEUTTS_DEVICE": self.neutts.device,
            "NEUTTS_REF_AUDIO": self.neutts.ref_audio or "",
            "NEUTTS_REF_TEXT": self.neutts.ref_text or "",
            "AUDIO_OUTPUT_DEVICE_NAME": self.audio.output_device_name,
            "AUDIO_SAMPLE_RATE": str(self.audio.sample_rate),
            "HOST": self.server.host,
            "PORT": str(self.server.port),
            "LOG_LEVEL": self.server.log_level,
            "SPEAK_PARTIALS": "true" if self.ui.speak_partials else "false",
        }

    def save(self, path: str = ".env") -> None:
        env_data = self.to_env()
        lines = [f"{key}={value}" for key, value in env_data.items()]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def update_from_payload(self, payload: Dict[str, Any]) -> None:
        if "ollama_model" in payload:
            self.llm.model = str(payload["ollama_model"])
        if "tts_backend" in payload:
            self.tts_backend = str(payload["tts_backend"])
        if "output_device" in payload:
            self.audio.output_device_name = str(payload["output_device"])
        if "speak_partials" in payload:
            self.ui.speak_partials = bool(payload["speak_partials"])
        if "ollama_host" in payload:
            self.llm.host = str(payload["ollama_host"])


def _empty_to_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None
