"""Mappings between LLM semantics, TTS params, and UE curves."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

EMOTION_ORDER = ["neutral", "happy", "sad", "angry", "surprised"]
RATE_VALUES = {"slow": 0.3, "normal": 0.6, "fast": 0.9}
INTENSITY_VALUES = {"calm": 0.3, "normal": 0.6, "excited": 0.9}
RATE_TIME_STRETCH = {"slow": 0.92, "normal": 1.0, "fast": 1.08}
INTENSITY_GAIN_DB = {"calm": -2.0, "normal": 0.0, "excited": 2.0}


@dataclass
class EmotionState:
    emotion: str = "neutral"
    rate: str = "normal"
    intensity: str = "normal"

    def to_curve_payload(self) -> Dict[str, float]:
        payload = {f"Emotion_{name.capitalize()}": 0.0 for name in EMOTION_ORDER}
        payload[f"Emotion_{self.emotion.capitalize()}"] = 1.0
        payload["Prosody_Rate"] = RATE_VALUES.get(self.rate, 0.6)
        payload["Prosody_Intensity"] = INTENSITY_VALUES.get(self.intensity, 0.6)
        return payload

    def normalized(self) -> "EmotionState":
        if self.emotion not in EMOTION_ORDER:
            self.emotion = "neutral"
        if self.rate not in RATE_VALUES:
            self.rate = "normal"
        if self.intensity not in INTENSITY_VALUES:
            self.intensity = "normal"
        return self


def emotion_to_curves(emotion: str) -> Dict[str, float]:
    result = {name: 0.0 for name in EMOTION_ORDER}
    if emotion in result:
        result[emotion] = 1.0
    else:
        result["neutral"] = 1.0
    return result


def prosody_scalars(rate: str, intensity: str) -> Tuple[float, float]:
    return RATE_VALUES.get(rate, 0.6), INTENSITY_VALUES.get(intensity, 0.6)


def tts_controls(rate: str, intensity: str) -> Tuple[float, float]:
    stretch = RATE_TIME_STRETCH.get(rate, 1.0)
    gain = INTENSITY_GAIN_DB.get(intensity, 0.0)
    return stretch, gain
