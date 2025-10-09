import importlib
import sys
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")


def test_neutts_stream_emits_pcm_frames(monkeypatch):
    chunk_ms = 20
    sample_rate = 16000

    def fake_stream_synthesize(**kwargs):
        assert kwargs["chunk_size"] == int(sample_rate * (chunk_ms / 1000.0))
        yield np.linspace(-1.0, 1.0, kwargs["chunk_size"], dtype=np.float32)
        yield (np.zeros(kwargs["chunk_size"], dtype=np.float32), kwargs["sample_rate"])

    fake_module = SimpleNamespace(stream_synthesize=fake_stream_synthesize)
    monkeypatch.setitem(sys.modules, "neuttsair", fake_module)
    neutts_wrapper = importlib.import_module("rt_tts.neutts_wrapper")
    importlib.reload(neutts_wrapper)
    monkeypatch.setattr(neutts_wrapper, "_adapter", None, raising=False)
    neutts_wrapper.load_adapter(neutts_wrapper.NeuTTSConfig(chunk_ms=chunk_ms))

    frames = list(
        neutts_wrapper.synth_stream(
            "hello",
            sr=sample_rate,
            chunk_ms=chunk_ms,
            emotion="neutral",
            rate="normal",
            intensity="normal",
            lookahead=2,
            lookback=2,
            interpolate=True,
        )
    )
    assert len(frames) == 2
    expected_bytes = int(sample_rate * (chunk_ms / 1000.0) * 2)
    for frame in frames:
        assert isinstance(frame, bytes)
        assert len(frame) == expected_bytes
