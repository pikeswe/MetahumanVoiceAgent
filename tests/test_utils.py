from utils import text_chunker, mapping


def test_chunk_text_splits_on_punctuation():
    text = "Hello there! How are you doing today? I'm fine."
    chunks = text_chunker.chunk_text(text, max_words=5)
    assert any("Hello" in c for c in chunks)
    assert any("How are you" in c for c in chunks)


def test_mapping_curves():
    state = mapping.EmotionState(emotion="happy", rate="fast", intensity="excited").normalized()
    payload = state.to_curve_payload()
    assert payload["Emotion_Happy"] == 1.0
    assert payload["Prosody_Rate"] == mapping.RATE_VALUES["fast"]
    assert payload["Prosody_Intensity"] == mapping.INTENSITY_VALUES["excited"]
