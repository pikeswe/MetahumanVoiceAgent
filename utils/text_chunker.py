"""Punctuation-aware chunking utilities."""
from __future__ import annotations

import re
from typing import Iterable, Iterator, List

_SENTENCE_END = re.compile(r"([.!?]+\s+)")
_WORD_BOUNDARY = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def chunk_text(text: str, max_words: int = 15) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    sentences = _SENTENCE_END.split(text)
    chunks: List[str] = []
    buffer: List[str] = []
    word_count = 0
    for segment in sentences:
        if not segment:
            continue
        words = segment.split()
        for word in words:
            buffer.append(word)
            word_count += 1
            if word.endswith(tuple(".!?")) or word_count >= max_words:
                chunks.append(" ".join(buffer))
                buffer.clear()
                word_count = 0
        if buffer and segment.endswith(tuple(".!?")):
            chunks.append(" ".join(buffer))
            buffer.clear()
            word_count = 0
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks


def rolling_chunks(tokens: Iterable[str], max_words: int = 15) -> Iterator[str]:
    buffer: List[str] = []
    for token in tokens:
        buffer.append(token)
        if len(buffer) >= max_words or token.endswith(tuple(".!?")):
            yield " ".join(buffer)
            buffer.clear()
    if buffer:
        yield " ".join(buffer)
