from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


def chunk_text(text: str, source: str, chunk_size: int = 512, overlap: int = 50) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    idx = 0
    step = chunk_size - overlap

    while start < len(words):
        window = words[start:start + chunk_size]
        chunks.append(Chunk(text=" ".join(window), source=source, chunk_index=idx))
        idx += 1
        start += step

    return chunks