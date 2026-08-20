# interface for index types

from __future__ import annotations
from typing import Protocol          #keeping it as a protocol means that the client doesn't need to know what index they are talking to
import numpy as np

class VectorIndex(Protocol):
    def build(self, vectors: np.ndarray, ids: list[str]) -> None:
        ...               # the three dots just means this is a signature.. like interfaces in java

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        ...

    def remove(self, vectors: np.ndarray, ids: list[str]) -> None:
        ...

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        ...