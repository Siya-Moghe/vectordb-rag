from __future__ import annotations
import numpy as np

_MODEL = None
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, EMBED_DIM), dtype=np.float32)
    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.astype(np.float32)


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]