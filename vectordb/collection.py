# A named collection of vectors + metadata, backed by one index.


from __future__ import annotations
import numpy as np
from vectordb.index.flat import FlatIndex
from vectordb.index.ivf import IVFIndex


class Collection:
    def __init__(self, name: str, dim: int, metric: str = "cosine", index: str = "flat", **index_kwargs):
        self.name = name
        self.dim = dim
        self.metric = metric
        self.index_type = index

        if index == "flat":
            self._index = FlatIndex(dim=dim, metric=metric)
        elif index == "ivf":
            self._index = IVFIndex(dim=dim, metric=metric, **index_kwargs)
        else:
            raise ValueError(f"Unknown index type '{index}'. Choose 'flat' or 'ivf'.")

        self._built = index == "flat"
        # id -> metadata dict (e.g. {"text": "...", "source": "manual.pdf"})
        self._metadata: dict[str, dict] = {}

    def insert(self, ids: list[str], vectors: np.ndarray, metadata: list[dict] | None = None) -> None:
        if metadata is None:
            metadata = [{} for _ in ids]
        if len(metadata) != len(ids):
            raise ValueError("metadata and ids must be the same length")

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if not self._built:
            # first insert into an IVF collection: train clusters on this batch
            self._index.build(vectors, ids)
            self._built = True
        else:
            self._index.add(vectors, ids)

        for _id, meta in zip(ids, metadata):
            self._metadata[_id] = meta

    def delete(self, ids: list[str]) -> None:
        self._index.remove(ids)
        for _id in ids:
            self._metadata.pop(_id, None)

    def search(self, query_vector: np.ndarray, k: int = 5, filter: dict | None = None) -> list[dict]:
        # returns a list of {"id", "score", "metadata"} dicts, best match first.
        fetch_k = k if filter is None else min(len(self._index), max(k * 5, 50))
        raw_results = self._index.search(query_vector, fetch_k)

        results = []
        for _id, score in raw_results:
            meta = self._metadata.get(_id, {})
            if filter and not all(meta.get(key) == val for key, val in filter.items()):
                continue
            results.append({"id": _id, "score": score, "metadata": meta})
            if len(results) >= k:
                break

        return results

    def __len__(self) -> int:
        return len(self._index)