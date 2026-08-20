# this is a collection of vectors and metadata

from __future__ import annotations
import numpy as np
from vectordb.index.flat import FlatIndex

class Collection:
    def __init__(self, name: str, dim: int, metric: str = "cosine"):
        self.name = name
        self.dim = dim
        self.metric = metric
        self._index = FlatIndex(dim=dim, metric=metric)
        self._metadata: dict[str, dict] = {} # id, metadata dict

    def insert(self, ids: list[str], vectors: np.ndarray, metadata: list[dict] | None) -> None:
        if metadata is None:
            metadata = [{} for i in ids]
        if len(metadata)!=len(ids):
            raise ValueError("metadata and ids have to be the same length")

        self._index.add(vectors, ids)
        for _id, meta in zip(ids, metadata):
            self._metadata[_id] = meta

    def delete(self, ids: list[str]) -> None:
        self._index.remove(ids)
        for _id in ids:
            self._metadata.pop(_id, None)

    def search(self, query_vector: np.ndarray, k: int=3, filter: dict | None = None) -> list[dict]:
        # returns a list of {id, score, metadata} dicts with the best match first
        fetchk = k if filter is None else min(len(self._index), max(k*5, 50))
        raw_results = self._index.search(query_vector, fetchk)

        results = []
        for _id, score in raw_results:
            meta = self._metadata.get(_id, {})
            if filter and not all(meta.get(key)==val for key,val in filter.items()):
                continue
            results.append({"id":_id, "score":score, "metadata":meta})
            if len(results)>=k:
                break
        return results

    def __len__(self) -> int:
        return len(self._index)