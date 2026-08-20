from __future__ import annotations
import numpy as np
from vectordb.distance import METRICS, metric_is_similarity

class FlatIndex:
    def __init__(self, dim: int, metric: str="cosine"):
        if metric not in METRICS:
            raise ValueError("Unknown metric")
        self.dim = dim
        self.metric = metric
        self._scorer = METRICS[metric]
        self._higher_is_better = metric_is_similarity(metric)
        self._ids: list[str] = []
        self._vectors: np.ndarray = np.empty((0,dim),dtype=np.float32)
        self._id_to_row: dict[str, int] = {}

    def build(self, vectors: np.ndarray, ids: list[str]) -> None:
        self._ids = []
        self._id_to_row = {}
        self._vectors = np.empty((0,self.dim),dtyoe=np.float32)
        self.add(vectors, ids)

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1,-1)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        if len(ids) != vectors.shape[0]:
            raise ValueError(f"ids and vectors must be same length")

        for i, _id in enumerate(ids):
            if _id in self._id_to_row:
                row = self._id_to_row[_id]
                self._vectors[row] = vectors[i]
            else:
                self._id_to_row[_id] = len(self._ids)
                self._ids.append(_id)
                self._vectors = np.vstack([self._vectors, vectors[i]])

    def remove(self, ids: list[str]) -> None:
        remove_set = set(ids)
        keep_rows = [i for i,_id in enumerate(self._ids) if _id not in remove_set]
        self._ids = [self.ids[i] for i in keep_rows]
        self._vectors = self._vectors[keep_rows] if keep_rows else np.empty((0,self.dim),dtype=np.float32)
        self._id_to_row = {_id: i for i,_id in enumerate(self._ids)}

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if len(self._ids) == 0:
            return []

        query = np.asarray(query, dtype = np.float32).reshape(-1)
        scores = self._scorer(query, self._vectors)

        k = min(k, len(self._ids))
        if self._higher_is_better:
            topk = np.argpartition(-scores, k-1)[:k]
        else:
            topk = np.argpartition(scores, k-1)[:k]
        topk = topk[np.argsort(scores[topk] if self._higher_is_better else scores[topk])]

        if self._higher_is_better:
            topk = topk[::-1]
        return [(self._ids[i], float(scores[i])) for i in topk]

    def __len__(self) -> int:
        return len(self._ids)