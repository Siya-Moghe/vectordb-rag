"""Top-level entry point for vectordb.

Usage:
    from vectordb import Client

    db = Client()
    col = db.create_collection("docs", dim=384, metric="cosine")
    col.insert(ids=["a"], vectors=embeddings, metadata=[{"text": "..."}])
    results = col.search(query_vector, k=5)

    # or, for the approximate/faster index at scale:
    col = db.create_collection("docs", dim=384, index="ivf", nlist=100, nprobe=8)
"""

from __future__ import annotations
from vectordb.collection import Collection

class Client:
    def __init__(self, path: str | None = None):
        # `path` is accepted now so the API shape won't need to change once
        # persistence (step 4) lands -- it's unused until then.
        self.path = path
        self._collections: dict[str, Collection] = {}

    def create_collection(self, name: str, dim: int, metric: str = "cosine", index: str = "flat", **index_kwargs) -> Collection:
        if name in self._collections:
            raise ValueError(f"Collection '{name}' already exists")
        col = Collection(name=name, dim=dim, metric=metric, index=index, **index_kwargs)
        self._collections[name] = col
        return col

    def get_collection(self, name: str) -> Collection:
        if name not in self._collections:
            raise KeyError(f"Collection '{name}' does not exist")
        return self._collections[name]

    def list_collections(self) -> list[str]:
        return list(self._collections.keys())