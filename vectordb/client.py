# entry point for vectordb
from __future__ import annotations
from vectordb.collection import Collection


class Client:
    def __init__(self, path: str | None = None):
        self.path = path
        self._collections: dict[str, Collection] = {}

    def create_collection(self, name: str, dim: int, metric: str = "cosine") -> Collection:
        if name in self._collections:
            raise ValueError(f"Collection '{name}' already exists")
        col = Collection(name=name, dim=dim, metric=metric)
        self._collections[name] = col
        return col

    def get_collection(self, name: str) -> Collection:
        if name not in self._collections:
            raise KeyError(f"Collection '{name}' does not exist")
        return self._collections[name]

    def list_collections(self) -> list[str]:
        return list(self._collections.keys())