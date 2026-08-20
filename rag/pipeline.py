from __future__ import annotations
from vectordb.client import Client
import rag.chunker as chunker 
import rag.embedder as embedder
import rag.generator as generator


class RagPipeline:
    def __init__(self, collection_name: str = "docs"):
        self.db = Client()
        self.collection = self.db.create_collection(
            name=collection_name, dim=embedder.EMBED_DIM, metric="cosine"
        )
        self._next_id = 0

    def ingest_text(self, text: str, source: str, chunk_size: int = 512, overlap: int = 50) -> int:
        chunks = chunker.chunk_text(text, source=source, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return 0

        vectors = embedder.embed([c.text for c in chunks])
        ids = [f"{source}::{c.chunk_index}::{self._next_id + i}" for i, c in enumerate(chunks)]
        self._next_id += len(chunks)

        metadata = [{"text": c.text, "source": c.source, "chunk_index": c.chunk_index} for c in chunks]
        self.collection.insert(ids=ids, vectors=vectors, metadata=metadata)
        return len(chunks)

    def retrieve(self, question: str, k: int = 5, filter: dict | None = None) -> list[dict]:
        query_vector = embedder.embed_one(question)
        return self.collection.search(query_vector, k=k, filter=filter)

    def ask(self, question: str, k: int = 5, filter: dict | None = None) -> dict:
        results = self.retrieve(question, k=k, filter=filter)
        context_chunks = [r["metadata"]["text"] for r in results]
        answer = generator.generate(question, context_chunks)
        return {"answer": answer, "context": results}