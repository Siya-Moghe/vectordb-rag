# vectordb-rag

A vector database built from scratch, with a RAG pipeline on top using
Gemini (free tier) for generation.

See `EXPLANATIONS.md` for a walkthrough of what's implemented and why.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

export GEMINI_API_KEY="your-key"   # free tier: https://aistudio.google.com/apikey
```

## Run the demo

```bash
python demo.py
```

This ingests `data/corpus/sample.txt`, embeds it locally, and answers a
question by retrieving relevant chunks and calling Gemini.

## Use it yourself

```python
from rag import RagPipeline

pipeline = RagPipeline()
pipeline.ingest_text(open("my_doc.txt").read(), source="my_doc.txt")

result = pipeline.ask("What does this document say about X?")
print(result["answer"])
print(result["context"])  # the chunks the answer was grounded in
```

## Status

- [x] Flat (brute-force) vector index
- [x] In-memory Client / Collection / CRUD API
- [x] Chunking + local embedding (sentence-transformers)
- [x] RAG pipeline with Gemini generation
- [ ] Persistence (WAL + snapshot)
- [ ] IVF index
- [ ] Benchmark harness
- [ ] Evaluation harness (recall@k, MRR, faithfulness)