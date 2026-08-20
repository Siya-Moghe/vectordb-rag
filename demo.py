from pathlib import Path
from rag.pipeline import RagPipeline


def main():
    pipeline = RagPipeline()

    corpus_dir = Path("data/corpus")
    for path in corpus_dir.glob("*.txt"):
        text = path.read_text()
        n_chunks = pipeline.ingest_text(text, source=path.name)
        print(f"Ingested {path.name}: {n_chunks} chunks")

    print(f"\nTotal chunks in collection: {len(pipeline.collection)}")

    question = "What is this project about?"
    result = pipeline.ask(question, k=3)

    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    print("\nRetrieved context:")
    for r in result["context"]:
        print(f"  [{r['score']:.3f}] {r['metadata']['source']} chunk#{r['metadata']['chunk_index']}")


if __name__ == "__main__":
    main()