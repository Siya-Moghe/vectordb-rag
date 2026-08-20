"""Compare IVFIndex against FlatIndex on random data.

Run: python benchmarks/bench_ivf_vs_flat.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from vectordb.index.flat import FlatIndex
from vectordb.index.ivf import IVFIndex


def recall_at_k(flat_results: list[tuple[str, float]], ivf_results: list[tuple[str, float]]) -> float:
    """Fraction of FlatIndex's top-k ids that also appear in IVFIndex's top-k."""
    flat_ids = {_id for _id, _ in flat_results}
    ivf_ids = {_id for _id, _ in ivf_results}
    if not flat_ids:
        return 1.0
    return len(flat_ids & ivf_ids) / len(flat_ids)


def run_benchmark(n_vectors=5000, dim=64, k=10, nlist=50, nprobe_values=(1, 4, 8, 20), n_queries=50):
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(n_vectors, dim)).astype(np.float32)
    ids = [f"vec_{i}" for i in range(n_vectors)]
    queries = rng.normal(size=(n_queries, dim)).astype(np.float32)

    print(f"Dataset: {n_vectors} vectors, dim={dim}, k={k}, nlist={nlist}\n")

    # ground truth
    flat = FlatIndex(dim=dim, metric="cosine")
    flat.build(vectors, ids)

    t0 = time.perf_counter()
    flat_results = [flat.search(q, k) for q in queries]
    flat_time = time.perf_counter() - t0
    print(f"FlatIndex:  {flat_time*1000/n_queries:.3f} ms/query (exact, this is ground truth)\n")

    print(f"{'nprobe':>8} | {'recall@'+str(k):>10} | {'ms/query':>10} | {'speedup':>8}")
    print("-" * 46)

    for nprobe in nprobe_values:
        ivf = IVFIndex(dim=dim, metric="cosine", nlist=nlist, nprobe=nprobe)
        ivf.build(vectors, ids)

        t0 = time.perf_counter()
        ivf_results = [ivf.search(q, k) for q in queries]
        ivf_time = time.perf_counter() - t0

        recalls = [recall_at_k(fr, ir) for fr, ir in zip(flat_results, ivf_results)]
        avg_recall = sum(recalls) / len(recalls)
        speedup = flat_time / ivf_time if ivf_time > 0 else float("inf")

        print(f"{nprobe:>8} | {avg_recall:>10.1%} | {ivf_time*1000/n_queries:>10.3f} | {speedup:>7.2f}x")



if __name__ == "__main__":
    run_benchmark()