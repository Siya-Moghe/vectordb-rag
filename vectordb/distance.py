from __future__ import annotations
import numpy as np

def cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # cosine similarity
    query = query.astype(np.float32)
    matrix = matrix.astype(np.float32)

    query_norm = np.linalg.norm(query)
    matrix_norm = np.linalg.norm(matrix, axis=1)

    denominator = matrix_norm*query_norm
    denominator[denominator==0] = 1e-10 #if the value is zero, replace it with a near zero value --> to prevent division by zero
    numerator = matrix @ query
    return numerator/denominator


def dot(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # dot product
    query = query.astype(np.float32)
    matrix = matrix.astype(np.float32)
    return matrix @ query

def euclidean(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # euclidean distance
    query = query.astype(np.float32)
    matrix = matrix.astype(np.float32)
    diff = matrix - query
    return np.linalg.norm(diff, axis=1)

METRICS = { "cosine": cosine, "dot": dot, "euclidean": euclidean }

def metric_is_similarity(metric: str) -> bool:
    return metric in ("cosine", "dot")