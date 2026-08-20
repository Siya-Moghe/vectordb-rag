from __future__ import annotations
import numpy as np
from scipy.cluster.vq import kmeans2

def train_kmeans(vectors: np.ndarray, n_clusters: int, seed: int=0) -> tuple[np.ndarray, np.ndarray]:
    #returns centroids : (n_clusters,dim) and assignments where assignments[i] is which cluster vector i belongs to 
    n_clusters = min(n_clusters, len(vectors))
    #first do random initialization
    centroids, assignments = kmeans2(vectors.astype(np.float64), n_clusters, minit="++", seed=seed)
    centroids, assignments = _fix_empty_clusters(vectors, centroids, assignments)
    return centroids.astype(np.float32), assignments

def _fix_empty_clusters(vectors: np.ndarray, centroids: np.ndarray, assignments: np.ndarray):
    #kmeans2 can leave a centroid with zero point assigned to it and this would make a cluster useless
    #so we detect this and re-seed each empty centroid on a random data point
    n_clusters = len(centroids)
    counts = np.bincount(assignments, minlength=n_clusters)       #gives list where result[i] = number of times i appears in the input
    empty = np.where(counts==0)[0]

    if len(empty)==0:
        return centroids, assignments

    rng = np.random.default_rng(0)
    replacement_points = rng.choice(len(vectors), size=len(empty), replace=False)
    centroids[empty] = vectors[replacement_points]

    dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)  
    # vectors becomes (N,1,D) and centroids becomes (1,K,D)
    # broadcasting like this makes them (N, K, D) essentially, so we get it as vector 0 - centroid 1, vector 0 - centroidd2 and so on
    # now, we calculate the vector norm across the feature dimension (Axis = 2)
    # so, the shape changes from (N,K,D) --> (N,K) and we will get the distance like
    # [ [distance(vector0, centroid0), distance(vector0, centroid1)],
    # [distance(vector1, centroid0), distance(vector1, centroid1)],
    # [distance(vector2, centroid0), distance(vector2, centroid1)] ]
    # assignments takes min across columns (axis=1) and so we get the centroid at minimum distance
    assignments = np.argmin(dists, axis=1)
    return centroids, assignments