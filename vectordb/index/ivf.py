# IVF clusters all vectors into nlist gorups via k means clustering
# at search time, it finds nprobe cluster centroids closest to the query, and only brut force searches inside those clusters
# each cluster is just a FlatIndex over its own subset of vectors

from __future__ import annotations
import numpy as np
from vectordb.distance import METRICS, metric_is_similarity
from vectordb.index.flat import FlatIndex
from vectordb.index.kmeans import train_kmeans

class IVFIndex:
    def __init__(self, dim: int, metric: str="cosine", nlist: int=100, nprobe: int=5):
        if metric not in METRICS:
            raise ValueError("metric not found")
        self.dim = dim
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self._scorer = METRICS[metric]
        self._higher_is_better = metric_is_similarity(metric)
        self._centroids: np.ndarray = np.empty((0,dim),dtype=np.float32)
        self._clusters: list[FlatIndex] = []  #each cluster is a flatindex
        self._id_to_cluster: dict[str, int] = {}

    def build(self, vectors: np.ndarray, ids: list[str]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)

        if len(vectors)==0:
            raise ValueError("there are zero vectors, can't build IVF")

        self._centroids, assignments = train_kmeans(vectors, n_clusters=self.nlist)
        n_clusters = len(self._centroids)

        self._clusters = [FlatIndex(dim = self.dim, metric = self.metric) for i in range(n_clusters)]
        self._id_to_cluster = {}

        for cluster_idx in range(n_clusters):
            rows = np.where(assignments == cluster_idx)
            """
            say assignments is [2,0,2,1,0,2] that means, vector 0 is cluster 2, vector 1 is in cluster 0 and so on
            so if assignments is equal to cluster index (say 2), we get an answer like [T,F,T,F,F,T]
            and then the np.where gives the positions where its true so rows = [0,2,5] which means vecotrs at rows 0,2,5 belong to cluster 2
            the we just do [0] to extract the actual array
            """
            if len(rows) == 0: 
                continue

            cluster_vectors = vectors[rows]
            cluster_ids = [ids[i] for i in rows]
            self._clusters[cluster_idx].build(cluster_vectors, cluster_ids)   #this does build for flat index, so builds a flat index per cluster

            for _id in cluster_ids:
                self._id_to_cluster[_id] = cluster_idx

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        # assign new vectors to one of the existing centroids
        if len(self._centroids) == 0:
            raise RuntimeError("adding elements before the ivf index was created")

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors.reshape(1,-1)    #[1,2,3].reshape(1,-1) becomes [[1,2,3]]

        for i, _id in enumerate(ids):
            vector = vectors[i]
            nearest_cluster = self._nearest_centroids(vector, n=1)[0]
            self._clusters[nearest_cluster].add(vector.reshape(1,-1),[_id])
            self._id_to_cluster[_id] = nearest_cluster

    def _nearest_centroids(self, vector: np.ndarray, n: int) -> np.ndarray:
        scores = self._scorer(vector, self._centroids)
        n = min(n, len(self._centroids))
        if self._higher_is_better:
            order = np.argsort(-scores)
        else:
            order = np.argsort(scores)
        return order[:n]

    def remove(self, ids: list[str]) -> None:
        by_cluster: dict[int, list[str]] = {}      #this will be like, cluster 0 : ["A","F"] so remove a and f

        for _id in ids:
            cluster_idx = self._id_to_cluster.get(_id)
            if cluster_idx is not None:
                by_cluster.setdefault(cluster_idx,[]).append(_id)

            for cluster_idx, cluster_ids in by_cluster.items():
                self._clusters[cluster_idx].remove[cluster_ids]
                for _id in cluster_ids:
                    del self._id_to_cluster[_id]

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if len(self._centroids) == 0:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        nprobe = min(self.nprobe, len(self._centroids))
        probe_clusters = self._nearest_centroids(query, n=nprobe)

        candidates: list[tuple[str,float]] = []
        for cluster_idx in probe_clusters:
            candidates.extend(self._clusters[cluster_idx].search(query,k))
            candidates.sort(key = lambda pair: pair[1], reverse = self._higher_is_better)
        return candidates[:k]

    def __len__(self) -> int:
        return len(self._id_to_cluster)