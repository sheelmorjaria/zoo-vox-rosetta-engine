#!/usr/bin/env python3
"""
Cluster Benchmark Suite for Zoo Vox Rosetta Engine

Compares clustering algorithms on:
1. Zoo Vox metrics (SVS, LRN depth, vocabulary utilization)
2. Graded continuity metrics (neighborhood consistency, noise rate)
3. Computational metrics (RAM, time)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
"""

import json
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional imports with graceful fallback
try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, pairwise_distances
    from sklearn.mixture import BayesianGaussianMixture
    from sklearn.neighbors import NearestNeighbors

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import hdbscan
    import umap

    UMAP_HDBSCAN_AVAILABLE = True
except ImportError:
    UMAP_HDBSCAN_AVAILABLE = False

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import igraph as ig
    import leidenalg

    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False

try:
    import hnswlib

    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False

# Check if sklearn has HDBSCAN (added in 1.3+)
try:
    from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

    SKLEARN_HDBSCAN_AVAILABLE = True
except ImportError:
    SKLEARN_HDBSCAN_AVAILABLE = False

# Local imports
import sys

from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional NgramCorpusStats import
try:
    from analysis.rosetta_stone.ngram_stats import NgramCorpusStats

    NGRAM_STATS_AVAILABLE = True
except ImportError:
    NGRAM_STATS_AVAILABLE = False
    print("Warning: NgramCorpusStats not available, using placeholder metrics")


@dataclass
class ClusterResult:
    """Result from a single clustering method."""

    method_name: str
    labels: np.ndarray  # Hard cluster assignments
    soft_labels: Optional[np.ndarray] = None  # Probabilities (n_samples, n_clusters)
    fit_time: float = 0.0
    peak_ram_mb: float = 0.0


@dataclass
class BenchmarkMetrics:
    """Metrics for a single clustering method."""

    method_name: str

    # Computational metrics
    fit_time_seconds: float
    peak_ram_mb: float

    # Cluster statistics
    n_clusters: int
    noise_rate: float  # Fraction of noise points (-1 labels)

    # Zoo Vox metrics
    shared_vocabulary_score: float  # SVS
    lrn_depth: int  # Longest repeated N-gram
    vocabulary_utilization: float  # Unique clusters / Total clusters

    # Graded continuity metrics
    neighborhood_consistency: float  # ARI on k-NN
    avg_neighbor_kl_divergence: float  # For soft clustering
    noise_classification_rate: float  # % of data labeled as noise


class ClusterBenchmarkSuite:
    """Benchmark clustering algorithms for graded phrase mining."""

    def __init__(self, n_neighbors: int = 30, graded_threshold: float = 0.3, svs_window: int = 3):
        self.n_neighbors = n_neighbors
        self.graded_threshold = graded_threshold
        self.svs_window = svs_window

    def run(
        self, features_112d: np.ndarray, sequences: List[List[int]], methods: List[str] = None
    ) -> Dict[str, BenchmarkMetrics]:
        """Run benchmark on specified clustering methods.

        Args:
            features_112d: 112D feature vectors (n_samples, 112)
            sequences: List of cluster sequences for N-gram analysis
            methods: List of methods to run. Options: "kmeans", "umap_hdbscan", "bgmm", "faiss_leiden"

        Returns:
            Dictionary mapping method name to BenchmarkMetrics
        """
        if methods is None:
            methods = ["kmeans", "umap_hdbscan", "bgmm"]

        print("╔═══════════════════════════════════════════════════════════════════════════╗")
        print("║     Cluster Benchmark Suite - Zoo Vox Rosetta Engine                     ║")
        print("╚═══════════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"Dataset: {features_112d.shape[0]:,} samples × {features_112d.shape[1]}D")
        print(f"Sequences: {len(sequences):,}")
        print(f"Methods: {', '.join(methods)}")
        print()

        results = {}

        for method in methods:
            print(f"{'=' * 70}")
            print(f"Method: {method}")
            print(f"{'=' * 70}")

            try:
                # Fit clustering
                cluster_result = self._fit_method(method, features_112d)

                # Compute metrics
                metrics = self._compute_metrics(method, cluster_result, features_112d, sequences)

                results[method] = metrics

                # Print summary
                self._print_metrics(metrics)

            except Exception as e:
                print(f"✗ Error running {method}: {e}")
                import traceback

                traceback.print_exc()

        # Compare methods
        if len(results) > 1:
            self._compare_methods(results)

        return results

    def _fit_method(self, method: str, features: np.ndarray) -> ClusterResult:
        """Fit a specific clustering method."""
        tracemalloc.start()
        start_time = time.time()

        if method == "kmeans":
            labels, soft_labels = self._fit_kmeans(features)
        elif method == "umap_hdbscan":
            labels, soft_labels = self._fit_umap_hdbscan(features)
        elif method == "bgmm":
            labels, soft_labels = self._fit_bgmm(features)
        elif method == "faiss_leiden":
            labels, soft_labels = self._fit_faiss_leiden(features)
        elif method == "birch":
            labels, soft_labels = self._fit_birch(features)
        elif method == "pca_bgmm":
            labels, soft_labels = self._fit_pca_bgmm(features)
        elif method == "hnsw_hdbscan":
            labels, soft_labels = self._fit_hnsw_hdbscan(features)
        elif method == "hnsw_leiden":
            labels, soft_labels = self._fit_hnsw_leiden(features)
        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return ClusterResult(
            method_name=method,
            labels=labels,
            soft_labels=soft_labels,
            fit_time=elapsed,
            peak_ram_mb=peak / 10**6,
        )

    def _fit_kmeans(self, features: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit MiniBatchKMeans (baseline)."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn not available")

        n_clusters = min(200, len(features) // 1000)

        print(f"  Fitting MiniBatchKMeans (k={n_clusters})...")
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=10000, max_iter=100, random_state=42
        )
        labels = kmeans.fit_predict(features)

        # Hard clustering only (no soft labels)
        return labels, None

    def _fit_umap_hdbscan(self, features: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit UMAP + HDBSCAN - the standard bioacoustics stack."""
        if not UMAP_HDBSCAN_AVAILABLE:
            raise ImportError("umap-learn or hdbscan not available")

        # Reduce 112D to lower dimension where HDBSCAN works efficiently
        print("  Reducing 112D → 10D with UMAP (preserves graded manifolds)...")
        reducer = umap.UMAP(
            n_components=10,
            n_neighbors=30,  # Balances local vs global structure
            min_dist=0.0,  # Tighter clusters for HDBSCAN
            metric="cosine",  # Better for high-dim audio features
            random_state=42,
        )
        embedding = reducer.fit_transform(features)

        print("  Clustering low-D embedding with HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=50,
            min_samples=10,
            prediction_data=True,
            cluster_selection_method="eom",
        )
        clusterer.fit(embedding)
        labels = clusterer.labels_

        # Get soft clustering probabilities (for graded boundaries)
        soft_labels = hdbscan.all_points_membership_vectors(clusterer)

        return labels, soft_labels

    def _fit_bgmm(self, features: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit Bayesian Gaussian Mixture Model - best theoretical fit for graded signals.

        Graded vocalizations are mixtures of categories (e.g., 60% alarm, 40% contact).
        GMMs natively output probabilistic memberships - the mathematically correct way
        to represent a graded continuum.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn not available")

        n_components = min(200, len(features) // 1000)

        print(f"  Fitting Bayesian GMM (n_components={n_components})...")
        print("    Note: Using 'diag' covariance to avoid OOM in 112D space")
        bgmm = BayesianGaussianMixture(
            n_components=n_components,
            covariance_type="diag",  # Critical: 'full' covariance in 112D will OOM
            max_iter=300,
            weight_concentration_prior=0.01,  # Encourages pruning to find optimal k
            random_state=42,
        )
        bgmm.fit(features)

        # Soft labels (probabilities for graded phrases)
        soft_labels = bgmm.predict_proba(features)

        # Hard labels (primary cluster)
        labels = soft_labels.argmax(axis=1)

        return labels, soft_labels

    def _fit_faiss_leiden(self, features: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit FAISS + Leiden community detection."""
        if not FAISS_AVAILABLE or not LEIDEN_AVAILABLE:
            raise ImportError("faiss or python-igraph/leidenalg not available")

        n_neighbors = 30

        print(f"  Building kNN graph with FAISS (k={n_neighbors})...")
        index = faiss.IndexFlatL2(features.shape[1])
        index.add(features.astype(np.float32))
        distances, indices = index.search(features.astype(np.float32), n_neighbors)

        print("  Building igraph...")
        edges = []
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if i != j:
                    edges.append((i, j))

        g = ig.Graph(edges=edges)

        print("  Running Leiden community detection...")
        partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition)
        labels = np.array(partition.membership)

        # No soft labels for Leiden
        return labels, None

    def _fit_birch(self, features: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies).

        BIRCH is designed for very large datasets and is memory-efficient.
        It builds a CF-tree (Clustering Feature tree) that incrementally
        compresses data into subclusters.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn not available")

        from sklearn.cluster import Birch

        n_clusters = min(100, len(features) // 1000)

        print(f"  Fitting BIRCH (n_clusters={n_clusters})...")
        print("    BIRCH is incremental and memory-efficient")
        birch = Birch(
            n_clusters=n_clusters,
            threshold=0.5,  # Distance threshold for subcluster merging
            branching_factor=50,  # Maximum number of CF subclusters in a node
        )
        labels = birch.fit_predict(features)

        # BIRCH doesn't provide soft labels natively
        # We can estimate soft labels using distances to cluster centroids
        # Get subcluster centroids (if available)
        if hasattr(birch, "subcluster_centers_"):
            from sklearn.metrics.pairwise import euclidean_distances

            centroids = birch.subcluster_centers_
            distances = euclidean_distances(features, centroids)
            # Convert distances to probabilities (inverse distance weighting)
            soft_labels = 1.0 / (distances + 1e-6)
            soft_labels = soft_labels / soft_labels.sum(axis=1, keepdims=True)
        else:
            soft_labels = None

        return labels, soft_labels

    def _fit_pca_bgmm(self, features: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit PCA + Bayesian GMM - optimized pipeline for graded signals.

        Reduces 112D → 30D using PCA before BGMM to dramatically speed up fitting
        and reduce memory usage while preserving global variance structure.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn not available")

        n_components = 30  # Reduce 112D → 30D for faster GMM fitting
        n_bgmm_components = min(200, len(features) // 1000)

        print(f"  Reducing 112D → {n_components}D with PCA...")
        pca = PCA(n_components=n_components, random_state=42)
        features_reduced = pca.fit_transform(features)
        print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

        print(
            f"  Fitting Bayesian GMM on {n_components}D data (n_components={n_bgmm_components})..."
        )
        bgmm = BayesianGaussianMixture(
            n_components=n_bgmm_components,
            covariance_type="full",  # Can use 'full' in 30D space (would OOM in 112D)
            max_iter=300,
            weight_concentration_prior=0.01,
            random_state=42,
        )
        bgmm.fit(features_reduced)

        # Soft labels (probabilities for graded phrases)
        soft_labels = bgmm.predict_proba(features_reduced)

        # Hard labels (primary cluster)
        labels = soft_labels.argmax(axis=1)

        return labels, soft_labels

    def _fit_hnsw_hdbscan(self, features: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit HNSW + Sklearn HDBSCAN - Numba-free alternative to UMAP+HDBSCAN.

        Uses HNSW to build an approximate k-NN graph in native 112D space,
        then feeds the sparse distance matrix to sklearn's HDBSCAN (Cython, no Numba).

        Advantages:
        - No metric distortion (preserves true 112D Euclidean distances)
        - Sub-linear O(N log N) scaling vs O(N^2) for exact k-NN
        - No WSL segfault (sklearn HDBSCAN is pure Cython)
        """
        if not HNSW_AVAILABLE:
            raise ImportError("hnswlib not available")
        if not SKLEARN_HDBSCAN_AVAILABLE:
            raise ImportError("sklearn.cluster.HDBSCAN not available (requires sklearn >= 1.3)")

        n_samples = features.shape[0]
        min_cluster_size = max(50, n_samples // 1000)
        # k_neighbors must be >= min_samples (defaults to min_cluster_size in sklearn)
        # We'll use k = max(60, min_cluster_size) to ensure enough neighbors
        k_neighbors = min(max(60, min_cluster_size + 10), n_samples - 1)

        print(f"  Building HNSW index in {features.shape[1]}D space (k={k_neighbors})...")

        # --- STEP 1: HNSW Approximate kNN in 112D ---
        index = hnswlib.Index(space="l2", dim=features.shape[1])
        index.init_index(max_elements=n_samples, ef_construction=200, M=16)
        index.add_items(features.astype(np.float32), np.arange(n_samples))

        # Query the k nearest neighbors
        index.set_ef(max(100, k_neighbors * 2))  # ef should always be > k
        indices, distances = index.knn_query(features.astype(np.float32), k=k_neighbors)

        print("  Building sparse distance matrix...")

        # --- STEP 2: Build Sparse Distance Matrix ---
        row_ind = np.repeat(np.arange(n_samples), k_neighbors)
        col_ind = indices.flatten()
        data = distances.flatten()

        sparse_dist_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(n_samples, n_samples))

        # Ensure symmetry (HNSW is directed, HDBSCAN needs undirected)
        sparse_dist_matrix = sparse_dist_matrix.maximum(sparse_dist_matrix.T)

        print(f"  Clustering with sklearn HDBSCAN (min_cluster_size={min_cluster_size})...")

        # --- STEP 3: Sklearn HDBSCAN (NO NUMBA) ---
        # Use min_samples smaller than k to avoid the "fewer than X neighbors" error
        clusterer = SklearnHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min(5, min_cluster_size - 1),  # Smaller than k_neighbors
            metric="precomputed",
            allow_single_cluster=True,
            cluster_selection_method="eom",
        )

        cluster_labels = clusterer.fit_predict(sparse_dist_matrix)

        # Sklearn HDBSCAN provides probabilities (confidence scores)
        soft_probs = clusterer.probabilities_

        # Normalize probabilities to sum to 1 (including noise as a "cluster")
        # For HDBSCAN, points labeled as noise (-1) have low probabilities
        if soft_probs is not None:
            # Convert to soft labels format (n_samples, n_clusters+1 for noise)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            soft_labels = np.zeros((len(cluster_labels), n_clusters + 1))
            for i, (label, prob) in enumerate(zip(cluster_labels, soft_probs)):
                if label == -1:
                    soft_labels[i, -1] = 1.0  # Noise cluster
                else:
                    soft_labels[i, label] = prob
                    soft_labels[i, -1] = 1.0 - prob  # Remainder to noise
        else:
            soft_labels = None

        return cluster_labels, soft_labels

    def _fit_hnsw_leiden(self, features: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit HNSW + Leiden - graph clustering that respects graded continua.

        Leiden community detection finds "tightly knit sub-networks" rather than
        "dense islands in sparse oceans" like HDBSCAN. This makes it better
        for graded vocalization streams where HDBSCAN would classify too much as noise.

        Uses HNSW for fast k-NN graph construction, then Leiden for graph clustering.
        """
        if not HNSW_AVAILABLE:
            raise ImportError("hnswlib not available")
        if not LEIDEN_AVAILABLE:
            raise ImportError("python-igraph or leidenalg not available")

        n_samples = features.shape[0]
        k_neighbors = min(30, n_samples - 1)

        print(f"  Building HNSW index in {features.shape[1]}D space (k={k_neighbors})...")

        # --- STEP 1: HNSW kNN Graph ---
        index = hnswlib.Index(space="l2", dim=features.shape[1])
        index.init_index(max_elements=n_samples, ef_construction=200, M=16)
        index.add_items(features.astype(np.float32), np.arange(n_samples))

        index.set_ef(50)
        indices, _ = index.knn_query(features.astype(np.float32), k=k_neighbors)

        print("  Building igraph for Leiden clustering...")

        # --- STEP 2: Build iGraph Network ---
        edges = []
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if i != j:  # No self-loops
                    edges.append((i, j))

        # Remove duplicate edges (HNSW may create duplicates)
        edges = list(set(edges))

        g = ig.Graph(n=n_samples, edges=edges)

        print("  Running Leiden community detection...")

        # --- STEP 3: Leiden Clustering ---
        # Resolution parameter controls cluster granularity
        # Higher = more clusters, Lower = fewer clusters
        resolution = 0.5
        partition = leidenalg.RBConfigurationVertexPartition(g, resolution_parameter=resolution)

        cluster_labels = np.array(partition.membership)

        print(f"  Found {len(set(cluster_labels))} clusters")

        # No soft labels for Leiden (hard partition only)
        return cluster_labels, None

    def _compute_metrics(
        self, method: str, result: ClusterResult, features: np.ndarray, sequences: List[List[int]]
    ) -> BenchmarkMetrics:
        """Compute all metrics for a clustering result."""
        print("  Computing metrics...")

        # 1. Computational metrics (already in result)
        fit_time = result.fit_time
        peak_ram = result.peak_ram_mb

        # 2. Cluster statistics
        unique_labels = set(result.labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_rate = list(result.labels).count(-1) / len(result.labels)

        # 3. Zoo Vox metrics (using NgramCorpusStats if available)
        # Create sequences with cluster labels
        clustered_sequences = self._apply_cluster_labels_to_sequences(sequences, result.labels)

        if NGRAM_STATS_AVAILABLE:
            try:
                ngram_stats = NgramCorpusStats(clustered_sequences)
                svs = self._calculate_svs(ngram_stats)
                lrn_depth = (
                    ngram_stats.max_ngram_length if hasattr(ngram_stats, "max_ngram_length") else 0
                )
            except:
                svs = 0.0
                lrn_depth = 0
        else:
            # Placeholder metrics
            svs = self._calculate_placeholder_svs(clustered_sequences)
            lrn_depth = self._calculate_placeholder_lrn(clustered_sequences)

        vocab_util = n_clusters / len(unique_labels) if unique_labels else 0

        # 4. Graded continuity metrics
        neighborhood_consistency = self._calculate_neighborhood_consistency(features, result.labels)

        avg_kl_div = 0.0
        if result.soft_labels is not None:
            avg_kl_div = self._calculate_neighbor_kl_divergence(features, result.soft_labels)

        return BenchmarkMetrics(
            method_name=method,
            fit_time_seconds=fit_time,
            peak_ram_mb=peak_ram,
            n_clusters=n_clusters,
            noise_rate=noise_rate,
            shared_vocabulary_score=svs,
            lrn_depth=lrn_depth,
            vocabulary_utilization=vocab_util,
            neighborhood_consistency=neighborhood_consistency,
            avg_neighbor_kl_divergence=avg_kl_div,
            noise_classification_rate=noise_rate,
        )

    def _apply_cluster_labels_to_sequences(
        self, sequences: List[List[int]], labels: np.ndarray
    ) -> List[List[int]]:
        """Apply cluster labels to sequences (placeholder for actual implementation)."""
        # This is a simplified version - actual implementation would map
        # segment indices to cluster labels
        return sequences

    def _calculate_svs(self, ngram_stats) -> float:
        """Calculate Shared Vocabulary Score."""
        # Placeholder - use actual SVS calculation from ngram_stats
        if hasattr(ngram_stats, "shared_vocabulary_score"):
            return ngram_stats.shared_vocabulary_score
        return 0.5

    def _calculate_placeholder_svs(self, sequences: List[List[int]]) -> float:
        """Calculate placeholder SVS when NgramCorpusStats is unavailable."""
        if not sequences:
            return 0.0

        # Count unique n-grams across all sequences
        ngram_sets = []
        for seq in sequences:
            ngrams = set()
            for i in range(len(seq) - 2):
                ngrams.add(tuple(seq[i : i + 3]))
            if ngrams:
                ngram_sets.append(ngrams)

        if not ngram_sets:
            return 0.0

        # Calculate shared vocabulary (Jaccard-like similarity)
        all_ngrams = set()
        for ngram_set in ngram_sets:
            all_ngrams.update(ngram_set)

        # Average pairwise similarity
        similarities = []
        for i in range(len(ngram_sets)):
            for j in range(i + 1, len(ngram_sets)):
                intersection = len(ngram_sets[i] & ngram_sets[j])
                union = len(ngram_sets[i] | ngram_sets[j])
                if union > 0:
                    similarities.append(intersection / union)

        return np.mean(similarities) if similarities else 0.0

    def _calculate_placeholder_lrn(self, sequences: List[List[int]]) -> int:
        """Calculate placeholder longest repeated N-gram when NgramCorpusStats is unavailable."""
        if not sequences:
            return 0

        from collections import Counter

        max_n = 10
        for n in range(max_n, 0, -1):
            ngrams = []
            for seq in sequences:
                for i in range(len(seq) - n + 1):
                    ngrams.append(tuple(seq[i : i + n]))

            counts = Counter(ngrams)
            if counts and counts.most_common(1)[0][1] > 1:
                return n

        return 0

    def _calculate_neighborhood_consistency(
        self, features: np.ndarray, labels: np.ndarray
    ) -> float:
        """Calculate neighborhood consistency using ARI on k-NN."""
        if not SKLEARN_AVAILABLE:
            return 0.0

        # Get k-nearest neighbors for each point
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(features)
        distances, indices = nn.kneighbors(features)

        # Calculate ARI between point labels and neighbor labels
        ari_scores = []
        for i in range(min(1000, len(features))):  # Sample for speed
            point_label = labels[i]
            neighbor_labels = labels[indices[i][1:]]  # Exclude self

            # Count neighbors with same label
            same_label = sum(neighbor_labels == point_label)
            consistency = same_label / self.n_neighbors
            ari_scores.append(consistency)

        return np.mean(ari_scores)

    def _calculate_neighbor_kl_divergence(
        self, features: np.ndarray, soft_labels: np.ndarray
    ) -> float:
        """Calculate average KL divergence between neighboring points."""
        if not SKLEARN_AVAILABLE:
            return 0.0

        # Get k-nearest neighbors
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(features)
        distances, indices = nn.kneighbors(features)

        # Calculate KL divergence for neighbors
        kl_divs = []
        epsilon = 1e-10  # Small constant to avoid log(0)

        for i in range(min(1000, len(features))):
            p = soft_labels[i] + epsilon
            p = p / p.sum()  # Normalize

            neighbor_kls = []
            for j in indices[i][1:]:  # Exclude self
                q = soft_labels[j] + epsilon
                q = q / q.sum()

                # KL divergence
                kl = np.sum(p * np.log(p / q))
                neighbor_kls.append(kl)

            kl_divs.append(np.mean(neighbor_kls))

        return np.mean(kl_divs)

    def _print_metrics(self, metrics: BenchmarkMetrics):
        """Print metrics summary."""
        print(f"\n  Results for {metrics.method_name}:")
        print("    Computational:")
        print(f"      Time: {metrics.fit_time_seconds:.1f}s")
        print(f"      RAM: {metrics.peak_ram_mb:.1f} MB")
        print("    Clusters:")
        print(f"      Count: {metrics.n_clusters}")
        print(f"      Noise rate: {metrics.noise_rate * 100:.1f}%")
        print("    Zoo Vox metrics:")
        print(f"      SVS: {metrics.shared_vocabulary_score:.3f}")
        print(f"      LRN depth: {metrics.lrn_depth}")
        print(f"      Vocab utilization: {metrics.vocabulary_utilization:.3f}")
        print("    Graded continuity:")
        print(f"      Neighborhood consistency: {metrics.neighborhood_consistency:.3f}")
        if metrics.avg_neighbor_kl_divergence > 0:
            print(f"      Avg neighbor KL divergence: {metrics.avg_neighbor_kl_divergence:.3f}")

    def _compare_methods(self, results: Dict[str, BenchmarkMetrics]):
        """Print comparison table of all methods."""
        print(f"\n{'=' * 70}")
        print("Comparison Table")
        print(f"{'=' * 70}")

        # Header
        print(
            f"\n{'Method':<15} {'Time(s)':<10} {'RAM(MB)':<10} {'SVS':<8} {'LRN':<6} {'Clusters':<10} {'Noise%':<8} {'Neighbor':<10}"
        )
        print("-" * 85)

        # Rows
        for metrics in results.values():
            print(
                f"{metrics.method_name:<15} "
                f"{metrics.fit_time_seconds:<10.1f} "
                f"{metrics.peak_ram_mb:<10.1f} "
                f"{metrics.shared_vocabulary_score:<8.3f} "
                f"{metrics.lrn_depth:<6} "
                f"{metrics.n_clusters:<10} "
                f"{metrics.noise_rate * 100:<8.1f} "
                f"{metrics.neighborhood_consistency:<10.3f}"
            )

        # Find best method for each metric
        print(f"\n{'=' * 70}")
        print("Best by Category")
        print(f"{'=' * 70}")

        # Fastest
        fastest = min(results.values(), key=lambda x: x.fit_time_seconds)
        print(f"  Fastest: {fastest.method_name} ({fastest.fit_time_seconds:.1f}s)")

        # Lowest RAM
        lowest_ram = min(results.values(), key=lambda x: x.peak_ram_mb)
        print(f"  Lowest RAM: {lowest_ram.method_name} ({lowest_ram.peak_ram_mb:.1f} MB)")

        # Highest SVS
        best_svs = max(results.values(), key=lambda x: x.shared_vocabulary_score)
        print(f"  Best SVS: {best_svs.method_name} ({best_svs.shared_vocabulary_score:.3f})")

        # Deepest LRN
        deepest_lrn = max(results.values(), key=lambda x: x.lrn_depth)
        print(f"  Deepest LRN: {deepest_lrn.method_name} (depth {deepest_lrn.lrn_depth})")

        # Best neighborhood consistency
        best_consistency = max(results.values(), key=lambda x: x.neighborhood_consistency)
        print(
            f"  Best neighborhood: {best_consistency.method_name} ({best_consistency.neighborhood_consistency:.3f})"
        )

    def export_results(self, results: Dict[str, BenchmarkMetrics], output_path: str):
        """Export benchmark results to JSON."""
        import numpy as np

        def convert_to_serializable(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        output_data = {
            method: convert_to_serializable(asdict(metrics)) for method, metrics in results.items()
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults exported to {output_path}")


def main():
    """Run benchmark on extracted 112D features."""
    # Check if extraction is complete
    feature_path = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/extraction_112d_no_cluster.json"

    if not Path(feature_path).exists():
        print(f"Error: Feature file not found: {feature_path}")
        print("Run extraction first: cargo run --release --example bat_112d_extraction_no_cluster")
        return

    print("Loading features...")
    with open(feature_path, "r") as f:
        data = json.load(f)

    # Sample for benchmark (tiny sample to avoid WSL crash)
    n_samples = min(20000, len(data["segments"]))
    print(f"Sampling {n_samples:,} segments for benchmark...")

    features_list = []
    for i, seg in enumerate(data["segments"][:n_samples]):
        features_list.append(seg["features_112d"])

    features_112d = np.array(features_list, dtype=np.float32)

    # Create dummy sequences (in real use, these would come from file ordering)
    sequences = []
    seq = []
    cluster_id = 0
    for i in range(len(features_list)):
        seq.append(cluster_id)
        if len(seq) >= 10 and np.random.random() > 0.7:
            sequences.append(seq)
            seq = []
            cluster_id = (cluster_id + 1) % 10
    if seq:
        sequences.append(seq)

    # Run benchmark - test all available methods
    suite = ClusterBenchmarkSuite()

    # Determine which methods are available
    # Skip UMAP+HDBSCAN due to WSL segfault issues
    available_methods = ["kmeans"]  # Always available (sklearn)

    # if UMAP_HDBSCAN_AVAILABLE:
    #     available_methods.append("umap_hdbscan")  # DISABLED - causes WSL segfault

    if SKLEARN_AVAILABLE:
        available_methods.append("bgmm")
        available_methods.append("birch")
        available_methods.append("pca_bgmm")  # Optimized PCA + BGMM pipeline

    if FAISS_AVAILABLE and LEIDEN_AVAILABLE:
        available_methods.append("faiss_leiden")

    # New HNSW-based methods (Numba-free alternatives)
    if HNSW_AVAILABLE:
        if SKLEARN_HDBSCAN_AVAILABLE:
            available_methods.append("hnsw_hdbscan")  # HNSW + sklearn HDBSCAN (no Numba!)
        if LEIDEN_AVAILABLE:
            available_methods.append("hnsw_leiden")  # HNSW + Leiden

    print(f"Available methods: {', '.join(available_methods)}")
    print("Note: UMAP+HDBSCAN disabled due to WSL compatibility issues")
    print("      HNSW+HDBSCAN uses sklearn's Numba-free implementation")
    print()

    results = suite.run(features_112d, sequences, methods=available_methods)

    # Export
    output_path = "/mnt/c/Users/sheel/Desktop/src/analysis/results/cluster_benchmark_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    suite.export_results(results, output_path)


if __name__ == "__main__":
    main()
