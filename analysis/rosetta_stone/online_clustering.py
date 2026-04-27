#!/usr/bin/env python3
"""
Online Clustering - Direction 8: Online/Incremental Clustering

This module implements incremental K-means clustering for real-time vocabulary
adaptation in the closed-loop agent. Unlike batch K-means, this algorithm
updates centroids incrementally as new data arrives.

Key Features:
- Incremental centroid updates via partial_fit()
- Automatic cluster spawning for novel patterns
- Forgetting mechanism via decay and pruning
- Concept drift detection
- Model persistence (pickle/joblib)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OnlineKMeans:
    """
    Incremental K-means for real-time vocabulary adaptation.

    This clusterer supports online learning where data arrives in streams
    rather than batches. It can spawn new clusters for novel patterns,
    prune stale clusters, and detect concept drift.

    Usage:
        clusterer = OnlineKMeans(initial_k=100, max_k=2000)
        clusterer.partial_fit(features_batch)
        labels = clusterer.predict(new_features)
    """

    def __init__(
        self,
        initial_k: int = 100,
        max_k: int = 2000,
        spawn_threshold: float = 3.0,
        merge_threshold: float = 1.0,
        decay_window_ms: int = 10000,
        decay_rate: float = 0.0,
        drift_threshold: float = 2.0,
        auto_spawn: bool = False,
        random_state: Optional[int] = None,
        min_samples_for_init: int = 2,
    ):
        """
        Initialize the OnlineKMeans clusterer.

        Args:
            initial_k: Initial number of clusters
            max_k: Maximum number of clusters (for spawning)
            spawn_threshold: Distance threshold for spawning new clusters
            merge_threshold: Distance threshold for merging nearby clusters
            decay_window_ms: Time window for considering clusters stale (milliseconds)
            decay_rate: Rate at which cluster counts decay (0.0 = no decay)
            drift_threshold: Centroid shift threshold for concept drift detection
            auto_spawn: Automatically spawn clusters during partial_fit
            random_state: Random seed for reproducibility
            min_samples_for_init: Minimum samples required before initialization
        """
        self.initial_k = initial_k
        self.max_k = max_k
        self.spawn_threshold = spawn_threshold
        self.merge_threshold = merge_threshold
        self.decay_window_ms = decay_window_ms
        self.decay_rate = decay_rate
        self.drift_threshold = drift_threshold
        self.auto_spawn = auto_spawn
        self.random_state = random_state
        self.min_samples_for_init = min_samples_for_init

        # Core clustering model
        self.model: Optional[MiniBatchKMeans] = None
        self.feature_dim: int = 0

        # Cluster metadata
        self.centroids: np.ndarray = np.array([])
        self.cluster_counts: np.ndarray = np.array([])
        self.last_seen: np.ndarray = np.array([])  # Timestamps (ms)
        self._is_fitted: bool = False

        # Concept drift tracking
        self._initial_centroids: Optional[np.ndarray] = None

        # Buffer for initialization
        self._init_buffer: list = []

    def partial_fit(self, features: np.ndarray) -> None:
        """
        Update centroids with new batch of features.

        Args:
            features: Feature matrix (n_samples, n_dimensions)
        """
        if features.shape[0] == 0:
            return

        n_samples, n_features = features.shape

        # Initialize if first call
        if self.model is None:
            self.feature_dim = n_features

            # Buffer samples until we have enough for initialization
            self._init_buffer.append(features)
            total_buffered = sum(f.shape[0] for f in self._init_buffer)

            if total_buffered < self.min_samples_for_init:
                # Not enough samples yet - keep buffering
                logger.debug(
                    f"Buffering samples: {total_buffered}/{self.min_samples_for_init} "
                    f"required for initialization"
                )
                return

            # Concatenate buffered samples and initialize
            buffered_features = np.vstack(self._init_buffer)
            n_samples = buffered_features.shape[0]  # Update to use buffered count
            self._init_buffer.clear()  # Clear buffer after use

            # Adjust initial_k if we have fewer samples
            initial_k = min(self.initial_k, max(1, n_samples - 1))

            self.model = MiniBatchKMeans(
                n_clusters=initial_k,
                batch_size=min(100, n_samples),
                random_state=self.random_state,
                max_iter=100,
                n_init=3,
            )
            self.model.fit(buffered_features)

            # Initialize metadata - use the ACTUAL number of clusters, not self.initial_k
            # This ensures centroids, cluster_counts, and last_seen always have matching lengths
            self.centroids = self.model.cluster_centers_.copy()
            self.cluster_counts = np.zeros(initial_k)  # Use initial_k, not self.initial_k
            self.last_seen = np.full(initial_k, time.time() * 1000)  # Use initial_k, not self.initial_k

            # Count initial assignments
            labels = self.model.predict(buffered_features)
            for label in labels:
                self.cluster_counts[label] += 1
                self.last_seen[label] = time.time() * 1000

            self._is_fitted = True
            self._initial_centroids = self.centroids.copy()

            logger.info(f"Initialized OnlineKMeans with k={initial_k} from {n_samples} samples")
        else:
            # Apply decay to existing counts
            if self.decay_rate > 0:
                self.cluster_counts *= (1.0 - self.decay_rate)

            current_time = time.time() * 1000

            # Auto-spawn mode: Detect novel samples BEFORE updating model
            # to prevent double-counting distant points
            if self.auto_spawn and self.should_spawn_cluster(features):
                known_features, novel_features = self._split_known_novel(features)

                # Update model only with known (non-distant) samples
                if len(known_features) > 0:
                    self.model.partial_fit(known_features)
                    self.centroids = self.model.cluster_centers_

                    # Update counts and timestamps only for known samples
                    labels = self.model.predict(known_features)
                    for label in labels:
                        self.cluster_counts[label] += 1
                        self.last_seen[label] = current_time

                # Spawn new cluster from novel samples (not counted against old clusters)
                if len(novel_features) > 0:
                    self.spawn_cluster(novel_features)
            else:
                # Standard path: update model with all features
                self.model.partial_fit(features)
                self.centroids = self.model.cluster_centers_

                # Update counts and timestamps
                labels = self.model.predict(features)
                for label in labels:
                    self.cluster_counts[label] += 1
                    self.last_seen[label] = current_time

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Assign cluster IDs to features.

        Args:
            features: Feature matrix (n_samples, n_dimensions)

        Returns:
            Array of cluster labels
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call partial_fit() first.")

        return self.model.predict(features)

    def should_spawn_cluster(
        self, features: np.ndarray, threshold: Optional[float] = None
    ) -> bool:
        """
        Check if new cluster should be spawned based on distance.

        Args:
            features: Feature matrix to check
            threshold: Distance threshold (uses spawn_threshold if None)

        Returns:
            True if cluster should be spawned
        """
        if not self._is_fitted:
            return False

        if len(self.centroids) >= self.max_k:
            return False

        threshold = threshold or self.spawn_threshold

        # Find minimum distance to any centroid
        min_distances = np.min(
            euclidean_distances(features, self.centroids), axis=1
        )

        # Spawn if any point is beyond threshold
        return np.any(min_distances > threshold)

    def spawn_cluster(self, features: np.ndarray) -> bool:
        """
        Spawn a new cluster from the given features.

        Args:
            features: Feature matrix for new cluster

        Returns:
            True if cluster was spawned
        """
        if not self._is_fitted:
            return False

        if len(self.centroids) >= self.max_k:
            logger.warning(f"Cannot spawn: max_k={self.max_k} reached")
            return False

        # Compute new centroid as mean of features
        new_centroid = np.mean(features, axis=0).reshape(1, -1)

        # Add to centroids
        self.centroids = np.vstack([self.centroids, new_centroid])

        # Update model with new centroids
        self._rebuild_model()

        # Add metadata for new cluster
        new_count = len(features)
        self.cluster_counts = np.append(self.cluster_counts, new_count)
        self.last_seen = np.append(self.last_seen, time.time() * 1000)

        logger.info(f"Spawned new cluster: total={len(self.centroids)}")

        return True

    def merge_nearby_clusters(self, threshold: Optional[float] = None) -> bool:
        """
        Merge clusters that are too close to each other.

        Args:
            threshold: Distance threshold for merging (uses merge_threshold if None)

        Returns:
            True if any clusters were merged
        """
        if not self._is_fitted or len(self.centroids) < 2:
            return False

        threshold = threshold or self.merge_threshold

        # Compute pairwise distances
        distances = euclidean_distances(self.centroids)

        # Find closest pair (excluding diagonal)
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)

        if min_dist < threshold:
            # Merge clusters
            i, j = min_idx

            # Weighted merge by count
            count_i = self.cluster_counts[i]
            count_j = self.cluster_counts[j]
            total_count = count_i + count_j

            merged_centroid = (
                self.centroids[i] * count_i + self.centroids[j] * count_j
            ) / total_count

            # Update centroid i
            self.centroids[i] = merged_centroid
            self.cluster_counts[i] = total_count

            # Remove centroid j
            self.centroids = np.delete(self.centroids, j, axis=0)
            self.cluster_counts = np.delete(self.cluster_counts, j)
            self.last_seen = np.delete(self.last_seen, j)

            # Rebuild model
            self._rebuild_model()

            logger.info(f"Merged clusters {i} and {j}: distance={min_dist:.2f}")
            return True

        return False

    def prune_stale_clusters(self) -> int:
        """
        Remove clusters that haven't been seen recently.

        Returns:
            Number of clusters pruned
        """
        if not self._is_fitted:
            return 0

        current_time = time.time() * 1000
        stale_mask = (current_time - self.last_seen) > self.decay_window_ms

        n_stale = np.sum(stale_mask)

        if n_stale > 0:
            # Remove stale clusters
            keep_mask = ~stale_mask
            self.centroids = self.centroids[keep_mask]
            self.cluster_counts = self.cluster_counts[keep_mask]
            self.last_seen = self.last_seen[keep_mask]

            # Only rebuild if we still have clusters
            if len(self.centroids) > 0:
                self._rebuild_model()
            else:
                # Reset to unfitted state if all clusters pruned
                self.model = None
                self._is_fitted = False

            logger.info(f"Pruned {n_stale} stale clusters")

        return int(n_stale)

    def prune_empty_clusters(self, min_count: float = 1.0) -> int:
        """
        Remove clusters with very low counts.

        Args:
            min_count: Minimum count threshold

        Returns:
            Number of clusters pruned
        """
        if not self._is_fitted:
            return 0

        empty_mask = self.cluster_counts < min_count
        n_empty = np.sum(empty_mask)

        if n_empty > 0:
            keep_mask = ~empty_mask
            self.centroids = self.centroids[keep_mask]
            self.cluster_counts = self.cluster_counts[keep_mask]
            self.last_seen = self.last_seen[keep_mask]

            # Only rebuild if we still have clusters
            if len(self.centroids) > 0:
                self._rebuild_model()
            else:
                # Reset to unfitted state if all clusters pruned
                self.model = None
                self._is_fitted = False

            logger.info(f"Pruned {n_empty} empty clusters")

        return int(n_empty)

    def detect_concept_drift(self) -> bool:
        """
        Detect if concept drift has occurred.

        Concept drift is detected when centroids have shifted significantly
        from their initial positions.

        Returns:
            True if concept drift detected
        """
        if not self._is_fitted or self._initial_centroids is None:
            return False

        drift_magnitude = self.get_drift_magnitude(self._initial_centroids)

        return drift_magnitude > self.drift_threshold

    def get_drift_magnitude(self, reference_centroids: np.ndarray) -> float:
        """
        Get the magnitude of centroid drift from reference.

        Args:
            reference_centroids: Reference centroids to compare against

        Returns:
            Mean drift distance
        """
        if not self._is_fitted:
            return 0.0

        # Match centroids by minimum distance
        max_clusters = min(len(reference_centroids), len(self.centroids))

        if max_clusters == 0:
            return 0.0

        # Compute distances and find best matching pairs
        distances = euclidean_distances(reference_centroids[:max_clusters], self.centroids[:max_clusters])

        # Mean of minimum distances for each reference centroid
        drift_per_cluster = np.min(distances, axis=1)
        mean_drift = np.mean(drift_per_cluster)

        return float(mean_drift)

    def save(self, path: str) -> None:
        """
        Save the clusterer state to disk.

        Args:
            path: File path to save to
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        state = {
            "centroids": self.centroids,
            "cluster_counts": self.cluster_counts,
            "last_seen": self.last_seen,
            "feature_dim": self.feature_dim,
            "initial_k": self.initial_k,
            "max_k": self.max_k,
            "spawn_threshold": self.spawn_threshold,
            "merge_threshold": self.merge_threshold,
            "decay_window_ms": self.decay_window_ms,
            "decay_rate": self.decay_rate,
            "drift_threshold": self.drift_threshold,
            "auto_spawn": self.auto_spawn,
            "random_state": self.random_state,
            "min_samples_for_init": self.min_samples_for_init,
            "_initial_centroids": self._initial_centroids,
        }

        path_obj = Path(path)

        if path_obj.suffix == ".joblib":
            joblib.dump(state, path)
        else:
            with open(path, "wb") as f:
                pickle.dump(state, f)

        logger.info(f"Saved OnlineKMeans to {path}")

    @classmethod
    def load(cls, path: str) -> "OnlineKMeans":
        """
        Load a clusterer from disk.

        Args:
            path: File path to load from

        Returns:
            Loaded OnlineKMeans instance
        """
        path_obj = Path(path)

        if path_obj.suffix == ".joblib":
            state = joblib.load(path)
        else:
            with open(path, "rb") as f:
                state = pickle.load(f)

        # Create new instance
        clusterer = cls(
            initial_k=state["initial_k"],
            max_k=state["max_k"],
            spawn_threshold=state["spawn_threshold"],
            merge_threshold=state["merge_threshold"],
            decay_window_ms=state["decay_window_ms"],
            decay_rate=state["decay_rate"],
            drift_threshold=state["drift_threshold"],
            auto_spawn=state["auto_spawn"],
            random_state=state["random_state"],
            min_samples_for_init=state.get("min_samples_for_init", 2),  # Default to 2 for backward compatibility
        )

        # Restore state
        clusterer.centroids = state["centroids"]
        clusterer.cluster_counts = state["cluster_counts"]
        clusterer.last_seen = state["last_seen"]
        clusterer.feature_dim = state["feature_dim"]
        clusterer._initial_centroids = state.get("_initial_centroids")
        clusterer._is_fitted = True

        # Rebuild model
        clusterer._rebuild_model()

        logger.info(f"Loaded OnlineKMeans from {path}")

        return clusterer

    def _rebuild_model(self) -> None:
        """Rebuild the sklearn model from current centroids."""
        n_clusters = len(self.centroids)

        # Create a new model and fit on centroids to initialize all internal state
        self.model = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=max(1, n_clusters),
            random_state=self.random_state,
            max_iter=1,
            n_init=1,
        )
        self.model.fit(self.centroids)

        # Ensure centroids match exactly (fit may move them slightly)
        self.model.cluster_centers_ = self.centroids.copy()

        # Sync our tracking
        self.centroids = self.model.cluster_centers_

    def _split_known_novel(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split features into known (close to centroids) and novel (distant).

        Args:
            features: Feature matrix to split

        Returns:
            Tuple of (known_features, novel_features)
        """
        min_distances = np.min(
            euclidean_distances(features, self.centroids), axis=1
        )
        distant_mask = min_distances > self.spawn_threshold

        novel_features = features[distant_mask]
        known_features = features[~distant_mask]

        return known_features, novel_features

    def _auto_spawn_if_needed(self, features: np.ndarray) -> None:
        """Automatically spawn cluster if distant data detected."""
        if self.should_spawn_cluster(features):
            # Find points beyond threshold
            min_distances = np.min(
                euclidean_distances(features, self.centroids), axis=1
            )
            distant_mask = min_distances > self.spawn_threshold

            if np.any(distant_mask):
                distant_features = features[distant_mask]
                self.spawn_cluster(distant_features)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_online_clusterer(
    initial_k: int = 100,
    max_k: int = 2000,
    random_state: Optional[int] = 42,
) -> OnlineKMeans:
    """
    Convenience function to create an OnlineKMeans clusterer.

    Args:
        initial_k: Initial number of clusters
        max_k: Maximum number of clusters
        random_state: Random seed

    Returns:
        Configured OnlineKMeans instance
    """
    return OnlineKMeans(
        initial_k=initial_k,
        max_k=max_k,
        random_state=random_state,
    )


def main():
    """Demo CLI for OnlineKMeans."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Online K-means clustering for streaming data"
    )
    parser.add_argument(
        "--features", "-f", required=True, help="Path to features .npy file"
    )
    parser.add_argument(
        "--initial-k", type=int, default=100, help="Initial number of clusters"
    )
    parser.add_argument(
        "--max-k", type=int, default=2000, help="Maximum number of clusters"
    )
    parser.add_argument(
        "--output", "-o", help="Save model to this path"
    )

    args = parser.parse_args()

    # Load features
    logger.info(f"Loading features from {args.features}")
    features = np.load(args.features)

    # Create and fit clusterer
    clusterer = OnlineKMeans(
        initial_k=args.initial_k,
        max_k=args.max_k,
        random_state=42,
    )

    logger.info("Fitting clusterer...")
    clusterer.partial_fit(features)

    print(f"\nResults:")
    print(f"  Clusters: {len(clusterer.centroids)}")
    print(f"  Feature dim: {clusterer.feature_dim}")
    print(f"  Total samples: {int(np.sum(clusterer.cluster_counts))}")

    # Save if requested
    if args.output:
        clusterer.save(args.output)
        print(f"  Model saved to: {args.output}")


if __name__ == "__main__":
    main()
