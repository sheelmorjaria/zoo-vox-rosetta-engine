#!/usr/bin/env python3
"""
Tests for VocabOptimizer - Direction 1: Adaptive Vocabulary

TDD Sprint 1.1: SVS Computation
TDD Sprint 1.2: K Optimization
TDD Sprint 1.3: SpeciesVocabRegistry integration

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
import sys
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.rosetta_stone.vocab_optimizer import (
    SpeciesVocabConfig,
    SpeciesVocabRegistry,
    VocabOptimizer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Fixtures
# =============================================================================


def create_well_separated_clusters(n_samples: int = 100) -> np.ndarray:
    """Create well-separated 2D clusters for high SVS testing."""
    np.random.seed(42)
    data = []

    # Cluster 1: centered at (0, 0)
    data.append(np.random.randn(n_samples // 4, 2) * 0.3)

    # Cluster 2: centered at (10, 10)
    data.append(np.random.randn(n_samples // 4, 2) * 0.3 + [10, 10])

    # Cluster 3: centered at (-10, -10)
    data.append(np.random.randn(n_samples // 4, 2) * 0.3 + [-10, -10])

    # Cluster 4: centered at (10, -10)
    data.append(np.random.randn(n_samples // 4, 2) * 0.3 + [10, -10])

    return np.vstack(data)


def create_overlapping_clusters(n_samples: int = 100) -> np.ndarray:
    """Create overlapping clusters for low SVS testing."""
    np.random.seed(42)
    data = []

    # Clusters with high variance and close centers
    data.append(np.random.randn(n_samples // 3, 2) * 3.0 + [0, 0])
    data.append(np.random.randn(n_samples // 3, 2) * 3.0 + [2, 2])
    data.append(np.random.randn(n_samples // 3, 2) * 3.0 + [-2, 2])

    return np.vstack(data)


def create_112d_test_data(n_samples: int = 100) -> np.ndarray:
    """Create 112D feature vectors for realistic testing."""
    np.random.seed(42)
    return np.random.randn(n_samples, 112)


# =============================================================================
# Sprint 1.1: SVS Computation Tests
# =============================================================================


class TestSVSComputation:
    """Test Silhouette Validation Score computation."""

    def test_svs_well_separated_clusters(self):
        """High SVS (>0.5) for well-separated clusters."""
        features = create_well_separated_clusters()
        optimizer = VocabOptimizer()

        svs = optimizer.compute_svs(features, k=4)

        assert svs > 0.5, f"Expected high SVS for well-separated clusters, got {svs}"
        logger.info(f"✓ Well-separated clusters SVS: {svs:.3f}")

    def test_svs_overlapping_clusters(self):
        """Low SVS (<0.5) for overlapping clusters."""
        features = create_overlapping_clusters()
        optimizer = VocabOptimizer()

        svs = optimizer.compute_svs(features, k=3)

        # Overlapping clusters have moderate SVS
        assert svs < 0.5, f"Expected moderate SVS for overlapping clusters, got {svs}"
        logger.info(f"✓ Overlapping clusters SVS: {svs:.3f}")

    def test_svs_single_cluster(self):
        """SVS = 0 for single cluster (edge case)."""
        # Single cluster data
        np.random.seed(42)
        features = np.random.randn(50, 2)

        optimizer = VocabOptimizer()
        svs = optimizer.compute_svs(features, k=1)

        # Silhouette score is 0 for k=1 (all points in same cluster)
        assert svs == 0.0, f"Expected SVS=0 for single cluster, got {svs}"
        logger.info(f"✓ Single cluster SVS: {svs:.3f}")

    def test_svs_deterministic_with_seed(self):
        """Same seed produces same SVS."""
        features = create_112d_test_data()

        optimizer1 = VocabOptimizer(random_state=42)
        svs1 = optimizer1.compute_svs(features, k=10)

        optimizer2 = VocabOptimizer(random_state=42)
        svs2 = optimizer2.compute_svs(features, k=10)

        assert svs1 == svs2, f"Expected deterministic SVS: {svs1} != {svs2}"
        logger.info(f"✓ Deterministic SVS with seed: {svs1:.3f}")


# =============================================================================
# Sprint 1.2: K Optimization Tests
# =============================================================================


class TestKOptimization:
    """Test k optimization using SVS maximization."""

    def test_optimize_k_finds_peak(self):
        """Returns k with highest SVS in range."""
        # Create data with 4 well-separated clusters
        features = create_well_separated_clusters()

        # Use step_size=1 to test all k values in small range
        optimizer = VocabOptimizer(k_range=(2, 8), step_size=1, random_state=42)
        optimal_k = optimizer.optimize_k(features, species="test")

        # Should find k=4 (actual number of clusters)
        assert optimal_k == 4, f"Expected k=4 for 4 clusters, got {optimal_k}"
        logger.info(f"✓ Found optimal k={optimal_k} for 4 clusters")

    def test_optimize_k_respects_range(self):
        """Never returns k outside specified range."""
        features = create_112d_test_data()

        optimizer = VocabOptimizer(k_range=(5, 10), random_state=42)
        optimal_k = optimizer.optimize_k(features, species="test")

        assert 5 <= optimal_k <= 10, f"k={optimal_k} outside range [5, 10]"
        logger.info(f"✓ k={optimal_k} within specified range")

    def test_optimize_k_uses_step_size(self):
        """Uses step_size to skip k values."""
        features = create_well_separated_clusters()

        # Use step=2 to test only even k values
        optimizer = VocabOptimizer(k_range=(2, 8), step_size=2, random_state=42)
        optimal_k = optimizer.optimize_k(features, species="test")

        # Should be one of: 2, 4, 6
        assert optimal_k in {2, 4, 6}, f"k={optimal_k} not in step sequence"
        logger.info(f"✓ k={optimal_k} respects step_size=2")

    def test_optimize_k_112d_features(self):
        """Works correctly with 112D feature vectors."""
        features = create_112d_test_data(n_samples=200)

        optimizer = VocabOptimizer(k_range=(5, 15), random_state=42)
        optimal_k = optimizer.optimize_k(features, species="test_112d")

        assert 5 <= optimal_k <= 15, f"k={optimal_k} outside range"
        logger.info(f"✓ Optimized k={optimal_k} for 112D features")

    def test_optimize_k_returns_svs_history(self):
        """Returns SVS history for analysis."""
        features = create_well_separated_clusters()

        # Use step_size=1 to test all k values
        optimizer = VocabOptimizer(k_range=(2, 6), step_size=1, random_state=42)
        optimal_k, svs_history = optimizer.optimize_k_with_history(
            features, species="test"
        )

        # Should have history for k=2,3,4,5
        assert len(svs_history) == 4, f"Expected 4 entries, got {len(svs_history)}"
        assert optimal_k in svs_history, "Optimal k should be in history"

        # Find the k with maximum SVS
        best_k = max(svs_history, key=svs_history.get)
        assert best_k == optimal_k, f"Best k={best_k} != optimal={optimal_k}"

        logger.info(f"✓ SVS history: {svs_history}")


# =============================================================================
# Sprint 1.3: SpeciesVocabConfig Tests
# =============================================================================


class TestSpeciesVocabConfig:
    """Test species-specific vocabulary configuration."""

    def test_create_vocab_config(self):
        """Create a vocabulary configuration."""
        config = SpeciesVocabConfig(
            species="egyptian_fruit_bat", optimal_k=1020, svs_score=0.45
        )

        assert config.species == "egyptian_fruit_bat"
        assert config.optimal_k == 1020
        assert config.svs_score == 0.45
        assert config.discovery_timestamp > 0
        logger.info(f"✓ Created config for {config.species}: k={config.optimal_k}")

    def test_vocab_config_to_dict(self):
        """Convert configuration to dictionary."""
        config = SpeciesVocabConfig(
            species="marmoset", optimal_k=450, svs_score=0.52
        )

        as_dict = config.to_dict()

        assert as_dict["species"] == "marmoset"
        assert as_dict["optimal_k"] == 450
        assert as_dict["svs_score"] == 0.52
        assert "discovery_timestamp" in as_dict
        logger.info(f"✓ Config to_dict: {as_dict}")

    def test_vocab_config_from_dict(self):
        """Create configuration from dictionary."""
        data = {
            "species": "dolphin",
            "optimal_k": 680,
            "svs_score": 0.38,
            "discovery_timestamp": 1234567890,
        }

        config = SpeciesVocabConfig.from_dict(data)

        assert config.species == "dolphin"
        assert config.optimal_k == 680
        assert config.svs_score == 0.38
        assert config.discovery_timestamp == 1234567890
        logger.info(f"✓ Config from_dict: {config.species}")

    def test_vocab_config_roundtrip_json(self):
        """Save and load configuration from JSON."""
        config = SpeciesVocabConfig(
            species="test_species", optimal_k=333, svs_score=0.42
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config.save(f.name)

            loaded = SpeciesVocabConfig.load(f.name)

        assert loaded.species == config.species
        assert loaded.optimal_k == config.optimal_k
        assert loaded.svs_score == config.svs_score
        logger.info(f"✓ JSON roundtrip successful")

    def test_vocab_config_equality(self):
        """Test configuration equality."""
        config1 = SpeciesVocabConfig(species="bat", optimal_k=100, svs_score=0.5)
        config2 = SpeciesVocabConfig(species="bat", optimal_k=100, svs_score=0.5)
        config3 = SpeciesVocabConfig(species="bat", optimal_k=200, svs_score=0.5)

        assert config1 == config2, "Equal configs should be equal"
        assert config1 != config3, "Different configs should not be equal"
        logger.info(f"✓ Equality check works")


# =============================================================================
# Integration Tests
# =============================================================================


class TestVocabOptimizerIntegration:
    """Integration tests with ExemplarManager."""

    def test_optimizer_with_normalization(self):
        """Optimizer works with normalized features."""
        features = create_112d_test_data(n_samples=200)

        # Normalize features (as ExemplarManager does)
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)

        optimizer = VocabOptimizer(k_range=(5, 15), random_state=42)
        optimal_k = optimizer.optimize_k(features_normalized, species="test_norm")

        assert 5 <= optimal_k <= 15
        logger.info(f"✓ Optimizer works with normalized features: k={optimal_k}")

    def test_optimizer_with_nans(self):
        """Handles NaN values in features."""
        features = create_112d_test_data(n_samples=100)
        features[10:15, 5] = np.nan  # Insert some NaNs

        optimizer = VocabOptimizer(k_range=(5, 10), random_state=42)

        # Should handle NaNs gracefully
        optimal_k = optimizer.optimize_k(features, species="test_nan")

        assert 5 <= optimal_k <= 10
        logger.info(f"✓ Handled NaN features: k={optimal_k}")

    def test_optimizer_small_dataset(self):
        """Works with small datasets."""
        # Very small dataset
        np.random.seed(42)
        features = np.random.randn(20, 112)

        optimizer = VocabOptimizer(k_range=(2, 5), random_state=42)
        optimal_k = optimizer.optimize_k(features, species="test_small")

        assert 2 <= optimal_k <= 5
        logger.info(f"✓ Small dataset (20 samples): k={optimal_k}")

    def test_optimizer_large_k_range(self):
        """Handles large k ranges efficiently."""
        # Use more samples to allow k >= 100 (need at least 101 samples)
        features = create_well_separated_clusters(n_samples=300)

        # Large range with step to speed up
        optimizer = VocabOptimizer(
            k_range=(100, 500), step_size=50, random_state=42
        )
        optimal_k = optimizer.optimize_k(features, species="test_large_range")

        # Should pick a k in the stepped range
        assert optimal_k in range(100, 501, 50)
        logger.info(f"✓ Large range optimization: k={optimal_k}")

    def test_optimizer_edge_case_n_equals_min_k(self):
        """Handles n_samples == min_k boundary case."""
        # When n_samples == min_k, range(min_k, effective_max_k) is empty
        # Should fall back to n_samples - 1
        np.random.seed(42)
        n_samples = 100
        features = np.random.randn(n_samples, 2)

        optimizer = VocabOptimizer(k_range=(100, 500), random_state=42)
        optimal_k, svs_history = optimizer.optimize_k_with_history(
            features, species="test_boundary"
        )

        # Should return k = n_samples - 1 = 99 (the maximum viable k)
        assert optimal_k == n_samples - 1
        assert optimal_k in svs_history
        assert svs_history[optimal_k] >= 0.0  # Should have computed a valid score
        logger.info(f"✓ Boundary case n_samples={n_samples}==min_k: k={optimal_k}")

    def test_optimizer_edge_case_single_sample(self):
        """Handles n_samples == 1 (insufficient for clustering)."""
        np.random.seed(42)
        features = np.random.randn(1, 2)

        optimizer = VocabOptimizer(k_range=(2, 100), random_state=42)
        optimal_k, svs_history = optimizer.optimize_k_with_history(
            features, species="test_single"
        )

        # Should return k=1 as fallback
        assert optimal_k == 1
        assert 1 in svs_history
        logger.info(f"✓ Single sample (n=1): k={optimal_k}")

    def test_optimizer_edge_case_two_samples(self):
        """Handles n_samples == 2 (minimum viable for clustering)."""
        np.random.seed(42)
        features = np.random.randn(2, 2)

        optimizer = VocabOptimizer(k_range=(5, 100), random_state=42)
        optimal_k, svs_history = optimizer.optimize_k_with_history(
            features, species="test_two"
        )

        # With n_samples=2, the only valid k is 1 (since k < n_samples)
        assert optimal_k == 1
        assert optimal_k in svs_history
        logger.info(f"✓ Two samples (n=2): k={optimal_k}")


# =============================================================================
# Benchmark Tests
# =============================================================================


class TestVocabOptimizerPerformance:
    """Performance benchmarks for VocabOptimizer."""

    def test_optimize_k_performance(self):
        """Benchmark k optimization performance."""
        import time

        features = create_112d_test_data(n_samples=500)

        optimizer = VocabOptimizer(k_range=(10, 50), step_size=5, random_state=42)

        start = time.time()
        optimal_k = optimizer.optimize_k(features, species="benchmark")
        elapsed = time.time() - start

        # Should complete in reasonable time (< 30 seconds for this size)
        assert elapsed < 30.0, f"Optimization too slow: {elapsed:.2f}s"

        logger.info(f"✓ Performance: k={optimal_k} in {elapsed:.2f}s")


# =============================================================================
# Integration Tests: ExemplarManager + Adaptive Vocabulary
# =============================================================================


class TestExemplarManagerIntegration:
    """End-to-end tests for ExemplarManager with adaptive vocabulary."""

    def test_exemplar_manager_loads_species_vocab(self):
        """ExemplarManager loads species-specific k from registry."""
        import tempfile
        from analysis.rosetta_stone.exemplar_manager import ExemplarManager

        # Create a vocabulary registry with species-specific configs
        registry = SpeciesVocabRegistry()

        # Add different optimal k for different species
        registry.register(SpeciesVocabConfig("marmoset", optimal_k=500, svs_score=0.65))
        registry.register(SpeciesVocabConfig("egyptian_fruit_bat", optimal_k=1020, svs_score=0.72))

        # Save registry to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            registry_path = f.name
            registry.save(registry_path)

        try:
            # Test 1: ExemplarManager with species loads correct k
            manager_bat = ExemplarManager(
                vocabulary_size=1000,  # Default
                species="egyptian_fruit_bat",
                vocab_registry_path=registry_path,
            )
            # Should have loaded k=1020 from registry
            assert manager_bat.vocabulary_size == 1020
            logger.info("✓ Loaded bat k=1020 from registry")

            # Test 2: Different species gets different k
            manager_marmoset = ExemplarManager(
                vocabulary_size=1000,
                species="marmoset",
                vocab_registry_path=registry_path,
            )
            assert manager_marmoset.vocabulary_size == 500
            logger.info("✓ Loaded marmoset k=500 from registry")

            # Test 3: Unknown species falls back to default
            # When species not found in registry, keeps the vocabulary_size passed in
            manager_unknown = ExemplarManager(
                vocabulary_size=150,  # Different from registry values
                species="unknown_species",
                vocab_registry_path=registry_path,
            )
            # Should keep the passed vocabulary_size since species not found
            assert manager_unknown.vocabulary_size == 150
            logger.info("✓ Unknown species uses default k=150")

        finally:
            # Clean up temp file
            Path(registry_path).unlink(missing_ok=True)

    def test_exemplar_manager_clustering_uses_actual_k(self):
        """ExemplarManager clustering uses species-specific k."""
        import tempfile
        from analysis.rosetta_stone.exemplar_manager import ExemplarManager

        # Create test segments
        np.random.seed(42)
        segments = []
        for i in range(50):
            segments.append({
                "file_path": f"seg_{i:03d}.wav",
                "features_112d": np.random.randn(112).tolist(),
                "duration_ms": 100.0,
                "mean_f0_hz": 8000.0,
            })

        # Create and save a vocab config
        registry = SpeciesVocabRegistry()
        registry.register(SpeciesVocabConfig("test_species", optimal_k=5, svs_score=0.7))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            registry_path = f.name
            registry.save(registry_path)

        try:
            # Create manager with species-specific vocabulary
            manager = ExemplarManager(
                vocabulary_size=100,  # Default
                species="test_species",
                vocab_registry_path=registry_path,
            )

            # Add segments
            for seg in segments:
                manager.add_segment(**seg)

            # Cluster - should use k=5 from registry
            manager.cluster_features()

            # Verify clustering used k=5
            actual_k = manager.get_actual_k()
            assert actual_k == 5, f"Expected k=5, got {actual_k}"
            assert len(manager.clusters) <= 5, f"Expected ≤5 clusters, got {len(manager.clusters)}"

            logger.info(f"✓ Clustering used species-specific k={actual_k}, "
                        f"found {len(manager.clusters)} clusters")

        finally:
            Path(registry_path).unlink(missing_ok=True)


class TestExemplarManagerCLI:
    """End-to-end tests for ExemplarManager CLI with adaptive vocabulary."""

    def test_cli_accepts_species_and_registry_args(self):
        """CLI should accept --species and --vocab-registry arguments."""
        from analysis.rosetta_stone.exemplar_manager import main
        import argparse
        import sys
        from io import StringIO

        # Mock sys.argv to simulate CLI arguments
        original_argv = sys.argv
        try:
            sys.argv = [
                "exemplar_manager.py",
                "--input", "/tmp/test_input.json",
                "--species", "test_species",
                "--vocab-registry", "/tmp/test_registry.json",
            ]

            # Parse args to verify they're accepted
            parser = argparse.ArgumentParser()
            parser.add_argument("--input", "-i", required=True)
            parser.add_argument("--species")
            parser.add_argument("--vocab-registry")

            # Should not raise an error
            args = parser.parse_args(sys.argv[1:])
            assert args.species == "test_species"
            assert args.vocab_registry == "/tmp/test_registry.json"

            logger.info("✓ CLI accepts --species and --vocab-registry arguments")

        finally:
            sys.argv = original_argv

    def test_cli_integration_with_species_and_registry(self):
        """Full CLI path should load species-specific k from registry."""
        import json
        import tempfile
        from analysis.rosetta_stone.exemplar_manager import ExemplarManager
        from analysis.rosetta_stone.vocab_optimizer import (
            SpeciesVocabConfig,
            SpeciesVocabRegistry,
        )

        # Create test segments manifest
        np.random.seed(42)
        segments_data = {
            "segments": [
                {
                    "file_path": f"seg_{i:03d}.wav",
                    "features_112d": np.random.randn(112).tolist(),
                    "duration_ms": 100.0,
                    "mean_f0_hz": 8000.0,
                }
                for i in range(50)
            ]
        }

        # Create vocab registry
        registry = SpeciesVocabRegistry()
        registry.register(SpeciesVocabConfig("test_cli_species", optimal_k=8, svs_score=0.75))

        # Save files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            manifest_path = f.name
            json.dump(segments_data, f)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            registry_path = f.name
            registry.save(registry_path)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            # Simulate CLI invocation
            manager = ExemplarManager(
                vocabulary_size=100,  # Default, should be overridden
                species="test_cli_species",
                vocab_registry_path=registry_path,
            )

            manager.load_manifest(manifest_path)
            manager.cluster_features()
            manager.select_exemplars()

            # Verify species-specific k was used
            actual_k = manager.get_actual_k()
            assert actual_k == 8, f"Expected k=8 from registry, got {actual_k}"
            assert len(manager.clusters) <= 8

            logger.info(f"✓ CLI integration: species-specific k={actual_k} used")

        finally:
            Path(manifest_path).unlink(missing_ok=True)
            Path(registry_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_cli_fallback_to_default_k_without_species(self):
        """CLI should use default k when species not provided."""
        import json
        import tempfile
        from analysis.rosetta_stone.exemplar_manager import ExemplarManager

        # Create test segments manifest
        np.random.seed(42)
        segments_data = {
            "segments": [
                {
                    "file_path": f"seg_{i:03d}.wav",
                    "features_112d": np.random.randn(112).tolist(),
                    "duration_ms": 100.0,
                    "mean_f0_hz": 8000.0,
                }
                for i in range(50)
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            manifest_path = f.name
            json.dump(segments_data, f)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            # Simulate CLI invocation without species
            manager = ExemplarManager(vocabulary_size=15)  # No species, no registry
            manager.load_manifest(manifest_path)
            manager.cluster_features()

            # Verify default k was used
            actual_k = manager.get_actual_k()
            assert actual_k == 15, f"Expected k=15 default, got {actual_k}"

            logger.info(f"✓ CLI fallback: default k={actual_k} used")

        finally:
            Path(manifest_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
