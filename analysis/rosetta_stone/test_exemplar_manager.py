#!/usr/bin/env python3
"""
Tests for Exemplar Manager - Stage 3 of Synthesis Pipeline
===========================================================

TDD Tests for the Python-side of the Rust/Python bridge:
- Loading segments_manifest.json from Rust
- Clustering 112D features
- Selecting best exemplars (closest to centroid)
- Outputting clusters.json for Rust synthesis

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import pytest
import json
import tempfile
import numpy as np
from pathlib import Path

from analysis.rosetta_stone.exemplar_manager import (
    ExemplarManager,
    SegmentInfo,
    ClusterInfo
)


class TestSegmentInfo:
    """Tests for SegmentInfo dataclass."""

    def test_segment_info_creation(self):
        """Test basic SegmentInfo creation."""
        segment = SegmentInfo(
            file_path="test.wav",
            features_112d=[0.5] * 112,
            duration_ms=100.0,
            mean_f0_hz=8000.0
        )
        assert segment.file_path == "test.wav"
        assert len(segment.features_112d) == 112
        assert segment.cluster_id is None

    def test_segment_info_with_cluster_id(self):
        """Test SegmentInfo with cluster ID assigned."""
        segment = SegmentInfo(
            file_path="test.wav",
            features_112d=[0.5] * 112,
            duration_ms=100.0,
            mean_f0_hz=8000.0,
            cluster_id=42
        )
        assert segment.cluster_id == 42


class TestClusterInfo:
    """Tests for ClusterInfo dataclass."""

    def test_cluster_info_creation(self):
        """Test basic ClusterInfo creation."""
        cluster = ClusterInfo(
            cluster_id=0,
            centroid_112d=[0.5] * 112,
            exemplar_audio="seg_042.wav",
            exemplar_features_112d=[0.6] * 112,
            num_segments=15,
            mean_distance_to_centroid=0.85
        )
        assert cluster.cluster_id == 0
        assert len(cluster.centroid_112d) == 112
        assert cluster.exemplar_audio == "seg_042.wav"


class TestExemplarManager:
    """Tests for ExemplarManager class."""

    def test_manager_creation(self):
        """Test basic ExemplarManager creation."""
        manager = ExemplarManager(vocabulary_size=100)
        assert manager.vocabulary_size == 100
        assert len(manager.segments) == 0
        assert len(manager.clusters) == 0

    def test_add_segment(self):
        """Test adding segments manually."""
        manager = ExemplarManager(vocabulary_size=100)

        manager.add_segment(
            file_path="seg_001.wav",
            features_112d=[0.5] * 112,
            duration_ms=100.0,
            mean_f0_hz=8000.0
        )

        assert len(manager.segments) == 1
        assert manager.segments[0].file_path == "seg_001.wav"
        assert manager.segments[0].cluster_id is None

    def test_add_multiple_segments(self):
        """Test adding multiple segments."""
        manager = ExemplarManager(vocabulary_size=100)

        for i in range(10):
            manager.add_segment(
                file_path=f"seg_{i:03d}.wav",
                features_112d=[0.5 + i * 0.01] * 112,
                duration_ms=100.0 + i * 10,
                mean_f0_hz=8000.0 + i * 100
            )

        assert len(manager.segments) == 10

    def test_load_manifest(self):
        """Test loading segments from JSON manifest."""
        manager = ExemplarManager(vocabulary_size=100)

        # Create test manifest
        manifest_data = {
            "version": "1.0",
            "sample_rate": 44100,
            "segments": [
                {
                    "file_path": "seg_001.wav",
                    "features_112d": [0.5] * 112,
                    "duration_ms": 100.0,
                    "mean_f0_hz": 8000.0
                },
                {
                    "file_path": "seg_002.wav",
                    "features_112d": [0.6] * 112,
                    "duration_ms": 150.0,
                    "mean_f0_hz": 8500.0
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(manifest_data, f)
            manifest_path = f.name

        try:
            count = manager.load_manifest(manifest_path)
            assert count == 2
            assert len(manager.segments) == 2
            assert manager.segments[0].file_path == "seg_001.wav"
        finally:
            Path(manifest_path).unlink()

    def test_cluster_features_small(self):
        """Test clustering with small dataset."""
        manager = ExemplarManager(vocabulary_size=3)

        # Add 10 segments in 3 distinct clusters
        for i in range(10):
            cluster_offset = (i % 3) * 0.2
            manager.add_segment(
                file_path=f"seg_{i:03d}.wav",
                features_112d=[0.5 + cluster_offset] * 112,
                duration_ms=100.0,
                mean_f0_hz=8000.0
            )

        manager.cluster_features(k=3)

        # All segments should have cluster IDs
        for segment in manager.segments:
            assert segment.cluster_id is not None
            assert 0 <= segment.cluster_id < 3

    def test_select_exemplars(self):
        """Test exemplar selection after clustering."""
        manager = ExemplarManager(vocabulary_size=3)

        # Add segments
        for i in range(15):
            cluster_offset = (i % 3) * 0.3
            manager.add_segment(
                file_path=f"seg_{i:03d}.wav",
                features_112d=[0.5 + cluster_offset + (i // 3) * 0.01] * 112,
                duration_ms=100.0,
                mean_f0_hz=8000.0
            )

        manager.cluster_features(k=3)
        clusters = manager.select_exemplars()

        assert len(clusters) == 3
        for cluster_id, cluster_info in clusters.items():
            assert cluster_info.exemplar_audio is not None
            assert cluster_info.num_segments > 0
            assert len(cluster_info.centroid_112d) == 112

    def test_get_exemplar_for_cluster(self):
        """Test retrieving exemplar by cluster ID."""
        manager = ExemplarManager(vocabulary_size=3)

        # Add and cluster segments
        for i in range(9):
            manager.add_segment(
                file_path=f"seg_{i:03d}.wav",
                features_112d=[0.5 + (i % 3) * 0.2] * 112,
                duration_ms=100.0,
                mean_f0_hz=8000.0
            )

        manager.cluster_features(k=3)
        manager.select_exemplars()

        # Get exemplar for cluster 0
        exemplar = manager.get_exemplar_for_cluster(0)
        assert exemplar is not None
        assert exemplar.cluster_id == 0

        # Non-existent cluster
        assert manager.get_exemplar_for_cluster(999) is None

    def test_get_exemplar_audio_path(self):
        """Test getting audio path for cluster exemplar."""
        manager = ExemplarManager(vocabulary_size=3)

        for i in range(9):
            manager.add_segment(
                file_path=f"seg_{i:03d}.wav",
                features_112d=[0.5 + (i % 3) * 0.2] * 112,
                duration_ms=100.0,
                mean_f0_hz=8000.0
            )

        manager.cluster_features(k=3)
        manager.select_exemplars()

        # Get audio path for cluster 0
        path = manager.get_exemplar_audio_path(0)
        assert path is not None
        assert path.endswith('.wav')

    def test_save_and_load_exemplars(self):
        """Test saving and loading exemplar data."""
        manager = ExemplarManager(vocabulary_size=3)

        # Create and cluster
        for i in range(9):
            manager.add_segment(
                file_path=f"seg_{i:03d}.wav",
                features_112d=[0.5 + (i % 3) * 0.2] * 112,
                duration_ms=100.0,
                mean_f0_hz=8000.0
            )

        manager.cluster_features(k=3)
        manager.select_exemplars()

        # Save
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name

        try:
            manager.save_exemplars(output_path)

            # Load into new manager
            new_manager = ExemplarManager()
            new_manager.load_exemplars(output_path)

            assert len(new_manager.clusters) == 3
            assert 0 in new_manager.clusters
            assert new_manager.clusters[0].exemplar_audio is not None
        finally:
            Path(output_path).unlink()

    def test_create_synthesis_manifest(self):
        """Test creating Rust-compatible synthesis manifest."""
        manager = ExemplarManager(vocabulary_size=3)

        for i in range(9):
            manager.add_segment(
                file_path=f"seg_{i:03d}.wav",
                features_112d=[0.5 + (i % 3) * 0.2] * 112,
                duration_ms=100.0,
                mean_f0_hz=8000.0
            )

        manager.cluster_features(k=3)
        manager.select_exemplars()

        # Create synthesis manifest
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name

        try:
            manager.create_synthesis_manifest(output_path)

            # Load and verify
            with open(output_path, 'r') as f:
                data = json.load(f)

            assert 'vocabulary_size' in data
            assert 'exemplars' in data
            assert len(data['exemplars']) == 3

            # Check exemplar structure
            exemplar = data['exemplars'][0]
            assert 'cluster_id' in exemplar
            assert 'audio_path' in exemplar
            assert 'metadata' in exemplar
        finally:
            Path(output_path).unlink()


class TestExemplarManagerIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self):
        """Test the full Stage 3 pipeline: load -> cluster -> select -> save."""
        manager = ExemplarManager(vocabulary_size=5)

        # Create test manifest
        manifest_data = {
            "version": "1.0",
            "sample_rate": 44100,
            "segments": [
                {
                    "file_path": f"seg_{i:03d}.wav",
                    "features_112d": [0.5 + (i % 5) * 0.15 + np.random.randn() * 0.01] * 112,
                    "duration_ms": 100.0 + i * 10,
                    "mean_f0_hz": 8000.0 + i * 100
                }
                for i in range(25)  # 25 segments, 5 clusters
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "segments_manifest.json"
            clusters_path = Path(tmpdir) / "clusters.json"
            synthesis_path = Path(tmpdir) / "synthesis_manifest.json"

            # Write manifest
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f)

            # Run pipeline
            manager.load_manifest(str(manifest_path))
            manager.cluster_features(k=5)
            manager.select_exemplars()
            manager.save_exemplars(str(clusters_path))
            manager.create_synthesis_manifest(str(synthesis_path))

            # Verify outputs exist
            assert clusters_path.exists()
            assert synthesis_path.exists()

            # Verify clusters.json structure
            with open(clusters_path, 'r') as f:
                clusters_data = json.load(f)

            assert clusters_data['vocabulary_size'] == 5
            assert len(clusters_data['clusters']) == 5

            # Verify synthesis_manifest.json structure
            with open(synthesis_path, 'r') as f:
                synth_data = json.load(f)

            assert synth_data['vocabulary_size'] == 5
            assert len(synth_data['exemplars']) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
