#!/usr/bin/env python3
"""
Tests for Hybrid Persona Architecture in Universal Rosetta Stone

Tests the 3-tier architecture:
- Tier 1: Unsupervised DBSCAN clustering
- Tier 2: Acoustic persona mapping (post-hoc)
- Tier 3: Contextual validation (deferred)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.rosetta_stone.universal_rosetta_stone import (
    Modality,
    PhraseSignature,
    UniversalRosettaStone,
)

# Check if persona support is available
try:
    from analysis.rosetta_stone.acoustic_similarity_for_atomic_phrase_candidates import (
        ACOUSTIC_PERSONAS,
    )

    HAS_PERSONA_SUPPORT = True
except ImportError:
    HAS_PERSONA_SUPPORT = False


@pytest.fixture
def sample_rate():
    """Standard sample rate for tests."""
    return 48000


@pytest.fixture
def rosetta_stone(sample_rate):
    """Create UniversalRosettaStone instance for testing."""
    return UniversalRosettaStone(sample_rate=sample_rate)


@pytest.fixture
def harmonic_phrases(sample_rate):
    """
    Create synthetic harmonic phrases with different characteristics.

    Returns phrases with varying HNR, attack times, and spectral flatness
    to test persona detection.
    """
    phrases = []

    # 1. PURE-like phrase (high HNR, low flatness, slow attack)
    duration = int(0.2 * sample_rate)
    t = np.linspace(0, 0.2, duration)
    pure_audio = 0.5 * np.sin(2 * np.pi * 7000 * t)
    # Slow attack envelope
    attack_samples = 50
    attack_envelope = np.linspace(0, 1, attack_samples)
    sustain_envelope = np.ones(duration - attack_samples)
    pure_audio *= np.concatenate([attack_envelope, sustain_envelope])

    pure_phrase = PhraseSignature(
        Modality.HARMONIC, pure_audio, timestamp=0.0, sample_rate=sample_rate
    )
    # Manually set features to match PURE persona
    pure_phrase.features = {
        "f0_mean": 7000.0,
        "f0_std": 50.0,
        "f0_range": 100.0,
        "harmonicity": 0.95,
        "harmonic_to_noise_ratio": 25.0,  # High HNR
        "spectral_flatness": 0.1,  # Low flatness
        "attack_time_ms": 25.0,  # Slow attack
        "decay_time_ms": 80.0,
        "duration_ms": 200.0,
    }
    phrases.append(pure_phrase)

    # 2. GRITTY-like phrase (low HNR, high flatness, fast attack)
    gritty_audio = np.random.randn(duration) * 0.3  # Noisy
    gritty_audio[:5] *= np.linspace(0, 1, 5)  # Very fast attack

    gritty_phrase = PhraseSignature(
        Modality.HARMONIC, gritty_audio, timestamp=0.5, sample_rate=sample_rate
    )
    gritty_phrase.features = {
        "f0_mean": 6500.0,
        "f0_std": 200.0,
        "f0_range": 500.0,
        "harmonicity": 0.3,
        "harmonic_to_noise_ratio": 2.0,  # Low HNR
        "spectral_flatness": 0.6,  # High flatness
        "attack_time_ms": 5.0,  # Fast attack
        "decay_time_ms": 20.0,
        "duration_ms": 200.0,
    }
    phrases.append(gritty_phrase)

    # 3. Another PURE-like phrase (should cluster with first)
    pure2_audio = 0.5 * np.sin(2 * np.pi * 7100 * t)
    attack_samples = 60
    attack_envelope = np.linspace(0, 1, attack_samples)
    sustain_envelope = np.ones(duration - attack_samples)
    pure2_audio *= np.concatenate([attack_envelope, sustain_envelope])

    pure2_phrase = PhraseSignature(
        Modality.HARMONIC, pure2_audio, timestamp=1.0, sample_rate=sample_rate
    )
    pure2_phrase.features = {
        "f0_mean": 7100.0,
        "f0_std": 45.0,
        "f0_range": 90.0,
        "harmonicity": 0.93,
        "harmonic_to_noise_ratio": 22.0,
        "spectral_flatness": 0.12,
        "attack_time_ms": 28.0,
        "decay_time_ms": 75.0,
        "duration_ms": 200.0,
    }
    phrases.append(pure2_phrase)

    # 4. SHARP-like phrase (fast attack, fast decay, high spectral contrast)
    sharp_audio = np.random.randn(duration // 2) * 0.5
    sharp_envelope = np.concatenate(
        [
            np.linspace(0, 1, 10),  # Fast attack
            np.linspace(1, 0, 30),  # Fast decay
        ]
    )
    sharp_audio = sharp_audio[:40] * sharp_envelope

    sharp_phrase = PhraseSignature(
        Modality.HARMONIC, sharp_audio, timestamp=1.5, sample_rate=sample_rate
    )
    sharp_phrase.features = {
        "f0_mean": 8000.0,
        "f0_std": 100.0,
        "f0_range": 200.0,
        "harmonicity": 0.7,
        "harmonic_to_noise_ratio": 8.0,
        "spectral_flatness": 0.25,
        "attack_time_ms": 3.0,  # Very fast attack
        "decay_time_ms": 15.0,  # Fast decay
        "spectral_contrast": 18.0,  # High spectral contrast
        "duration_ms": 80.0,
    }
    phrases.append(sharp_phrase)

    return phrases


class TestHybridPersonaArchitecture:
    """Test suite for hybrid persona architecture."""

    @pytest.mark.skipif(not HAS_PERSONA_SUPPORT, reason="Persona support not available")
    def test_compute_cluster_persona_score_pure(self, rosetta_stone, harmonic_phrases):
        """Test persona scoring for PURE-like cluster."""

        # Get the two PURE-like phrases
        pure_phrases = [harmonic_phrases[0], harmonic_phrases[2]]

        # Compute PURE persona score
        score = rosetta_stone.compute_cluster_persona_score(pure_phrases, "pure")

        # Should have moderate-to-high score for PURE persona
        assert score > 0.15, f"Expected moderate PURE score, got {score}"

    @pytest.mark.skipif(not HAS_PERSONA_SUPPORT, reason="Persona support not available")
    def test_compute_cluster_persona_score_gritty(self, rosetta_stone, harmonic_phrases):
        """Test persona scoring for GRITTY-like cluster."""

        gritty_phrases = [harmonic_phrases[1]]

        score = rosetta_stone.compute_cluster_persona_score(gritty_phrases, "gritty")

        # Should have moderate score for GRITTY persona
        assert score > 0.3, f"Expected moderate GRITTY score, got {score}"

    def test_compute_cluster_persona_score_invalid_persona(self, rosetta_stone, harmonic_phrases):
        """Test that invalid persona returns 0."""
        score = rosetta_stone.compute_cluster_persona_score(harmonic_phrases, "invalid_persona")
        assert score == 0.0

    def test_build_vocabulary_with_personas_structure(self, rosetta_stone, harmonic_phrases):
        """Test that hybrid vocabulary has correct structure."""
        clusters = rosetta_stone.build_vocabulary_with_personas(
            harmonic_phrases, eps=0.5, min_samples=1, enable_persona_mapping=HAS_PERSONA_SUPPORT
        )

        # Should return a dictionary
        assert isinstance(clusters, dict)

        # Each cluster should have required keys
        for cluster_id, cluster_data in clusters.items():
            assert "phrases" in cluster_data
            assert "dominant_persona" in cluster_data
            assert "persona_scores" in cluster_data
            assert "cluster_size" in cluster_data
            assert "mean_features" in cluster_data

            # Verify data types
            assert isinstance(cluster_data["phrases"], list)
            assert isinstance(cluster_data["dominant_persona"], str)
            assert isinstance(cluster_data["persona_scores"], dict)
            assert isinstance(cluster_data["cluster_size"], int)
            assert isinstance(cluster_data["mean_features"], dict)

    def test_build_vocabulary_with_personas_clustering(self, rosetta_stone, harmonic_phrases):
        """Test that DBSCAN clustering still works with personas."""
        clusters = rosetta_stone.build_vocabulary_with_personas(
            harmonic_phrases, eps=0.5, min_samples=1
        )

        # Should have at least one cluster
        assert len(clusters) >= 1

        # Total phrases across all clusters should equal input
        total_phrases = sum(c["cluster_size"] for c in clusters.values())
        assert total_phrases == len(harmonic_phrases)

    @pytest.mark.skipif(not HAS_PERSONA_SUPPORT, reason="Persona support not available")
    def test_find_phrases_by_persona(self, rosetta_stone, harmonic_phrases):
        """Test semantic phrase search by persona."""
        # Build vocabulary with personas
        clusters = rosetta_stone.build_vocabulary_with_personas(
            harmonic_phrases, eps=0.5, min_samples=1, enable_persona_mapping=True
        )

        # Find PURE phrases
        pure_matches = rosetta_stone.find_phrases_by_persona(clusters, "pure", min_score=0.3)

        # Should return list of tuples
        assert isinstance(pure_matches, list)

        # Each match should be (cluster_id, phrases, score) tuple
        for match in pure_matches:
            assert len(match) == 3
            assert isinstance(match[0], int)  # cluster_id
            assert isinstance(match[1], list)  # phrases
            assert isinstance(match[2], (int, float))  # score

    @pytest.mark.skipif(not HAS_PERSONA_SUPPORT, reason="Persona support not available")
    def test_find_phrases_by_persona_threshold(self, rosetta_stone, harmonic_phrases):
        """Test that min_score threshold filters results correctly."""
        clusters = rosetta_stone.build_vocabulary_with_personas(
            harmonic_phrases, eps=0.5, min_samples=1
        )

        # Higher threshold should return fewer or equal results
        matches_low = rosetta_stone.find_phrases_by_persona(clusters, "pure", min_score=0.1)
        matches_high = rosetta_stone.find_phrases_by_persona(clusters, "pure", min_score=0.5)

        assert len(matches_high) <= len(matches_low)

    @pytest.mark.skipif(not HAS_PERSONA_SUPPORT, reason="Persona support not available")
    def test_get_persona_summary(self, rosetta_stone, harmonic_phrases):
        """Test persona summary generation."""
        clusters = rosetta_stone.build_vocabulary_with_personas(
            harmonic_phrases, eps=0.5, min_samples=1
        )

        summary = rosetta_stone.get_persona_summary(clusters)

        # Should return a dictionary
        assert isinstance(summary, dict)

        # Should have entries for all personas
        if HAS_PERSONA_SUPPORT:
            expected_personas = list(ACOUSTIC_PERSONAS.keys()) + ["unclassified"]
            for persona in expected_personas:
                assert persona in summary
                assert "cluster_count" in summary[persona]
                assert "total_phrases" in summary[persona]
                assert "avg_score" in summary[persona]

    @pytest.mark.skipif(not HAS_PERSONA_SUPPORT, reason="Persona support not available")
    def test_get_persona_summary_aggregation(self, rosetta_stone, harmonic_phrases):
        """Test that persona summary correctly aggregates cluster data."""
        clusters = rosetta_stone.build_vocabulary_with_personas(
            harmonic_phrases, eps=0.5, min_samples=1
        )

        summary = rosetta_stone.get_persona_summary(clusters)

        # Total clusters in summary should match input
        total_summary_clusters = sum(s["cluster_count"] for s in summary.values())
        assert total_summary_clusters == len(clusters)

    def test_persona_mapping_disabled(self, rosetta_stone, harmonic_phrases):
        """Test that persona mapping can be disabled."""
        clusters = rosetta_stone.build_vocabulary_with_personas(
            harmonic_phrases, eps=0.5, min_samples=1, enable_persona_mapping=False
        )

        # All clusters should be 'unclassified'
        for cluster_data in clusters.values():
            assert cluster_data["dominant_persona"] == "unclassified"
            assert cluster_data["persona_scores"] == {}

    def test_backward_compatibility_build_vocabulary(self, rosetta_stone, harmonic_phrases):
        """Test that original build_vocabulary() still works."""
        # Original method should still work
        clusters = rosetta_stone.build_vocabulary(harmonic_phrases, eps=0.5, min_samples=1)

        # Should return dictionary
        assert isinstance(clusters, dict)

    def test_empty_phrases_list(self, rosetta_stone):
        """Test behavior with empty phrases list."""
        clusters = rosetta_stone.build_vocabulary_with_personas([], eps=0.5, min_samples=1)

        assert clusters == {}

    def test_single_phrase(self, rosetta_stone, harmonic_phrases):
        """Test behavior with single phrase (below min_samples)."""
        single_phrase = [harmonic_phrases[0]]

        # With min_samples=2, single phrase won't form a cluster
        clusters = rosetta_stone.build_vocabulary_with_personas(
            single_phrase, eps=0.5, min_samples=2
        )

        # DBSCAN requires min_samples, so should be empty
        assert len(clusters) == 0


class TestPersonaFeatureExtraction:
    """Test that micro-dynamics features are available for persona scoring."""

    def test_harmonic_phrase_features(self, rosetta_stone, sample_rate):
        """Test that harmonic phrases have required features."""
        duration = int(0.1 * sample_rate)
        audio = 0.5 * np.sin(2 * np.pi * 7000 * np.linspace(0, 0.1, duration))

        phrase = PhraseSignature(Modality.HARMONIC, audio, timestamp=0.0, sample_rate=sample_rate)

        # Should have extracted features
        assert isinstance(phrase.features, dict)
        assert len(phrase.features) > 0

        # Should have fundamental frequency features
        assert "f0_mean" in phrase.features or "mean_freq" in phrase.features


class TestIntegration:
    """Integration tests for hybrid architecture."""

    @pytest.mark.skipif(not HAS_PERSONA_SUPPORT, reason="Persona support not available")
    def test_full_workflow(self, rosetta_stone, harmonic_phrases):
        """Test complete workflow: cluster -> persona search -> summary."""
        # Step 1: Build vocabulary with personas
        clusters = rosetta_stone.build_vocabulary_with_personas(
            harmonic_phrases, eps=0.5, min_samples=1
        )

        # Step 2: Search for specific persona
        matches = rosetta_stone.find_phrases_by_persona(clusters, "pure", min_score=0.3)

        # Step 3: Generate summary
        summary = rosetta_stone.get_persona_summary(clusters)

        # Verify workflow completed successfully
        assert len(clusters) >= 1
        assert isinstance(matches, list)
        assert isinstance(summary, dict)

    @pytest.mark.skipif(not HAS_PERSONA_SUPPORT, reason="Persona support not available")
    def test_cross_species_persona_consistency(self, rosetta_stone, sample_rate):
        """Test that personas work across different phrase characteristics."""
        # Create phrases with different F0 but similar persona characteristics
        phrases = []

        for f0 in [5000, 7000, 9000]:
            duration = int(0.1 * sample_rate)
            t = np.linspace(0, 0.1, duration)
            audio = 0.5 * np.sin(2 * np.pi * f0 * t)

            phrase = PhraseSignature(
                Modality.HARMONIC, audio, timestamp=float(f0), sample_rate=sample_rate
            )
            # Same persona features, different F0
            phrase.features = {
                "f0_mean": float(f0),
                "f0_std": 50.0,
                "f0_range": 100.0,
                "harmonic_to_noise_ratio": 20.0,
                "spectral_flatness": 0.12,
                "attack_time_ms": 25.0,
                "decay_time_ms": 75.0,
                "duration_ms": 100.0,
            }
            phrases.append(phrase)

        # Build vocabulary - should cluster by persona despite different F0
        clusters = rosetta_stone.build_vocabulary_with_personas(
            phrases,
            eps=1.0,  # Higher eps to allow F0 variation
            min_samples=1,
        )

        # With high eps, should form clusters based on persona features
        assert len(clusters) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
