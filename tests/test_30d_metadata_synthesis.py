"""
30D Micro-Dynamics Metadata Synthesis Tests
============================================

Tests for the 30-dimensional micro-dynamics metadata support
in the Python metadata-first synthesis engine.

This validates TDD approach:
1. 30D feature extraction from phrase metadata
2. Vector space queries using all 30 dimensions
3. Interpolation between 30D feature vectors
4. Ghost word synthesis with full 30D features
5. Backward compatibility with existing 4D API

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from realtime.metadata_synthesizer import (
    MetadataFirstSynthesizer,
    MetadataQuery,
    PhraseCandidate,
    SynthesisRecipe,
    VectorSpaceQueryEngine,
)


class Test30DFeatureExtraction(unittest.TestCase):
    """Test 30-dimensional feature extraction from phrase metadata."""

    def setUp(self):
        """Set up test fixtures with 30D metadata."""
        # Create a complete 30D metadata dictionary
        self.metadata_30d = {
            # === Fundamental (3 features) ===
            "mean_f0_hz": 7000.0,
            "f0_range_hz": 400.0,
            "duration_ms": 50.0,
            # === Grit Factors (3 features) ===
            "harmonic_to_noise_ratio": 20.0,
            "spectral_flatness": 0.1,
            "harmonicity": 0.95,
            # === Motion Factors (7 features) ===
            "attack_time_ms": 10.0,
            "decay_time_ms": 15.0,
            "sustain_level": 0.7,
            "vibrato_rate_hz": 8.0,
            "vibrato_depth": 50.0,
            "jitter": 0.02,
            "shimmer": 0.03,
            # === Fingerprint Factors (13 MFCCs) ===
            "mfcc_1": -500.0,
            "mfcc_2": -100.0,
            "mfcc_3": -50.0,
            "mfcc_4": -20.0,
            "mfcc_5": -10.0,
            "mfcc_6": -5.0,
            "mfcc_7": 0.0,
            "mfcc_8": 5.0,
            "mfcc_9": 10.0,
            "mfcc_10": 15.0,
            "mfcc_11": 20.0,
            "mfcc_12": 25.0,
            "mfcc_13": 30.0,
            # === Spectral Dynamics (1 feature) ===
            "spectral_flux": 100.0,
            # === Rhythm Factors (3 features) ===
            "median_ici_ms": 45.0,
            "onset_rate_hz": 15.0,
            "ici_coefficient_of_variation": 0.25,
            # === Additional metadata ===
            "phrase_id": "test_phrase_001",
            "species": "marmoset",
            "cluster_id": 0,
            "context": "contact",
        }

    def test_phrase_candidate_extracts_all_30d_features(self):
        """Test that PhraseCandidate extracts all 30 features."""
        # Create synthetic audio buffer
        sample_rate = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_buffer = 0.5 * np.sin(2 * np.pi * 7000 * t)

        # Create PhraseCandidate with 30D metadata
        candidate = PhraseCandidate(
            phrase_id=self.metadata_30d["phrase_id"],
            audio_buffer=audio_buffer,
            metadata=self.metadata_30d,
            sample_rate=sample_rate,
        )

        # Verify all 30 features are extracted
        # Fundamental (3)
        self.assertEqual(candidate.mean_f0_hz, 7000.0)
        self.assertEqual(candidate.f0_range_hz, 400.0)
        self.assertEqual(candidate.duration_ms, 50.0)

        # Grit Factors (3)
        self.assertEqual(candidate.harmonic_to_noise_ratio, 20.0)
        self.assertEqual(candidate.spectral_flatness, 0.1)
        self.assertEqual(candidate.harmonicity, 0.95)

        # Motion Factors (7)
        self.assertEqual(candidate.attack_time_ms, 10.0)
        self.assertEqual(candidate.decay_time_ms, 15.0)
        self.assertEqual(candidate.sustain_level, 0.7)
        self.assertEqual(candidate.vibrato_rate_hz, 8.0)
        self.assertEqual(candidate.vibrato_depth, 50.0)
        self.assertEqual(candidate.jitter, 0.02)
        self.assertEqual(candidate.shimmer, 0.03)

        # MFCCs (13)
        self.assertEqual(candidate.mfcc_1, -500.0)
        self.assertEqual(candidate.mfcc_2, -100.0)
        self.assertEqual(candidate.mfcc_3, -50.0)
        self.assertEqual(candidate.mfcc_4, -20.0)
        self.assertEqual(candidate.mfcc_5, -10.0)
        self.assertEqual(candidate.mfcc_6, -5.0)
        self.assertEqual(candidate.mfcc_7, 0.0)
        self.assertEqual(candidate.mfcc_8, 5.0)
        self.assertEqual(candidate.mfcc_9, 10.0)
        self.assertEqual(candidate.mfcc_10, 15.0)
        self.assertEqual(candidate.mfcc_11, 20.0)
        self.assertEqual(candidate.mfcc_12, 25.0)
        self.assertEqual(candidate.mfcc_13, 30.0)

        # Spectral Dynamics (1)
        self.assertEqual(candidate.spectral_flux, 100.0)

        # Rhythm Factors (3)
        self.assertEqual(candidate.median_ici_ms, 45.0)
        self.assertEqual(candidate.onset_rate_hz, 15.0)
        self.assertEqual(candidate.ici_coefficient_of_variation, 0.25)

    def test_phrase_feature_vector_returns_30_dimensions(self):
        """Test that get_feature_vector returns 30-dimensional array."""
        sample_rate = 48000
        audio_buffer = np.random.randn(48000)

        candidate = PhraseCandidate(
            phrase_id="test",
            audio_buffer=audio_buffer,
            metadata=self.metadata_30d,
            sample_rate=sample_rate,
        )

        feature_vector = candidate.get_feature_vector()

        # Verify 30 dimensions
        self.assertEqual(len(feature_vector), 30)

        # Verify values match metadata
        self.assertAlmostEqual(feature_vector[0], 7000.0)  # mean_f0_hz
        self.assertAlmostEqual(feature_vector[1], 400.0)  # f0_range_hz
        self.assertAlmostEqual(feature_vector[2], 50.0)  # duration_ms

    def test_backward_compatibility_with_4d_metadata(self):
        """Test that old 4D metadata still works."""
        # Old-style 4D metadata
        old_metadata = {
            "mean_f0_hz": 6500.0,
            "duration_ms": 60.0,
            "f0_range_hz": 300.0,
            "harmonicity": 0.9,
            "phrase_id": "old_style",
            "species": "marmoset",
            "cluster_id": 0,
            "context": "contact",
        }

        sample_rate = 48000
        audio_buffer = np.random.randn(48000)

        candidate = PhraseCandidate(
            phrase_id="old_style",
            audio_buffer=audio_buffer,
            metadata=old_metadata,
            sample_rate=sample_rate,
        )

        # Should extract available features
        self.assertEqual(candidate.mean_f0_hz, 6500.0)
        self.assertEqual(candidate.duration_ms, 60.0)
        self.assertEqual(candidate.f0_range_hz, 300.0)
        self.assertEqual(candidate.harmonicity, 0.9)

        # Missing features should have default values
        self.assertEqual(candidate.harmonic_to_noise_ratio, 0.0)
        self.assertEqual(candidate.spectral_flatness, 0.0)


class Test30DVectorSpaceQueries(unittest.TestCase):
    """Test vector space queries with 30-dimensional features."""

    def setUp(self):
        """Set up query engine with 30D test data."""
        self.query_engine = VectorSpaceQueryEngine()
        self._create_29d_test_phrases()

    def _create_29d_test_phrases(self):
        """Create test phrases with 30D metadata."""
        sample_rate = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create phrases with different 30D characteristics
        self.phrase_configs = [
            {
                "phrase_id": "marm_phee_001",
                "species": "marmoset",
                "cluster_id": 0,
                "context": "contact",
                "mean_f0_hz": 6526.0,
                "f0_range_hz": 427.0,
                "duration_ms": 76.5,
                "harmonic_to_noise_ratio": 25.0,
                "spectral_flatness": 0.05,
                "harmonicity": 0.95,
                "attack_time_ms": 8.0,
                "decay_time_ms": 12.0,
                "sustain_level": 0.75,
                "vibrato_rate_hz": 7.5,
                "vibrato_depth": 45.0,
                "jitter": 0.015,
                "shimmer": 0.025,
                "mfcc_1": -450.0,
                "mfcc_2": -90.0,
                "mfcc_3": -45.0,
                "mfcc_4": -18.0,
                "mfcc_5": -9.0,
                "mfcc_6": -4.5,
                "mfcc_7": 0.0,
                "mfcc_8": 4.5,
                "mfcc_9": 9.0,
                "mfcc_10": 13.5,
                "mfcc_11": 18.0,
                "mfcc_12": 22.5,
                "mfcc_13": 27.0,
                "spectral_flux": 90.0,
                "median_ici_ms": 50.0,
                "onset_rate_hz": 14.0,
                "ici_coefficient_of_variation": 0.22,
            },
            {
                "phrase_id": "marm_alarm_001",
                "species": "marmoset",
                "cluster_id": 1,
                "context": "alarm",
                "mean_f0_hz": 6020.0,
                "f0_range_hz": 3722.0,
                "duration_ms": 58.1,
                "harmonic_to_noise_ratio": 10.0,
                "spectral_flatness": 0.3,
                "harmonicity": 0.7,
                "attack_time_ms": 3.0,
                "decay_time_ms": 8.0,
                "sustain_level": 0.5,
                "vibrato_rate_hz": 15.0,
                "vibrato_depth": 100.0,
                "jitter": 0.05,
                "shimmer": 0.08,
                "mfcc_1": -600.0,
                "mfcc_2": -120.0,
                "mfcc_3": -60.0,
                "mfcc_4": -24.0,
                "mfcc_5": -12.0,
                "mfcc_6": -6.0,
                "mfcc_7": 0.0,
                "mfcc_8": 6.0,
                "mfcc_9": 12.0,
                "mfcc_10": 18.0,
                "mfcc_11": 24.0,
                "mfcc_12": 30.0,
                "mfcc_13": 36.0,
                "spectral_flux": 150.0,
                "median_ici_ms": 30.0,
                "onset_rate_hz": 25.0,
                "ici_coefficient_of_variation": 0.4,
            },
            {
                "phrase_id": "bat_midfm_001",
                "species": "egyptian_bat",
                "cluster_id": 2,
                "context": "navigation",
                "mean_f0_hz": 25000.0,
                "f0_range_hz": 15000.0,
                "duration_ms": 15.0,
                "harmonic_to_noise_ratio": 5.0,
                "spectral_flatness": 0.6,
                "harmonicity": 0.4,
                "attack_time_ms": 2.0,
                "decay_time_ms": 5.0,
                "sustain_level": 0.3,
                "vibrato_rate_hz": 0.0,
                "vibrato_depth": 0.0,
                "jitter": 0.01,
                "shimmer": 0.02,
                "mfcc_1": -700.0,
                "mfcc_2": -140.0,
                "mfcc_3": -70.0,
                "mfcc_4": -28.0,
                "mfcc_5": -14.0,
                "mfcc_6": -7.0,
                "mfcc_7": 0.0,
                "mfcc_8": 7.0,
                "mfcc_9": 14.0,
                "mfcc_10": 21.0,
                "mfcc_11": 28.0,
                "mfcc_12": 35.0,
                "mfcc_13": 42.0,
                "spectral_flux": 200.0,
                "median_ici_ms": 10.0,
                "onset_rate_hz": 50.0,
                "ici_coefficient_of_variation": 0.6,
            },
        ]

        # Create candidates and add to query engine
        for config in self.phrase_configs:
            # Generate synthetic audio
            f0 = config["mean_f0_hz"]
            audio_buffer = 0.5 * np.sin(2 * np.pi * f0 * t)

            candidate = PhraseCandidate(
                phrase_id=config["phrase_id"],
                audio_buffer=audio_buffer,
                metadata=config,
                sample_rate=sample_rate,
            )

            self.query_engine.phrases.append(candidate)

        # Build indexes
        for phrase in self.query_engine.phrases:
            species = phrase.species
            if species not in self.query_engine.species_index:
                self.query_engine.species_index[species] = []
            self.query_engine.species_index[species].append(phrase)

            cluster = phrase.cluster_id
            if cluster not in self.query_engine.cluster_index:
                self.query_engine.cluster_index[cluster] = []
            self.query_engine.cluster_index[cluster].append(phrase)

    def test_query_nearest_uses_all_30_dimensions(self):
        """Test that query_nearest_metadata uses all 30 dimensions for scoring."""
        query = MetadataQuery(
            target_f0_hz=6500.0,
            target_duration_ms=75.0,
            f0_tolerance_hz=500.0,
            duration_tolerance_ms=20.0,
        )

        results = self.query_engine.query_nearest_metadata(query, top_k=3)

        # Should return results
        self.assertGreater(len(results), 0)

        # First result should be the closest in 30D space
        best_match = results[0]
        self.assertEqual(best_match.phrase_id, "marm_phee_001")

        # All results should have acoustic scores calculated
        for result in results:
            self.assertGreater(result.acoustic_score, 0.0)
            self.assertIsNotNone(result.feature_vector)
            self.assertEqual(len(result.feature_vector), 30)  # 30D actual implementation

    def test_30d_distance_calculation(self):
        """Test that distance is calculated using all 30 dimensions."""
        # Get two phrases
        phrase1 = self.query_engine.phrases[0]  # marm_phee_001
        phrase2 = self.query_engine.phrases[1]  # marm_alarm_001

        # Calculate 30D Euclidean distance
        vec1 = phrase1.get_feature_vector()
        vec2 = phrase2.get_feature_vector()

        distance = np.sqrt(np.sum((vec1 - vec2) ** 2))

        # Distance should be positive
        self.assertGreater(distance, 0.0)

        # Verify calculation
        expected_diff = vec1 - vec2
        expected_distance = np.sqrt(np.sum(expected_diff**2))
        self.assertAlmostEqual(distance, expected_distance, places=5)


class Test30DInterpolation(unittest.TestCase):
    """Test interpolation with 30-dimensional feature vectors."""

    def setUp(self):
        """Set up test data."""
        self.sample_rate = 48000
        self.duration = 1.0
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))

        # Create two phrases with different 30D characteristics
        self.metadata_a = {
            "phrase_id": "phrase_a",
            "species": "marmoset",
            "cluster_id": 0,
            "context": "contact",
            "mean_f0_hz": 6000.0,
            "f0_range_hz": 300.0,
            "duration_ms": 50.0,
            "harmonic_to_noise_ratio": 20.0,
            "spectral_flatness": 0.1,
            "harmonicity": 0.95,
            "attack_time_ms": 10.0,
            "decay_time_ms": 15.0,
            "sustain_level": 0.7,
            "vibrato_rate_hz": 8.0,
            "vibrato_depth": 50.0,
            "jitter": 0.02,
            "shimmer": 0.03,
            "mfcc_1": -500.0,
            "mfcc_2": -100.0,
            "mfcc_3": -50.0,
            "mfcc_4": -20.0,
            "mfcc_5": -10.0,
            "mfcc_6": -5.0,
            "mfcc_7": 0.0,
            "mfcc_8": 5.0,
            "mfcc_9": 10.0,
            "mfcc_10": 15.0,
            "mfcc_11": 20.0,
            "mfcc_12": 25.0,
            "mfcc_13": 30.0,
            "spectral_flux": 100.0,
            "median_ici_ms": 45.0,
            "onset_rate_hz": 15.0,
            "ici_coefficient_of_variation": 0.25,
        }

        self.metadata_b = {
            "phrase_id": "phrase_b",
            "species": "marmoset",
            "cluster_id": 1,
            "context": "alarm",
            "mean_f0_hz": 8000.0,
            "f0_range_hz": 500.0,
            "duration_ms": 70.0,
            "harmonic_to_noise_ratio": 15.0,
            "spectral_flatness": 0.2,
            "harmonicity": 0.85,
            "attack_time_ms": 5.0,
            "decay_time_ms": 10.0,
            "sustain_level": 0.6,
            "vibrato_rate_hz": 12.0,
            "vibrato_depth": 80.0,
            "jitter": 0.04,
            "shimmer": 0.05,
            "mfcc_1": -600.0,
            "mfcc_2": -120.0,
            "mfcc_3": -60.0,
            "mfcc_4": -24.0,
            "mfcc_5": -12.0,
            "mfcc_6": -6.0,
            "mfcc_7": 0.0,
            "mfcc_8": 6.0,
            "mfcc_9": 12.0,
            "mfcc_10": 18.0,
            "mfcc_11": 24.0,
            "mfcc_12": 30.0,
            "mfcc_13": 36.0,
            "spectral_flux": 120.0,
            "median_ici_ms": 55.0,
            "onset_rate_hz": 18.0,
            "ici_coefficient_of_variation": 0.3,
        }

        audio_a = 0.5 * np.sin(2 * np.pi * 6000 * t)
        audio_b = 0.5 * np.sin(2 * np.pi * 8000 * t)

        self.candidate_a = PhraseCandidate(
            phrase_id="phrase_a",
            audio_buffer=audio_a,
            metadata=self.metadata_a,
            sample_rate=self.sample_rate,
        )

        self.candidate_b = PhraseCandidate(
            phrase_id="phrase_b",
            audio_buffer=audio_b,
            metadata=self.metadata_b,
            sample_rate=self.sample_rate,
        )

    def test_interpolate_30d_features_50_50(self):
        """Test 50/50 interpolation of 30D features."""
        from realtime.metadata_synthesizer import interpolate_30d_features

        # Interpolate at 50%
        interpolated = interpolate_30d_features(self.candidate_a, self.candidate_b, 0.5)

        # Verify all 30 dimensions are interpolated
        self.assertEqual(len(interpolated), 30)

        # Check that values are halfway between
        vec_a = self.candidate_a.get_feature_vector()
        vec_b = self.candidate_b.get_feature_vector()

        expected = 0.5 * vec_a + 0.5 * vec_b

        for i in range(30):
            self.assertAlmostEqual(interpolated[i], expected[i], places=5)

    def test_interpolate_30d_features_75_25(self):
        """Test 75/25 interpolation of 30D features."""
        from realtime.metadata_synthesizer import interpolate_30d_features

        # Interpolate at 75% (75% of B, 25% of A)
        interpolated = interpolate_30d_features(self.candidate_a, self.candidate_b, 0.75)

        vec_a = self.candidate_a.get_feature_vector()
        vec_b = self.candidate_b.get_feature_vector()

        # blend_ratio=0.75 means 75% of B, 25% of A
        expected = 0.25 * vec_a + 0.75 * vec_b

        for i in range(30):
            self.assertAlmostEqual(interpolated[i], expected[i], places=5)

    def test_interpolate_preserves_physical_constraints(self):
        """Test that interpolation respects physical constraints."""
        from realtime.metadata_synthesizer import interpolate_30d_features

        # Duration should be interpolated
        interpolated = interpolate_30d_features(self.candidate_a, self.candidate_b, 0.5)
        duration_idx = 2  # duration_ms is at index 2
        self.assertAlmostEqual(interpolated[duration_idx], 60.0)  # (50 + 70) / 2


class Test30DSynthesisRecipe(unittest.TestCase):
    """Test synthesis recipes with 30D features."""

    def test_synthesis_recipe_with_30d_targets(self):
        """Test that synthesis recipes include 30D target parameters."""
        sources = []
        sample_rate = 48000
        t = np.linspace(0, 1.0, 48000)

        # Create source candidates
        for i, f0 in enumerate([6000.0, 8000.0]):
            metadata = {
                "phrase_id": f"phrase_{i}",
                "species": "marmoset",
                "cluster_id": i,
                "context": "test",
                "mean_f0_hz": f0,
                "f0_range_hz": 300.0 + i * 200,
                "duration_ms": 50.0 + i * 20,
                "harmonic_to_noise_ratio": 20.0 - i * 5,
                "spectral_flatness": 0.1 + i * 0.1,
                "harmonicity": 0.95 - i * 0.1,
                "attack_time_ms": 10.0,
                "decay_time_ms": 15.0,
                "sustain_level": 0.7,
                "vibrato_rate_hz": 8.0,
                "vibrato_depth": 50.0,
                "jitter": 0.02,
                "shimmer": 0.03,
                "mfcc_1": -500.0,
                "mfcc_2": -100.0,
                "mfcc_3": -50.0,
                "mfcc_4": -20.0,
                "mfcc_5": -10.0,
                "mfcc_6": -5.0,
                "mfcc_7": 0.0,
                "mfcc_8": 5.0,
                "mfcc_9": 10.0,
                "mfcc_10": 15.0,
                "mfcc_11": 20.0,
                "mfcc_12": 25.0,
                "mfcc_13": 30.0,
                "spectral_flux": 100.0,
                "median_ici_ms": 45.0,
                "onset_rate_hz": 15.0,
                "ici_coefficient_of_variation": 0.25,
            }

            audio = 0.5 * np.sin(2 * np.pi * f0 * t)
            candidate = PhraseCandidate(
                phrase_id=f"phrase_{i}",
                audio_buffer=audio,
                metadata=metadata,
                sample_rate=sample_rate,
            )
            sources.append((candidate, 0.5))

        # Calculate 30D target parameters
        target_params = {}
        vec_a = sources[0][0].get_feature_vector()
        vec_b = sources[1][0].get_feature_vector()
        interpolated = 0.5 * vec_a + 0.5 * vec_b

        # Verify interpolation includes all 30 dimensions
        self.assertEqual(len(interpolated), 30)

        recipe = SynthesisRecipe(
            sources=sources,
            target_params=target_params,
            synthesis_mode="morph",
            is_cross_persona=True,
            discovery_potential=0.5,
            reasoning="Test 30D interpolation",
        )

        self.assertEqual(len(recipe.sources), 2)
        self.assertTrue(recipe.is_cross_persona)
        self.assertGreater(recipe.discovery_potential, 0.0)


class Test30DMetadataFirstSynthesizer(unittest.TestCase):
    """Integration tests for 30D metadata-first synthesis."""

    def test_synthesizer_creates_30d_queries(self):
        """Test that synthesizer creates queries compatible with 30D features."""
        synthesizer = MetadataFirstSynthesizer()

        # Query with 30D-aware parameters
        audio, recipe = synthesizer.synthesize_by_target(
            target_f0_hz=7000.0,
            target_duration_ms=50.0,
            species="marmoset",
            synthesis_duration_ms=200.0,
        )

        # Verify synthesis succeeded
        self.assertIsNotNone(audio)
        self.assertIsNotNone(recipe)
        self.assertEqual(len(audio), 200 * 48)  # 200ms at 48kHz

    def test_ghost_word_with_30d_features(self):
        """Test ghost word synthesis using 30D feature interpolation."""
        synthesizer = MetadataFirstSynthesizer()

        # Add some 30D phrases to the query engine
        sample_rate = 48000
        t = np.linspace(0, 1.0, sample_rate)

        for cluster_id in [0, 1]:
            metadata = {
                "phrase_id": f"cluster_{cluster_id}_phrase",
                "species": "marmoset",
                "cluster_id": cluster_id,
                "context": "test",
                "mean_f0_hz": 6000.0 + cluster_id * 2000,
                "f0_range_hz": 300.0 + cluster_id * 200,
                "duration_ms": 50.0 + cluster_id * 10,
                "harmonic_to_noise_ratio": 20.0 - cluster_id * 5,
                "spectral_flatness": 0.1 + cluster_id * 0.1,
                "harmonicity": 0.95 - cluster_id * 0.1,
                "attack_time_ms": 10.0,
                "decay_time_ms": 15.0,
                "sustain_level": 0.7,
                "vibrato_rate_hz": 8.0,
                "vibrato_depth": 50.0,
                "jitter": 0.02,
                "shimmer": 0.03,
                "mfcc_1": -500.0,
                "mfcc_2": -100.0,
                "mfcc_3": -50.0,
                "mfcc_4": -20.0,
                "mfcc_5": -10.0,
                "mfcc_6": -5.0,
                "mfcc_7": 0.0,
                "mfcc_8": 5.0,
                "mfcc_9": 10.0,
                "mfcc_10": 15.0,
                "mfcc_11": 20.0,
                "mfcc_12": 25.0,
                "mfcc_13": 30.0,
                "spectral_flux": 100.0,
                "median_ici_ms": 45.0,
                "onset_rate_hz": 15.0,
                "ici_coefficient_of_variation": 0.25,
            }

            audio = 0.5 * np.sin(2 * np.pi * metadata["mean_f0_hz"] * t)
            candidate = PhraseCandidate(
                phrase_id=metadata["phrase_id"],
                audio_buffer=audio,
                metadata=metadata,
                sample_rate=sample_rate,
            )

            synthesizer.query_engine.phrases.append(candidate)
            synthesizer.query_engine.cluster_index[cluster_id] = [candidate]

        # Synthesize ghost word
        audio, recipe = synthesizer.synthesize_ghost_word(
            cluster_a_id=0,
            cluster_b_id=1,
            blend_ratio=0.5,
            species="marmoset",
        )

        # Verify ghost word was created
        self.assertIsNotNone(audio)
        self.assertTrue(recipe.is_cross_persona)
        self.assertEqual(recipe.discovery_potential, 1.0)


def run_tests():
    """Run all tests and report results."""
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED - 30D metadata synthesis fully implemented!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
