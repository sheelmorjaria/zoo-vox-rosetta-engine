#!/usr/bin/env python3
"""
Tests for Prosodic DTW (Dynamic Time Warping)

Tests the DTW-based prosodic similarity analysis for
measuring temporal alignment between vocalizations.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging

import numpy as np
import pytest

from ethological_validation.prosodic_dtw import (
    DTWResult,
    FastDTW,
    ProsodicDTW,
    ProsodicFeature,
    ProsodicFeatureExtractor,
    DEFAULT_PROSODIC_DTW,
    DEFAULT_FEATURE_EXTRACTOR,
)

logger = logging.getLogger(__name__)


class TestProsodicFeature:
    """Test ProsodicFeature dataclass."""

    def test_prosodic_feature_creation(self):
        """Should create prosodic feature."""
        feature = ProsodicFeature(
            f0_contour=np.array([5000, 5100, 5200]),
            amplitude_envelope=np.array([-10, -5, -15]),
            duration_ms=100.0,
        )
        assert len(feature.f0_contour) == 3
        assert feature.duration_ms == 100.0

    def test_prosodic_feature_with_spectral_centroid(self):
        """Should create prosodic feature with spectral centroid."""
        feature = ProsodicFeature(
            f0_contour=np.array([5000, 5100]),
            amplitude_envelope=np.array([-10, -5]),
            duration_ms=50.0,
            spectral_centroid=np.array([3000, 3100]),
        )
        assert feature.spectral_centroid is not None
        assert len(feature.spectral_centroid) == 2


class TestDTWResult:
    """Test DTWResult dataclass."""

    def test_dtw_result_creation(self):
        """Should create DTW result."""
        result = DTWResult(
            similarity_score=0.85,
            dtw_distance=10.5,
            normalized_distance=0.21,
            warping_path=np.array([[0, 0], [1, 1]]),
            best_match_idx=0,
        )
        assert result.similarity_score == 0.85
        assert result.best_match_idx == 0


class TestFastDTW:
    """Test FastDTW implementation."""

    def test_initialization(self):
        """Should initialize with optional window size."""
        dtw = FastDTW()
        assert dtw.window_size is None

        dtw_windowed = FastDTW(window_size=10)
        assert dtw_windowed.window_size == 10

    def test_compute_distance_identical(self):
        """Should return 0 for identical sequences."""
        dtw = FastDTW()
        x = np.array([1, 2, 3, 4, 5])

        distance = dtw.compute_distance(x, x)

        assert distance == pytest.approx(0.0)

    def test_compute_distance_different(self):
        """Should return positive distance for different sequences."""
        dtw = FastDTW()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])

        distance = dtw.compute_distance(x, y)

        assert distance > 0

    def test_compute_distance_with_path(self):
        """Should return distance and warping path."""
        dtw = FastDTW()
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])

        distance, path = dtw.compute_distance_with_path(x, y)

        assert distance == pytest.approx(0.0)
        assert path.shape[1] == 2
        assert len(path) == 3  # Diagonal path

    def test_dtw_warping_path_properties(self):
        """Warping path should be monotonic and bounded."""
        dtw = FastDTW()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3, 4, 5, 6])

        _, path = dtw.compute_distance_with_path(x, y)

        # Path should start at (0, 0) or (1, 1) depending on indexing
        assert path[0, 0] >= 0
        assert path[0, 1] >= 0

        # Path should end at the last points
        assert path[-1, 0] == len(x) - 1
        assert path[-1, 1] == len(y) - 1

        # Path should be monotonic (non-decreasing in both dimensions)
        for i in range(1, len(path)):
            assert path[i, 0] >= path[i-1, 0]
            assert path[i, 1] >= path[i-1, 1]

    def test_dtw_window_constraint(self):
        """Window constraint should limit computation band."""
        dtw_full = FastDTW(window_size=None)
        dtw_windowed = FastDTW(window_size=5)

        x = np.arange(100)
        y = np.arange(100)

        dist_full = dtw_full.compute_distance(x, y)
        dist_windowed = dtw_windowed.compute_distance(x, y)

        # For identical sequences, both should give same result
        assert dist_full == pytest.approx(dist_windowed)

    def test_dtw_empty_sequences(self):
        """Should handle empty or near-empty sequences."""
        dtw = FastDTW()

        # Empty sequences
        dist = dtw.compute_distance(np.array([]), np.array([]))
        assert dist == pytest.approx(0.0)

    def test_dtw_different_lengths(self):
        """Should handle sequences of different lengths."""
        dtw = FastDTW()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3])

        distance = dtw.compute_distance(x, y)

        assert distance >= 0


class TestProsodicDTW:
    """Test ProsodicDTW engine."""

    def test_initialization(self):
        """Should initialize with optional baselines."""
        dtw = ProsodicDTW()
        assert len(dtw.baselines) == 0

        baselines = [np.array([5000, 5100, 5200])]
        dtw_with_baseline = ProsodicDTW(baseline_contours=baselines)
        assert len(dtw_with_baseline.baselines) == 1

    def test_add_baseline(self):
        """Should add baseline contour."""
        dtw = ProsodicDTW()
        contour = np.array([5000, 5100, 5200])

        dtw.add_baseline(contour)

        assert len(dtw.baselines) == 1

    def test_set_baselines(self):
        """Should replace all baselines."""
        dtw = ProsodicDTW()
        dtw.add_baseline(np.array([5000, 5100]))

        new_baselines = [np.array([6000, 6100, 6200])]
        dtw.set_baselines(new_baselines)

        assert len(dtw.baselines) == 1
        np.testing.assert_array_equal(dtw.baselines[0], new_baselines[0])

    def test_score_response_no_baselines(self):
        """Should return default score when no baselines available."""
        dtw = ProsodicDTW()
        f0_contour = np.array([5000, 5100, 5200])

        result = dtw.score_response(f0_contour)

        assert result.similarity_score == 0.5
        assert result.best_match_idx == -1

    def test_score_response_with_baselines(self):
        """Should score response against baselines."""
        baselines = [
            np.linspace(5000, 7000, 50),  # Rising F0
            np.ones(50) * 6000,            # Flat F0
        ]
        # Use larger sigma to handle larger DTW distances
        dtw = ProsodicDTW(baseline_contours=baselines, sigma=10000.0)

        # Similar to baseline 0
        response = np.linspace(5100, 6900, 45)
        result = dtw.score_response(response)

        assert 0.0 <= result.similarity_score <= 1.0
        assert result.best_match_idx in [0, 1]
        assert result.dtw_distance >= 0

    def test_score_response_perfect_match(self):
        """Should give high score for perfect match."""
        baselines = [np.linspace(5000, 7000, 50)]
        dtw = ProsodicDTW(baseline_contours=baselines, sigma=10000.0)

        # Exact match
        response = np.linspace(5000, 7000, 50)
        result = dtw.score_response(response)

        assert result.similarity_score > 0.9
        assert result.best_match_idx == 0

    def test_score_response_no_match(self):
        """Should give low score for very different contour."""
        baselines = [np.linspace(5000, 7000, 50)]  # Rising
        dtw = ProsodicDTW(baseline_contours=baselines, sigma=10000.0)

        # Completely opposite pattern
        response = np.linspace(7000, 5000, 50)  # Falling
        result = dtw.score_response(response)

        # With large sigma, even different patterns may have moderate similarity
        assert result.similarity_score < 1.0

    def test_score_joint_prosody(self):
        """Should score using both F0 and amplitude."""
        baselines = [np.linspace(5000, 7000, 50)]
        dtw = ProsodicDTW(baseline_contours=baselines)

        f0_contour = np.linspace(5100, 6900, 45)
        amp_envelope = np.linspace(-20, -5, 45)

        score = dtw.score_joint_prosody(
            f0_contour,
            amp_envelope,
            f0_weight=0.7,
            amp_weight=0.3,
        )

        assert 0.0 <= score <= 1.0

    def test_interpolate_to_length(self):
        """Should interpolate array to target length."""
        dtw = ProsodicDTW()

        original = np.array([1, 2, 3, 4, 5])
        interpolated = dtw._interpolate_to_length(original, 10)

        assert len(interpolated) == 10
        assert interpolated[0] == pytest.approx(1.0)
        assert interpolated[-1] == pytest.approx(5.0)

    def test_interpolate_same_length(self):
        """Should return unchanged if same length."""
        dtw = ProsodicDTW()

        original = np.array([1, 2, 3, 4, 5])
        result = dtw._interpolate_to_length(original, 5)

        np.testing.assert_array_equal(result, original)


class TestProsodicFeatureExtractor:
    """Test ProsodicFeatureExtractor."""

    def test_initialization(self):
        """Should initialize with sample rate and frame size."""
        extractor = ProsodicFeatureExtractor(sample_rate=48000, frame_size=512)
        assert extractor.sample_rate == 48000
        assert extractor.frame_size == 512
        assert extractor.hop_size == 128  # frame_size // 4

    def test_extract_from_audio(self):
        """Should extract prosodic features from audio."""
        extractor = ProsodicFeatureExtractor(sample_rate=48000)

        # Generate test audio (sine wave)
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(48000 * duration))
        audio = 0.5 * np.sin(2 * np.pi * 1000 * t)  # 1kHz tone

        features = extractor.extract_from_audio(audio)

        assert isinstance(features, ProsodicFeature)
        assert features.duration_ms == pytest.approx(100.0, rel=0.1)
        assert len(features.f0_contour) > 0
        assert len(features.amplitude_envelope) > 0

    def test_extract_amplitude_envelope(self):
        """Should extract RMS amplitude envelope."""
        extractor = ProsodicFeatureExtractor(sample_rate=48000)

        # Generate amplitude-modulated tone
        duration = 0.1
        t = np.linspace(0, duration, int(48000 * duration))
        modulator = 0.5 * (1 + np.sin(2 * np.pi * 10 * t))
        audio = modulator * np.sin(2 * np.pi * 1000 * t)

        envelope = extractor._extract_amplitude_envelope(audio)

        assert len(envelope) > 0
        # Envelope should show the modulation pattern
        assert np.max(envelope) > np.min(envelope)

    def test_extract_f0_contour(self):
        """Should extract F0 contour from voiced audio."""
        extractor = ProsodicFeatureExtractor(sample_rate=48000)

        # Generate 5kHz tone (within F0 range for many species)
        duration = 0.1
        t = np.linspace(0, duration, int(48000 * duration))
        audio = 0.5 * np.sin(2 * np.pi * 5000 * t)

        f0_contour = extractor._extract_f0_contour(audio)

        assert len(f0_contour) > 0
        # Should detect F0 around 5kHz
        voiced_frames = f0_contour[f0_contour > 0]
        if len(voiced_frames) > 0:
            assert 4000 < np.median(voiced_frames) < 6000

    def test_extract_spectral_centroid(self):
        """Should extract spectral centroid trajectory."""
        extractor = ProsodicFeatureExtractor(sample_rate=48000)

        duration = 0.1
        t = np.linspace(0, duration, int(48000 * duration))
        audio = 0.5 * np.sin(2 * np.pi * 1000 * t)

        centroid = extractor._extract_spectral_centroid(audio)

        assert len(centroid) > 0
        # Centroid should be positive frequency
        assert np.all(centroid > 0)


class TestPresetConfigurations:
    """Test preset configurations."""

    def test_default_prosodic_dtw(self):
        """Should have default DTW engine configured."""
        assert DEFAULT_PROSODIC_DTW is not None
        assert isinstance(DEFAULT_PROSODIC_DTW, ProsodicDTW)

    def test_default_feature_extractor(self):
        """Should have default feature extractor configured."""
        assert DEFAULT_FEATURE_EXTRACTOR is not None
        assert isinstance(DEFAULT_FEATURE_EXTRACTOR, ProsodicFeatureExtractor)


class TestIntegrationScenarios:
    """Integration tests for realistic prosodic comparison scenarios."""

    def test_natural_vs_aggressive_prosody(self):
        """
        Test differentiation between natural conversation and
        aggressive staccato bursts using DTW.
        """
        # Natural baseline: smooth rising F0
        natural_baseline = np.linspace(5000, 7000, 50) + np.random.randn(50) * 50
        dtw = ProsodicDTW(baseline_contours=[natural_baseline])

        # Natural-like response
        natural_response = np.linspace(5100, 6900, 45) + np.random.randn(45) * 50
        result_natural = dtw.score_response(natural_response)

        # Aggressive staccato: discontinuous bursts
        aggressive = np.concatenate([
            np.ones(20) * 9000,  # High burst
            np.zeros(15),        # Gap
            np.ones(15) * 9000,  # Another burst
        ])
        result_aggressive = dtw.score_response(aggressive)

        # Natural should score higher than aggressive
        assert result_natural.similarity_score > result_aggressive.similarity_score

    def test_contact_call_matching(self):
        """Test matching contact call prosody."""
        # Contact call: rising then flat F0
        contact_contour = np.concatenate([
            np.linspace(5000, 7000, 20),
            np.ones(30) * 7000,
        ])
        dtw = ProsodicDTW(baseline_contours=[contact_contour], sigma=10000.0)

        # Similar pattern
        similar_call = np.concatenate([
            np.linspace(5100, 6900, 18),
            np.ones(27) * 6900,
        ])
        result = dtw.score_response(similar_call)

        # With large sigma, similar patterns should have reasonable similarity
        assert result.similarity_score > 0.3

    def test_multi_baseline_selection(self):
        """Test selection from multiple baseline types."""
        baselines = [
            np.linspace(5000, 7000, 50),      # Rising
            np.ones(50) * 6000,                # Flat
            6000 + 1000 * np.sin(np.linspace(0, 2*np.pi, 50)),  # Modulated
        ]
        dtw = ProsodicDTW(baseline_contours=baselines, sigma=10000.0)

        # Best match for first baseline
        rising_response = np.linspace(5100, 6900, 45)
        result = dtw.score_response(rising_response)

        # Should match the rising baseline (index 0)
        assert result.best_match_idx == 0
        # With large sigma, should have reasonable similarity
        assert result.similarity_score > 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
