#!/usr/bin/env python3
"""
Tests for Acoustic Convergence Engine with DTW

Tests the ethological validation metrics for measuring acoustic
convergence between animal and system vocalizations.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging

import numpy as np
import pytest

from ethological_validation.acoustic_convergence import (
    AcousticConvergenceEngine,
    ConvergenceResult,
    MultiDimensionalConvergence,
    compute_batch_convergence,
    compute_convergence_from_affect_vectors,
    DEFAULT_CONVERGENCE_ENGINE,
    MULTI_DIM_CONVERGENCE,
)

logger = logging.getLogger(__name__)


class TestConvergenceResult:
    """Test ConvergenceResult dataclass."""

    def test_convergence_result_creation(self):
        """Should create a convergence result."""
        result = ConvergenceResult(
            convergence_score=0.8,
            raw_convergence=0.5,
            direction="toward",
            pre_distance=1.0,
            post_distance=0.5,
        )
        assert result.convergence_score == 0.8
        assert result.raw_convergence == 0.5
        assert result.direction == "toward"
        assert result.pre_distance == 1.0
        assert result.post_distance == 0.5


class TestAcousticConvergenceEngine:
    """Test AcousticConvergenceEngine."""

    def test_initialization_default(self):
        """Should initialize with cosine distance by default."""
        engine = AcousticConvergenceEngine()
        assert engine.distance_metric == 'cosine'

    def test_initialization_euclidean(self):
        """Should initialize with euclidean distance."""
        engine = AcousticConvergenceEngine(distance_metric='euclidean')
        assert engine.distance_metric == 'euclidean'

    def test_initialization_mahalanobis(self):
        """Should initialize with mahalanobis distance when covariance provided."""
        cov = np.eye(16)
        engine = AcousticConvergenceEngine(
            distance_metric='mahalanobis',
            covariance_matrix=cov
        )
        assert engine.distance_metric == 'mahalanobis'

    def test_initialization_mahalanobis_no_covariance(self):
        """Should raise error when mahalanobis requested without covariance."""
        with pytest.raises(ValueError, match="covariance_matrix required"):
            AcousticConvergenceEngine(distance_metric='mahalanobis')

    def test_calculate_convergence_toward(self):
        """Should detect convergence when animal moves toward AI."""
        # Use euclidean distance for magnitude-based convergence
        engine = AcousticConvergenceEngine(distance_metric='euclidean')

        animal_pre = np.ones(16) * 0.1
        ai_output = np.ones(16) * 0.5
        animal_post = np.ones(16) * 0.4  # Moved toward AI

        result = engine.calculate_convergence(animal_pre, ai_output, animal_post)

        assert result.direction == "toward"
        assert result.convergence_score > 0.5
        assert result.post_distance < result.pre_distance

    def test_calculate_convergence_away(self):
        """Should detect divergence when animal moves away from AI."""
        # Use euclidean distance for magnitude-based convergence
        engine = AcousticConvergenceEngine(distance_metric='euclidean')

        animal_pre = np.ones(16) * 0.4
        ai_output = np.ones(16) * 0.5
        animal_post = np.ones(16) * 0.1  # Moved away from AI

        result = engine.calculate_convergence(animal_pre, ai_output, animal_post)

        assert result.direction == "away"
        assert result.convergence_score < 0.5
        assert result.post_distance > result.pre_distance

    def test_calculate_convergence_neutral(self):
        """Should detect neutral when animal doesn't change much."""
        engine = AcousticConvergenceEngine(distance_metric='cosine')

        animal_pre = np.ones(16) * 0.5
        ai_output = np.ones(16) * 0.5
        animal_post = np.ones(16) * 0.49  # Very small change

        result = engine.calculate_convergence(animal_pre, ai_output, animal_post)

        assert result.direction == "neutral"
        assert 0.4 < result.convergence_score < 0.6

    def test_cosine_distance(self):
        """Should compute cosine distance correctly."""
        engine = AcousticConvergenceEngine(distance_metric='cosine')

        v1 = np.ones(16)
        v2 = np.ones(16)
        assert engine._cosine_distance(v1, v2) == pytest.approx(0.0)

        v3 = np.ones(16)
        v4 = -np.ones(16)
        assert engine._cosine_distance(v3, v4) == pytest.approx(2.0)

    def test_euclidean_distance(self):
        """Should compute euclidean distance correctly."""
        engine = AcousticConvergenceEngine(distance_metric='euclidean')

        v1 = np.zeros(16)
        v2 = np.zeros(16)
        assert engine._euclidean_distance(v1, v2) == 0.0

        v3 = np.zeros(16)
        v4 = np.ones(16)
        assert engine._euclidean_distance(v3, v4) == pytest.approx(4.0, rel=0.01)

    def test_mahalanobis_distance(self):
        """Should compute mahalanobis distance correctly."""
        cov = np.eye(16)
        engine = AcousticConvergenceEngine(
            distance_metric='mahalanobis',
            covariance_matrix=cov
        )

        v1 = np.zeros(16)
        v2 = np.zeros(16)
        assert engine._mahalanobis_distance(v1, v2) == 0.0

    def test_convergence_score_range(self):
        """Should produce scores in [0, 1] range."""
        engine = AcousticConvergenceEngine()

        for _ in range(10):
            animal_pre = np.random.randn(16)
            ai_output = np.random.randn(16)
            animal_post = np.random.randn(16)

            result = engine.calculate_convergence(animal_pre, ai_output, animal_post)

            assert 0.0 <= result.convergence_score <= 1.0


class TestMultiDimensionalConvergence:
    """Test MultiDimensionalConvergence."""

    def test_initialization(self):
        """Should initialize with dimension extractors."""
        mdc = MultiDimensionalConvergence()
        assert 'f0' in mdc.dimensions
        assert 'harmonics' in mdc.dimensions
        assert 'noise' in mdc.dimensions
        assert 'affect' in mdc.dimensions

    def test_calculate_dimensional_convergence(self):
        """Should calculate convergence for each dimension."""
        mdc = MultiDimensionalConvergence()

        # Use positive F0 values to avoid log issues
        animal_pre = np.abs(np.random.randn(112)) + 1.0
        ai_output = np.abs(np.random.randn(112)) + 1.0
        animal_post = animal_pre * 0.7 + ai_output * 0.3  # Convergence

        results = mdc.calculate_dimensional_convergence(
            animal_pre, ai_output, animal_post
        )

        assert len(results) == 4
        for dim_name, result in results.items():
            if result is not None:
                assert isinstance(result, ConvergenceResult)
                assert 0.0 <= result.convergence_score <= 1.0


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_compute_convergence_from_affect_vectors(self):
        """Should compute convergence from 16D affect vectors."""
        # Use different direction vectors for proper cosine distance
        pre = np.array([0.1] * 8 + [0.9] * 8)
        ai = np.ones(16) * 0.5
        post = np.array([0.3] * 8 + [0.7] * 8)  # Moved toward AI

        score = compute_convergence_from_affect_vectors(pre, ai, post)

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should detect convergence

    def test_compute_batch_convergence_empty(self):
        """Should handle empty batch."""
        result = compute_batch_convergence([])
        assert result['count'] == 0

    def test_compute_batch_convergence(self):
        """Should compute batch convergence statistics."""
        batch = [
            {
                'animal_pre': np.ones(16) * 0.1,
                'ai_output': np.ones(16) * 0.5,
                'animal_post': np.ones(16) * 0.4,
            },
            {
                'animal_pre': np.ones(16) * 0.1,
                'ai_output': np.ones(16) * 0.5,
                'animal_post': np.ones(16) * 0.35,
            },
        ]

        result = compute_batch_convergence(batch)

        assert result['count'] == 2
        assert 'mean_score' in result
        assert 'std_score' in result
        assert 'toward_rate' in result
        assert 'away_rate' in result
        assert 'neutral_rate' in result


class TestPresetConfigurations:
    """Test preset configurations."""

    def test_default_convergence_engine(self):
        """Should have default engine configured."""
        assert DEFAULT_CONVERGENCE_ENGINE is not None
        assert DEFAULT_CONVERGENCE_ENGINE.distance_metric == 'cosine'

    def test_multi_dim_convergence(self):
        """Should have multi-dimensional analyzer configured."""
        assert MULTI_DIM_CONVERGENCE is not None
        assert len(MULTI_DIM_CONVERGENCE.dimensions) == 4


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    def test_vocal_learning_convergence_scenario(self):
        """
        Test realistic vocal learning scenario:
        - Animal starts with baseline call
        - AI produces target dialect
        - Animal shifts toward target (vocal convergence)
        """
        engine = AcousticConvergenceEngine(distance_metric='cosine')

        # Baseline: Low arousal, calm
        animal_pre = np.array([0.2, 0.0, 0.3] + [0.0] * 13)

        # AI target: Moderate arousal, engaged
        ai_target = np.array([0.5, 0.3, 0.5] + [0.1] * 13)

        # Animal response: Shifts toward AI (acceptance)
        animal_post = np.array([0.4, 0.2, 0.4] + [0.05] * 13)

        result = engine.calculate_convergence(animal_pre, ai_target, animal_post)

        # Should detect convergence
        assert result.direction == "toward"
        assert result.convergence_score > 0.6

    def test_aggressive_divergence_scenario(self):
        """
        Test aggressive response scenario:
        - Animal starts neutral
        - AI produces call
        - Animal shifts away (aggression/rejection)
        """
        engine = AcousticConvergenceEngine(distance_metric='cosine')

        animal_pre = np.array([0.5, 0.0, 0.5] + [0.0] * 13)
        ai_target = np.array([0.5, 0.0, 0.5] + [0.0] * 13)

        # Animal becomes highly aroused, negative valence (aggression)
        animal_post = np.array([0.9, -0.8, 0.7] + [0.2] * 13)

        result = engine.calculate_convergence(animal_pre, ai_target, animal_post)

        # Should detect divergence
        assert result.direction == "away"
        assert result.convergence_score < 0.5

    def test_deescalation_convergence(self):
        """
        Test de-escalation scenario:
        - High arousal animal
        - AI produces calming response
        - Animal de-escalates toward calm
        """
        engine = AcousticConvergenceEngine(distance_metric='cosine')

        # Highly aroused animal
        animal_pre = np.array([0.9, -0.5, 0.8] + [0.3] * 13)

        # Calming AI response
        ai_calm = np.array([0.3, 0.5, 0.3] + [0.0] * 13)

        # Animal de-escalates
        animal_post = np.array([0.5, 0.0, 0.5] + [0.1] * 13)

        result = engine.calculate_convergence(animal_pre, ai_calm, animal_post)

        # Should detect convergence toward calm
        assert result.direction == "toward"

    def test_full_112d_convergence(self):
        """Test convergence on full 112D RosettaFeatures."""
        engine = AcousticConvergenceEngine(distance_metric='cosine')

        animal_pre = np.random.randn(112) * 0.1
        ai_output = np.random.randn(112) * 0.5
        animal_post = animal_pre * 0.6 + ai_output * 0.4

        result = engine.calculate_convergence(animal_pre, ai_output, animal_post)

        assert 0.0 <= result.convergence_score <= 1.0
        assert result.pre_distance > 0
        assert result.post_distance > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
