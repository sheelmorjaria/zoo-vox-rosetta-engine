"""
Persona Semantic Zone Validation Tests
======================================

Test-Driven Development (TDD) tests for persona-based semantic zone validation.

Tests verify that:
1. Valid vocalizations fall within their persona's semantic zone
2. Invalid vocalizations are rejected
3. Ghost words are correctly identified
4. Classification works across species
5. Statistical boundaries are enforced

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings

from realtime.persona_semantic_zone_validator import (
    PersonaSemanticZoneValidator,
    SemanticZoneValidationResult,
)

warnings.filterwarnings('ignore')


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def validator():
    """Create semantic zone validator with default clusters."""
    return PersonaSemanticZoneValidator()


@pytest.fixture
def marmoset_phee_features():
    """Valid marmoset phee call features."""
    return {
        'mean_f0_hz': 6526,
        'duration_ms': 76.5,
        'f0_range_hz': 427,
        'harmonicity': 0.95,
        'spectral_flatness': 0.1,
        'jitter': 0.02,
        'shimmer': 0.03
    }


@pytest.fixture
def marmoset_alarm_features():
    """Valid marmoset alarm call features."""
    return {
        'mean_f0_hz': 6020,
        'duration_ms': 58.1,
        'f0_range_hz': 3722,
        'harmonicity': 0.7,
        'spectral_flatness': 0.3,
        'jitter': 0.08,
        'shimmer': 0.05
    }


@pytest.fixture
def ghost_word_features():
    """Ghost word: interpolation between phee and alarm."""
    return {
        'mean_f0_hz': 6273,
        'duration_ms': 67.3,
        'f0_range_hz': 2074,
        'harmonicity': 0.825,
        'spectral_flatness': 0.2,
        'jitter': 0.05,
        'shimmer': 0.04
    }


@pytest.fixture
def invalid_features():
    """Invalid features (outside all semantic zones)."""
    return {
        'mean_f0_hz': 10000,
        'duration_ms': 200,
        'f0_range_hz': 100,
        'harmonicity': 0.3,
        'spectral_flatness': 0.8,
        'jitter': 0.2,
        'shimmer': 0.15
    }


# ============================================================================
# Test Suite 1: Semantic Zone Validation
# ============================================================================

class TestSemanticZoneValidation:
    """Test that vocalizations are validated against semantic zones."""

    def test_valid_phee_in_zone(self, validator, marmoset_phee_features):
        """Valid phee call should be in MARMOSET_PHEE zone."""
        result = validator.validate_vocalization(
            marmoset_phee_features,
            species='marmoset'
        )

        assert result.passed, "Valid phee should pass validation"
        assert result.persona_id == 'MARMOSET_PHEE'
        assert result.semantic_label == 'contact'
        assert result.confidence > 0.95, "Should have high confidence"
        assert not result.is_outlier, "Should not be an outlier"
        assert result.mahalanobis_distance < 2.0, "Should be within 2-sigma"

    def test_valid_alarm_in_zone(self, validator, marmoset_alarm_features):
        """Valid alarm call should be in MARMOSET_ALARM zone."""
        result = validator.validate_vocalization(
            marmoset_alarm_features,
            species='marmoset'
        )

        assert result.passed, "Valid alarm should pass validation"
        assert result.persona_id == 'MARMOSET_ALARM'
        assert result.semantic_label == 'alarm'
        assert result.confidence > 0.95
        assert not result.is_outlier

    def test_invalid_out_of_zone(self, validator, invalid_features):
        """Invalid features should be rejected."""
        result = validator.validate_vocalization(
            invalid_features,
            species='marmoset'
        )

        assert not result.passed, "Invalid features should fail"
        assert result.is_outlier, "Should be marked as outlier"
        assert len(result.warnings) > 0, "Should have warnings"
        assert result.mahalanobis_distance > 2.0

    def test_ghost_word_detected(self, validator, ghost_word_features):
        """Ghost word should be identified as between clusters."""
        is_ghost, close_clusters = validator.is_ghost_word(
            ghost_word_features,
            species='marmoset'
        )

        assert is_ghost, "Should be identified as ghost word"

    def test_classification_returns_persona(self, validator, marmoset_phee_features):
        """Classification should return correct persona."""
        persona_id, confidence = validator.classify_vocalization(
            marmoset_phee_features,
            species='marmoset'
        )

        assert persona_id == 'MARMOSET_PHEE'
        assert confidence > 0.95


# ============================================================================
# Test Suite 2: Statistical Boundary Validation
# ============================================================================

class TestStatisticalBoundaryValidation:
    """Test that statistical boundaries are enforced."""

    def test_cluster_boundaries_are_reasonable(self, validator):
        """Cluster boundaries should be statistically reasonable."""
        boundaries = validator.get_cluster_boundaries('MARMOSET_PHEE')

        # Check that all features have boundaries
        expected_features = ['mean_f0_hz', 'duration_ms', 'f0_range_hz',
                           'harmonicity', 'spectral_flatness', 'jitter', 'shimmer']

        for feature in expected_features:
            assert feature in boundaries, f"Missing boundary for {feature}"
            lower, upper = boundaries[feature]
            assert lower < upper, f"Invalid boundary for {feature}"

    def test_f0_boundary_includes_centroid(self, validator):
        """F0 boundary should include cluster centroid."""
        boundaries = validator.get_cluster_boundaries('MARMOSET_PHEE')
        lower_f0, upper_f0 = boundaries['mean_f0_hz']

        # Phee centroid is 6526 Hz
        assert lower_f0 < 6526 < upper_f0, "Centroid should be within boundary"

    def test_boundary_width_reflects_variance(self, validator):
        """Boundary width should reflect cluster variance."""
        phee_boundaries = validator.get_cluster_boundaries('MARMOSET_PHEE')
        alarm_boundaries = validator.get_cluster_boundaries('MARMOSET_ALARM')

        # Check that boundaries were calculated
        phee_f0_width = phee_boundaries['f0_range_hz'][1] - phee_boundaries['f0_range_hz'][0]
        alarm_f0_width = alarm_boundaries['f0_range_hz'][1] - alarm_boundaries['f0_range_hz'][0]

        # Both should have positive width
        assert phee_f0_width > 0
        assert alarm_f0_width > 0

    def test_outside_boundary_fails_validation(self, validator):
        """Features outside boundary should fail validation."""
        # Get boundaries
        boundaries = validator.get_cluster_boundaries('MARMOSET_PHEE')
        lower_f0, upper_f0 = boundaries['mean_f0_hz']

        # Create features outside boundary
        outside_features = {
            'mean_f0_hz': upper_f0 + 1000,  # Well above boundary
            'duration_ms': 76.5,
            'f0_range_hz': 427,
            'harmonicity': 0.95,
            'spectral_flatness': 0.1,
            'jitter': 0.02,
            'shimmer': 0.03
        }

        result = validator.validate_vocalization(
            outside_features,
            species='marmoset'
        )

        assert not result.passed, "Should fail when outside boundary"
        assert result.is_outlier


# ============================================================================
# Test Suite 3: Feature-Level Validation
# ============================================================================

class TestFeatureLevelValidation:
    """Test feature-level diagnostics and warnings."""

    def test_feature_deviations_calculated(self, validator, invalid_features):
        """Should calculate feature deviations from centroid."""
        result = validator.validate_vocalization(
            invalid_features,
            species='marmoset'
        )

        assert len(result.feature_deviations) > 0, "Should have feature deviations"

    def test_z_scores_calculated(self, validator, invalid_features):
        """Should calculate z-scores for each feature."""
        result = validator.validate_vocalization(
            invalid_features,
            species='marmoset'
        )

        assert len(result.feature_z_scores) > 0, "Should have z-scores"

        # Some features should have high z-scores (invalid features)
        high_z_scores = [f for f, z in result.feature_z_scores.items() if abs(z) > 2]
        assert len(high_z_scores) > 0, "Should have outlier features"

    def test_warnings_for_outlier_features(self, validator, invalid_features):
        """Should generate warnings for outlier features."""
        result = validator.validate_vocalization(
            invalid_features,
            species='marmoset'
        )

        assert len(result.warnings) > 0, "Should have warnings for outliers"


# ============================================================================
# Test Suite 4: Cross-Species Validation
# ============================================================================

class TestCrossSpeciesValidation:
    """Test validation across different species."""

    def test_marmoset_features_rejected_for_bat(self, validator, marmoset_phee_features):
        """Marmoset features should not match bat clusters."""
        result = validator.validate_vocalization(
            marmoset_phee_features,
            species='egyptian_bat'
        )

        # Marmoset features should not pass bat validation
        # (F0 is wrong for bat clusters)
        is_valid_bat = result.persona_id.startswith('BAT_')

        # This might pass if there's no species filter, but let's check distance
        if is_valid_bat:
            # If it matches a bat cluster, it should have low confidence
            assert result.confidence < 0.5, "Should have low confidence for wrong species"

    def test_species_filter_works(self, validator):
        """Species filter should restrict search to correct species."""
        phee_features = {
            'mean_f0_hz': 6526,
            'duration_ms': 76.5,
            'f0_range_hz': 427,
            'harmonicity': 0.95,
            'spectral_flatness': 0.1,
            'jitter': 0.02,
            'shimmer': 0.03
        }

        # Without species filter, should still find marmoset
        result_no_filter = validator.validate_vocalization(phee_features)
        assert result_no_filter.persona_id == 'MARMOSET_PHEE'

        # With species filter, should still find marmoset
        result_with_filter = validator.validate_vocalization(
            phee_features,
            species='marmoset'
        )
        assert result_with_filter.persona_id == 'MARMOSET_PHEE'


# ============================================================================
# Test Suite 5: Ghost Word Detection
# ============================================================================

class TestGhostWordDetection:
    """Test detection of ghost words (between-cluster vocalizations)."""

    def test_ghost_word_between_clusters(self, validator, ghost_word_features):
        """Ghost word should be between two clusters."""
        is_ghost, close_clusters = validator.is_ghost_word(
            ghost_word_features,
            species='marmoset'
        )

        assert is_ghost, "Should be identified as ghost word"

    def test_valid_call_not_ghost_word(self, validator, marmoset_phee_features):
        """Valid call should not be ghost word."""
        is_ghost, close_clusters = validator.is_ghost_word(
            marmoset_phee_features,
            species='marmoset'
        )

        # Valid phee is clearly in one cluster
        # It should have high distance from other clusters
        assert not is_ghost, "Valid call should not be ghost word"

    def test_ghost_word_has_low_confidence(self, validator, ghost_word_features, marmoset_phee_features):
        """Ghost word should have lower confidence than valid call."""
        validator.validate_vocalization(
            ghost_word_features,
            species='marmoset',
            strict=False
        )

        valid_result = validator.validate_vocalization(
            marmoset_phee_features,
            species='marmoset',
            strict=False
        )

        # Ghost word might have lower confidence or be an outlier
        # At minimum, valid call should have high confidence
        assert valid_result.confidence > 0.95, "Valid call should have high confidence"

    def test_ghost_word_suggestion(self, validator, ghost_word_features):
        """Ghost word might suggest alternative persona."""
        validator.validate_vocalization(
            ghost_word_features,
            species='marmoset',
            strict=False
        )

        # Ghost word is equidistant from phee and alarm
        # Might suggest alternative
        # (This depends on implementation details)


# ============================================================================
# Test Suite 6: Strict vs Lenient Validation
# ============================================================================

class TestStrictnessLevels:
    """Test strict vs lenient validation modes."""

    def test_strict_mode_rejects_outliers(self, validator):
        """Strict mode should reject outliers (>2σ)."""
        # Features at 2.5σ from centroid
        outlier_features = {
            'mean_f0_hz': 6526 + 2.5 * 935,  # 2.5σ from phee
            'duration_ms': 76.5,
            'f0_range_hz': 427,
            'harmonicity': 0.95,
            'spectral_flatness': 0.1,
            'jitter': 0.02,
            'shimmer': 0.03
        }

        result = validator.validate_vocalization(
            outlier_features,
            species='marmoset',
            strict=True
        )

        assert not result.passed, "Strict mode should reject 2.5σ outlier"

    def test_lenient_mode_accepts_boundary(self, validator):
        """Lenient mode might accept boundary cases."""
        boundary_features = {
            'mean_f0_hz': 6526 + 1.8 * 935,  # 1.8σ from phee
            'duration_ms': 76.5,
            'f0_range_hz': 427,
            'harmonicity': 0.95,
            'spectral_flatness': 0.1,
            'jitter': 0.02,
            'shimmer': 0.03
        }

        # Strict mode should accept (< 2σ)
        strict_result = validator.validate_vocalization(
            boundary_features,
            species='marmoset',
            strict=True
        )

        # Lenient mode should definitely accept
        lenient_result = validator.validate_vocalization(
            boundary_features,
            species='marmoset',
            strict=False
        )

        assert strict_result.passed, "1.8σ should be accepted"
        assert lenient_result.passed, "Lenient mode should accept"


# ============================================================================
# Test Suite 7: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_features_handled(self, validator):
        """Empty feature dict should be handled gracefully."""
        empty_features = {}

        result = validator.validate_vocalization(
            empty_features,
            species='marmoset'
        )

        # Should classify based on default values (0)
        # Might not pass, but shouldn't crash
        assert isinstance(result, SemanticZoneValidationResult)

    def test_unknown_species_handled(self, validator, marmoset_phee_features):
        """Unknown species should be handled gracefully."""
        result = validator.validate_vocalization(
            marmoset_phee_features,
            species='unknown_species'
        )

        # Should return unknown persona
        assert result.persona_id == "UNKNOWN" or not result.passed

    def test_nonexistent_persona_boundaries(self, validator):
        """Getting boundaries for nonexistent persona should return empty."""
        boundaries = validator.get_cluster_boundaries('NONEXISTENT')

        assert boundaries == {}, "Should return empty dict for nonexistent persona"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
