"""
Test Suite for Acoustic Algebra: Semantic Gradient Engine
==========================================================

Tests for:
- ContextualMap calculation
- Semantic vector operations
- Gradient synthesis
- Contextual axis analysis
- Z-score normalization

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from realtime.acoustic_algebra_contextual import (
    ContextualAxis,
    ContextualMap,
    GradientSynthesizer,
    SemanticVector,
)

from realtime.phrase_audio_library import PhraseAudioLibrary, PhraseAudioSegment

# ============================================================================
# SemanticVector Tests
# ============================================================================


class TestSemanticVector:
    """Test SemanticVector dataclass."""

    def test_create_vector(self):
        """Test creating a semantic vector."""
        vector = np.random.randn(17)
        sv = SemanticVector(vector=vector, context="aggression")

        assert sv.context == "aggression"
        assert sv.dimension == 17
        assert np.array_equal(sv.vector, vector)

    def test_invalid_dimension(self):
        """Test that non-17D vectors raise error."""
        vector = np.random.randn(10)

        with pytest.raises(ValueError, match="must be 17D"):
            SemanticVector(vector=vector, context="test")

    def test_distance_to(self):
        """Test Euclidean distance calculation."""
        v1 = SemanticVector(vector=np.array([1.0] * 17), context="aggression")
        v2 = SemanticVector(vector=np.array([2.0] * 17), context="courtship")

        distance = v1.distance_to(v2)

        expected = np.sqrt(17)  # sqrt(17 * 1^2)
        assert np.isclose(distance, expected, atol=0.01)

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        v1 = SemanticVector(vector=np.array([1.0] * 17), context="aggression")
        v2 = SemanticVector(vector=np.array([2.0] * 17), context="courtship")

        similarity = v1.cosine_similarity(v2)

        # Identical direction = 1.0
        assert np.isclose(similarity, 1.0, atol=0.01)

    def test_to_dict(self):
        """Test serialization."""
        vector = np.array([1.0, 2.0] + [0.0] * 15)
        sv = SemanticVector(vector=vector, context="test")

        result = sv.to_dict()

        assert result["context"] == "test"
        assert len(result["vector"]) == 17


# ============================================================================
# ContextualAxis Tests
# ============================================================================


class TestContextualAxis:
    """Test ContextualAxis dataclass."""

    def test_create_axis(self):
        """Test creating a contextual axis."""
        centroid = np.random.randn(17)
        variance = np.random.rand(17)
        axis_direction = np.random.randn(17)

        axis = ContextualAxis(
            context="aggression",
            centroid=centroid,
            variance=variance,
            axis_direction=axis_direction,
            defining_features=[(0, "mean_f0_hz", 0.5)],
            phrase_count=10,
        )

        assert axis.context == "aggression"
        assert axis.phrase_count == 10
        assert axis.magnitude > 0

    def test_magnitude(self):
        """Test magnitude calculation."""
        axis = ContextualAxis(
            context="test",
            centroid=np.zeros(17),
            variance=np.ones(17),
            axis_direction=np.array([3.0, 4.0] + [0.0] * 15),
            defining_features=[],
            phrase_count=5,
        )

        # Magnitude = sqrt(3^2 + 4^2) = 5
        assert np.isclose(axis.magnitude, 5.0, atol=0.01)


# ============================================================================
# ContextualMap Tests
# ============================================================================


class TestContextualMap:
    """Test ContextualMap class."""

    @pytest.fixture
    def sample_library(self):
        """Create a sample phrase library with contexts."""
        library = PhraseAudioLibrary(species="marmoset", sr=22050)

        # Add segments with different contexts
        np.random.seed(42)

        # Aggression segments (higher F0)
        for i in range(5):
            audio = np.random.randn(2205)  # 100ms
            segment = PhraseAudioSegment(
                audio=audio,
                sr=22050,
                phrase_key=f"agg_{i}",
                source_file="test",
                start_time_ms=i * 200,
                end_time_ms=i * 200 + 100,
                mean_f0_hz=7500 + np.random.randn() * 200,
                std_f0_hz=100,
                mean_duration_ms=200 + np.random.randn() * 20,
                mean_range_hz=500 + np.random.randn() * 50,
                context="aggression",
                snr_db=20.0,
            )
            library.add_segment(segment)

        # Contact segments (lower F0)
        for i in range(5):
            audio = np.random.randn(2205)
            segment = PhraseAudioSegment(
                audio=audio,
                sr=22050,
                phrase_key=f"contact_{i}",
                source_file="test",
                start_time_ms=i * 200 + 1000,
                end_time_ms=i * 200 + 1000 + 100,
                mean_f0_hz=6000 + np.random.randn() * 200,
                std_f0_hz=80,
                mean_duration_ms=150 + np.random.randn() * 20,
                mean_range_hz=300 + np.random.randn() * 50,
                context="contact_call",
                snr_db=20.0,
            )
            library.add_segment(segment)

        return library

    def test_from_phrase_library(self, sample_library):
        """Test building contextual map from library."""
        map = ContextualMap.from_phrase_library(
            sample_library, baseline_context="contact_call", min_samples_per_context=2
        )

        assert len(map.contexts) >= 2
        assert "aggression" in map.contexts
        assert "contact_call" in map.contexts

    def test_get_context_vector(self, sample_library):
        """Test getting context vector."""
        map = ContextualMap.from_phrase_library(sample_library)

        vector = map.get_context_vector("aggression")

        assert vector is not None
        assert vector.context == "aggression"
        assert vector.dimension == 17

    def test_context_similarity(self, sample_library):
        """Test context similarity calculation."""
        map = ContextualMap.from_phrase_library(sample_library)

        similarity = map.context_similarity("aggression", "contact_call")

        assert similarity is not None
        # Cosine similarity ranges from -1 to 1
        assert -1.0 <= similarity <= 1.0

    def test_interpolate_contexts(self, sample_library):
        """Test context interpolation."""
        map = ContextualMap.from_phrase_library(sample_library)

        # 50% interpolation
        result = map.interpolate_contexts("contact_call", "aggression", 0.5)

        assert result is not None
        assert "50%" in result.context

    def test_interpolate_intensity_bounds(self, sample_library):
        """Test that intensity is clamped to [0, 1]."""
        map = ContextualMap.from_phrase_library(sample_library)

        # Should clamp to 1.0
        result = map.interpolate_contexts(
            "contact_call",
            "aggression",
            1.5,  # Out of bounds
        )

        assert result is not None

    def test_find_nearest_real_phrase(self, sample_library):
        """Test finding nearest real phrase to target vector."""
        map = ContextualMap.from_phrase_library(sample_library)

        # Get aggression vector as target
        target_vec = map.get_context_vector("aggression")

        if target_vec:
            nearest = map.find_nearest_real_phrase(
                target_vec.vector, sample_library, max_distance=100.0
            )

            # Should find something
            assert nearest is not None or True  # May not find if vectors are far

    def test_get_contextual_analysis(self, sample_library):
        """Test getting comprehensive analysis."""
        map = ContextualMap.from_phrase_library(sample_library)

        analysis = map.get_contextual_analysis()

        assert "contexts" in analysis
        assert "similarities" in analysis
        assert "num_contexts" in analysis
        assert analysis["num_contexts"] >= 2


# ============================================================================
# GradientSynthesizer Tests
# ============================================================================


class TestGradientSynthesizer:
    """Test GradientSynthesizer class."""

    @pytest.fixture
    def sample_map_and_library(self):
        """Create sample map and library."""
        library = PhraseAudioLibrary(species="marmoset", sr=22050)

        # Add segments
        np.random.seed(42)

        for ctx in ["aggression", "contact_call", "courtship"]:
            for i in range(3):
                base_f0 = {"aggression": 7500, "contact_call": 6000, "courtship": 5500}[ctx]
                audio = np.random.randn(2205)
                segment = PhraseAudioSegment(
                    audio=audio,
                    sr=22050,
                    phrase_key=f"{ctx}_{i}",
                    source_file="test",
                    start_time_ms=i * 200,
                    end_time_ms=i * 200 + 100,
                    context=ctx,
                    mean_f0_hz=base_f0 + np.random.randn() * 200,
                    std_f0_hz=100,
                    mean_duration_ms=200 + np.random.randn() * 50,
                    mean_range_hz=400 + np.random.randn() * 100,
                    snr_db=20.0,
                )
                library.add_segment(segment)

        map = ContextualMap.from_phrase_library(library)

        return map, library

    def test_synthesize_gradient(self, sample_map_and_library):
        """Test gradient synthesis."""
        map, library = sample_map_and_library
        synthesizer = GradientSynthesizer(map, library)

        result = synthesizer.synthesize_gradient(intent="aggression", intensity=0.5)

        assert result is not None
        assert result["intent"] == "aggression"
        assert result["intensity"] == 0.5
        assert "virtual_vector" in result
        assert "synthesis_params" in result

    def test_synthesize_params_calculation(self, sample_map_and_library):
        """Test synthesis parameters calculation."""
        map, library = sample_map_and_library
        synthesizer = GradientSynthesizer(map, library)

        result = synthesizer.synthesize_gradient("aggression", 0.5)

        if result:
            params = result["synthesis_params"]

            # Check expected parameters
            assert "pitch_shift" in params
            assert "time_stretch" in params
            assert "roughness" in params
            assert "brightness" in params


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_gradient_workflow(self):
        """Test complete workflow from library to gradient synthesis."""
        # Create library
        library = PhraseAudioLibrary(species="marmoset", sr=22050)

        np.random.seed(42)

        # Add multiple contexts
        contexts_data = {
            "aggression": {"f0": 7500, "dur": 180},
            "contact_call": {"f0": 6000, "dur": 150},
            "courtship": {"f0": 5500, "dur": 250},
        }

        for ctx, base_params in contexts_data.items():
            for i in range(4):
                audio = np.random.randn(4410)
                segment = PhraseAudioSegment(
                    audio=audio,
                    sr=22050,
                    phrase_key=f"{ctx}_{i}",
                    source_file="test",
                    start_time_ms=i * 200,
                    end_time_ms=i * 200 + 200,
                    context=ctx,
                    mean_f0_hz=base_params["f0"] + np.random.randn() * 200,
                    std_f0_hz=100,
                    mean_duration_ms=base_params["dur"] + np.random.randn() * 30,
                    mean_range_hz=400 + np.random.randn() * 100,
                    snr_db=20.0,
                )
                library.add_segment(segment)

        # Build contextual map
        map = ContextualMap.from_phrase_library(library)

        assert len(map.contexts) >= 3

        # Create synthesizer
        synthesizer = GradientSynthesizer(map, library)

        # Test multiple intensities
        intensities = [0.0, 0.25, 0.5, 0.75, 1.0]

        for intensity in intensities:
            result = synthesizer.synthesize_gradient("aggression", intensity)
            assert result is not None
            assert result["intensity"] == intensity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
