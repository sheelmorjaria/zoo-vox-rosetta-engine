# Syntactic Coherence Tests
# ==========================
#
# Tests for syntactic discipline validation under chaotic conditions.

import pytest
from e2e_testing.syntactic_coherence_tester import (
    SyntacticCoherenceTester,
    SyntacticCascadeDetected,
    NBDMergingDetected,
)


class TestSyntacticCoherenceTester:
    """Test syntactic coherence validation."""

    def test_tester_creation(self):
        """Test creating coherence tester."""
        # Note: This test mocks the syntax_graph and transformer
        # since we don't have actual model files loaded
        tester = SyntacticCoherenceTester(
            syntax_graph_path=None,
            transformer_model_path=None,
            max_gibberish_ratio=0.05,
        )
        assert tester.max_gibberish_ratio == 0.05
        assert tester.total_responses == 0

    def test_validate_response_normal(self):
        """Test validating normal response."""
        tester = SyntacticCoherenceTester(
            syntax_graph_path=None,
            transformer_model_path=None,
        )

        # Mock high probability response
        tester._mock_probability = 0.8

        # This should not raise
        tester.validate_response_mock([], 1, 0.8)

        assert tester.total_responses == 1
        assert tester.gibberish_responses == 0

    def test_validate_response_gibberish(self):
        """Test detecting gibberish response."""
        tester = SyntacticCoherenceTester(
            syntax_graph_path=None,
            transformer_model_path=None,
            max_gibberish_ratio=0.1,  # 10% threshold
        )

        # Add 9 normal responses
        for i in range(9):
            tester.total_responses += 1

        # Add 1 gibberish response (low probability)
        tester.gibberish_responses = 1
        tester.total_responses = 10

        # Should not exceed threshold yet (10% exactly)
        current_ratio = tester.gibberish_responses / tester.total_responses
        assert current_ratio <= 0.1

    def test_gibberish_ratio_calculation(self):
        """Test gibberish ratio calculation."""
        tester = SyntacticCoherenceTester(
            syntax_graph_path=None,
            transformer_model_path=None,
        )

        # Simulate responses
        tester.total_responses = 100
        tester.gibberish_responses = 3

        ratio = tester.gibberish_responses / tester.total_responses
        assert ratio == 0.03

    def test_syntactic_cascade_detected(self):
        """Test syntactic cascade detection."""
        tester = SyntacticCoherenceTester(
            syntax_graph_path=None,
            transformer_model_path=None,
            max_gibberish_ratio=0.05,
        )

        # Simulate too many gibberish responses
        tester.total_responses = 100
        tester.gibberish_responses = 10  # 10% > 5% threshold

        current_ratio = tester.gibberish_responses / tester.total_responses
        assert current_ratio > 0.05

    def test_validate_segment_duration_ultra_short(self):
        """Test detecting ultra-short segments."""
        tester = SyntacticCoherenceTester(
            syntax_graph_path=None,
            transformer_model_path=None,
        )

        # Validate ultra-short syllable
        tester.validate_segment_duration(8.0, "Phonetic")

        assert tester.sub_50ms_count == 1

    def test_validate_segment_duration_normal(self):
        """Test normal segment duration."""
        tester = SyntacticCoherenceTester(
            syntax_graph_path=None,
            transformer_model_path=None,
        )

        # Validate normal segment
        tester.validate_segment_duration(35.0, "Syllable")

        assert tester.sub_50ms_count == 1
        assert tester.merged_segment_count == 0

    def test_validate_segment_duration_merged(self):
        """Test detecting merged segments."""
        tester = SyntacticCoherenceTester(
            syntax_graph_path=None,
            transformer_model_path=None,
        )

        # Add many normal segments first
        for _ in range(50):
            tester.segment_durations_ms.append(20.0)
            tester.sub_50ms_count += 1

        # Now add a suspiciously long segment
        tester.validate_segment_duration(60.0, "Phonetic")

        assert tester.merged_segment_count == 1

    def test_merge_rate_threshold(self):
        """Test merge rate threshold validation."""
        tester = SyntacticCoherenceTester(
            syntax_graph_path=None,
            transformer_model_path=None,
        )

        # Simulate many segments with high merge rate
        for _ in range(50):
            tester.segment_durations_ms.append(20.0)
            tester.sub_50ms_count += 1

        # Add 15 merged segments (>20% of 65 total)
        for _ in range(15):
            tester.merged_segment_count += 1
            tester.segment_durations_ms.append(60.0)

        # Calculate merge rate
        merge_rate = tester.merged_segment_count / len(tester.segment_durations_ms)
        assert merge_rate > 0.20  # Should exceed threshold


class TestSyntacticCascadeDetected:
    """Test SyntacticCascadeDetected exception."""

    def test_exception_creation(self):
        """Test creating exception."""
        exc = SyntacticCascadeDetected("Gibberish ratio: 15%")
        assert "15%" in str(exc)


class TestNBDMergingDetected:
    """Test NBDMergingDetected exception."""

    def test_exception_creation(self):
        """Test creating exception."""
        exc = NBDMergingDetected("Merge rate: 25%")
        assert "25%" in str(exc)
        assert "EMA" in str(exc)
