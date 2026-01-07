#!/usr/bin/env python3
"""
Unit tests for harmonic_affirmation module using TDD methodology.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from harmonic_affirmation import HarmonicAffirmation, HarmonicGroup


class TestHarmonicAffirmation:
    """Test suite for HarmonicAffirmation class."""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.harmonic_affirmation = HarmonicAffirmation()
        self.sample_phrases = [
            {'f0_mean': 440.0, 'features': {'f0_mean': 440.0}},
            {'f0_mean': 880.0, 'features': {'f0_mean': 880.0}},
            {'f0_mean': 1320.0, 'features': {'f0_mean': 1320.0}},
            {'f0_mean': 2000.0, 'features': {'f0_mean': 2000.0}},
            {'f0_mean': 3000.0, 'features': {'f0_mean': 3000.0}},
        ]

    def test_initialization(self):
        """Test HarmonicAffirmation initializes correctly."""
        assert self.harmonic_affirmation.harmonic_threshold_ratio == 0.2
        assert self.harmonic_affirmation.min_harmonic_group_size == 2
        assert self.harmonic_affirmation.max_harmonic_deviation == 0.15

    def test_analyze_harmonic_series(self):
        """Test basic harmonic series analysis."""
        # Perfect harmonic series: 440Hz, 880Hz, 1320Hz (1st, 2nd, 3rd harmonics)
        result = self.harmonic_affirmation.analyze_harmonic_series(self.sample_phrases)

        assert 'total_harmonic_phrases' in result
        assert 'fundamental_freq' in result
        assert 'harmonic_groups' in result
        assert 'harmonic_ratio' in result
        assert 'threshold' in result

        # Should identify 440Hz as fundamental
        assert result['fundamental_freq'] == 440.0
        # Should group 440, 880, 1320 as harmonics
        assert 'harmonic_1' in result['harmonic_groups']  # 440 Hz
        assert 'harmonic_2' in result['harmonic_groups']  # 880 Hz
        assert 'harmonic_3' in result['harmonic_groups']  # 1320 Hz

    def test_harmonic_group_analysis(self):
        """Test HarmonicGroup class functionality."""
        group = HarmonicGroup('test_group', [0, 1, 2])

        assert group.name == 'test_group'
        assert group.phrase_indices == [0, 1, 2]
        assert group.size == 3

        # Test harmonic calculation
        f0_values = [440.0, 880.0, 1320.0]
        avg_f0 = group.calculate_average_f0(f0_values)
        assert avg_f0 == 880.0

        # Test harmonic deviation
        deviation = group.calculate_harmonic_deviation(f0_values, 880.0)
        assert abs(deviation) < 0.01  # Should be very small for perfect harmonics

    def test_analyze_with_noise(self):
        """Test analysis with harmonic noise."""
        # Add non-harmonic frequency
        noisy_phrases = self.sample_phrases + [
            {'f0_mean': 2500.0, 'features': {'f0_mean': 2500.0}}  # Not a harmonic of 440
        ]

        result = self.harmonic_affirmation.analyze_harmonic_series(noisy_phrases)

        # Should still identify harmonic groups
        assert result['harmonic_ratio'] < 1.0  # Not all phrases are harmonic
        assert 'non_harmonic' in result['harmonic_groups']
        assert len(result['harmonic_groups']['non_harmonic']) > 0

    def test_analyze_perfect_vs_imperfect_harmonics(self):
        """Test analysis of perfect vs imperfect harmonics."""
        # Perfect harmonics
        perfect_phrases = [
            {'f0_mean': 500.0, 'features': {'f0_mean': 500.0}},
            {'f0_mean': 1000.0, 'features': {'f0_mean': 1000.0}},
            {'f0_mean': 1500.0, 'features': {'f0_mean': 1500.0}},
        ]

        # Imperfect harmonics (5% deviation)
        imperfect_phrases = [
            {'f0_mean': 500.0, 'features': {'f0_mean': 500.0}},
            {'f0_mean': 1020.0, 'features': {'f0_mean': 1020.0}},  # 4% deviation
            {'f0_mean': 1480.0, 'features': {'f0_mean': 1480.0}},  # 1.3% deviation
        ]

        perfect_result = self.harmonic_affirmation.analyze_harmonic_series(perfect_phrases)
        imperfect_result = self.harmonic_affirmation.analyze_harmonic_series(imperfect_phrases)

        # Perfect should have higher harmonic ratio
        assert perfect_result['harmonic_ratio'] >= imperfect_result['harmonic_ratio']

    def test_empty_input_handling(self):
        """Test handling of empty input."""
        result = self.harmonic_affirmation.analyze_harmonic_series([])

        assert result['total_harmonic_phrases'] == 0
        assert result['fundamental_freq'] is None
        assert result['harmonic_ratio'] == 0.0

    def test_single_phrase_handling(self):
        """Test handling of single phrase."""
        single_phrase = [{'f0_mean': 440.0, 'features': {'f0_mean': 440.0}}]
        result = self.harmonic_affirmation.analyze_harmonic_series(single_phrase)

        assert result['total_harmonic_phrases'] == 1
        assert result['harmonic_ratio'] == 1.0

    def test_threshold_configuration(self):
        """Test threshold configuration affects analysis."""
        # Use very strict threshold
        strict_affirmation = HarmonicAffirmation(harmonic_threshold_ratio=0.05)
        result = strict_affirmation.analyze_harmonic_series(self.sample_phrases)

        # With strict threshold, may not identify all harmonics
        assert 'harmonic_groups' in result

    def test_f0_range_validation(self):
        """Test F0 range validation."""
        # Test with extreme values
        extreme_phrases = [
            {'f0_mean': 20.0, 'features': {'f0_mean': 20.0}},  # Very low
            {'f0_mean': 20000.0, 'features': {'f0_mean': 20000.0}},  # Very high
        ]

        result = self.harmonic_affirmation.analyze_harmonic_series(extreme_phrases)
        # Should handle without errors
        assert 'total_harmonic_phrases' in result

    def test_harmonic_distance_calculation(self):
        """Test harmonic distance calculation between frequencies."""
        # Test perfect harmonic relationship
        distance = self.harmonic_affirmation.calculate_harmonic_distance(880.0, 440.0)
        assert distance == 0.0  # Perfect 2nd harmonic

        # Test non-harmonic relationship
        distance = self.harmonic_affirmation.calculate_harmonic_distance(900.0, 440.0)
        assert distance > 0.0

        # Test octave equivalence
        distance = self.harmonic_affirmation.calculate_harmonic_distance(880.0, 220.0)
        assert distance == 0.0  # Perfect octave

    def test_harmonic_grouping_algorithm(self):
        """Test the harmonic grouping algorithm."""
        # Multiple fundamentals
        multi_fundamental_phrases = [
            {'f0_mean': 440.0, 'features': {'f0_mean': 440.0}},  # Fundamental 1
            {'f0_mean': 880.0, 'features': {'f0_mean': 880.0}},  # 2nd harmonic of 440
            {'f0_mean': 660.0, 'features': {'f0_mean': 660.0}},  # Fundamental 2
            {'f0_mean': 1320.0, 'features': {'f0_mean': 1320.0}},  # 2nd harmonic of 660
            {'f0_mean': 1000.0, 'features': {'f0_mean': 1000.0}},  # Non-harmonic
        ]

        result = self.harmonic_affirmation.analyze_harmonic_series(multi_fundamental_phrases)

        # Should identify harmonic groups (only the lowest frequency will be fundamental)
        harmonic_groups = [f for f in result['harmonic_groups'].keys() if f.startswith('harmonic_')]
        assert len(harmonic_groups) >= 1  # At least one harmonic group
        # Should have non-harmonic group
        assert 'non_harmonic' in result['harmonic_groups']

    def test_analysis_report_generation(self):
        """Test comprehensive analysis report generation."""
        result = self.harmonic_affirmation.generate_analysis_report(self.sample_phrases)

        # Check report structure
        required_fields = [
            'summary', 'detailed_analysis', 'recommendations',
            'confidence_score', 'metadata'
        ]

        for field in required_fields:
            assert field in result

        # Check content quality
        assert isinstance(result['confidence_score'], float)
        assert 0.0 <= result['confidence_score'] <= 1.0
        assert isinstance(result['summary'], str)
        assert len(result['summary']) > 0
