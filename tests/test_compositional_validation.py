#!/usr/bin/env python3
"""
Unit tests for compositional_validation module using TDD methodology.
"""

import pytest
import numpy as np
from typing import Dict, List, Any
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from compositional_validation import CompositionalValidator, ValidationRule, ValidationResult


class TestCompositionalValidator:
    """Test suite for CompositionalValidator class."""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.validator = CompositionalValidator()
        self.sample_grammar = {
            'phrase1': {'phrase2': 10, 'phrase3': 5},
            'phrase2': {'phrase4': 8, 'phrase1': 2},
            'phrase3': {'phrase1': 3, 'phrase4': 7},
            'phrase4': {'phrase1': 1}
        }
        self.sample_phrases = [
            {'key': 'phrase1', 'sequence_position': 0, 'context': 'start'},
            {'key': 'phrase2', 'sequence_position': 1, 'context': 'middle'},
            {'key': 'phrase4', 'sequence_position': 2, 'context': 'end'}
        ]

    def test_initialization(self):
        """Test CompositionalValidator initializes correctly."""
        assert self.validator.min_transitions == 3
        assert self.validator.significance_threshold == 0.05
        assert self.validator.min_sequence_length == 2
        assert hasattr(self.validator, 'grammar')
        assert hasattr(self.validator, 'validation_rules')

    def test_chi_squared_test_uniformity(self):
        """Test chi-squared test for uniformity in grammar transitions."""
        # Perfectly uniform grammar
        uniform_grammar = {
            'A': {'B': 5, 'C': 5},
            'B': {'A': 5, 'C': 5},
            'C': {'A': 5, 'B': 5}
        }

        result = self.validator.perform_chi_squared_test(uniform_grammar)

        assert 'chi_squared' in result
        assert 'p_value' in result
        assert 'degrees_of_freedom' in result
        assert 'is_significant' in result
        assert result['degrees_of_freedom'] == 2  # 3 categories - 1

    def test_chi_squared_test_non_uniform(self):
        """Test chi-squared test for non-uniform grammar."""
        # Highly non-uniform grammar
        non_uniform_grammar = {
            'A': {'B': 95, 'C': 5},
            'B': {'A': 10, 'C': 90},
            'C': {'A': 5, 'B': 5}
        }

        result = self.validator.perform_chi_squared_test(non_uniform_grammar)

        assert result['is_significant']  # Should be significant for non-uniform data

    def test_sequence_validation(self):
        """Test sequence validation against grammar rules."""
        result = self.validator.validate_sequence(self.sample_phrases, self.sample_grammar)

        assert 'is_valid' in result
        assert 'violations' in result['validation_details']
        assert 'confidence' in result
        assert 'validation_details' in result

        # Should have validation details
        assert 'transitions_analyzed' in result['validation_details']
        assert 'transitions_valid' in result['validation_details']

    def test_validate_grammar_structure(self):
        """Test grammar structure validation."""
        result = self.validator.validate_grammar_structure(self.sample_grammar)

        assert 'is_valid' in result
        assert 'errors' in result
        assert 'warnings' in result
        assert 'statistics' in result

        # Should check for missing transitions, cycles, etc.
        assert 'total_transitions' in result['statistics']
        assert 'unique_phrases' in result['statistics']

    def test_build_contingency_table(self):
        """Test contingency table building for chi-squared test."""
        contingency = self.validator._build_contingency_table(self.sample_grammar)

        assert isinstance(contingency, dict)
        assert 'from_phrases' in contingency
        assert 'to_phrases' in contingency
        assert 'observed_counts' in contingency
        assert 'expected_counts' in contingency

    def test_generate_validation_report(self):
        """Test comprehensive validation report generation."""
        result = self.validator.generate_validation_report(
            self.sample_phrases,
            self.sample_grammar
        )

        assert 'summary' in result
        assert 'grammar_analysis' in result
        assert 'sequence_validation' in result
        assert 'recommendations' in result
        assert 'metadata' in result

        # Check report content
        assert isinstance(result['summary'], str)
        assert isinstance(result['recommendations'], list)
        assert 'timestamp' in result['metadata']

    def test_empty_grammar_handling(self):
        """Test handling of empty grammar."""
        result = self.validator.validate_grammar_structure({})

        assert result['is_valid'] == False
        assert len(result['errors']) > 0
        assert 'Grammar is empty' in result['errors']

    def test_single_phrase_sequence(self):
        """Test handling of single phrase sequence."""
        single_phrase = [{'key': 'phrase1', 'sequence_position': 0}]

        result = self.validator.validate_sequence(single_phrase, self.sample_grammar)

        assert result['is_valid'] == True  # Single phrase is always valid
        assert result['validation_details']['transitions_analyzed'] == 0

    def test_cycle_detection(self):
        """Test cycle detection in grammar."""
        # Grammar with cycles
        cyclic_grammar = {
            'A': {'B': 10},
            'B': {'C': 10},
            'C': {'A': 10}  # Creates A->B->C->A cycle
        }

        result = self.validator.validate_grammar_structure(cyclic_grammar)

        # Should detect cycles as warnings
        assert 'potential cycle' in str(result['warnings'])

    def test_validation_rule_creation(self):
        """Test ValidationRule class functionality."""
        rule = ValidationRule(
            name='test_rule',
            description='Test validation rule',
            validator_func=lambda x: len(x) > 0,
            severity='error'
        )

        assert rule.name == 'test_rule'
        assert rule.description == 'Test validation rule'
        assert rule.severity == 'error'
        assert rule.validator_func(['a', 'b']) == True

    def test_validation_result_class(self):
        """Test ValidationResult class functionality."""
        result = ValidationResult(
            is_valid=True,
            confidence=0.95,
            message='Validation passed'
        )

        assert result.is_valid == True
        assert result.confidence == 0.95
        assert result.message == 'Validation passed'

        # Test add_violation
        result.add_violation('test_violation', 'warning', 'warning')
        assert len(result.violations) == 1
        assert result.violations[0]['type'] == 'test_violation'

    def test_probability_calculation(self):
        """Test probability calculation for transitions."""
        # Test with known probabilities
        from_phrase = 'phrase1'
        to_phrase = 'phrase2'

        prob = self.validator.calculate_transition_probability(
            from_phrase, to_phrase, self.sample_grammar
        )

        assert 0 <= prob <= 1
        assert prob == 10 / 15  # 10 transitions from phrase1 to phrase2 out of 15 total from phrase1

    def test_normalize_transitions(self):
        """Test transition normalization."""
        # Unnormalized transitions
        unnormalized = {
            'phrase1': {'phrase2': 10, 'phrase3': 5, 'phrase4': 3},
            'phrase2': {'phrase1': 2}
        }

        normalized = self.validator._normalize_transitions(unnormalized)

        # Check that rows sum to 1 (within floating point tolerance)
        for from_phrase, transitions in normalized.items():
            assert abs(sum(transitions.values()) - 1.0) < 1e-10

    def test_edge_cases_handling(self):
        """Test edge case handling."""
        # None/empty inputs
        result1 = self.validator.validate_sequence([], self.sample_grammar)
        result2 = self.validator.validate_grammar_structure(None)

        assert result1['is_valid'] == True  # Empty sequence is valid
        assert result2['is_valid'] == False  # None grammar is invalid

    def test_comprehensive_validation_pipeline(self):
        """Test complete validation pipeline."""
        # Create more complex test data
        complex_grammar = {
            'start': {'middle': 20, 'end': 5},
            'middle': {'middle': 10, 'end': 15},
            'end': {'start': 2}  # Small back-reference
        }

        complex_sequence = [
            {'key': 'start', 'position': 0},
            {'key': 'middle', 'position': 1},
            {'key': 'middle', 'position': 2},
            {'key': 'end', 'position': 3}
        ]

        # Run full validation pipeline
        grammar_result = self.validator.validate_grammar_structure(complex_grammar)
        sequence_result = self.validator.validate_sequence(complex_sequence, complex_grammar)
        report = self.validator.generate_validation_report(complex_sequence, complex_grammar)

        # Validate results
        assert grammar_result['is_valid'] == True
        assert sequence_result['is_valid'] == True
        assert 'summary' in report
        assert len(report['recommendations']) >= 0  # Could be 0 or more

        # Check statistical analysis
        assert 'statistics' in report['grammar_analysis']
        assert 'validation_details' in report['sequence_validation']