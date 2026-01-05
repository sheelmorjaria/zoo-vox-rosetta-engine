"""
Compositional Validation Module

This module provides functionality for validating sentence-phrase compositions
using statistical methods and grammar rule analysis.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict, Counter
import math
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Represents a validation rule for compositional analysis."""

    name: str
    description: str
    validator_func: Callable
    severity: str = 'error'  # 'error', 'warning', 'info'

    def validate(self, data: Any) -> bool:
        """Validate data using this rule."""
        try:
            return self.validator_func(data)
        except Exception as e:
            logger.warning(f"Validation rule '{self.name}' failed: {e}")
            return False


@dataclass
class ValidationResult:
    """Represents the result of a validation operation."""

    is_valid: bool
    confidence: float
    message: str
    violations: List[Dict] = None

    def __post_init__(self):
        if self.violations is None:
            self.violations = []

    def add_violation(self, violation_type: str, message: str, severity: str = 'warning'):
        """Add a violation to the result."""
        self.violations.append({
            'type': violation_type,
            'message': message,
            'severity': severity
        })
        self.is_valid = False


class CompositionalValidator:
    """
    Validates sentence-phrase compositions using statistical methods.

    This class implements the compositional validation functionality that was missing
    from the Rosetta Stone analysis pipeline.
    """

    def __init__(self,
                 min_transitions: int = 3,
                 significance_threshold: float = 0.05,
                 min_sequence_length: int = 2):
        """
        Initialize CompositionalValidator with configuration parameters.

        Args:
            min_transitions: Minimum number of transitions for meaningful analysis
            significance_threshold: Threshold for statistical significance (p-value)
            min_sequence_length: Minimum sequence length for validation
        """
        self.min_transitions = min_transitions
        self.significance_threshold = significance_threshold
        self.min_sequence_length = min_sequence_length
        self.grammar: Optional[Dict] = None
        self.validation_rules: List[ValidationRule] = []

    def validate_grammar_structure(self, grammar: Dict) -> Dict[str, Any]:
        """
        Validate the structure and integrity of a grammar.

        Args:
            grammar: Grammar dictionary with transition counts

        Returns:
            Dictionary with validation results
        """
        if grammar is None:
            return {
                'is_valid': False,
                'errors': ['Grammar is None'],
                'warnings': [],
                'statistics': {}
            }

        if not grammar:
            return {
                'is_valid': False,
                'errors': ['Grammar is empty'],
                'warnings': [],
                'statistics': {}
            }

        errors = []
        warnings = []
        statistics = self._calculate_grammar_statistics(grammar)

        # Check for circular references
        cycles = self._detect_cycles(grammar)
        if cycles:
            warnings.append(f"Detected {len(cycles)} potential cycle(s)")

        # Check for disconnected phrases
        disconnected = self._find_disconnected_phrases(grammar)
        if disconnected:
            warnings.append(f"Found {len(disconnected)} disconnected phrase(s)")

        # Check for missing transitions from some phrases
        sources_without_targets = self._find_sources_without_targets(grammar)
        if sources_without_targets:
            warnings.append(f"Found {len(sources_without_targets)} source phrase(s) with no targets")

        # Check grammar completeness
        total_transitions = statistics['total_transitions']
        if total_transitions < self.min_transitions:
            errors.append(f"Insufficient transitions ({total_transitions} < {self.min_transitions})")

        # Check for balance (each phrase should have some incoming transitions)
        phrases_without_incoming = self._find_phrases_without_incoming(grammar)
        if phrases_without_incoming and len(phrases_without_incoming) < len(grammar) * 0.3:
            warnings.append(f"Found {len(phrases_without_incoming)} phrase(s) without incoming transitions")

        # Add unique_phrases to statistics
        statistics['unique_phrases'] = len(set(grammar.keys()))

        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'statistics': statistics
        }

    def validate_sequence(self, sequence: List[Dict], grammar: Dict) -> Dict[str, Any]:
        """
        Validate a sequence against grammar rules.

        Args:
            sequence: List of phrase dictionaries with sequence information
            grammar: Grammar dictionary with transition counts

        Returns:
            Dictionary with sequence validation results
        """
        if not sequence:
            return {
                'is_valid': True,
                'confidence': 1.0,
                'message': 'Empty sequence is valid',
                'validation_details': {'transitions_analyzed': 0, 'transitions_valid': 0}
            }

        if len(sequence) < self.min_sequence_length:
            return {
                'is_valid': True,
                'confidence': 0.5,
                'message': f'Short sequence (length {len(sequence)} < {self.min_sequence_length})',
                'validation_details': {'transitions_analyzed': 0, 'transitions_valid': 0}
            }

        # Calculate transition probabilities
        transition_probs = self._normalize_transitions(grammar)

        # Validate each transition
        transitions_analyzed = 0
        transitions_valid = 0
        violations = []
        confidence_scores = []

        for i in range(len(sequence) - 1):
            from_phrase = sequence[i]['key']
            to_phrase = sequence[i + 1]['key']

            if from_phrase in grammar and to_phrase in grammar[from_phrase]:
                transitions_analyzed += 1

                # Get probability of this transition
                if from_phrase in transition_probs and to_phrase in transition_probs[from_phrase]:
                    prob = transition_probs[from_phrase][to_phrase]
                    confidence_scores.append(prob)
                else:
                    prob = 0.0

                # Consider transition valid if probability is reasonable
                if prob > 0.01:  # Minimum threshold
                    transitions_valid += 1
                else:
                    violations.append({
                        'type': 'low_probability_transition',
                        'from': from_phrase,
                        'to': to_phrase,
                        'probability': prob
                    })
            else:
                transitions_analyzed += 1
                violations.append({
                    'type': 'missing_transition',
                    'from': from_phrase,
                    'to': to_phrase,
                    'probability': 0.0
                })

        # Calculate overall confidence
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

        # Adjust confidence based on transition validity
        if transitions_analyzed > 0:
            transition_ratio = transitions_valid / transitions_analyzed
            confidence = 0.5 * transition_ratio + 0.5 * avg_confidence
        else:
            confidence = 1.0  # No transitions to validate

        return {
            'is_valid': transitions_valid == transitions_analyzed and len(violations) == 0,
            'confidence': float(confidence),
            'message': f'Sequence validation: {transitions_valid}/{transitions_analyzed} transitions valid',
            'validation_details': {
                'transitions_analyzed': transitions_analyzed,
                'transitions_valid': transitions_valid,
                'violations': violations
            }
        }

    def perform_chi_squared_test(self, grammar: Dict) -> Dict[str, Any]:
        """
        Perform chi-squared test for uniformity of grammar transitions.

        Args:
            grammar: Grammar dictionary with transition counts

        Returns:
            Dictionary with chi-squared test results
        """
        contingency = self._build_contingency_table(grammar)

        if not contingency['observed_counts']:
            return {
                'chi_squared': 0.0,
                'p_value': 1.0,
                'degrees_of_freedom': 0,
                'is_significant': False
            }

        observed = np.array(contingency['observed_counts'])
        expected = np.array(contingency['expected_counts'])

        # Calculate chi-squared statistic
        chi_squared = np.sum((observed - expected) ** 2 / expected)

        # Degrees of freedom
        df = len(observed) - 1

        # Calculate p-value (simplified - in practice would use scipy.stats)
        # Using approximation for chi-squared distribution
        p_value = 1.0 - self._chi2_cdf(chi_squared, df)

        return {
            'chi_squared': float(chi_squared),
            'p_value': float(p_value),
            'degrees_of_freedom': df,
            'is_significant': p_value < self.significance_threshold
        }

    def generate_validation_report(self, sequence: List[Dict], grammar: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.

        Args:
            sequence: Sequence to validate
            grammar: Grammar rules to use for validation

        Returns:
            Comprehensive validation report
        """
        # Grammar analysis
        grammar_result = self.validate_grammar_structure(grammar)

        # Statistical analysis
        chi_squared_result = self.perform_chi_squared_test(grammar)

        # Sequence validation
        sequence_result = self.validate_sequence(sequence, grammar)

        # Generate summary
        summary_parts = []
        if grammar_result['is_valid']:
            summary_parts.append("Grammar structure is valid")
        else:
            summary_parts.append("Grammar structure has issues")

        if chi_squared_result['is_significant']:
            summary_parts.append("Transition patterns are statistically significant")
        else:
            summary_parts.append("Transition patterns appear random")

        if sequence_result['is_valid']:
            summary_parts.append("Sequence follows grammar rules")
        else:
            summary_parts.append("Sequence has violations")

        summary = ". ".join(summary_parts) + "."

        # Generate recommendations
        recommendations = []

        if not grammar_result['is_valid']:
            recommendations.append("Fix grammar structure issues first")

        if chi_squared_result['p_value'] > 0.5:
            recommendations.append("Consider collecting more training data for better grammar modeling")

        if sequence_result['confidence'] < 0.5:
            recommendations.append("Sequence confidence is low - consider alternative grammars")

        if grammar_result['statistics']['total_transitions'] < 10:
            recommendations.append("Grammar is under-sampled; collect more data")

        # Metadata
        metadata = {
            'analysis_type': 'compositional_validation',
            'timestamp': None,  # Will be set by caller
            'parameters': {
                'min_transitions': self.min_transitions,
                'significance_threshold': self.significance_threshold,
                'min_sequence_length': self.min_sequence_length
            },
            'sequence_length': len(sequence),
            'grammar_size': len(grammar)
        }

        return {
            'summary': summary,
            'grammar_analysis': grammar_result,
            'statistical_analysis': chi_squared_result,
            'sequence_validation': sequence_result,
            'recommendations': recommendations,
            'metadata': metadata
        }

    def calculate_transition_probability(self, from_phrase: str, to_phrase: str, grammar: Dict) -> float:
        """
        Calculate probability of transition between phrases.

        Args:
            from_phrase: Source phrase
            to_phrase: Target phrase
            grammar: Grammar dictionary with transition counts

        Returns:
            Transition probability
        """
        if from_phrase not in grammar:
            return 0.0

        total_transitions = sum(grammar[from_phrase].values())
        if total_transitions == 0:
            return 0.0

        return grammar[from_phrase].get(to_phrase, 0) / total_transitions

    def _calculate_grammar_statistics(self, grammar: Dict) -> Dict[str, Any]:
        """Calculate statistics about the grammar."""
        total_transitions = sum(sum(transitions.values()) for transitions in grammar.values())
        total_phrases = len(grammar)

        # Calculate out-degree for each phrase
        out_degrees = [len(transitions) for transitions in grammar.values()]
        avg_out_degree = np.mean(out_degrees) if out_degrees else 0

        # Calculate in-degree for each phrase
        in_degree_counter = Counter()
        for transitions in grammar.values():
            for target_phrase in transitions:
                in_degree_counter[target_phrase] += 1

        return {
            'total_transitions': total_transitions,
            'total_phrases': total_phrases,
            'average_out_degree': float(avg_out_degree),
            'max_out_degree': max(out_degrees) if out_degrees else 0,
            'unique_transitions': sum(len(set(transitions.keys())) for transitions in grammar.values())
        }

    def _detect_cycles(self, grammar: Dict) -> List[List[str]]:
        """Detect cycles in the grammar graph."""
        cycles = []
        visited = set()
        recursion_stack = set()
        path = []

        def dfs(node):
            if node in recursion_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:]
                cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            recursion_stack.add(node)
            path.append(node)

            if node in grammar:
                for neighbor in grammar[node]:
                    dfs(neighbor)

            path.pop()
            recursion_stack.remove(node)

        for phrase in grammar:
            if phrase not in visited:
                dfs(phrase)

        return cycles

    def _find_disconnected_phrases(self, grammar: Dict) -> List[str]:
        """Find phrases with no connections."""
        all_phrases = set(grammar.keys())
        connected_phrases = set()

        # Find all phrases that appear as targets
        for transitions in grammar.values():
            connected_phrases.update(transitions.keys())

        # Find phrases that are not connected to anything
        return list(all_phrases - connected_phrases)

    def _find_sources_without_targets(self, grammar: Dict) -> List[str]:
        """Find source phrases with no targets."""
        return [phrase for phrase, transitions in grammar.items() if not transitions]

    def _find_phrases_without_incoming(self, grammar: Dict) -> List[str]:
        """Find phrases with no incoming transitions."""
        all_phrases = set(grammar.keys())
        connected_phrases = set()

        # Find all phrases that appear as targets
        for transitions in grammar.values():
            connected_phrases.update(transitions.keys())

        return list(all_phrases - connected_phrases)

    def _normalize_transitions(self, grammar: Dict) -> Dict[str, Dict[str, float]]:
        """Normalize transition counts to probabilities."""
        normalized = {}

        for from_phrase, transitions in grammar.items():
            total = sum(transitions.values())
            if total > 0:
                normalized[from_phrase] = {
                    to_phrase: count / total
                    for to_phrase, count in transitions.items()
                }

        return normalized

    def _build_contingency_table(self, grammar: Dict) -> Dict[str, Any]:
        """Build contingency table for chi-squared test."""
        # Get all unique from-phrases
        from_phrases = list(grammar.keys())
        if not from_phrases:
            return {
                'from_phrases': [],
                'to_phrases': [],
                'observed_counts': [],
                'expected_counts': []
            }

        # Get all unique to-phrases
        to_phrases = set()
        for transitions in grammar.values():
            to_phrases.update(transitions.keys())
        to_phrases = sorted(list(to_phrases))

        # Build observed counts matrix
        observed_counts = []
        for from_phrase in from_phrases:
            row = []
            for to_phrase in to_phrases:
                row.append(grammar[from_phrase].get(to_phrase, 0))
            observed_counts.append(row)

        # Calculate expected counts (uniform distribution)
        total_transitions = sum(sum(row) for row in observed_counts)
        expected_count_per_cell = total_transitions / (len(from_phrases) * len(to_phrases))

        expected_counts = [
            [expected_count_per_cell for _ in to_phrases]
            for _ in from_phrases
        ]

        return {
            'from_phrases': from_phrases,
            'to_phrases': to_phrases,
            'observed_counts': [sum(row) for row in observed_counts],  # Marginal totals
            'expected_counts': [sum(row) for row in expected_counts]   # Marginal totals
        }

    def _chi2_cdf(self, x: float, k: int) -> float:
        """Approximation of chi-squared cumulative distribution function."""
        # Using Wilson-Hilferty approximation for CDF
        # This is a simplified approximation - in production use scipy.stats
        if k <= 0:
            return 0.0

        z = ((x / k) ** (1/3) - 1 + 2/(9*k)) / math.sqrt(2/(9*k))

        # Approximate normal CDF
        if z > 0:
            return 0.5 * (1 + math.erf(z / math.sqrt(2)))
        else:
            return 0.5 * (1 - math.erf(-z / math.sqrt(2)))


# Example usage and testing
if __name__ == "__main__":
    # Test data
    test_grammar = {
        'phrase1': {'phrase2': 10, 'phrase3': 5},
        'phrase2': {'phrase4': 8, 'phrase1': 2},
        'phrase3': {'phrase1': 3, 'phrase4': 7},
        'phrase4': {'phrase1': 1}
    }

    test_sequence = [
        {'key': 'phrase1', 'position': 0},
        {'key': 'phrase2', 'position': 1},
        {'key': 'phrase4', 'position': 2}
    ]

    # Create validator and test
    validator = CompositionalValidator()
    result = validator.validate_grammar_structure(test_grammar)
    print("Grammar Validation Result:", result)

    sequence_result = validator.validate_sequence(test_sequence, test_grammar)
    print("Sequence Validation Result:", sequence_result)