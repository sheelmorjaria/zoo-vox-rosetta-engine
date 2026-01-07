"""
Harmonic Affirmation Module

This module provides functionality for analyzing harmonic relationships in animal vocalizations.
It identifies harmonic series, validates harmonic structures, and generates analysis reports.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np


class HarmonicGroup:
    """Represents a group of frequencies that are harmonically related."""

    def __init__(self, name: str, phrase_indices: List[int]):
        self.name = name
        self.phrase_indices = phrase_indices
        self.size = len(phrase_indices)

    def calculate_average_f0(self, f0_values: List[float]) -> float:
        """Calculate average F0 for this harmonic group."""
        if not f0_values or not self.phrase_indices:
            return 0.0

        group_f0s = [f0_values[i] for i in self.phrase_indices if i < len(f0_values)]
        return np.mean(group_f0s) if group_f0s else 0.0

    def calculate_harmonic_deviation(self, f0_values: List[float], expected_f0: float) -> float:
        """Calculate harmonic deviation from expected frequency."""
        if not self.phrase_indices or expected_f0 == 0:
            return 0.0

        group_f0 = self.calculate_average_f0(f0_values)
        if group_f0 == 0:
            return 0.0

        # Calculate relative deviation
        return abs(group_f0 - expected_f0) / expected_f0


class HarmonicAffirmation:
    """
    Analyzes harmonic relationships in animal vocalizations using TDD methodology.

    This class implements the harmonic affirmation functionality that was missing
    from the Rosetta Stone analysis pipeline.
    """

    def __init__(self,
                 harmonic_threshold_ratio: float = 0.2,
                 min_harmonic_group_size: int = 2,
                 max_harmonic_deviation: float = 0.15):
        """
        Initialize HarmonicAffirmation with configuration parameters.

        Args:
            harmonic_threshold_ratio: Maximum allowed deviation for harmonic relationships (default: 20%)
            min_harmonic_group_size: Minimum number of phrases to form a harmonic group
            max_harmonic_deviation: Maximum allowed harmonic deviation for validation
        """
        self.harmonic_threshold_ratio = harmonic_threshold_ratio
        self.min_harmonic_group_size = min_harmonic_group_size
        self.max_harmonic_deviation = max_harmonic_deviation

    def analyze_harmonic_series(self, phrases: List[Dict]) -> Dict[str, Any]:
        """
        Analyze a series of phrases for harmonic relationships.

        Args:
            phrases: List of phrase dictionaries with 'f0_mean' and 'features' keys

        Returns:
            Dictionary containing harmonic analysis results
        """
        if not phrases:
            return {
                'total_harmonic_phrases': 0,
                'fundamental_freq': None,
                'harmonic_groups': {},
                'harmonic_ratio': 0.0,
                'threshold': 0.0
            }

        # Extract F0 values
        f0_values = [p.get('features', {}).get('f0_mean', 0) or p.get('f0_mean', 0)
                      for p in phrases if p.get('features', {}).get('f0_mean', 0) or p.get('f0_mean', 0) > 0]

        if not f0_values:
            return {
                'total_harmonic_phrases': 0,
                'fundamental_freq': None,
                'harmonic_groups': {},
                'harmonic_ratio': 0.0,
                'threshold': 0.0
            }

        # Find fundamental frequency (use the lowest frequency)
        fundamental_freq = min(f0_values)
        harmonic_threshold = fundamental_freq * self.harmonic_threshold_ratio

        # Group frequencies by harmonic relationship
        harmonic_groups = defaultdict(list)
        non_harmonic_count = 0

        for i, f0 in enumerate(f0_values):
            # Check if frequency is close to a harmonic of the fundamental
            harmonic_found = False

            # Check for harmonics up to 10th order
            for harmonic_order in range(1, 11):
                expected_freq = harmonic_order * fundamental_freq
                if abs(f0 - expected_freq) <= harmonic_threshold:
                    group_name = f'harmonic_{harmonic_order}'
                    harmonic_groups[group_name].append(i)
                    harmonic_found = True
                    break

            if not harmonic_found:
                harmonic_groups['non_harmonic'].append(i)
                non_harmonic_count += 1

        # Calculate harmonic ratio (exclude non_harmonic and multiple fundamentals)
        harmonic_keys = [key for key in harmonic_groups.keys()
                        if key != 'non_harmonic' and not key.startswith('fundamental_')]
        total_harmonic_phrases = sum(len(harmonic_groups[key]) for key in harmonic_keys)
        harmonic_ratio = total_harmonic_phrases / len(f0_values) if f0_values else 0.0

        return {
            'total_harmonic_phrases': total_harmonic_phrases,
            'fundamental_freq': float(fundamental_freq),
            'harmonic_groups': dict(harmonic_groups),
            'harmonic_ratio': float(harmonic_ratio),
            'threshold': float(harmonic_threshold)
        }

    def calculate_harmonic_distance(self, freq1: float, freq2: float) -> float:
        """
        Calculate harmonic distance between two frequencies.

        Args:
            freq1: First frequency
            freq2: Second frequency

        Returns:
            Harmonic distance (0 for perfect harmonics, >0 for deviations)
        """
        if freq1 == 0 or freq2 == 0:
            return float('inf')

        # Calculate harmonic ratio
        ratio = freq1 / freq2

        # Find closest integer harmonic
        closest_harmonic = round(ratio)

        # Calculate relative deviation
        if closest_harmonic == 0:
            return float('inf')

        deviation = abs(ratio - closest_harmonic) / closest_harmonic
        return deviation

    def generate_analysis_report(self, phrases: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report for harmonic affirmation.

        Args:
            phrases: List of phrase dictionaries

        Returns:
            Comprehensive analysis report with recommendations
        """
        # Basic harmonic analysis
        harmonic_analysis = self.analyze_harmonic_series(phrases)

        # Calculate confidence score
        confidence_score = harmonic_analysis['harmonic_ratio']
        if harmonic_analysis['total_harmonic_phrases'] < self.min_harmonic_group_size:
            confidence_score *= 0.5  # Penalize small groups

        # Generate summary
        if confidence_score > 0.8:
            summary = "Strong harmonic structure detected with clear harmonic relationships."
        elif confidence_score > 0.5:
            summary = "Moderate harmonic structure with some harmonic relationships."
        elif confidence_score > 0.2:
            summary = "Weak harmonic structure with limited harmonic relationships."
        else:
            summary = "No significant harmonic structure detected."

        # Generate recommendations
        recommendations = []

        if harmonic_analysis['harmonic_ratio'] < 0.5:
            recommendations.append("Consider analyzing with different threshold values for harmonic detection.")

        if harmonic_analysis['total_harmonic_phrases'] < self.min_harmonic_group_size:
            recommendations.append("More data needed to establish reliable harmonic patterns.")

        if 'non_harmonic' in harmonic_analysis['harmonic_groups']:
            non_harmonic_count = len(harmonic_analysis['harmonic_groups']['non_harmonic'])
            if non_harmonic_count > len(phrases) * 0.5:
                recommendations.append("High proportion of non-harmonic frequencies detected.")

        # Check for multiple fundamental frequencies
        fundamentals = [key for key in harmonic_analysis['harmonic_groups'].keys()
                       if key.startswith('fundamental_') or key == 'fundamental']

        if len(fundamentals) > 1:
            recommendations.append("Multiple harmonic series detected - consider analyzing separately.")

        # Metadata
        metadata = {
            'analysis_type': 'harmonic_affirmation',
            'timestamp': None,  # Will be set by caller
            'parameters': {
                'harmonic_threshold_ratio': self.harmonic_threshold_ratio,
                'min_harmonic_group_size': self.min_harmonic_group_size,
                'max_harmonic_deviation': self.max_harmonic_deviation
            },
            'total_phrases_analyzed': len(phrases)
        }

        return {
            'summary': summary,
            'detailed_analysis': harmonic_analysis,
            'recommendations': recommendations,
            'confidence_score': float(confidence_score),
            'metadata': metadata
        }


# Example usage and testing
if __name__ == "__main__":
    # Test data
    test_phrases = [
        {'f0_mean': 440.0, 'features': {'f0_mean': 440.0}},
        {'f0_mean': 880.0, 'features': {'f0_mean': 880.0}},
        {'f0_mean': 1320.0, 'features': {'f0_mean': 1320.0}},
        {'f0_mean': 2000.0, 'features': {'f0_mean': 2000.0}},
    ]

    # Create analyzer and test
    analyzer = HarmonicAffirmation()
    result = analyzer.analyze_harmonic_series(test_phrases)
    print("Harmonic Analysis Result:", result)

    # Test report generation
    report = analyzer.generate_analysis_report(test_phrases)
    print("\nAnalysis Report:", report)
