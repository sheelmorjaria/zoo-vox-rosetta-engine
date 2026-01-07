"""
Persona-Based Semantic Zone Validator
=====================================

Validates whether synthesized or natural audio vocalizations fall within
valid "semantic zones" defined by persona cluster boundaries.

This is the POST-SYNTHESIS validation step that answers:
"Is this sound something that could actually exist in nature given this
species' vocal repertoire?"

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)


@dataclass
class PersonaCluster:
    """Statistical definition of a persona's semantic zone."""
    persona_id: str
    species: str
    cluster_id: int

    # Cluster centroid (mean feature vector)
    centroid: np.ndarray  # [f0, duration, f0_range, harmonicity, spectral_flatness, jitter, shimmer]

    # Cluster boundary (covariance matrix)
    covariance: np.ndarray

    # Feature names for interpretation
    feature_names: List[str] = field(default_factory=list)

    # Validation thresholds (statistical bounds)
    std_multiplier: float = 2.0  # 2-sigma boundary (95% confidence)

    # Sample size (for weighting)
    sample_size: int = 0

    # Semantic label
    semantic_label: str = ""  # e.g., "contact", "alarm", "navigation"

    def __post_init__(self):
        if len(self.feature_names) == 0:
            self.feature_names = [
                'mean_f0_hz', 'duration_ms', 'f0_range_hz',
                'harmonicity', 'spectral_flatness', 'jitter', 'shimmer'
            ]


@dataclass
class SemanticZoneValidationResult:
    """Result of semantic zone validation."""
    passed: bool
    persona_id: str
    semantic_label: str
    confidence: float  # 0-1, how confident are we this belongs to this persona
    mahalanobis_distance: float  # Distance from cluster centroid
    is_outlier: bool  # Beyond 2-sigma boundary

    # Feature-level diagnostics
    feature_deviations: Dict[str, float] = field(default_factory=dict)
    feature_z_scores: Dict[str, float] = field(default_factory=dict)

    # Recommendations
    warnings: List[str] = field(default_factory=list)
    suggested_persona: Optional[str] = None

    def __str__(self):
        status = "✅ IN ZONE" if self.passed else "❌ OUT OF ZONE"
        output = [
            f"{status}: {self.persona_id} ({self.semantic_label})",
            f"  Confidence: {self.confidence:.2%}",
            f"  Mahalanobis Distance: {self.mahalanobis_distance:.2f}",
        ]

        if self.is_outlier:
            output.append("  ⚠️  OUTLIER: Beyond 2-sigma boundary")

        if self.warnings:
            output.append("  Warnings:")
            for w in self.warnings:
                output.append(f"    - {w}")

        if self.suggested_persona:
            output.append(f"  Suggested: {self.suggested_persona}")

        return "\n".join(output)


class PersonaSemanticZoneValidator:
    """
    Validates vocalizations against persona semantic zones.

    Uses multivariate Gaussian mixture models to represent
    each persona's "semantic zone" in feature space.

    A vocalization is valid if it falls within the statistical
    boundaries (typically 2-sigma) of at least one persona cluster.
    """

    def __init__(self, persona_invariants_path: Optional[str] = None):
        """
        Initialize the semantic zone validator.

        Args:
            persona_invariants_path: Path to persona_invariants.json
        """
        self.clusters: Dict[str, PersonaCluster] = {}
        self.species_to_personas: Dict[str, List[str]] = {}

        if persona_invariants_path:
            self.load_persona_invariants(persona_invariants_path)
        else:
            self._create_default_clusters()

        logger.info(f"Loaded {len(self.clusters)} persona semantic zones")

    def load_persona_invariants(self, json_path: str):
        """Load persona cluster definitions from JSON."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            personas = data.get('personas', {})

            for persona_id, persona_data in personas.items():
                # Extract mean and std for each feature
                features = ['mean_f0_hz', 'duration_ms', 'f0_range_hz',
                           'harmonicity', 'spectral_flatness', 'jitter', 'shimmer']

                # Build centroid vector
                centroid = np.array([
                    persona_data['acoustic_profile'].get(f, 0) for f in features
                ])

                # Build covariance matrix (diagonal for now, assumes feature independence)
                # In production, this should be computed from actual data
                std_values = []
                for f in features:
                    # Estimate std from cluster data
                    # For now, use heuristic: std = mean * 0.1 (10% CV)
                    std_values.append(centroid[features.index(f)] * 0.1)

                covariance = np.diag(std_values) ** 2

                cluster = PersonaCluster(
                    persona_id=persona_id,
                    species=persona_data['species'],
                    cluster_id=persona_data['cluster_id'],
                    centroid=centroid,
                    covariance=covariance,
                    feature_names=features,
                    sample_size=100,  # Default
                    semantic_label=persona_data.get('usage', '').split()[0].lower()
                )

                self.clusters[persona_id] = cluster

            # Build species index
            for persona_id, cluster in self.clusters.items():
                species = cluster.species
                if species not in self.species_to_personas:
                    self.species_to_personas[species] = []
                self.species_to_personas[species].append(persona_id)

            logger.info(f"Loaded {len(self.clusters)} persona clusters from {json_path}")

        except Exception as e:
            logger.warning(f"Failed to load persona invariants: {e}")
            self._create_default_clusters()

    def _create_default_clusters(self):
        """Create default persona clusters based on known statistics."""

        # Marmoset Phee (Cluster 0)
        phee_centroid = np.array([6526, 76.5, 427, 0.95, 0.1, 0.02, 0.03])
        phee_cov = np.diag([935, 57.6, 399, 0.05, 0.05, 0.01, 0.01]) ** 2

        self.clusters['MARMOSET_PHEE'] = PersonaCluster(
            persona_id='MARMOSET_PHEE',
            species='marmoset',
            cluster_id=0,
            centroid=phee_centroid,
            covariance=phee_cov,
            sample_size=576,
            semantic_label='contact'
        )

        # Marmoset Alarm (Cluster 1)
        alarm_centroid = np.array([6020, 58.1, 3722, 0.7, 0.3, 0.08, 0.05])
        alarm_cov = np.diag([701, 0.0, 163, 0.1, 0.1, 0.02, 0.02]) ** 2

        self.clusters['MARMOSET_ALARM'] = PersonaCluster(
            persona_id='MARMOSET_ALARM',
            species='marmoset',
            cluster_id=1,
            centroid=alarm_centroid,
            covariance=alarm_cov,
            sample_size=22,
            semantic_label='alarm'
        )

        # Bat Mid-FM (Cluster 1)
        bat_mid_centroid = np.array([7437, 17.4, 9755, 0.6, 0.4, 0.05, 0.04])
        bat_mid_cov = np.diag([1232, 0.0, 2583, 0.15, 0.15, 0.02, 0.02]) ** 2

        self.clusters['BAT_MID_FM'] = PersonaCluster(
            persona_id='BAT_MID_FM',
            species='egyptian_bat',
            cluster_id=1,
            centroid=bat_mid_centroid,
            covariance=bat_mid_cov,
            sample_size=233,
            semantic_label='navigation'
        )

        # Bat Social US (Cluster 2)
        bat_social_centroid = np.array([7408, 17.4, 24, 0.85, 0.1, 0.01, 0.01])
        bat_social_cov = np.diag([1383, 0.0, 22, 0.1, 0.05, 0.005, 0.005]) ** 2

        self.clusters['BAT_SOCIAL_US'] = PersonaCluster(
            persona_id='BAT_SOCIAL_US',
            species='egyptian_bat',
            cluster_id=2,
            centroid=bat_social_centroid,
            covariance=bat_social_cov,
            sample_size=213,
            semantic_label='social'
        )

        # Build species index
        for persona_id, cluster in self.clusters.items():
            species = cluster.species
            if species not in self.species_to_personas:
                self.species_to_personas[species] = []
            self.species_to_personas[species].append(persona_id)

    def validate_vocalization(
        self,
        features: Dict[str, float],
        species: Optional[str] = None,
        strict: bool = True
    ) -> SemanticZoneValidationResult:
        """
        Validate if a vocalization falls within a valid semantic zone.

        Args:
            features: Acoustic features of the vocalization
            species: Optional species filter
            strict: If True, must fall within 2-sigma boundary

        Returns:
            SemanticZoneValidationResult with persona assignment and confidence
        """
        # Build feature vector
        feature_names = ['mean_f0_hz', 'duration_ms', 'f0_range_hz',
                        'harmonicity', 'spectral_flatness', 'jitter', 'shimmer']

        feature_vector = np.array([features.get(f, 0) for f in feature_names])

        # Filter clusters by species if specified
        candidate_clusters = list(self.clusters.values())
        if species:
            candidate_clusters = [
                c for c in candidate_clusters if c.species == species
            ]

        if len(candidate_clusters) == 0:
            return SemanticZoneValidationResult(
                passed=False,
                persona_id="UNKNOWN",
                semantic_label="unknown",
                confidence=0.0,
                mahalanobis_distance=float('inf'),
                is_outlier=True,
                warnings=["No matching persona clusters for species"]
            )

        # Calculate Mahalanobis distance to each cluster
        results = []
        for cluster in candidate_clusters:
            distance = self._mahalanobis_distance(
                feature_vector,
                cluster.centroid,
                cluster.covariance
            )

            # Calculate confidence based on distance
            # Distance < 2 = within 2-sigma (95% confidence)
            # Distance < 3 = within 3-sigma (99.7% confidence)
            if distance < 2.0:
                confidence = 1.0 - (distance / 2.0) * 0.05  # 95%+ confidence
            elif distance < 3.0:
                confidence = 0.95 - ((distance - 2.0) / 1.0) * 0.047  # 90-95% confidence
            else:
                confidence = max(0.0, 0.5 - (distance - 3.0) * 0.1)  # Low confidence

            results.append((cluster, distance, confidence))

        # Sort by distance (closest = best match)
        results.sort(key=lambda x: x[1])

        best_cluster, best_distance, best_confidence = results[0]

        # Determine if outlier (beyond 2-sigma)
        is_outlier = best_distance > best_cluster.std_multiplier

        # Calculate feature-level deviations
        feature_deviations = {}
        feature_z_scores = {}

        for i, fname in enumerate(feature_names):
            deviation = feature_vector[i] - best_cluster.centroid[i]
            std = np.sqrt(best_cluster.covariance[i, i])

            feature_deviations[fname] = deviation
            feature_z_scores[fname] = deviation / std if std > 0 else 0

        # Generate warnings for outlier features
        warnings = []
        for fname, z_score in feature_z_scores.items():
            if abs(z_score) > 2.0:
                warnings.append(f"{fname}: {z_score:.1f}σ deviation")

        # Check if passed validation
        if strict:
            passed = not is_outlier
        else:
            passed = best_distance < 3.0  # Allow 3-sigma

        # Suggest alternative persona if confidence is low
        suggested_persona = None
        if len(results) > 1 and best_confidence < 0.7:
            second_best = results[1]
            if second_best[1] < best_distance * 1.5:  # Within 50% distance
                suggested_persona = second_best[0].persona_id

        return SemanticZoneValidationResult(
            passed=passed,
            persona_id=best_cluster.persona_id,
            semantic_label=best_cluster.semantic_label,
            confidence=best_confidence,
            mahalanobis_distance=best_distance,
            is_outlier=is_outlier,
            feature_deviations=feature_deviations,
            feature_z_scores=feature_z_scores,
            warnings=warnings,
            suggested_persona=suggested_persona
        )

    def _mahalanobis_distance(
        self,
        point: np.ndarray,
        centroid: np.ndarray,
        covariance: np.ndarray
    ) -> float:
        """
        Calculate Mahalanobis distance from point to cluster centroid.

        This is the multivariate equivalent of z-score, accounting for
        covariance between features.
        """
        try:
            # Add small regularization to avoid singular matrix
            reg = np.eye(covariance.shape[0]) * 1e-6
            inv_cov = np.linalg.inv(covariance + reg)

            delta = point - centroid
            distance = np.sqrt(delta.T @ inv_cov @ delta)

            return distance
        except np.linalg.LinAlgError:
            # Fallback to Euclidean distance if covariance is singular
            return euclidean(point, centroid)

    def classify_vocalization(
        self,
        features: Dict[str, float],
        species: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Classify a vocalization into a persona.

        Simpler version of validate_vocalization that just returns
        the best matching persona and confidence.

        Returns:
            (persona_id, confidence)
        """
        result = self.validate_vocalization(features, species, strict=False)
        return result.persona_id, result.confidence

    def get_cluster_boundaries(
        self,
        persona_id: str
    ) -> Dict[str, Tuple[float, float]]:
        """
        Get the statistical boundaries (±2σ) for each feature.

        Returns:
            Dict mapping feature_name -> (lower_bound, upper_bound)
        """
        if persona_id not in self.clusters:
            return {}

        cluster = self.clusters[persona_id]
        boundaries = {}

        for i, fname in enumerate(cluster.feature_names):
            mean = cluster.centroid[i]
            std = np.sqrt(cluster.covariance[i, i])

            lower = mean - 2 * std
            upper = mean + 2 * std

            boundaries[fname] = (max(0, lower), upper)  # Clamp negative values

        return boundaries

    def is_ghost_word(
        self,
        features: Dict[str, float],
        species: str,
        threshold: float = 1.5
    ) -> Tuple[bool, List[Tuple[str, float]]]:
        """
        Determine if a vocalization is a "ghost word" - falls between
        two or more persona clusters.

        A ghost word is:
        - Within 1.5-sigma of at least 2 different clusters
        - Not clearly belonging to any single cluster (< 70% confidence)

        Returns:
            (is_ghost_word, [(persona_id, distance), ...])
        """
        if species not in self.species_to_personas:
            return False, []

        # Get all distances to clusters for this species
        feature_names = ['mean_f0_hz', 'duration_ms', 'f0_range_hz',
                        'harmonicity', 'spectral_flatness', 'jitter', 'shimmer']
        feature_vector = np.array([features.get(f, 0) for f in feature_names])

        distances = []
        for persona_id in self.species_to_personas[species]:
            cluster = self.clusters[persona_id]
            distance = self._mahalanobis_distance(
                feature_vector,
                cluster.centroid,
                cluster.covariance
            )
            distances.append((persona_id, distance))

        # Sort by distance
        distances.sort(key=lambda x: x[1])

        # Check if within threshold of at least 2 clusters
        close_clusters = [(p, d) for p, d in distances if d < threshold]

        is_ghost = len(close_clusters) >= 2

        # Also check if best match has low confidence
        if len(distances) > 0:
            best_distance = distances[0][1]
            if best_distance > 1.0:  # Low confidence
                is_ghost = True

        return is_ghost, close_clusters


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PERSONA SEMANTIC ZONE VALIDATOR DEMONSTRATION")
    print("="*80)

    validator = PersonaSemanticZoneValidator()

    # Test 1: Natural marmoset phee (should pass)
    print("\n--- Test 1: Natural Marmoset Phee Call ---")
    phee_features = {
        'mean_f0_hz': 6526,
        'duration_ms': 76.5,
        'f0_range_hz': 427,
        'harmonicity': 0.95,
        'spectral_flatness': 0.1,
        'jitter': 0.02,
        'shimmer': 0.03
    }

    result = validator.validate_vocalization(phee_features, species='marmoset')
    print(result)

    # Test 2: Marmoset alarm (should pass)
    print("\n--- Test 2: Marmoset Alarm Call ---")
    alarm_features = {
        'mean_f0_hz': 6020,
        'duration_ms': 58.1,
        'f0_range_hz': 3722,
        'harmonicity': 0.7,
        'spectral_flatness': 0.3,
        'jitter': 0.08,
        'shimmer': 0.05
    }

    result = validator.validate_vocalization(alarm_features, species='marmoset')
    print(result)

    # Test 3: Ghost word (interpolation)
    print("\n--- Test 3: Ghost Word (Phee + Alarm Interpolation) ---")
    ghost_features = {
        'mean_f0_hz': 6273,  # Midpoint
        'duration_ms': 67.3,  # Midpoint
        'f0_range_hz': 2074,  # Midpoint
        'harmonicity': 0.825,  # Average
        'spectral_flatness': 0.2,
        'jitter': 0.05,
        'shimmer': 0.04
    }

    result = validator.validate_vocalization(ghost_features, species='marmoset')
    print(result)

    is_ghost, close_clusters = validator.is_ghost_word(ghost_features, 'marmoset')
    print("\nGhost Word Analysis:")
    print(f"  Is Ghost Word: {is_ghost}")
    print("  Close Clusters:")
    for persona_id, distance in close_clusters:
        print(f"    - {persona_id}: {distance:.2f}σ")

    # Test 4: Invalid vocalization (should fail)
    print("\n--- Test 4: Invalid Vocalization (Out of Range) ---")
    invalid_features = {
        'mean_f0_hz': 10000,  # Too high for marmoset
        'duration_ms': 200,  # Too long
        'f0_range_hz': 100,
        'harmonicity': 0.3,
        'spectral_flatness': 0.8,
        'jitter': 0.2,
        'shimmer': 0.15
    }

    result = validator.validate_vocalization(invalid_features, species='marmoset')
    print(result)

    # Test 5: Cluster boundaries
    print("\n--- Test 5: Cluster Boundaries (MARMOSET_PHEE) ---")
    boundaries = validator.get_cluster_boundaries('MARMOSET_PHEE')
    print("  Feature Boundaries (95% confidence):")
    for fname, (lower, upper) in boundaries.items():
        print(f"    {fname}: [{lower:.1f}, {upper:.1f}]")

    print("\n" + "="*80)
    print("\n🎯 Semantic Zone Validation ready for:")
    print("   ✓ Validating synthesized vocalizations")
    print("   ✓ Classifying natural recordings")
    print("   ✓ Detecting ghost words (between-cluster sounds)")
    print("   ✓ Ensuring scientific validity")
    print()
