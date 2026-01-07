"""
Contextual Map: Semantic Gradient Engine
========================================

Integrates high-dimensional acoustic algebra with annotation data to enable
continuous gradient synthesis (e.g., 30% Aggressive instead of Binary Aggressive/Not).

This is the "Semantic Vector" component that transforms the pipeline from:
- Discrete Retrieval: Play "Aggressive" phrase
- Continuous Generation: Generate "30% Aggressive" phrase

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from high_dimensional_acoustic_algebra import (
    AcousticFeatureVector17,
    HighDimensionalAcousticAlgebra,
)


@dataclass
class ContextCentroid:
    """
    A semantic centroid representing the "mean" acoustic signature of a context.

    Example:
        centroid_aggression = mean(all 17D vectors for "Aggression" labeled phrases)
        centroid_contact = mean(all 17D vectors for "Contact" labeled phrases)
    """
    context_name: str
    centroid_vector: AcousticFeatureVector17
    sample_count: int
    variance: Optional[np.ndarray] = None  # Per-feature variance
    std_dev: Optional[np.ndarray] = None   # Per-feature std dev


class ContextualMap:
    """
    Maintains contextual centroids and enables gradient synthesis.

    **Two Roles:**

    1. **Discovery Phase**: Calculate "Semantic Vectors" from annotation data
       - Groups phrases by annotation label (e.g., "aggression", "food", "alarm")
       - Computes centroid for each context (the "mean" of that emotion)
       - Stores baseline (typically "contact" or "neutral")

    2. **Synthesis Phase**: Generate "Virtual Phrases" via gradient interpolation
       - Interpolates between baseline and target context
       - Creates nuanced intensities (0.3 = 30% Aggressive)
       - Finds nearest real phrase for synthesis
    """

    def __init__(self, algebra: Optional[HighDimensionalAcousticAlgebra] = None):
        """
        Initialize contextual map.

        Args:
            algebra: High-dimensional algebra engine (creates default if None)
        """
        self.algebra = algebra or HighDimensionalAcousticAlgebra()
        self.centroids: Dict[str, ContextCentroid] = {}
        self.baseline_context: Optional[str] = None  # Typically "contact" or "neutral"

    def calculate_context_centroids(
        self,
        phrase_vectors: Dict[str, AcousticFeatureVector17],
        context_labels: Dict[str, str]
    ) -> Dict[str, ContextCentroid]:
        """
        Calculate semantic centroids for each context from annotated phrases.

        **Discovery Phase**: Call this after loading annotations to define
        the "meaning" of each context as a 17D vector.

        Args:
            phrase_vectors: Dict mapping phrase_key → 17D feature vector
            context_labels: Dict mapping phrase_key → context label
                (e.g., "contact", "aggression", "food", "alarm")

        Returns:
            Dict mapping context_name → ContextCentroid
        """
        # Group vectors by context
        context_groups: Dict[str, List[AcousticFeatureVector17]] = {}

        for phrase_key, vector in phrase_vectors.items():
            context = context_labels.get(phrase_key, "unknown")
            if context not in context_groups:
                context_groups[context] = []
            context_groups[context].append(vector)

        # Calculate centroids
        self.centroids = {}

        for context, vectors in context_groups.items():
            # Convert to numpy matrix (N x 17)
            matrix = np.stack([v.to_numpy() for v in vectors])

            # Calculate mean (centroid)
            centroid_vec = np.mean(matrix, axis=0)

            # Calculate variance and std
            variance = np.var(matrix, axis=0)
            std_dev = np.std(matrix, axis=0)

            # Create centroid
            self.centroids[context] = ContextCentroid(
                context_name=context,
                centroid_vector=AcousticFeatureVector17.from_numpy(centroid_vec),
                sample_count=len(vectors),
                variance=variance,
                std_dev=std_dev
            )

        # Set baseline to "contact" if available, otherwise first context
        if "contact" in self.centroids:
            self.baseline_context = "contact"
        elif "neutral" in self.centroids:
            self.baseline_context = "neutral"
        elif len(self.centroids) > 0:
            self.baseline_context = list(self.centroids.keys())[0]

        return self.centroids

    def set_baseline(self, context_name: str):
        """Set the baseline context (typically "contact" or "neutral")."""
        if context_name not in self.centroids:
            raise ValueError(f"Context '{context_name}' not found in centroids")
        self.baseline_context = context_name

    def generate_graded_phrase(
        self,
        target_context: str,
        intensity: float,
        baseline_context: Optional[str] = None
    ) -> AcousticFeatureVector17:
        """
        Generate a "Virtual Phrase" at specified intensity along semantic gradient.

        **Synthesis Phase**: Call this to create nuanced phrases.

        Math:
            V_result = V_baseline + (V_target - V_baseline) * intensity

        Args:
            target_context: Target context (e.g., "aggression")
            intensity: Interpolation factor (0.0 = baseline, 1.0 = full target)
            baseline_context: Override baseline (uses default if None)

        Returns:
            Interpolated "Virtual Phrase" vector
        """
        if target_context not in self.centroids:
            raise ValueError(f"Target context '{target_context}' not found")

        baseline = baseline_context or self.baseline_context
        if baseline not in self.centroids:
            raise ValueError(f"Baseline context '{baseline}' not found")

        vec_baseline = self.centroids[baseline].centroid_vector
        vec_target = self.centroids[target_context].centroid_vector

        # Interpolate: Start + (Direction * Distance)
        virtual_phrase = self.algebra.interpolate(
            vec_baseline,
            vec_target,
            alpha=intensity
        )

        return virtual_phrase

    def find_nearest_real_phrase(
        self,
        virtual_vector: AcousticFeatureVector17,
        phrase_vectors: Dict[str, AcousticFeatureVector17]
    ) -> Tuple[str, AcousticFeatureVector17, float]:
        """
        Find the nearest real phrase to a virtual vector.

        Since we can't synthesize a vector directly, we need to find the
        closest real phrase in the database to use as the source buffer.

        Args:
            virtual_vector: The virtual (interpolated) vector
            phrase_vectors: Dict mapping phrase_key → 17D feature vector

        Returns:
            (phrase_key, nearest_vector, distance)
        """
        best_key = None
        best_vector = None
        best_distance = float('inf')

        virtual_vec = virtual_vector.to_numpy()

        for phrase_key, vector in phrase_vectors.items():
            real_vec = vector.to_numpy()

            # Euclidean distance in Z-score normalized space
            z_virtual = self.algebra.normalizer.normalize(virtual_vec)
            z_real = self.algebra.normalizer.normalize(real_vec)

            distance = np.linalg.norm(z_virtual - z_real)

            if distance < best_distance:
                best_distance = distance
                best_key = phrase_key
                best_vector = vector

        return best_key, best_vector, best_distance

    def calculate_context_delta(
        self,
        context_a: str,
        context_b: str
    ) -> AcousticFeatureVector17:
        """
        Calculate the "delta" vector between two contexts.

        Useful for understanding "what makes X different from Y".

        Example:
            delta = aggression - contact
            # delta might show: +2ms attack, +500Hz F0 range, +5% jitter

        Args:
            context_a: First context
            context_b: Second context

        Returns:
            Difference vector (A - B)
        """
        if context_a not in self.centroids:
            raise ValueError(f"Context '{context_a}' not found")
        if context_b not in self.centroids:
            raise ValueError(f"Context '{context_b}' not found")

        vec_a = self.centroids[context_a].centroid_vector
        vec_b = self.centroids[context_b].centroid_vector

        return self.algebra.subtract(vec_a, vec_b)

    def get_context_variance_explained(
        self,
        context_name: str,
        feature_name: str
    ) -> float:
        """
        Get variance of a specific feature for a context.

        High variance = that feature is NOT diagnostic of this context
        Low variance = that feature IS diagnostic (consistent across examples)

        Args:
            context_name: Context to check
            feature_name: Name of feature (e.g., "mean_f0_hz", "jitter")

        Returns:
            Variance of that feature for this context
        """
        if context_name not in self.centroids:
            raise ValueError(f"Context '{context_name}' not found")

        centroid = self.centroids[context_name]

        if centroid.variance is None:
            return 0.0

        # Find index of feature
        feature_names = centroid.centroid_vector.feature_names()
        if feature_name not in feature_names:
            raise ValueError(f"Unknown feature: {feature_name}")

        idx = feature_names.index(feature_name)
        return float(centroid.variance[idx])

    def save(self, filepath: Path):
        """Save centroids to JSON file."""
        data = {
            'baseline_context': self.baseline_context,
            'centroids': {}
        }

        for ctx_name, centroid in self.centroids.items():
            data['centroids'][ctx_name] = {
                'context_name': centroid.context_name,
                'centroid_vector': centroid.centroid_vector.to_dict(),
                'sample_count': centroid.sample_count,
                'variance': centroid.variance.tolist() if centroid.variance is not None else None,
                'std_dev': centroid.std_dev.tolist() if centroid.std_dev is not None else None
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> 'ContextualMap':
        """Load centroids from JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        algebra = HighDimensionalAcousticAlgebra()
        map_obj = cls(algebra)
        map_obj.baseline_context = data.get('baseline_context')

        for ctx_name, ctx_data in data['centroids'].items():
            map_obj.centroids[ctx_name] = ContextCentroid(
                context_name=ctx_data['context_name'],
                centroid_vector=AcousticFeatureVector17.from_dict(ctx_data['centroid_vector']),
                sample_count=ctx_data['sample_count'],
                variance=np.array(ctx_data['variance']) if ctx_data.get('variance') else None,
                std_dev=np.array(ctx_data['std_dev']) if ctx_data.get('std_dev') else None
            )

        return map_obj

    def summarize(self):
        """Print summary of contextual map."""
        print("\n" + "="*70)
        print("CONTEXTUAL MAP SUMMARY")
        print("="*70)

        if self.baseline_context:
            print(f"\n📍 Baseline Context: {self.baseline_context}")

        print(f"\n📊 Contexts Defined: {len(self.centroids)}")
        print("-" * 70)

        for ctx_name, centroid in sorted(self.centroids.items()):
            is_baseline = " ← BASELINE" if ctx_name == self.baseline_context else ""
            print(f"\n{ctx_name.upper()}{is_baseline}")
            print(f"  Sample Count: {centroid.sample_count}")
            print(f"  Centroid: {centroid.centroid_vector}")

            if centroid.std_dev is not None:
                # Show 3 features with highest variance (least diagnostic)
                feature_names = centroid.centroid_vector.feature_names()
                variances = centroid.variance
                top_variance_idx = np.argsort(variances)[-3:][::-1]

                print("  Highest Variance Features (least diagnostic):")
                for idx in top_variance_idx:
                    print(f"    - {feature_names[idx]}: {variances[idx]:.2f}")

        print("\n" + "="*70)


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CONTEXTUAL MAP DEMONSTRATION")
    print("="*70)

    # Create example phrase vectors with context labels
    phrase_vectors = {
        'phrase_001': AcousticFeatureVector17(
            mean_f0_hz=6500, duration_ms=70, attack_ms=0.010, decay_ms=0.050,
            f0_range_hz=400, vibrato_rate_hz=8.0, vibrato_depth_hz=50.0,
            jitter=0.02, shimmer=0.03, harmonicity_hnr=20.0, spectral_flatness=0.1,
            spectral_centroid_hz=7000.0, spectral_rolloff_hz=13000.0,
            bandwidth_hz=5000.0, slope_db_per_octave=-8.0,
            rms_db=-20.0, peak_amplitude=0.15
        ),
        'phrase_002': AcousticFeatureVector17(
            mean_f0_hz=6400, duration_ms=75, attack_ms=0.012, decay_ms=0.045,
            f0_range_hz=450, vibrato_rate_hz=7.5, vibrato_depth_hz=45.0,
            jitter=0.018, shimmer=0.028, harmonicity_hnr=22.0, spectral_flatness=0.12,
            spectral_centroid_hz=7200.0, spectral_rolloff_hz=12500.0,
            bandwidth_hz=5200.0, slope_db_per_octave=-7.5,
            rms_db=-19.0, peak_amplitude=0.14
        ),
        'phrase_003': AcousticFeatureVector17(
            mean_f0_hz=6100, duration_ms=55, attack_ms=0.005, decay_ms=0.030,
            f0_range_hz=3500, vibrato_rate_hz=12.0, vibrato_depth_hz=150.0,
            jitter=0.08, shimmer=0.05, harmonicity_hnr=5.0, spectral_flatness=0.3,
            spectral_centroid_hz=8000.0, spectral_rolloff_hz=15000.0,
            bandwidth_hz=8000.0, slope_db_per_octave=-4.0,
            rms_db=-15.0, peak_amplitude=0.25
        ),
        'phrase_004': AcousticFeatureVector17(
            mean_f0_hz=6000, duration_ms=50, attack_ms=0.004, decay_ms=0.025,
            f0_range_hz=3800, vibrato_rate_hz=11.0, vibrato_depth_hz=140.0,
            jitter=0.075, shimmer=0.045, harmonicity_hnr=6.0, spectral_flatness=0.28,
            spectral_centroid_hz=8200.0, spectral_rolloff_hz=14500.0,
            bandwidth_hz=7500.0, slope_db_per_octave=-4.5,
            rms_db=-14.0, peak_amplitude=0.23
        ),
    }

    context_labels = {
        'phrase_001': 'contact',
        'phrase_002': 'contact',
        'phrase_003': 'aggression',
        'phrase_004': 'aggression',
    }

    # Create contextual map
    map_obj = ContextualMap()
    map_obj.calculate_context_centroids(phrase_vectors, context_labels)
    map_obj.summarize()

    # Generate graded phrases
    print("\n" + "="*70)
    print("GRADIENT GENERATION")
    print("="*70)

    for intensity in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print(f"\n🎯 Intensity {intensity*100:.0f}% (Contact → Aggression)")

        virtual = map_obj.generate_graded_phrase('aggression', intensity)
        print(f"  Virtual: {virtual}")

        nearest_key, nearest_vec, distance = map_obj.find_nearest_real_phrase(virtual, phrase_vectors)
        print(f"  Nearest: {nearest_key} (distance: {distance:.3f})")

    # Context delta
    print("\n" + "="*70)
    print("CONTEXT DELTA (What makes Aggression different from Contact?)")
    print("="*70)

    delta = map_obj.calculate_context_delta('aggression', 'contact')
    print(f"\nDelta Vector: {delta}")

    # Show most different features
    delta_vec = delta.to_numpy()
    feature_names = delta.feature_names()
    top_delta_idx = np.argsort(np.abs(delta_vec))[-5:][::-1]

    print("\nTop 5 Differentiating Features:")
    for idx in top_delta_idx:
        print(f"  - {feature_names[idx]}: {delta_vec[idx]:+.2f}")

    print("\n" + "="*70)
    print("\n✅ Contextual Map enables:")
    print("   ✓ Semantic centroids (define 'meaning' of contexts)")
    print("   ✓ Gradient generation (30% Aggressive, 50% Aggressive, etc.)")
    print("   ✓ Nearest-neighbor lookup (find real phrase for synthesis)")
    print("   ✓ Context delta analysis (what makes X different from Y?)")
    print()
