"""
Cognitive Interaction Engine (The "Brain" of the Hybrid Stack)

This module implements the cognitive layer that:
1. Calculates virtual targets from intents using Acoustic Algebra
2. Finds nearest neighbor phrases in the database
3. Applies safety clamping to prevent over-warping artifacts
4. Converts deltas to Rust synthesis parameters

Architecture: Python Logic Layer → Rust Execution Layer

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

# =============================================================================
# Data Models
# =============================================================================


@dataclass
class Vector17D:
    """30-dimensional acoustic feature vector (expanded from 17D/20D)

    Note: Named Vector17D for backwards compatibility, now contains 30 fields.
    Added features: harmonicity, shimmer, spectral_flux, mfcc_5-13 (9 new MFCCs)
    """

    # === Fundamental (3 features) ===
    mean_f0_hz: float
    duration_ms: float
    f0_range_hz: float

    # === Grit Factors (3 features) - Added harmonicity ===
    harmonic_to_noise_ratio: float
    spectral_flatness: float
    harmonicity: float  # NEW: Degree of periodicity vs noise

    # === Motion Factors (7 features) - Added shimmer ===
    attack_time_ms: float
    decay_time_ms: float
    sustain_level: float
    vibrato_rate_hz: float
    vibrato_depth: float
    jitter: float
    shimmer: float  # NEW: Amplitude instability

    # === Fingerprint Factors (13 features) - Expanded from 4 ===
    mfcc_1: float
    mfcc_2: float
    mfcc_3: float
    mfcc_4: float
    mfcc_5: float  # NEW
    mfcc_6: float  # NEW
    mfcc_7: float  # NEW
    mfcc_8: float  # NEW
    mfcc_9: float  # NEW
    mfcc_10: float  # NEW
    mfcc_11: float  # NEW
    mfcc_12: float  # NEW
    mfcc_13: float  # NEW
    spectral_contrast: float

    # === Spectral Dynamics (1 feature) - NEW ===
    spectral_flux: float  # NEW: Rate of spectral change

    # === Rhythm Factors (3 features) ===
    median_ici_ms: float
    onset_rate_hz: float
    ici_coefficient_of_variation: float


@dataclass
class VectorDelta:
    """Delta between two vectors (matches expanded Vector17D implementation)"""

    delta_mean_f0_hz: float
    delta_duration_ms: float
    delta_f0_range_hz: float
    delta_harmonic_to_noise_ratio: float
    delta_spectral_flatness: float
    delta_harmonicity: float  # NEW: Delta in harmonicity
    delta_attack_time_ms: float
    delta_decay_time_ms: float
    delta_sustain_level: float
    delta_vibrato_rate_hz: float
    delta_vibrato_depth: float
    delta_jitter: float
    delta_shimmer: float  # NEW: Delta in shimmer
    delta_mfcc_1: float
    delta_mfcc_2: float
    delta_mfcc_3: float
    delta_mfcc_4: float
    delta_mfcc_5: float  # NEW
    delta_mfcc_6: float  # NEW
    delta_mfcc_7: float  # NEW
    delta_mfcc_8: float  # NEW
    delta_mfcc_9: float  # NEW
    delta_mfcc_10: float  # NEW
    delta_mfcc_11: float  # NEW
    delta_mfcc_12: float  # NEW
    delta_mfcc_13: float  # NEW
    delta_spectral_contrast: float
    delta_spectral_flux: float  # NEW: Delta in spectral flux
    delta_median_ici_ms: float
    delta_onset_rate_hz: float
    delta_ici_coefficient_of_variation: float


@dataclass
class AudioPhrase:
    """Audio phrase from database"""

    key: str
    features: Vector17D
    species: str


# =============================================================================
# Cognitive Interaction Engine
# =============================================================================


class CognitiveInteractionEngine:
    """
    The "Brain" of the cognitive hybrid stack.

    Responsibilities:
    1. Intent → Virtual Target (via Acoustic Algebra)
    2. Virtual Target → Nearest Neighbor (via Phrase Database)
    3. Delta Calculation & Safety Clamping (prevents Uncanny Valley)
    4. Delta → Rust Synthesis Parameters
    """

    def __init__(self, algebra_map, phrase_db, synthesizer, max_safe_warp: float = 0.2):
        """
        Initialize the cognitive interaction engine.

        Args:
            algebra_map: AcousticAlgebraMap for calculating virtual targets
            phrase_db: PhraseDatabase for finding nearest neighbors
            synthesizer: Rust synthesizer interface (via PyO3)
            max_safe_warp: Maximum safe warp distance (default 0.2 = 20%)
        """
        self.algebra_map = algebra_map
        self.phrase_db = phrase_db
        self.synthesizer = synthesizer
        self.max_safe_warp = max_safe_warp
        self.logger = logging.getLogger(__name__)

    def generate_response(self, intent: str, intensity: float) -> Optional[Dict[str, Any]]:
        """
        Generate a response audio buffer from intent and intensity.

        Pipeline:
        1. Calculate virtual target from intent/intensity
        2. Find nearest phrase in database
        3. Calculate delta and apply safety clamping
        4. Send warp parameters to Rust synthesizer

        Args:
            intent: Semantic intent (e.g., "aggression", "alarm")
            intensity: Intensity value (0.0 to 1.0)

        Returns:
            Dictionary with synthesis results, or None if failed
        """
        try:
            # Step 1: Calculate virtual target
            virtual_target = self.algebra_map.generate_graded_vector(
                intent=intent, intensity=intensity
            )

            if virtual_target is None:
                self.logger.error(f"Failed to generate virtual target for intent={intent}")
                return None

            # Step 2: Find nearest phrase
            nearest_phrase = self.phrase_db.find_nearest(virtual_target)

            if nearest_phrase is None:
                self.logger.error("No phrases found in database")
                return None

            # Step 3: Calculate delta
            delta = self._calculate_delta(virtual_target, nearest_phrase.features)

            # Step 4: Apply safety clamping
            clamped_delta = self._apply_safety_clamp(delta, nearest_phrase.features, virtual_target)

            # Step 5: Convert delta to Rust parameters
            warp_params = self._delta_to_rust_parameters(clamped_delta, nearest_phrase.features)

            # Step 6: Send to synthesizer
            self.synthesizer.set_warp_delta(warp_params)

            return {
                "intent": intent,
                "intensity": intensity,
                "source_phrase": nearest_phrase.key,
                "virtual_target": virtual_target,
                "delta": clamped_delta,
                "warp_params": warp_params,
                "was_clamped": clamped_delta != delta,
            }

        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}")
            return None

    def _calculate_delta(self, target: Vector17D, anchor: Vector17D) -> VectorDelta:
        """Calculate delta between target and anchor vectors"""
        return VectorDelta(
            delta_mean_f0_hz=target.mean_f0_hz - anchor.mean_f0_hz,
            delta_duration_ms=target.duration_ms - anchor.duration_ms,
            delta_f0_range_hz=target.f0_range_hz - anchor.f0_range_hz,
            delta_harmonic_to_noise_ratio=target.harmonic_to_noise_ratio
            - anchor.harmonic_to_noise_ratio,
            delta_spectral_flatness=target.spectral_flatness - anchor.spectral_flatness,
            delta_harmonicity=target.harmonicity - anchor.harmonicity,
            delta_attack_time_ms=target.attack_time_ms - anchor.attack_time_ms,
            delta_decay_time_ms=target.decay_time_ms - anchor.decay_time_ms,
            delta_sustain_level=target.sustain_level - anchor.sustain_level,
            delta_vibrato_rate_hz=target.vibrato_rate_hz - anchor.vibrato_rate_hz,
            delta_vibrato_depth=target.vibrato_depth - anchor.vibrato_depth,
            delta_jitter=target.jitter - anchor.jitter,
            delta_shimmer=target.shimmer - anchor.shimmer,
            delta_mfcc_1=target.mfcc_1 - anchor.mfcc_1,
            delta_mfcc_2=target.mfcc_2 - anchor.mfcc_2,
            delta_mfcc_3=target.mfcc_3 - anchor.mfcc_3,
            delta_mfcc_4=target.mfcc_4 - anchor.mfcc_4,
            delta_mfcc_5=target.mfcc_5 - anchor.mfcc_5,  # NEW
            delta_mfcc_6=target.mfcc_6 - anchor.mfcc_6,  # NEW
            delta_mfcc_7=target.mfcc_7 - anchor.mfcc_7,  # NEW
            delta_mfcc_8=target.mfcc_8 - anchor.mfcc_8,  # NEW
            delta_mfcc_9=target.mfcc_9 - anchor.mfcc_9,  # NEW
            delta_mfcc_10=target.mfcc_10 - anchor.mfcc_10,  # NEW
            delta_mfcc_11=target.mfcc_11 - anchor.mfcc_11,  # NEW
            delta_mfcc_12=target.mfcc_12 - anchor.mfcc_12,  # NEW
            delta_mfcc_13=target.mfcc_13 - anchor.mfcc_13,  # NEW
            delta_spectral_contrast=target.spectral_contrast - anchor.spectral_contrast,
            delta_spectral_flux=target.spectral_flux - anchor.spectral_flux,
            delta_median_ici_ms=target.median_ici_ms - anchor.median_ici_ms,
            delta_onset_rate_hz=target.onset_rate_hz - anchor.onset_rate_hz,
            delta_ici_coefficient_of_variation=target.ici_coefficient_of_variation
            - anchor.ici_coefficient_of_variation,
        )

    def _apply_safety_clamp(
        self, delta: VectorDelta, anchor: Vector17D, target: Vector17D
    ) -> VectorDelta:
        """
        Apply safety clamping to prevent over-warping.

        If the delta represents a warp distance > max_safe_warp,
        scale it down to stay within safe limits.
        """
        # Calculate normalized distance (simplified metric using key dimensions)
        # Using a subset of key dimensions for distance calculation
        distance = self._calculate_distance(target, anchor)

        if distance <= self.max_safe_warp:
            # Safe! No clamping needed
            return delta
        else:
            # Too far! Apply clamping
            clamp_factor = self.max_safe_warp / distance
            self.logger.warning(
                f"Clamping extrapolation: distance {distance:.3f} exceeds "
                f"max_safe_warp {self.max_safe_warp:.3f}, applying factor {clamp_factor:.3f}"
            )

            return VectorDelta(
                delta_mean_f0_hz=delta.delta_mean_f0_hz * clamp_factor,
                delta_duration_ms=delta.delta_duration_ms * clamp_factor,
                delta_f0_range_hz=delta.delta_f0_range_hz * clamp_factor,
                delta_harmonic_to_noise_ratio=delta.delta_harmonic_to_noise_ratio * clamp_factor,
                delta_spectral_flatness=delta.delta_spectral_flatness * clamp_factor,
                delta_harmonicity=delta.delta_harmonicity * clamp_factor,
                delta_attack_time_ms=delta.delta_attack_time_ms * clamp_factor,
                delta_decay_time_ms=delta.delta_decay_time_ms * clamp_factor,
                delta_sustain_level=delta.delta_sustain_level * clamp_factor,
                delta_vibrato_rate_hz=delta.delta_vibrato_rate_hz * clamp_factor,
                delta_vibrato_depth=delta.delta_vibrato_depth * clamp_factor,
                delta_jitter=delta.delta_jitter * clamp_factor,
                delta_shimmer=delta.delta_shimmer * clamp_factor,
                delta_mfcc_1=delta.delta_mfcc_1 * clamp_factor,
                delta_mfcc_2=delta.delta_mfcc_2 * clamp_factor,
                delta_mfcc_3=delta.delta_mfcc_3 * clamp_factor,
                delta_mfcc_4=delta.delta_mfcc_4 * clamp_factor,
                delta_mfcc_5=delta.delta_mfcc_5 * clamp_factor,  # NEW
                delta_mfcc_6=delta.delta_mfcc_6 * clamp_factor,  # NEW
                delta_mfcc_7=delta.delta_mfcc_7 * clamp_factor,  # NEW
                delta_mfcc_8=delta.delta_mfcc_8 * clamp_factor,  # NEW
                delta_mfcc_9=delta.delta_mfcc_9 * clamp_factor,  # NEW
                delta_mfcc_10=delta.delta_mfcc_10 * clamp_factor,  # NEW
                delta_mfcc_11=delta.delta_mfcc_11 * clamp_factor,  # NEW
                delta_mfcc_12=delta.delta_mfcc_12 * clamp_factor,  # NEW
                delta_mfcc_13=delta.delta_mfcc_13 * clamp_factor,  # NEW
                delta_spectral_contrast=delta.delta_spectral_contrast * clamp_factor,
                delta_spectral_flux=delta.delta_spectral_flux * clamp_factor,
                delta_median_ici_ms=delta.delta_median_ici_ms * clamp_factor,
                delta_onset_rate_hz=delta.delta_onset_rate_hz * clamp_factor,
                delta_ici_coefficient_of_variation=delta.delta_ici_coefficient_of_variation
                * clamp_factor,
            )

    def _calculate_distance(self, v1: Vector17D, v2: Vector17D) -> float:
        """
        Calculate normalized Euclidean distance between two vectors.

        Uses a subset of key dimensions for safety distance calculation:
        F0, duration, HNR, spectral flatness. This is a simplified metric
        for clamp activation decisions, not the full 30D distance.
        """
        # Normalization ranges
        f0_range = 2000.0
        dur_range = 50.0
        hnr_range = 30.0
        flatness_range = 1.0

        # Calculate normalized differences
        f0_diff = (v1.mean_f0_hz - v2.mean_f0_hz) / f0_range
        dur_diff = (v1.duration_ms - v2.duration_ms) / dur_range
        hnr_diff = (v1.harmonic_to_noise_ratio - v2.harmonic_to_noise_ratio) / hnr_range
        flatness_diff = (v1.spectral_flatness - v2.spectral_flatness) / flatness_range

        # Euclidean distance
        return (f0_diff**2 + dur_diff**2 + hnr_diff**2 + flatness_diff**2) ** 0.5

    def _delta_to_rust_parameters(self, delta: VectorDelta, anchor: Vector17D) -> Dict[str, float]:
        """
        Convert 30D delta to Rust synthesis parameters.

        Maps delta values to granular synthesis parameters:
        - pitch_shift_ratio: Based on delta_mean_f0_hz
        - roughness_amount: Based on delta_spectral_flatness
        - duration_scale: Based on delta_duration_ms
        """
        # Calculate pitch shift ratio
        if anchor.mean_f0_hz > 0:
            pitch_shift_ratio = 1.0 + (delta.delta_mean_f0_hz / anchor.mean_f0_hz)
        else:
            pitch_shift_ratio = 1.0

        # Calculate roughness amount (clamp to [0, 1])
        roughness_amount = max(0.0, min(1.0, 0.3 + delta.delta_spectral_flatness))

        # Calculate duration scale
        if anchor.duration_ms > 0:
            duration_scale = 1.0 + (delta.delta_duration_ms / anchor.duration_ms)
        else:
            duration_scale = 1.0

        return {
            "pitch_shift_ratio": pitch_shift_ratio,
            "roughness_amount": roughness_amount,
            "duration_scale": duration_scale,
            "delta_mean_f0_hz": delta.delta_mean_f0_hz,
            "delta_spectral_flatness": delta.delta_spectral_flatness,
        }
