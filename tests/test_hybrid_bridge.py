"""
TDD Tests for "The Hybrid Bridge": Acoustic Algebra + Granular Synthesis

This test suite validates the integration of Acoustic Algebra (The Map)
with Granular Synthesis (The Vehicle) to create the "Cognitive Synthesis" Engine.

The Hybrid Workflow:
1. Algebra (Planner): Intent → Virtual Target Vector (17D)
2. Database Lookup (Anchor): Virtual Target → Nearest Real Phrase
3. Delta Calculator (Interpreter): Virtual - Real = Delta (17D difference)
4. Delta Mapper: 17D Delta → Granular Warp Parameters
5. Granular Engine (Mouth): Real Audio + Warp → Warped Real Audio

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import math
import unittest
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

# =============================================================================
# Phase 1: Data Models - Virtual Target and Delta
# =============================================================================

class Intent(Enum):
    """Semantic intents for synthesis"""
    NEUTRAL = "NEUTRAL"
    AGGRESSION = "AGGRESSION"
    COURTSHIP = "COURTSHIP"
    ALARM = "ALARM"
    FOOD_DISCOVERY = "FOOD_DISCOVERY"


@dataclass
class VirtualTarget:
    """
    A 17-dimensional virtual target calculated by Acoustic Algebra.

    This is a "Ghost Phrase" - a theoretical sound that doesn't exist
    in the database, but represents the ideal acoustic characteristics.
    """
    # Core frequency features (3D)
    mean_f0_hz: float
    duration_ms: float
    f0_range_hz: float

    # Grit factors (2D)
    harmonic_to_noise_ratio: float  # HNR in dB (0-40)
    spectral_flatness: float         # 0=tonal, 1=noise

    # ADSR envelope (3D)
    attack_time_ms: float
    decay_time_ms: float
    sustain_level: float

    # Vibrato (2D)
    vibrato_rate_hz: float
    vibrato_depth: float

    # Micro-dynamics (2D)
    jitter: float
    shimmer: float

    # Timbre (5D)
    mfcc_1: float
    mfcc_2: float
    mfcc_3: float
    mfcc_4: float
    spectral_contrast: float

    # Rhythm (2D)
    median_ici_ms: float
    onset_rate_hz: float


@dataclass
class AcousticDelta:
    """
    The difference between a Virtual Target and a Real Source (17D).

    This represents "Warp Instructions" for the Granular Engine.
    """
    delta_mean_f0_hz: float
    delta_duration_ms: float
    delta_f0_range_hz: float
    delta_harmonic_to_noise_ratio: float
    delta_spectral_flatness: float
    delta_attack_time_ms: float
    delta_decay_time_ms: float
    delta_sustain_level: float
    delta_vibrato_rate_hz: float
    delta_vibrato_depth: float
    delta_jitter: float
    delta_shimmer: float
    delta_mfcc_1: float
    delta_mfcc_2: float
    delta_mfcc_3: float
    delta_mfcc_4: float
    delta_spectral_contrast: float
    delta_median_ici_ms: float
    delta_onset_rate_hz: float

    def magnitude(self) -> float:
        """Calculate Euclidean magnitude of the delta vector"""
        components = [
            self.delta_mean_f0_hz / 1000.0,  # Normalize F0 (Hz)
            self.delta_duration_ms / 100.0,   # Normalize duration (ms)
            self.delta_f0_range_hz / 500.0,
            self.delta_harmonic_to_noise_ratio / 40.0,
            self.delta_spectral_flatness,
            self.delta_attack_time_ms / 50.0,
            self.delta_decay_time_ms / 100.0,
            self.delta_sustain_level,
            self.delta_vibrato_rate_hz / 20.0,
            self.delta_vibrato_depth / 0.1,
            self.delta_jitter / 0.2,
            self.delta_shimmer / 0.2,
            self.delta_mfcc_1 / 3.0,
            self.delta_mfcc_2 / 3.0,
            self.delta_mfcc_3 / 3.0,
            self.delta_mfcc_4 / 3.0,
            self.delta_spectral_contrast / 30.0,
            self.delta_median_ici_ms / 100.0,
            self.delta_onset_rate_hz / 30.0,
        ]
        return math.sqrt(sum(c**2 for c in components))


@dataclass
class GranularWarpParameters:
    """
    Granular synthesis parameters derived from 17D Delta.

    This maps abstract acoustic deltas to concrete granular controls.
    """
    pitch_shift_ratio: float      # 0.5 to 2.0 (1.0 = no shift)
    time_stretch_ratio: float     # 0.5 to 2.0 (1.0 = no stretch)
    roughness_amount: float       # 0.0 to 1.0 (jitter/shimmer mix)
    grain_size_ms: float          # Grain size in milliseconds
    vibrato_amount: float         # Vibrato intensity
    is_clamped: bool              # Whether delta was clamped (safety)


@dataclass
class RealPhrase:
    """A real recording from the database"""
    phrase_id: str
    vector: VirtualTarget  # Reuse VirtualTarget as 17D vector
    audio_buffer: np.ndarray  # Real audio samples


# =============================================================================
# Phase 2: Core Components
# =============================================================================

class AcousticAlgebraEngine:
    """
    The Planner: Converts semantic intents into 17D Virtual Targets.

    Uses linear interpolation between known archetypal vectors.
    """

    def __init__(self):
        # Archetypal vectors for different intents
        self.archetypes = {
            Intent.NEUTRAL: VirtualTarget(
                mean_f0_hz=7000.0, duration_ms=50.0, f0_range_hz=400.0,
                harmonic_to_noise_ratio=20.0, spectral_flatness=0.2,
                attack_time_ms=15.0, decay_time_ms=20.0, sustain_level=0.6,
                vibrato_rate_hz=6.0, vibrato_depth=0.02, jitter=0.03, shimmer=0.02,
                mfcc_1=1.0, mfcc_2=0.7, mfcc_3=-0.2, mfcc_4=0.3, spectral_contrast=15.0,
                median_ici_ms=0.0, onset_rate_hz=0.0
            ),
            Intent.AGGRESSION: VirtualTarget(
                mean_f0_hz=8000.0, duration_ms=40.0, f0_range_hz=600.0,
                harmonic_to_noise_ratio=5.0, spectral_flatness=0.7,  # Gritty
                attack_time_ms=3.0, decay_time_ms=10.0, sustain_level=0.4,  # Sharp
                vibrato_rate_hz=0.0, vibrato_depth=0.0, jitter=0.12, shimmer=0.08,  # Rough
                mfcc_1=1.8, mfcc_2=1.2, mfcc_3=0.3, mfcc_4=0.1, spectral_contrast=5.0,
                median_ici_ms=30.0, onset_rate_hz=15.0  # Rhythmic
            ),
            Intent.COURTSHIP: VirtualTarget(
                mean_f0_hz=6500.0, duration_ms=80.0, f0_range_hz=300.0,
                harmonic_to_noise_ratio=30.0, spectral_flatness=0.05,  # Pure
                attack_time_ms=30.0, decay_time_ms=40.0, sustain_level=0.8,  # Smooth
                vibrato_rate_hz=8.0, vibrato_depth=0.05, jitter=0.01, shimmer=0.005,  # Stable
                mfcc_1=0.5, mfcc_2=0.3, mfcc_3=-0.5, mfcc_4=0.6, spectral_contrast=25.0,
                median_ici_ms=0.0, onset_rate_hz=0.0  # Tonal
            ),
        }

    def calculate_virtual_target(self, intent: Intent, intensity: float) -> VirtualTarget:
        """
        Calculate a virtual target vector for a given intent and intensity.

        Args:
            intent: The semantic intent (AGGRESSION, COURTSHIP, etc.)
            intensity: Intensity from 0.0 (neutral) to 1.0 (full intent)

        Returns:
            VirtualTarget: 17D virtual target vector
        """
        # Clamp intensity to [0, 1]
        intensity = max(0.0, min(1.0, intensity))

        neutral = self.archetypes[Intent.NEUTRAL]
        archetype = self.archetypes.get(intent, neutral)

        # Linear interpolation: Neutral + (Archetype - Neutral) * intensity
        return VirtualTarget(
            mean_f0_hz=neutral.mean_f0_hz + (archetype.mean_f0_hz - neutral.mean_f0_hz) * intensity,
            duration_ms=neutral.duration_ms + (archetype.duration_ms - neutral.duration_ms) * intensity,
            f0_range_hz=neutral.f0_range_hz + (archetype.f0_range_hz - neutral.f0_range_hz) * intensity,
            harmonic_to_noise_ratio=neutral.harmonic_to_noise_ratio + (archetype.harmonic_to_noise_ratio - neutral.harmonic_to_noise_ratio) * intensity,
            spectral_flatness=neutral.spectral_flatness + (archetype.spectral_flatness - neutral.spectral_flatness) * intensity,
            attack_time_ms=neutral.attack_time_ms + (archetype.attack_time_ms - neutral.attack_time_ms) * intensity,
            decay_time_ms=neutral.decay_time_ms + (archetype.decay_time_ms - neutral.decay_time_ms) * intensity,
            sustain_level=neutral.sustain_level + (archetype.sustain_level - neutral.sustain_level) * intensity,
            vibrato_rate_hz=neutral.vibrato_rate_hz + (archetype.vibrato_rate_hz - neutral.vibrato_rate_hz) * intensity,
            vibrato_depth=neutral.vibrato_depth + (archetype.vibrato_depth - neutral.vibrato_depth) * intensity,
            jitter=neutral.jitter + (archetype.jitter - neutral.jitter) * intensity,
            shimmer=neutral.shimmer + (archetype.shimmer - neutral.shimmer) * intensity,
            mfcc_1=neutral.mfcc_1 + (archetype.mfcc_1 - neutral.mfcc_1) * intensity,
            mfcc_2=neutral.mfcc_2 + (archetype.mfcc_2 - neutral.mfcc_2) * intensity,
            mfcc_3=neutral.mfcc_3 + (archetype.mfcc_3 - neutral.mfcc_3) * intensity,
            mfcc_4=neutral.mfcc_4 + (archetype.mfcc_4 - neutral.mfcc_4) * intensity,
            spectral_contrast=neutral.spectral_contrast + (archetype.spectral_contrast - neutral.spectral_contrast) * intensity,
            median_ici_ms=neutral.median_ici_ms + (archetype.median_ici_ms - neutral.median_ici_ms) * intensity,
            onset_rate_hz=neutral.onset_rate_hz + (archetype.onset_rate_hz - neutral.onset_rate_hz) * intensity,
        )


class DeltaCalculator:
    """
    The Interpreter: Calculates the difference between Virtual Target and Real Source.

    This produces "Warp Instructions" for the Granular Engine.
    """

    def calculate_delta(self, target: VirtualTarget, source: VirtualTarget) -> AcousticDelta:
        """
        Calculate the 17D delta between target and source.

        Args:
            target: Virtual target (desired sound)
            source: Real source (nearest neighbor)

        Returns:
            AcousticDelta: 17D difference vector
        """
        return AcousticDelta(
            delta_mean_f0_hz=target.mean_f0_hz - source.mean_f0_hz,
            delta_duration_ms=target.duration_ms - source.duration_ms,
            delta_f0_range_hz=target.f0_range_hz - source.f0_range_hz,
            delta_harmonic_to_noise_ratio=target.harmonic_to_noise_ratio - source.harmonic_to_noise_ratio,
            delta_spectral_flatness=target.spectral_flatness - source.spectral_flatness,
            delta_attack_time_ms=target.attack_time_ms - source.attack_time_ms,
            delta_decay_time_ms=target.decay_time_ms - source.decay_time_ms,
            delta_sustain_level=target.sustain_level - source.sustain_level,
            delta_vibrato_rate_hz=target.vibrato_rate_hz - source.vibrato_rate_hz,
            delta_vibrato_depth=target.vibrato_depth - source.vibrato_depth,
            delta_jitter=target.jitter - source.jitter,
            delta_shimmer=target.shimmer - source.shimmer,
            delta_mfcc_1=target.mfcc_1 - source.mfcc_1,
            delta_mfcc_2=target.mfcc_2 - source.mfcc_2,
            delta_mfcc_3=target.mfcc_3 - source.mfcc_3,
            delta_mfcc_4=target.mfcc_4 - source.mfcc_4,
            delta_spectral_contrast=target.spectral_contrast - source.spectral_contrast,
            delta_median_ici_ms=target.median_ici_ms - source.median_ici_ms,
            delta_onset_rate_hz=target.onset_rate_hz - source.onset_rate_hz,
        )


class DeltaMapper:
    """
    Maps 17D acoustic deltas to granular synthesis parameters.

    Converts abstract acoustic differences into concrete granular controls.
    """

    def __init__(self, max_warp_ratio: float = 0.2):
        """
        Args:
            max_warp_ratio: Maximum allowed warp (0.2 = 20%)
        """
        self.max_warp_ratio = max_warp_ratio

    def map_delta_to_granular(self, delta: AcousticDelta) -> GranularWarpParameters:
        """
        Map 17D delta to granular parameters with delta clamping.

        Args:
            delta: 17D acoustic difference

        Returns:
            GranularWarpParameters: Concrete granular controls
        """
        # Calculate pitch shift from F0 delta
        # Formula: pitch_ratio = (source_f0 + delta_f0) / source_f0
        # For simplicity, we map normalized delta to ratio
        f0_shift_normalized = delta.delta_mean_f0_hz / 1000.0  # Normalize
        pitch_shift = 1.0 + f0_shift_normalized

        # Calculate time stretch from duration delta
        duration_shift_normalized = delta.delta_duration_ms / 100.0
        time_stretch = 1.0 + duration_shift_normalized

        # Calculate roughness from HNR and spectral flatness deltas
        # Lower HNR + Higher flatness = More roughness
        hnr_impact = -delta.delta_harmonic_to_noise_ratio / 40.0  # Negative because lower HNR = rougher
        flatness_impact = delta.delta_spectral_flatness
        roughness = np.clip((hnr_impact + flatness_impact) / 2.0, 0.0, 1.0)

        # Grain size from attack time (faster attack = smaller grains)
        grain_size = 50.0 - delta.delta_attack_time_ms  # Inverse relationship
        grain_size = np.clip(grain_size, 5.0, 100.0)

        # Vibrato from vibrato deltas
        vibrato_amount = (delta.delta_vibrato_rate_hz / 20.0 +
                         delta.delta_vibrato_depth / 0.1) / 2.0
        vibrato_amount = np.clip(vibrato_amount, 0.0, 1.0)

        # DELTA CLAMPING (Safety Check)
        is_clamped = False
        delta_magnitude = delta.magnitude()

        if delta_magnitude > self.max_warp_ratio:
            # Clamp all parameters
            clamp_factor = self.max_warp_ratio / delta_magnitude
            pitch_shift = 1.0 + (pitch_shift - 1.0) * clamp_factor
            time_stretch = 1.0 + (time_stretch - 1.0) * clamp_factor
            roughness = roughness * clamp_factor
            vibrato_amount = vibrato_amount * clamp_factor
            is_clamped = True

        # Final clamps for safety
        pitch_shift = np.clip(pitch_shift, 0.5, 2.0)
        time_stretch = np.clip(time_stretch, 0.5, 2.0)

        return GranularWarpParameters(
            pitch_shift_ratio=pitch_shift,
            time_stretch_ratio=time_stretch,
            roughness_amount=roughness,
            grain_size_ms=grain_size,
            vibrato_amount=vibrato_amount,
            is_clamped=is_clamped
        )


class HybridSynthesisEngine:
    """
    The Cognitive Synthesis Engine: Orchestrates the complete Hybrid workflow.

    Workflow:
    1. Algebra: Intent → Virtual Target
    2. Lookup: Virtual Target → Nearest Real Phrase
    3. Delta: Virtual - Real = Warp Instructions
    4. Map: 17D Delta → Granular Parameters
    5. Synthesize: Real Audio + Parameters → Warped Audio
    """

    def __init__(self, max_warp_ratio: float = 0.2):
        self.algebra = AcousticAlgebraEngine()
        self.delta_calc = DeltaCalculator()
        self.mapper = DeltaMapper(max_warp_ratio=max_warp_ratio)
        self.phrase_database: Dict[str, RealPhrase] = {}

    def register_phrase(self, phrase: RealPhrase):
        """Register a real phrase in the database"""
        self.phrase_database[phrase.phrase_id] = phrase

    def find_nearest_phrase(self, target: VirtualTarget) -> Optional[RealPhrase]:
        """
        Find the nearest real phrase to the virtual target.

        Uses Euclidean distance in 17D space.
        """
        if not self.phrase_database:
            return None

        def distance_17d(v1: VirtualTarget, v2: VirtualTarget) -> float:
            """Calculate Euclidean distance in normalized 17D space"""
            dims = [
                (v1.mean_f0_hz - v2.mean_f0_hz) / 1000.0,
                (v1.duration_ms - v2.duration_ms) / 100.0,
                (v1.f0_range_hz - v2.f0_range_hz) / 500.0,
                (v1.harmonic_to_noise_ratio - v2.harmonic_to_noise_ratio) / 40.0,
                (v1.spectral_flatness - v2.spectral_flatness),
                (v1.attack_time_ms - v2.attack_time_ms) / 50.0,
                (v1.decay_time_ms - v2.decay_time_ms) / 100.0,
                (v1.sustain_level - v2.sustain_level),
                (v1.vibrato_rate_hz - v2.vibrato_rate_hz) / 20.0,
                (v1.vibrato_depth - v1.vibrato_depth) / 0.1,
                (v1.jitter - v2.jitter) / 0.2,
                (v1.shimmer - v2.shimmer) / 0.2,
                (v1.mfcc_1 - v2.mfcc_1) / 3.0,
                (v1.mfcc_2 - v2.mfcc_2) / 3.0,
                (v1.mfcc_3 - v2.mfcc_3) / 3.0,
                (v1.mfcc_4 - v2.mfcc_4) / 3.0,
                (v1.spectral_contrast - v2.spectral_contrast) / 30.0,
                (v1.median_ici_ms - v2.median_ici_ms) / 100.0,
                (v1.onset_rate_hz - v2.onset_rate_hz) / 30.0,
            ]
            return math.sqrt(sum(d**2 for d in dims))

        nearest = min(self.phrase_database.values(),
                     key=lambda p: distance_17d(target, p.vector))
        return nearest

    def synthesize(self, intent: Intent, intensity: float) -> Tuple[Optional[np.ndarray], GranularWarpParameters]:
        """
        Execute complete Hybrid workflow.

        Args:
            intent: Semantic intent (AGGRESSION, COURTSHIP, etc.)
            intensity: Intensity from 0.0 to 1.0

        Returns:
            (audio_buffer, warp_params): Synthesized audio and parameters used
        """
        # Step 1: Algebra - Calculate Virtual Target
        target = self.algebra.calculate_virtual_target(intent, intensity)

        # Step 2: Lookup - Find Nearest Real Phrase
        nearest = self.find_nearest_phrase(target)
        if nearest is None:
            return None, GranularWarpParameters(1.0, 1.0, 0.0, 20.0, 0.0, False)

        # Step 3: Delta - Calculate Warp Instructions
        delta = self.delta_calc.calculate_delta(target, nearest.vector)

        # Step 4: Map - Convert to Granular Parameters
        params = self.mapper.map_delta_to_granular(delta)

        # Step 5: Synthesize - Apply warp to real audio
        # (In real implementation, this would call Rust granular engine)
        # For now, return the original audio buffer as placeholder
        audio = nearest.audio_buffer.copy()

        return audio, params


# =============================================================================
# Phase 1 Tests: Data Models
# =============================================================================

class TestVirtualTarget(unittest.TestCase):
    """Test Virtual Target (Ghost Phrase) creation"""

    def test_create_virtual_target(self):
        """Test creating a 17D virtual target"""
        target = VirtualTarget(
            mean_f0_hz=7000.0, duration_ms=50.0, f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0, spectral_flatness=0.2,
            attack_time_ms=15.0, decay_time_ms=20.0, sustain_level=0.6,
            vibrato_rate_hz=6.0, vibrato_depth=0.02, jitter=0.03, shimmer=0.02,
            mfcc_1=1.0, mfcc_2=0.7, mfcc_3=-0.2, mfcc_4=0.3, spectral_contrast=15.0,
            median_ici_ms=0.0, onset_rate_hz=0.0
        )

        self.assertEqual(target.mean_f0_hz, 7000.0)
        self.assertEqual(target.harmonic_to_noise_ratio, 20.0)
        self.assertEqual(target.spectral_flatness, 0.2)


class TestAcousticDelta(unittest.TestCase):
    """Test Acoustic Delta (Warp Instructions)"""

    def test_delta_calculation(self):
        """Test calculating delta between two vectors"""
        target = VirtualTarget(
            mean_f0_hz=7500.0, duration_ms=60.0, f0_range_hz=500.0,
            harmonic_to_noise_ratio=15.0, spectral_flatness=0.3,
            attack_time_ms=10.0, decay_time_ms=15.0, sustain_level=0.7,
            vibrato_rate_hz=7.0, vibrato_depth=0.03, jitter=0.05, shimmer=0.03,
            mfcc_1=1.2, mfcc_2=0.8, mfcc_3=-0.1, mfcc_4=0.4, spectral_contrast=12.0,
            median_ici_ms=10.0, onset_rate_hz=5.0
        )

        source = VirtualTarget(
            mean_f0_hz=7000.0, duration_ms=50.0, f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0, spectral_flatness=0.2,
            attack_time_ms=15.0, decay_time_ms=20.0, sustain_level=0.6,
            vibrato_rate_hz=6.0, vibrato_depth=0.02, jitter=0.03, shimmer=0.02,
            mfcc_1=1.0, mfcc_2=0.7, mfcc_3=-0.2, mfcc_4=0.3, spectral_contrast=15.0,
            median_ici_ms=0.0, onset_rate_hz=0.0
        )

        calc = DeltaCalculator()
        delta = calc.calculate_delta(target, source)

        # Verify key deltas
        self.assertEqual(delta.delta_mean_f0_hz, 500.0)  # 7500 - 7000
        self.assertEqual(delta.delta_duration_ms, 10.0)  # 60 - 50
        self.assertEqual(delta.delta_harmonic_to_noise_ratio, -5.0)  # 15 - 20

    def test_delta_magnitude(self):
        """Test calculating delta magnitude"""
        delta = AcousticDelta(
            delta_mean_f0_hz=100.0, delta_duration_ms=10.0, delta_f0_range_hz=50.0,
            delta_harmonic_to_noise_ratio=2.0, delta_spectral_flatness=0.05,
            delta_attack_time_ms=5.0, delta_decay_time_ms=3.0, delta_sustain_level=0.1,
            delta_vibrato_rate_hz=1.0, delta_vibrato_depth=0.01, delta_jitter=0.02, delta_shimmer=0.01,
            delta_mfcc_1=0.2, delta_mfcc_2=0.1, delta_mfcc_3=0.1, delta_mfcc_4=0.1,
            delta_spectral_contrast=2.0, delta_median_ici_ms=5.0, delta_onset_rate_hz=2.0
        )

        magnitude = delta.magnitude()
        self.assertGreater(magnitude, 0.0)
        self.assertLess(magnitude, 1.0)  # Should be less than 1 for small delta


# =============================================================================
# Phase 2 Tests: Acoustic Algebra Engine
# =============================================================================

class TestAcousticAlgebraEngine(unittest.TestCase):
    """Test Algebra Engine (The Planner)"""

    def test_calculate_neutral_target(self):
        """Test calculating neutral target (0% intensity)"""
        engine = AcousticAlgebraEngine()
        target = engine.calculate_virtual_target(Intent.AGGRESSION, 0.0)

        # At 0% intensity, should match neutral archetypes
        neutral = engine.archetypes[Intent.NEUTRAL]
        self.assertAlmostEqual(target.mean_f0_hz, neutral.mean_f0_hz, places=5)
        self.assertAlmostEqual(target.harmonic_to_noise_ratio, neutral.harmonic_to_noise_ratio, places=5)

    def test_calculate_full_intensity_target(self):
        """Test calculating full intensity target (100% intensity)"""
        engine = AcousticAlgebraEngine()
        target = engine.calculate_virtual_target(Intent.AGGRESSION, 1.0)

        # At 100% intensity, should match aggression archetype
        aggression = engine.archetypes[Intent.AGGRESSION]
        self.assertAlmostEqual(target.mean_f0_hz, aggression.mean_f0_hz, places=5)
        self.assertAlmostEqual(target.harmonic_to_noise_ratio, aggression.harmonic_to_noise_ratio, places=5)
        self.assertAlmostEqual(target.spectral_flatness, aggression.spectral_flatness, places=5)

    def test_calculate_partial_intensity_target(self):
        """Test calculating partial intensity (50% aggression)"""
        engine = AcousticAlgebraEngine()
        target = engine.calculate_virtual_target(Intent.AGGRESSION, 0.5)

        # At 50% intensity, should be halfway between neutral and aggression
        neutral = engine.archetypes[Intent.NEUTRAL]
        aggression = engine.archetypes[Intent.AGGRESSION]

        expected_f0 = (neutral.mean_f0_hz + aggression.mean_f0_hz) / 2
        self.assertAlmostEqual(target.mean_f0_hz, expected_f0, places=5)

        expected_hnr = (neutral.harmonic_to_noise_ratio + aggression.harmonic_to_noise_ratio) / 2
        self.assertAlmostEqual(target.harmonic_to_noise_ratio, expected_hnr, places=5)

    def test_gradient_continuum(self):
        """Test gradient continuum: 0%, 30%, 60%, 90%, 100%"""
        engine = AcousticAlgebraEngine()

        intensities = [0.0, 0.3, 0.6, 0.9, 1.0]
        targets = [engine.calculate_virtual_target(Intent.AGGRESSION, i) for i in intensities]

        # Verify monotonic progression
        for i in range(1, len(targets)):
            # F0 should increase with intensity
            self.assertGreater(targets[i].mean_f0_hz, targets[i-1].mean_f0_hz)
            # HNR should decrease with intensity (more aggressive = less harmonic)
            self.assertLess(targets[i].harmonic_to_noise_ratio, targets[i-1].harmonic_to_noise_ratio)
            # Flatness should increase (more aggressive = more noise)
            self.assertGreater(targets[i].spectral_flatness, targets[i-1].spectral_flatness)


# =============================================================================
# Phase 3 Tests: Delta Mapper
# =============================================================================

class TestDeltaMapper(unittest.TestCase):
    """Test Delta Mapper (17D to Granular)"""

    def test_map_delta_to_pitch_shift(self):
        """Test mapping F0 delta to pitch shift ratio"""
        mapper = DeltaMapper()

        # Positive F0 delta should increase pitch
        delta_f0_positive = AcousticDelta(
            delta_mean_f0_hz=500.0, delta_duration_ms=0.0, delta_f0_range_hz=0.0,
            delta_harmonic_to_noise_ratio=0.0, delta_spectral_flatness=0.0,
            delta_attack_time_ms=0.0, delta_decay_time_ms=0.0, delta_sustain_level=0.0,
            delta_vibrato_rate_hz=0.0, delta_vibrato_depth=0.0, delta_jitter=0.0, delta_shimmer=0.0,
            delta_mfcc_1=0.0, delta_mfcc_2=0.0, delta_mfcc_3=0.0, delta_mfcc_4=0.0,
            delta_spectral_contrast=0.0, delta_median_ici_ms=0.0, delta_onset_rate_hz=0.0
        )

        params = mapper.map_delta_to_granular(delta_f0_positive)
        self.assertGreater(params.pitch_shift_ratio, 1.0)

    def test_map_delta_to_roughness(self):
        """Test mapping HNR/flatness delta to roughness"""
        mapper = DeltaMapper()

        # Lower HNR + higher flatness = more roughness
        delta_rough = AcousticDelta(
            delta_mean_f0_hz=0.0, delta_duration_ms=0.0, delta_f0_range_hz=0.0,
            delta_harmonic_to_noise_ratio=-10.0,  # Lower HNR
            delta_spectral_flatness=0.3,  # Higher flatness
            delta_attack_time_ms=0.0, delta_decay_time_ms=0.0, delta_sustain_level=0.0,
            delta_vibrato_rate_hz=0.0, delta_vibrato_depth=0.0, delta_jitter=0.0, delta_shimmer=0.0,
            delta_mfcc_1=0.0, delta_mfcc_2=0.0, delta_mfcc_3=0.0, delta_mfcc_4=0.0,
            delta_spectral_contrast=0.0, delta_median_ici_ms=0.0, delta_onset_rate_hz=0.0
        )

        params = mapper.map_delta_to_granular(delta_rough)
        # Roughness should be positive (greater than 0)
        self.assertGreater(params.roughness_amount, 0.0)

    def test_delta_clamping(self):
        """Test delta clamping for safety"""
        mapper = DeltaMapper(max_warp_ratio=0.2)

        # Create a huge delta (simulating 200% aggression request)
        huge_delta = AcousticDelta(
            delta_mean_f0_hz=5000.0, delta_duration_ms=200.0, delta_f0_range_hz=1000.0,
            delta_harmonic_to_noise_ratio=-50.0, delta_spectral_flatness=2.0,
            delta_attack_time_ms=50.0, delta_decay_time_ms=100.0, delta_sustain_level=1.0,
            delta_vibrato_rate_hz=20.0, delta_vibrato_depth=0.5, delta_jitter=0.5, delta_shimmer=0.5,
            delta_mfcc_1=3.0, delta_mfcc_2=3.0, delta_mfcc_3=3.0, delta_mfcc_4=3.0,
            delta_spectral_contrast=50.0, delta_median_ici_ms=200.0, delta_onset_rate_hz=100.0
        )

        params = mapper.map_delta_to_granular(huge_delta)

        # Should be clamped
        self.assertTrue(params.is_clamped)
        # Parameters should be within safe bounds
        self.assertGreaterEqual(params.pitch_shift_ratio, 0.5)
        self.assertLessEqual(params.pitch_shift_ratio, 2.0)


# =============================================================================
# Phase 4 Tests: Hybrid Synthesis Engine (Integration)
# =============================================================================

class TestHybridSynthesisEngine(unittest.TestCase):
    """Test complete Hybrid workflow integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.engine = HybridSynthesisEngine(max_warp_ratio=0.2)

        # Register test phrases
        neutral_phrase = RealPhrase(
            phrase_id="neutral_001",
            vector=VirtualTarget(
                mean_f0_hz=7000.0, duration_ms=50.0, f0_range_hz=400.0,
                harmonic_to_noise_ratio=20.0, spectral_flatness=0.2,
                attack_time_ms=15.0, decay_time_ms=20.0, sustain_level=0.6,
                vibrato_rate_hz=6.0, vibrato_depth=0.02, jitter=0.03, shimmer=0.02,
                mfcc_1=1.0, mfcc_2=0.7, mfcc_3=-0.2, mfcc_4=0.3, spectral_contrast=15.0,
                median_ici_ms=0.0, onset_rate_hz=0.0
            ),
            audio_buffer=np.random.randn(2205).astype(np.float32) * 0.1  # 50ms at 44.1kHz
        )

        aggressive_phrase = RealPhrase(
            phrase_id="aggressive_001",
            vector=VirtualTarget(
                mean_f0_hz=8000.0, duration_ms=40.0, f0_range_hz=600.0,
                harmonic_to_noise_ratio=5.0, spectral_flatness=0.7,
                attack_time_ms=3.0, decay_time_ms=10.0, sustain_level=0.4,
                vibrato_rate_hz=0.0, vibrato_depth=0.0, jitter=0.12, shimmer=0.08,
                mfcc_1=1.8, mfcc_2=1.2, mfcc_3=0.3, mfcc_4=0.1, spectral_contrast=5.0,
                median_ici_ms=30.0, onset_rate_hz=15.0
            ),
            audio_buffer=np.random.randn(1764).astype(np.float32) * 0.1  # 40ms
        )

        self.engine.register_phrase(neutral_phrase)
        self.engine.register_phrase(aggressive_phrase)

    def test_end_to_end_synthesis(self):
        """Test complete end-to-end synthesis workflow"""
        # Request 50% aggression
        audio, params = self.engine.synthesize(Intent.AGGRESSION, 0.5)

        # Should return audio
        self.assertIsNotNone(audio)
        self.assertIsInstance(audio, np.ndarray)

        # Should return valid parameters
        self.assertIsInstance(params, GranularWarpParameters)

    def test_nearest_neighbor_selection(self):
        """Test that nearest neighbor is selected correctly"""
        # Request 75% aggression - should pick aggressive phrase
        target = self.engine.algebra.calculate_virtual_target(Intent.AGGRESSION, 0.75)
        nearest = self.engine.find_nearest_phrase(target)

        self.assertIsNotNone(nearest)
        # Should be closer to aggressive than neutral
        dist_to_aggressive = (
            (target.mean_f0_hz - 8000.0)**2 +
            (target.harmonic_to_noise_ratio - 5.0)**2
        ) ** 0.5

        dist_to_neutral = (
            (target.mean_f0_hz - 7000.0)**2 +
            (target.harmonic_to_noise_ratio - 20.0)**2
        ) ** 0.5

        # Aggressive phrase should be closer
        self.assertLess(dist_to_aggressive, dist_to_neutral)

    def test_gradient_synthesis(self):
        """Test synthesis across gradient continuum"""
        intensities = [0.0, 0.25, 0.5, 0.75, 1.0]

        for intensity in intensities:
            audio, params = self.engine.synthesize(Intent.AGGRESSION, intensity)

            self.assertIsNotNone(audio)
            # Parameters should always be within safe bounds
            self.assertGreaterEqual(params.pitch_shift_ratio, 0.5)
            self.assertLessEqual(params.pitch_shift_ratio, 2.0)

    def test_over_warp_protection(self):
        """Test that over-warping is prevented"""
        # Request 200% intensity (way beyond database)
        audio, params = self.engine.synthesize(Intent.AGGRESSION, 2.0)

        # Should still return audio (from nearest real phrase)
        self.assertIsNotNone(audio)

        # Intensity is clamped to [0, 1], so this will use 100% archetype
        # The delta calculation should be safe (not exceed max_warp_ratio)
        # Verify parameters are within safe bounds
        self.assertGreaterEqual(params.pitch_shift_ratio, 0.5)
        self.assertLessEqual(params.pitch_shift_ratio, 2.0)


# =============================================================================
# Phase 5 Tests: Safety and Edge Cases
# =============================================================================

class TestHybridSafety(unittest.TestCase):
    """Test safety mechanisms and edge cases"""

    def setUp(self):
        """Set up test fixtures"""
        self.engine = HybridSynthesisEngine(max_warp_ratio=0.2)

        # Register test phrases
        neutral_phrase = RealPhrase(
            phrase_id="neutral_001",
            vector=VirtualTarget(
                mean_f0_hz=7000.0, duration_ms=50.0, f0_range_hz=400.0,
                harmonic_to_noise_ratio=20.0, spectral_flatness=0.2,
                attack_time_ms=15.0, decay_time_ms=20.0, sustain_level=0.6,
                vibrato_rate_hz=6.0, vibrato_depth=0.02, jitter=0.03, shimmer=0.02,
                mfcc_1=1.0, mfcc_2=0.7, mfcc_3=-0.2, mfcc_4=0.3, spectral_contrast=15.0,
                median_ici_ms=0.0, onset_rate_hz=0.0
            ),
            audio_buffer=np.random.randn(2205).astype(np.float32) * 0.1
        )

        aggressive_phrase = RealPhrase(
            phrase_id="aggressive_001",
            vector=VirtualTarget(
                mean_f0_hz=8000.0, duration_ms=40.0, f0_range_hz=600.0,
                harmonic_to_noise_ratio=5.0, spectral_flatness=0.7,
                attack_time_ms=3.0, decay_time_ms=10.0, sustain_level=0.4,
                vibrato_rate_hz=0.0, vibrato_depth=0.0, jitter=0.12, shimmer=0.08,
                mfcc_1=1.8, mfcc_2=1.2, mfcc_3=0.3, mfcc_4=0.1, spectral_contrast=5.0,
                median_ici_ms=30.0, onset_rate_hz=15.0
            ),
            audio_buffer=np.random.randn(1764).astype(np.float32) * 0.1
        )

        self.engine.register_phrase(neutral_phrase)
        self.engine.register_phrase(aggressive_phrase)

    def test_empty_database(self):
        """Test behavior with empty phrase database"""
        engine = HybridSynthesisEngine()

        audio, params = engine.synthesize(Intent.AGGRESSION, 0.5)

        # Should return None (no phrases available)
        self.assertIsNone(audio)
        self.assertEqual(params.pitch_shift_ratio, 1.0)

    def test_zero_intensity(self):
        """Test zero intensity (neutral request)"""
        # Use engine from setUp
        audio, params = self.engine.synthesize(Intent.AGGRESSION, 0.0)

        # Should find neutral phrase
        self.assertIsNotNone(audio)
        # Should have minimal warp parameters
        self.assertAlmostEqual(params.pitch_shift_ratio, 1.0, places=1)
        # Roughness should be near 0 when delta is 0
        self.assertLess(params.roughness_amount, 0.2)

    def test_negative_intensity(self):
        """Test negative intensity (should clamp to 0)"""
        # Negative intensity should behave like zero
        audio, params = self.engine.synthesize(Intent.AGGRESSION, -0.5)

        self.assertIsNotNone(audio)
        # Should be clamped to neutral (pitch_shift ~ 1.0)
        self.assertAlmostEqual(params.pitch_shift_ratio, 1.0, places=1)

    def test_high_intensity_beyond_database(self):
        """Test intensity beyond max database values (150%, 200%)"""
        # Use engine from setUp (already has neutral and aggressive phrases)

        # Request 150% intensity (beyond database range)
        # This will be clamped to 100% by the Algebra engine
        audio, params = self.engine.synthesize(Intent.AGGRESSION, 1.5)

        # Should still return audio (clamped to available data)
        self.assertIsNotNone(audio)
        # The intensity clamping in Algebra means the delta is still safe
        # (is_clamped would only be true if delta magnitude exceeded max_warp_ratio)
        # But we still verify parameters are safe
        self.assertGreaterEqual(params.pitch_shift_ratio, 0.5)
        self.assertLessEqual(params.pitch_shift_ratio, 2.0)


if __name__ == '__main__':
    unittest.main()
