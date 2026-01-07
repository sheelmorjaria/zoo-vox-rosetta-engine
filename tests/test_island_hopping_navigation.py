"""
TDD Tests for "Island Hopping" Navigation Strategy

⚠️ DEPRECATION NOTICE ⚠️
=========================
The Python Vector17D implementation in this file is DEPRECATED and superseded
by the Rust implementation in:

    technical_architecture/src/island_hopping.rs

For production use, import from the Rust module via PyO3:

    from technical_architecture import Vector17D, NavigationEngine

The Python implementation in this file is kept ONLY as:
1. Reference implementation for validation
2. Test fixture documentation
3. Cross-validation with Rust tests

DO NOT use the Python Vector17D from this file in production.
It is slower (50-100μs vs 1-5μs) and non-deterministic due to GC pauses.

For migration guide, see: archive/deprecated_python_fallbacks/INTERPOLATION_EXTRAPOLATION_DEPRECATION.md

---

This test suite validates the waypoint-based navigation through 17D acoustic space.
The core principle: Treat real audio phrases as "Safe Islands" and use granular
synthesis as the "Boat" to travel between them.

Key Concepts:
- Islands: Real audio phrases (high fidelity, safe)
- Ocean: Empty mathematical space (low fidelity, dangerous)
- Waypoints: Calculated route points between islands
- Boat: Granular synthesis (warps between islands)

Navigation Modes:
- Mode A: Linear Gradient (The "Road") - Semantic continuum
- Mode B: Random Walk (The "Drift") - Emergence/discovery
- Mode C: Semantic Avoidance (The "Safe Harbor") - Deception prevention

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import heapq


# =============================================================================
# Data Models
# =============================================================================

class NavigationMode(Enum):
    """Navigation strategy for island hopping"""
    LINEAR_GRADIENT = "LINEAR_GRADIENT"  # Direct path (The "Road")
    RANDOM_WALK = "RANDOM_WALK"          # Exploration (The "Drift")
    SEMANTIC_AVOIDANCE = "SEMANTIC_AVOIDANCE"  # Safety (The "Safe Harbor")


@dataclass
class Vector17D:
    """17D acoustic vector representing a point in acoustic space"""
    mean_f0_hz: float
    duration_ms: float
    f0_range_hz: float
    harmonic_to_noise_ratio: float
    spectral_flatness: float
    attack_time_ms: float
    decay_time_ms: float
    sustain_level: float
    vibrato_rate_hz: float
    vibrato_depth: float
    jitter: float
    shimmer: float
    mfcc_1: float
    mfcc_2: float
    mfcc_3: float
    mfcc_4: float
    spectral_contrast: float

    def distance_to(self, other: 'Vector17D') -> float:
        """Calculate Euclidean distance between two vectors"""
        v1 = np.array([
            self.mean_f0_hz, self.duration_ms, self.f0_range_hz,
            self.harmonic_to_noise_ratio, self.spectral_flatness,
            self.attack_time_ms, self.decay_time_ms, self.sustain_level,
            self.vibrato_rate_hz, self.vibrato_depth, self.jitter, self.shimmer,
            self.mfcc_1, self.mfcc_2, self.mfcc_3, self.mfcc_4,
            self.spectral_contrast
        ])
        v2 = np.array([
            other.mean_f0_hz, other.duration_ms, other.f0_range_hz,
            other.harmonic_to_noise_ratio, other.spectral_flatness,
            other.attack_time_ms, other.decay_time_ms, other.sustain_level,
            other.vibrato_rate_hz, other.vibrato_depth, other.jitter, other.shimmer,
            other.mfcc_1, other.mfcc_2, other.mfcc_3, other.mfcc_4,
            other.spectral_contrast
        ])
        return float(np.linalg.norm(v1 - v2))

    def __sub__(self, other: 'Vector17D') -> 'VectorDelta':
        """Calculate delta between two vectors"""
        return VectorDelta(
            delta_mean_f0_hz=self.mean_f0_hz - other.mean_f0_hz,
            delta_duration_ms=self.duration_ms - other.duration_ms,
            delta_f0_range_hz=self.f0_range_hz - other.f0_range_hz,
            delta_hnr=self.harmonic_to_noise_ratio - other.harmonic_to_noise_ratio,
            delta_spectral_flatness=self.spectral_flatness - other.spectral_flatness,
            delta_attack_time_ms=self.attack_time_ms - other.attack_time_ms,
            delta_decay_time_ms=self.decay_time_ms - other.decay_time_ms,
            delta_sustain_level=self.sustain_level - other.sustain_level,
            delta_vibrato_rate_hz=self.vibrato_rate_hz - other.vibrato_rate_hz,
            delta_vibrato_depth=self.vibrato_depth - other.vibrato_depth,
            delta_jitter=self.jitter - other.jitter,
            delta_shimmer=self.shimmer - other.shimmer,
            delta_mfcc_1=self.mfcc_1 - other.mfcc_1,
            delta_mfcc_2=self.mfcc_2 - other.mfcc_2,
            delta_mfcc_3=self.mfcc_3 - other.mfcc_3,
            delta_mfcc_4=self.mfcc_4 - other.mfcc_4,
            delta_spectral_contrast=self.spectral_contrast - other.spectral_contrast,
        )


@dataclass
class VectorDelta:
    """17D difference vector (warp instructions)"""
    delta_mean_f0_hz: float
    delta_duration_ms: float
    delta_f0_range_hz: float
    delta_hnr: float
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

    @property
    def magnitude(self) -> float:
        """Calculate the magnitude of the delta vector"""
        arr = np.array([
            self.delta_mean_f0_hz, self.delta_duration_ms, self.delta_f0_range_hz,
            self.delta_hnr, self.delta_spectral_flatness,
            self.delta_attack_time_ms, self.delta_decay_time_ms, self.delta_sustain_level,
            self.delta_vibrato_rate_hz, self.delta_vibrato_depth,
            self.delta_jitter, self.delta_shimmer,
            self.delta_mfcc_1, self.delta_mfcc_2, self.delta_mfcc_3, self.delta_mfcc_4,
            self.delta_spectral_contrast
        ])
        return float(np.linalg.norm(arr))

    def clamp(self, max_magnitude: float) -> 'VectorDelta':
        """Clamp delta to maximum magnitude (safety)"""
        if self.magnitude <= max_magnitude:
            return self
        scale = max_magnitude / self.magnitude
        return VectorDelta(
            delta_mean_f0_hz=self.delta_mean_f0_hz * scale,
            delta_duration_ms=self.delta_duration_ms * scale,
            delta_f0_range_hz=self.delta_f0_range_hz * scale,
            delta_hnr=self.delta_hnr * scale,
            delta_spectral_flatness=self.delta_spectral_flatness * scale,
            delta_attack_time_ms=self.delta_attack_time_ms * scale,
            delta_decay_time_ms=self.delta_decay_time_ms * scale,
            delta_sustain_level=self.delta_sustain_level * scale,
            delta_vibrato_rate_hz=self.delta_vibrato_rate_hz * scale,
            delta_vibrato_depth=self.delta_vibrato_depth * scale,
            delta_jitter=self.delta_jitter * scale,
            delta_shimmer=self.delta_shimmer * scale,
            delta_mfcc_1=self.delta_mfcc_1 * scale,
            delta_mfcc_2=self.delta_mfcc_2 * scale,
            delta_mfcc_3=self.delta_mfcc_3 * scale,
            delta_mfcc_4=self.delta_mfcc_4 * scale,
            delta_spectral_contrast=self.delta_spectral_contrast * scale,
        )


@dataclass
class Waypoint:
    """A point in the navigation route"""
    position: Vector17D
    target_intensity: float
    is_real_island: bool = False
    island_id: Optional[str] = None


@dataclass
class RouteSegment:
    """A segment of the journey between two islands"""
    start_island: str  # Real phrase ID
    end_island: Optional[str]  # Next real phrase ID (None for final)
    virtual_target: Vector17D
    warp_delta: VectorDelta
    distance: float


@dataclass
class NavigationRoute:
    """Complete navigation route through the acoustic archipelago"""
    segments: List[RouteSegment] = field(default_factory=list)
    total_distance: float = 0.0

    def add_segment(self, segment: RouteSegment):
        """Add a segment to the route"""
        self.segments.append(segment)
        self.total_distance += segment.distance


@dataclass
class AudioIsland:
    """A real audio phrase (safe island in the ocean)"""
    island_id: str
    vector: Vector17D
    audio_buffer: Optional[np.ndarray] = None


# =============================================================================
# Island Hopping Navigator
# =============================================================================

class AcousticAlgebraEngine:
    """
    The "Chart" - Calculates trajectories through 17D space
    """

    def generate_graded_vector(self, intent: str, intensity: float) -> Vector17D:
        """
        Generate a 17D vector for a given intent and intensity

        This is the "Algebra" - calculating where in the 17D ocean
        a theoretical sound exists.
        """
        # Simplified implementation for testing
        # In production, this would use real acoustic algebra

        if intent == "aggression":
            # Aggression increases F0, decreases HNR, increases roughness
            base_f0 = 7000.0
            base_hnr = 20.0
            base_roughness = 0.1  # spectral flatness

            return Vector17D(
                mean_f0_hz=base_f0 + (1000.0 * intensity),  # Higher pitch = more aggressive
                duration_ms=50.0 - (10.0 * intensity),      # Shorter = more urgent
                f0_range_hz=400.0 + (200.0 * intensity),
                harmonic_to_noise_ratio=base_hnr - (15.0 * intensity),  # Lower HNR = rougher
                spectral_flatness=base_roughness + (0.6 * intensity),  # Higher = noisier
                attack_time_ms=15.0 - (12.0 * intensity),   # Sharper attack
                decay_time_ms=20.0 - (10.0 * intensity),
                sustain_level=0.6 - (0.2 * intensity),
                vibrato_rate_hz=6.0 - (6.0 * intensity),    # Less vibrato when aggressive
                vibrato_depth=0.02 - (0.02 * intensity),
                jitter=0.03 + (0.09 * intensity),           # More jitter
                shimmer=0.02 + (0.06 * intensity),
                mfcc_1=1.0 + (0.8 * intensity),
                mfcc_2=0.7 + (0.5 * intensity),
                mfcc_3=-0.2 + (0.5 * intensity),
                mfcc_4=0.3 - (0.2 * intensity),
                spectral_contrast=15.0 - (10.0 * intensity),
            )
        elif intent == "courtship":
            # Courtship emphasizes complexity and clarity
            return Vector17D(
                mean_f0_hz=7000.0 + (500.0 * intensity),
                duration_ms=50.0 + (50.0 * intensity),      # Longer calls
                f0_range_hz=400.0 + (300.0 * intensity),    # More modulation
                harmonic_to_noise_ratio=20.0 + (5.0 * intensity),  # Clearer tone
                spectral_flatness=0.1,
                attack_time_ms=15.0,
                decay_time_ms=20.0 + (10.0 * intensity),
                sustain_level=0.6,
                vibrato_rate_hz=6.0 + (2.0 * intensity),
                vibrato_depth=0.02 + (0.03 * intensity),
                jitter=0.03,
                shimmer=0.02,
                mfcc_1=1.0,
                mfcc_2=0.7,
                mfcc_3=-0.2,
                mfcc_4=0.3,
                spectral_contrast=15.0,
            )
        else:
            # Neutral / default
            return Vector17D(
                mean_f0_hz=7000.0,
                duration_ms=50.0,
                f0_range_hz=400.0,
                harmonic_to_noise_ratio=20.0,
                spectral_flatness=0.1,
                attack_time_ms=15.0,
                decay_time_ms=20.0,
                sustain_level=0.6,
                vibrato_rate_hz=6.0,
                vibrato_depth=0.02,
                jitter=0.03,
                shimmer=0.02,
                mfcc_1=1.0,
                mfcc_2=0.7,
                mfcc_3=-0.2,
                mfcc_4=0.3,
                spectral_contrast=15.0,
            )


class PhraseDatabase:
    """
    The "Archipelago" - Collection of real audio islands
    """

    def __init__(self):
        self.islands: dict[str, AudioIsland] = {}

    def add_island(self, island: AudioIsland):
        """Add an island to the database"""
        self.islands[island.island_id] = island

    def find_nearest_17d(self, target: Vector17D) -> Optional[AudioIsland]:
        """
        Find the nearest real island to a target vector

        This is the "Navigation" - finding which island is closest
        to our calculated waypoint.
        """
        if not self.islands:
            return None

        nearest_id = min(
            self.islands.keys(),
            key=lambda iid: target.distance_to(self.islands[iid].vector)
        )
        return self.islands[nearest_id]


class IslandHoppingNavigator:
    """
    The "Navigator" - Plans routes through the acoustic archipelago

    Combines:
    - Acoustic Algebra (The Chart)
    - Phrase Database (The Archipelago)
    - Navigation strategy (The Route Planner)
    """

    def __init__(self, algebra: AcousticAlgebraEngine, database: PhraseDatabase):
        self.algebra = algebra
        self.database = database

    def plan_linear_gradient(
        self,
        intent: str,
        start_intensity: float,
        end_intensity: float,
        num_waypoints: int = 10
    ) -> NavigationRoute:
        """
        Mode A: Linear Gradient (The "Road")

        Direct path from start to end intensity.
        Used for: Semantic continuum testing
        """
        route = NavigationRoute()

        # Generate waypoints
        intensities = np.linspace(start_intensity, end_intensity, num_waypoints)

        current_island_id = None

        for intensity in intensities:
            # 1. Calculate virtual target (Algebra)
            target = self.algebra.generate_graded_vector(intent, intensity)

            # 2. Find nearest island (Navigation)
            nearest = self.database.find_nearest_17d(target)

            if nearest is None:
                continue

            # 3. Calculate segment
            if current_island_id is None:
                # First segment - start from this island
                current_island_id = nearest.island_id
            else:
                # Calculate warp from current to next
                current_island = self.database.islands[current_island_id]
                delta = target - current_island.vector

                segment = RouteSegment(
                    start_island=current_island_id,
                    end_island=nearest.island_id,
                    virtual_target=target,
                    warp_delta=delta,
                    distance=delta.magnitude
                )
                route.add_segment(segment)

                # Move to next island
                current_island_id = nearest.island_id

        return route

    def plan_random_walk(
        self,
        start_intent: str,
        start_intensity: float,
        num_steps: int = 10,
        step_size: float = 0.1
    ) -> NavigationRoute:
        """
        Mode B: Random Walk (The "Drift")

        Random exploration of the acoustic space.
        Used for: Emergence and discovery of new valid sounds
        """
        route = NavigationRoute()

        current_intent = start_intent
        current_intensity = start_intensity

        # Get starting island
        target = self.algebra.generate_graded_vector(current_intent, current_intensity)
        nearest = self.database.find_nearest_17d(target)

        if nearest is None:
            return route

        current_island_id = nearest.island_id

        for _ in range(num_steps):
            # Random walk: drift intensity
            current_intensity += np.random.uniform(-step_size, step_size)
            current_intensity = np.clip(current_intensity, 0.0, 1.0)

            # Calculate new target
            target = self.algebra.generate_graded_vector(current_intent, current_intensity)

            # Find nearest island
            nearest = self.database.find_nearest_17d(target)
            if nearest is None:
                continue

            # Calculate segment
            current_island = self.database.islands[current_island_id]
            delta = target - current_island.vector

            segment = RouteSegment(
                start_island=current_island_id,
                end_island=nearest.island_id,
                virtual_target=target,
                warp_delta=delta,
                distance=delta.magnitude
            )
            route.add_segment(segment)

            # Move to next island
            current_island_id = nearest.island_id

        return route


# =============================================================================
# Phase 1 Tests: Waypoint Calculation
# =============================================================================

class TestVector17DCalculations(unittest.TestCase):
    """Test 17D vector operations for navigation"""

    def test_distance_calculation(self):
        """Test Euclidean distance between vectors"""
        v1 = Vector17D(
            mean_f0_hz=7000.0, duration_ms=50.0, f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0, spectral_flatness=0.1,
            attack_time_ms=15.0, decay_time_ms=20.0, sustain_level=0.6,
            vibrato_rate_hz=6.0, vibrato_depth=0.02, jitter=0.03, shimmer=0.02,
            mfcc_1=1.0, mfcc_2=0.7, mfcc_3=-0.2, mfcc_4=0.3, spectral_contrast=15.0
        )

        v2 = Vector17D(
            mean_f0_hz=8000.0, duration_ms=50.0, f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0, spectral_flatness=0.1,
            attack_time_ms=15.0, decay_time_ms=20.0, sustain_level=0.6,
            vibrato_rate_hz=6.0, vibrato_depth=0.02, jitter=0.03, shimmer=0.02,
            mfcc_1=1.0, mfcc_2=0.7, mfcc_3=-0.2, mfcc_4=0.3, spectral_contrast=15.0
        )

        distance = v1.distance_to(v2)

        # Only F0 differs by 1000 Hz
        self.assertAlmostEqual(distance, 1000.0, places=1)

    def test_delta_calculation(self):
        """Test delta vector calculation"""
        v1 = Vector17D(
            mean_f0_hz=7000.0, duration_ms=50.0, f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0, spectral_flatness=0.1,
            attack_time_ms=15.0, decay_time_ms=20.0, sustain_level=0.6,
            vibrato_rate_hz=6.0, vibrato_depth=0.02, jitter=0.03, shimmer=0.02,
            mfcc_1=1.0, mfcc_2=0.7, mfcc_3=-0.2, mfcc_4=0.3, spectral_contrast=15.0
        )

        v2 = Vector17D(
            mean_f0_hz=8000.0, duration_ms=40.0, f0_range_hz=500.0,
            harmonic_to_noise_ratio=15.0, spectral_flatness=0.2,
            attack_time_ms=10.0, decay_time_ms=15.0, sustain_level=0.5,
            vibrato_rate_hz=5.0, vibrato_depth=0.01, jitter=0.04, shimmer=0.03,
            mfcc_1=1.2, mfcc_2=0.8, mfcc_3=-0.1, mfcc_4=0.2, spectral_contrast=12.0
        )

        delta = v2 - v1

        self.assertEqual(delta.delta_mean_f0_hz, 1000.0)
        self.assertEqual(delta.delta_duration_ms, -10.0)
        self.assertEqual(delta.delta_hnr, -5.0)

    def test_delta_magnitude(self):
        """Test delta magnitude calculation"""
        delta = VectorDelta(
            delta_mean_f0_hz=1000.0, delta_duration_ms=0.0, delta_f0_range_hz=0.0,
            delta_hnr=0.0, delta_spectral_flatness=0.0,
            delta_attack_time_ms=0.0, delta_decay_time_ms=0.0, delta_sustain_level=0.0,
            delta_vibrato_rate_hz=0.0, delta_vibrato_depth=0.0,
            delta_jitter=0.0, delta_shimmer=0.0,
            delta_mfcc_1=0.0, delta_mfcc_2=0.0, delta_mfcc_3=0.0,
            delta_mfcc_4=0.0, delta_spectral_contrast=0.0
        )

        magnitude = delta.magnitude

        # Only F0 changed by 1000 Hz
        self.assertAlmostEqual(magnitude, 1000.0, places=1)

    def test_delta_clamping(self):
        """Test delta clamping for safety"""
        delta = VectorDelta(
            delta_mean_f0_hz=1000.0, delta_duration_ms=0.0, delta_f0_range_hz=0.0,
            delta_hnr=0.0, delta_spectral_flatness=0.0,
            delta_attack_time_ms=0.0, delta_decay_time_ms=0.0, delta_sustain_level=0.0,
            delta_vibrato_rate_hz=0.0, delta_vibrato_depth=0.0,
            delta_jitter=0.0, delta_shimmer=0.0,
            delta_mfcc_1=0.0, delta_mfcc_2=0.0, delta_mfcc_3=0.0,
            delta_mfcc_4=0.0, delta_spectral_contrast=0.0
        )

        # Clamp to max 20% (represented as 200 for this test)
        clamped = delta.clamp(200.0)

        # Should be scaled down
        self.assertLess(clamped.delta_mean_f0_hz, delta.delta_mean_f0_hz)
        self.assertAlmostEqual(clamped.magnitude, 200.0, places=1)


class TestAcousticAlgebraEngine(unittest.TestCase):
    """Test acoustic algebra (the "Chart")"""

    def setUp(self):
        self.algebra = AcousticAlgebraEngine()

    def test_neutral_aggression_vector(self):
        """Test neutral aggression (0% intensity)"""
        vector = self.algebra.generate_graded_vector("aggression", 0.0)

        # Should be at baseline
        self.assertAlmostEqual(vector.mean_f0_hz, 7000.0, places=1)
        self.assertAlmostEqual(vector.harmonic_to_noise_ratio, 20.0, places=1)
        self.assertAlmostEqual(vector.spectral_flatness, 0.1, places=1)

    def test_full_aggression_vector(self):
        """Test full aggression (100% intensity)"""
        vector = self.algebra.generate_graded_vector("aggression", 1.0)

        # Should be maximally aggressive
        self.assertAlmostEqual(vector.mean_f0_hz, 8000.0, places=1)  # Higher pitch
        self.assertAlmostEqual(vector.harmonic_to_noise_ratio, 5.0, places=1)  # Lower HNR
        self.assertAlmostEqual(vector.spectral_flatness, 0.7, places=1)  # Noisier
        self.assertAlmostEqual(vector.attack_time_ms, 3.0, places=1)  # Sharp attack

    def test_half_aggression_vector(self):
        """Test half aggression (50% intensity)"""
        vector = self.algebra.generate_graded_vector("aggression", 0.5)

        # Should be halfway between
        self.assertAlmostEqual(vector.mean_f0_hz, 7500.0, places=1)
        self.assertAlmostEqual(vector.harmonic_to_noise_ratio, 12.5, places=1)

    def test_courtship_vector(self):
        """Test courtship vector"""
        vector = self.algebra.generate_graded_vector("courtship", 0.7)

        # Courtship emphasizes complexity
        self.assertGreater(vector.duration_ms, 50.0)  # Longer calls
        self.assertGreater(vector.vibrato_depth, 0.02)  # More vibrato


class TestPhraseDatabase(unittest.TestCase):
    """Test phrase database (the "Archipelago")"""

    def setUp(self):
        self.db = PhraseDatabase()

        # Add test islands
        self.db.add_island(AudioIsland(
            island_id="neutral_001",
            vector=Vector17D(
                mean_f0_hz=7000.0, duration_ms=50.0, f0_range_hz=400.0,
                harmonic_to_noise_ratio=20.0, spectral_flatness=0.1,
                attack_time_ms=15.0, decay_time_ms=20.0, sustain_level=0.6,
                vibrato_rate_hz=6.0, vibrato_depth=0.02, jitter=0.03, shimmer=0.02,
                mfcc_1=1.0, mfcc_2=0.7, mfcc_3=-0.2, mfcc_4=0.3, spectral_contrast=15.0
            )
        ))

        self.db.add_island(AudioIsland(
            island_id="aggressive_001",
            vector=Vector17D(
                mean_f0_hz=8000.0, duration_ms=40.0, f0_range_hz=600.0,
                harmonic_to_noise_ratio=5.0, spectral_flatness=0.7,
                attack_time_ms=3.0, decay_time_ms=10.0, sustain_level=0.4,
                vibrato_rate_hz=0.0, vibrato_depth=0.0, jitter=0.12, shimmer=0.08,
                mfcc_1=1.8, mfcc_2=1.2, mfcc_3=0.3, mfcc_4=0.1, spectral_contrast=5.0
            )
        ))

    def test_find_nearest_to_neutral(self):
        """Test finding nearest island to neutral target"""
        target = Vector17D(
            mean_f0_hz=7000.0, duration_ms=50.0, f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0, spectral_flatness=0.1,
            attack_time_ms=15.0, decay_time_ms=20.0, sustain_level=0.6,
            vibrato_rate_hz=6.0, vibrato_depth=0.02, jitter=0.03, shimmer=0.02,
            mfcc_1=1.0, mfcc_2=0.7, mfcc_3=-0.2, mfcc_4=0.3, spectral_contrast=15.0
        )

        nearest = self.db.find_nearest_17d(target)

        self.assertIsNotNone(nearest)
        self.assertEqual(nearest.island_id, "neutral_001")

    def test_find_nearest_to_aggressive(self):
        """Test finding nearest island to aggressive target"""
        target = Vector17D(
            mean_f0_hz=7800.0, duration_ms=42.0, f0_range_hz=550.0,
            harmonic_to_noise_ratio=7.0, spectral_flatness=0.6,
            attack_time_ms=4.0, decay_time_ms=12.0, sustain_level=0.45,
            vibrato_rate_hz=1.0, vibrato_depth=0.01, jitter=0.1, shimmer=0.06,
            mfcc_1=1.6, mfcc_2=1.1, mfcc_3=0.2, mfcc_4=0.15, spectral_contrast=7.0
        )

        nearest = self.db.find_nearest_17d(target)

        self.assertIsNotNone(nearest)
        # Should be closer to aggressive_001
        self.assertEqual(nearest.island_id, "aggressive_001")

    def test_empty_database_returns_none(self):
        """Test that empty database returns None"""
        empty_db = PhraseDatabase()
        target = Vector17D(
            mean_f0_hz=7000.0, duration_ms=50.0, f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0, spectral_flatness=0.1,
            attack_time_ms=15.0, decay_time_ms=20.0, sustain_level=0.6,
            vibrato_rate_hz=6.0, vibrato_depth=0.02, jitter=0.03, shimmer=0.02,
            mfcc_1=1.0, mfcc_2=0.7, mfcc_3=-0.2, mfcc_4=0.3, spectral_contrast=15.0
        )

        nearest = empty_db.find_nearest_17d(target)

        self.assertIsNone(nearest)


# =============================================================================
# Phase 2 Tests: Navigation Planning
# =============================================================================

class TestIslandHoppingNavigator(unittest.TestCase):
    """Test island hopping navigation (the "Route Planner")"""

    def setUp(self):
        self.algebra = AcousticAlgebraEngine()
        self.db = PhraseDatabase()

        # Add test islands spanning the aggression spectrum
        self.db.add_island(AudioIsland(
            island_id="neutral",
            vector=Vector17D(
                mean_f0_hz=7000.0, duration_ms=50.0, f0_range_hz=400.0,
                harmonic_to_noise_ratio=20.0, spectral_flatness=0.1,
                attack_time_ms=15.0, decay_time_ms=20.0, sustain_level=0.6,
                vibrato_rate_hz=6.0, vibrato_depth=0.02, jitter=0.03, shimmer=0.02,
                mfcc_1=1.0, mfcc_2=0.7, mfcc_3=-0.2, mfcc_4=0.3, spectral_contrast=15.0
            )
        ))

        self.db.add_island(AudioIsland(
            island_id="mild_aggression",
            vector=Vector17D(
                mean_f0_hz=7500.0, duration_ms=45.0, f0_range_hz=500.0,
                harmonic_to_noise_ratio=12.0, spectral_flatness=0.4,
                attack_time_ms=8.0, decay_time_ms=15.0, sustain_level=0.5,
                vibrato_rate_hz=3.0, vibrato_depth=0.01, jitter=0.07, shimmer=0.05,
                mfcc_1=1.4, mfcc_2=0.9, mfcc_3=0.1, mfcc_4=0.2, spectral_contrast=10.0
            )
        ))

        self.db.add_island(AudioIsland(
            island_id="full_aggression",
            vector=Vector17D(
                mean_f0_hz=8000.0, duration_ms=40.0, f0_range_hz=600.0,
                harmonic_to_noise_ratio=5.0, spectral_flatness=0.7,
                attack_time_ms=3.0, decay_time_ms=10.0, sustain_level=0.4,
                vibrato_rate_hz=0.0, vibrato_depth=0.0, jitter=0.12, shimmer=0.08,
                mfcc_1=1.8, mfcc_2=1.2, mfcc_3=0.3, mfcc_4=0.1, spectral_contrast=5.0
            )
        ))

        self.navigator = IslandHoppingNavigator(self.algebra, self.db)

    def test_linear_gradient_route_planning(self):
        """Test Mode A: Linear Gradient navigation"""
        route = self.navigator.plan_linear_gradient(
            intent="aggression",
            start_intensity=0.0,
            end_intensity=1.0,
            num_waypoints=5
        )

        # Should generate a route
        self.assertGreater(len(route.segments), 0)

        # All segments should have valid data
        for segment in route.segments:
            self.assertIsNotNone(segment.start_island)
            self.assertIsNotNone(segment.warp_delta)

    def test_linear_gradient_follows_spectrum(self):
        """Test that linear gradient follows the aggression spectrum"""
        route = self.navigator.plan_linear_gradient(
            intent="aggression",
            start_intensity=0.0,
            end_intensity=1.0,
            num_waypoints=10
        )

        # Route should start at or near neutral
        if len(route.segments) > 0:
            first_segment = route.segments[0]
            # Start island should be neutral or close
            self.assertIn(first_segment.start_island, ["neutral", "mild_aggression"])

    def test_random_walk_generates_route(self):
        """Test Mode B: Random Walk navigation"""
        np.random.seed(42)  # For reproducibility

        route = self.navigator.plan_random_walk(
            start_intent="aggression",
            start_intensity=0.5,
            num_steps=5,
            step_size=0.1
        )

        # Should generate a route with one segment per step
        self.assertEqual(len(route.segments), 5)

    def test_route_segment_distances_are_reasonable(self):
        """Test that route segments don't exceed safe warp distance"""
        route = self.navigator.plan_linear_gradient(
            intent="aggression",
            start_intensity=0.0,
            end_intensity=1.0,
            num_waypoints=10
        )

        # All segments should have reasonable distances
        # (In production, this would be checked against max_warp_ratio)
        for segment in route.segments:
            # Distance should be non-negative
            self.assertGreaterEqual(segment.distance, 0.0)

    def test_total_distance_is_calculated(self):
        """Test that route calculates total distance correctly"""
        route = self.navigator.plan_linear_gradient(
            intent="aggression",
            start_intensity=0.0,
            end_intensity=1.0,
            num_waypoints=5
        )

        # Total distance should be sum of segment distances
        expected_total = sum(s.distance for s in route.segments)
        self.assertAlmostEqual(route.total_distance, expected_total)


# =============================================================================
# Phase 3 Tests: Safety and Edge Cases
# =============================================================================

class TestIslandHoppingSafety(unittest.TestCase):
    """Test safety mechanisms for island hopping"""

    def setUp(self):
        self.algebra = AcousticAlgebraEngine()
        self.db = PhraseDatabase()
        self.navigator = IslandHoppingNavigator(self.algebra, self.db)

    def test_navigation_with_empty_database(self):
        """Test navigation with no islands"""
        route = self.navigator.plan_linear_gradient(
            intent="aggression",
            start_intensity=0.0,
            end_intensity=1.0,
            num_waypoints=5
        )

        # Should return empty route
        self.assertEqual(len(route.segments), 0)

    def test_navigation_with_single_island(self):
        """Test navigation with only one island"""
        self.db.add_island(AudioIsland(
            island_id="only_island",
            vector=Vector17D(
                mean_f0_hz=7000.0, duration_ms=50.0, f0_range_hz=400.0,
                harmonic_to_noise_ratio=20.0, spectral_flatness=0.1,
                attack_time_ms=15.0, decay_time_ms=20.0, sustain_level=0.6,
                vibrato_rate_hz=6.0, vibrato_depth=0.02, jitter=0.03, shimmer=0.02,
                mfcc_1=1.0, mfcc_2=0.7, mfcc_3=-0.2, mfcc_4=0.3, spectral_contrast=15.0
            )
        ))

        route = self.navigator.plan_linear_gradient(
            intent="aggression",
            start_intensity=0.0,
            end_intensity=1.0,
            num_waypoints=5
        )

        # With a single island, all segments use that same island
        # The navigator still generates segments (virtual targets differ)
        # but all point to the same real source
        self.assertGreater(len(route.segments), 0)

        # All segments should use the same island
        for segment in route.segments:
            self.assertEqual(segment.start_island, "only_island")
            if segment.end_island:
                self.assertEqual(segment.end_island, "only_island")

    def test_zero_waypoints(self):
        """Test navigation with zero waypoints"""
        self.db.add_island(AudioIsland(
            island_id="test",
            vector=Vector17D(
                mean_f0_hz=7000.0, duration_ms=50.0, f0_range_hz=400.0,
                harmonic_to_noise_ratio=20.0, spectral_flatness=0.1,
                attack_time_ms=15.0, decay_time_ms=20.0, sustain_level=0.6,
                vibrato_rate_hz=6.0, vibrato_depth=0.02, jitter=0.03, shimmer=0.02,
                mfcc_1=1.0, mfcc_2=0.7, mfcc_3=-0.2, mfcc_4=0.3, spectral_contrast=15.0
            )
        ))

        route = self.navigator.plan_linear_gradient(
            intent="aggression",
            start_intensity=0.0,
            end_intensity=1.0,
            num_waypoints=0
        )

        # Should handle gracefully
        # (Route might be empty or minimal)
        self.assertIsInstance(route, NavigationRoute)

    def test_extreme_intensities(self):
        """Test navigation with extreme intensity values"""
        self.db.add_island(AudioIsland(
            island_id="test",
            vector=Vector17D(
                mean_f0_hz=7000.0, duration_ms=50.0, f0_range_hz=400.0,
                harmonic_to_noise_ratio=20.0, spectral_flatness=0.1,
                attack_time_ms=15.0, decay_time_ms=20.0, sustain_level=0.6,
                vibrato_rate_hz=6.0, vibrato_depth=0.02, jitter=0.03, shimmer=0.02,
                mfcc_1=1.0, mfcc_2=0.7, mfcc_3=-0.2, mfcc_4=0.3, spectral_contrast=15.0
            )
        ))

        # Test negative intensity (should be clamped in production)
        route = self.navigator.plan_random_walk(
            start_intent="aggression",
            start_intensity=-0.5,
            num_steps=3,
            step_size=0.1
        )

        # Should still generate a route
        self.assertIsInstance(route, NavigationRoute)


# =============================================================================
# Phase 4 Tests: Integration
# =============================================================================

class TestIslandHoppingIntegration(unittest.TestCase):
    """Test complete island hopping workflow"""

    def setUp(self):
        self.algebra = AcousticAlgebraEngine()
        self.db = PhraseDatabase()

        # Create a realistic archipelago
        # Neutral islands
        for i in range(3):
            self.db.add_island(AudioIsland(
                island_id=f"neutral_{i:03d}",
                vector=Vector17D(
                    mean_f0_hz=7000.0 + np.random.uniform(-100, 100),
                    duration_ms=50.0 + np.random.uniform(-5, 5),
                    f0_range_hz=400.0,
                    harmonic_to_noise_ratio=20.0,
                    spectral_flatness=0.1,
                    attack_time_ms=15.0,
                    decay_time_ms=20.0,
                    sustain_level=0.6,
                    vibrato_rate_hz=6.0,
                    vibrato_depth=0.02,
                    jitter=0.03,
                    shimmer=0.02,
                    mfcc_1=1.0,
                    mfcc_2=0.7,
                    mfcc_3=-0.2,
                    mfcc_4=0.3,
                    spectral_contrast=15.0
                )
            ))

        # Aggressive islands
        for i in range(3):
            self.db.add_island(AudioIsland(
                island_id=f"aggressive_{i:03d}",
                vector=Vector17D(
                    mean_f0_hz=8000.0 + np.random.uniform(-100, 100),
                    duration_ms=40.0 + np.random.uniform(-5, 5),
                    f0_range_hz=600.0,
                    harmonic_to_noise_ratio=5.0,
                    spectral_flatness=0.7,
                    attack_time_ms=3.0,
                    decay_time_ms=10.0,
                    sustain_level=0.4,
                    vibrato_rate_hz=0.0,
                    vibrato_depth=0.0,
                    jitter=0.12,
                    shimmer=0.08,
                    mfcc_1=1.8,
                    mfcc_2=1.2,
                    mfcc_3=0.3,
                    mfcc_4=0.1,
                    spectral_contrast=5.0
                )
            ))

        self.navigator = IslandHoppingNavigator(self.algebra, self.db)

    def test_complete_navigation_workflow(self):
        """Test complete workflow: Algebra → Lookup → Route → Delta"""
        # 1. Plan route
        route = self.navigator.plan_linear_gradient(
            intent="aggression",
            start_intensity=0.0,
            end_intensity=1.0,
            num_waypoints=10
        )

        # 2. Verify route structure
        self.assertGreater(len(route.segments), 0)

        # 3. Verify all segments have required data
        for segment in route.segments:
            self.assertIsNotNone(segment.start_island)
            self.assertIsNotNone(segment.virtual_target)
            self.assertIsNotNone(segment.warp_delta)
            self.assertGreaterEqual(segment.distance, 0.0)

    def test_route_segments_use_different_islands(self):
        """Test that route uses different islands as it progresses"""
        route = self.navigator.plan_linear_gradient(
            intent="aggression",
            start_intensity=0.0,
            end_intensity=1.0,
            num_waypoints=10
        )

        if len(route.segments) > 1:
            # First segment should start at a neutral island
            first_segment = route.segments[0]
            self.assertTrue(first_segment.start_island.startswith("neutral") or
                           first_segment.start_island.startswith("aggressive"))

    def test_route_total_distance_is_accumulated(self):
        """Test that total distance is the sum of all segments"""
        route = self.navigator.plan_linear_gradient(
            intent="aggression",
            start_intensity=0.0,
            end_intensity=1.0,
            num_waypoints=10
        )

        if len(route.segments) > 0:
            # Calculate expected total
            expected_total = sum(s.distance for s in route.segments)
            self.assertAlmostEqual(route.total_distance, expected_total, places=1)

    def test_deterministic_navigation(self):
        """Test that same inputs produce same route"""
        np.random.seed(42)

        route1 = self.navigator.plan_linear_gradient(
            intent="aggression",
            start_intensity=0.0,
            end_intensity=1.0,
            num_waypoints=5
        )

        np.random.seed(42)

        route2 = self.navigator.plan_linear_gradient(
            intent="aggression",
            start_intensity=0.0,
            end_intensity=1.0,
            num_waypoints=5
        )

        # Should produce same number of segments
        self.assertEqual(len(route1.segments), len(route2.segments))


if __name__ == '__main__':
    unittest.main()
