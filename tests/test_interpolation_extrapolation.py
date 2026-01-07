"""
TDD Tests for Interpolation vs Extrapolation in Island Hopping

⚠️ DEPRECATION NOTICE ⚠️
=========================
The Python Vector17D and NavigationEngine implementations in this file are
DEPRECATED and superseded by the Rust implementation in:

    technical_architecture/src/island_hopping.rs

For production use, import from the Rust module via PyO3:

    from technical_architecture import Vector17D, NavigationEngine

The Python implementation in this file is kept ONLY as:
1. Reference implementation for validation
2. Test fixture documentation
3. Cross-validation with Rust tests

DO NOT use the Python Vector17D or NavigationEngine from this file in production.
They are slower (50-100μs vs 1-5μs) and non-deterministic due to GC pauses.

For migration guide, see: archive/deprecated_python_fallbacks/INTERPOLATION_EXTRAPOLATION_DEPRECATION.md

---

This test suite validates the two navigation modes:
- Interpolation (Bridge Builder): Safe navigation between known islands
- Extrapolation (Ocean Explorer): Risky navigation beyond known islands
- Delta Clamping (The Leash): Safety valve to prevent over-warping

Key Principles:
- Interpolation = High fidelity (supported by two anchor islands)
- Extrapolation = Risky (only one anchor, can enter Uncanny Valley)
- Delta Clamping = Safety (max 20% warp distance)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import math


# =============================================================================
# Data Models (Reusing from island_hopping_navigation.py)
# =============================================================================


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

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for calculations"""
        return np.array(
            [
                self.mean_f0_hz,
                self.duration_ms,
                self.f0_range_hz,
                self.harmonic_to_noise_ratio,
                self.spectral_flatness,
                self.attack_time_ms,
                self.decay_time_ms,
                self.sustain_level,
                self.vibrato_rate_hz,
                self.vibrato_depth,
                self.jitter,
                self.shimmer,
                self.mfcc_1,
                self.mfcc_2,
                self.mfcc_3,
                self.mfcc_4,
                self.spectral_contrast,
            ]
        )

    def distance_to(self, other: "Vector17D") -> float:
        """
        Calculate normalized Euclidean distance between two vectors.

        Returns distance normalized to [0, 1] range by dividing by
        the maximum expected range for each dimension.
        """
        v1 = self.to_numpy()
        v2 = other.to_numpy()

        # Normalization ranges for each dimension
        ranges = np.array(
            [
                2000.0,  # mean_f0_hz: 0-2000 Hz typical range
                50.0,  # duration_ms: 0-50 ms
                500.0,  # f0_range_hz: 0-500 Hz
                25.0,  # harmonic_to_noise_ratio: 0-25 dB
                1.0,  # spectral_flatness: 0-1
                30.0,  # attack_time_ms: 0-30 ms
                50.0,  # decay_time_ms: 0-50 ms
                1.0,  # sustain_level: 0-1
                10.0,  # vibrato_rate_hz: 0-10 Hz
                0.1,  # vibrato_depth: 0-0.1
                0.15,  # jitter: 0-0.15
                0.15,  # shimmer: 0-0.15
                2.0,  # mfcc_1: -1 to 1
                2.0,  # mfcc_2: -1 to 1
                2.0,  # mfcc_3: -1 to 1
                2.0,  # mfcc_4: -1 to 1
                20.0,  # spectral_contrast: 0-20 dB
            ]
        )

        # Normalize differences and calculate distance
        diff = (v1 - v2) / ranges
        return float(np.linalg.norm(diff))

    def __add__(self, other: "Vector17D") -> "Vector17D":
        """Add two vectors (for interpolation)"""
        return Vector17D(
            mean_f0_hz=self.mean_f0_hz + other.mean_f0_hz,
            duration_ms=self.duration_ms + other.duration_ms,
            f0_range_hz=self.f0_range_hz + other.f0_range_hz,
            harmonic_to_noise_ratio=self.harmonic_to_noise_ratio + other.harmonic_to_noise_ratio,
            spectral_flatness=self.spectral_flatness + other.spectral_flatness,
            attack_time_ms=self.attack_time_ms + other.attack_time_ms,
            decay_time_ms=self.decay_time_ms + other.decay_time_ms,
            sustain_level=self.sustain_level + other.sustain_level,
            vibrato_rate_hz=self.vibrato_rate_hz + other.vibrato_rate_hz,
            vibrato_depth=self.vibrato_depth + other.vibrato_depth,
            jitter=self.jitter + other.jitter,
            shimmer=self.shimmer + other.shimmer,
            mfcc_1=self.mfcc_1 + other.mfcc_1,
            mfcc_2=self.mfcc_2 + other.mfcc_2,
            mfcc_3=self.mfcc_3 + other.mfcc_3,
            mfcc_4=self.mfcc_4 + other.mfcc_4,
            spectral_contrast=self.spectral_contrast + other.spectral_contrast,
        )

    def __sub__(self, other: "Vector17D") -> "Vector17D":
        """Subtract two vectors (for delta calculation)"""
        return Vector17D(
            mean_f0_hz=self.mean_f0_hz - other.mean_f0_hz,
            duration_ms=self.duration_ms - other.duration_ms,
            f0_range_hz=self.f0_range_hz - other.f0_range_hz,
            harmonic_to_noise_ratio=self.harmonic_to_noise_ratio - other.harmonic_to_noise_ratio,
            spectral_flatness=self.spectral_flatness - other.spectral_flatness,
            attack_time_ms=self.attack_time_ms - other.attack_time_ms,
            decay_time_ms=self.decay_time_ms - other.decay_time_ms,
            sustain_level=self.sustain_level - other.sustain_level,
            vibrato_rate_hz=self.vibrato_rate_hz - other.vibrato_rate_hz,
            vibrato_depth=self.vibrato_depth - other.vibrato_depth,
            jitter=self.jitter - other.jitter,
            shimmer=self.shimmer - other.shimmer,
            mfcc_1=self.mfcc_1 - other.mfcc_1,
            mfcc_2=self.mfcc_2 - other.mfcc_2,
            mfcc_3=self.mfcc_3 - other.mfcc_3,
            mfcc_4=self.mfcc_4 - other.mfcc_4,
            spectral_contrast=self.spectral_contrast - other.spectral_contrast,
        )

    def __mul__(self, scalar: float) -> "Vector17D":
        """Multiply vector by scalar (for weighting)"""
        return Vector17D(
            mean_f0_hz=self.mean_f0_hz * scalar,
            duration_ms=self.duration_ms * scalar,
            f0_range_hz=self.f0_range_hz * scalar,
            harmonic_to_noise_ratio=self.harmonic_to_noise_ratio * scalar,
            spectral_flatness=self.spectral_flatness * scalar,
            attack_time_ms=self.attack_time_ms * scalar,
            decay_time_ms=self.decay_time_ms * scalar,
            sustain_level=self.sustain_level * scalar,
            vibrato_rate_hz=self.vibrato_rate_hz * scalar,
            vibrato_depth=self.vibrato_depth * scalar,
            jitter=self.jitter * scalar,
            shimmer=self.shimmer * scalar,
            mfcc_1=self.mfcc_1 * scalar,
            mfcc_2=self.mfcc_2 * scalar,
            mfcc_3=self.mfcc_3 * scalar,
            mfcc_4=self.mfcc_4 * scalar,
            spectral_contrast=self.spectral_contrast * scalar,
        )

    def normalized(self) -> "Vector17D":
        """Return normalized vector (unit length)"""
        arr = self.to_numpy()
        norm = np.linalg.norm(arr)
        if norm == 0:
            return self * 0.0
        return self * (1.0 / norm)


@dataclass
class NavigationWaypoint:
    """A waypoint in the navigation route"""

    target: Vector17D
    mode: str  # "interpolation" or "extrapolation"
    anchor_island: Optional[str]  # Nearest real island
    distance_to_anchor: float
    was_clamped: bool = False
    original_target: Optional[Vector17D] = None  # If clamped, store original


@dataclass
class AudioIsland:
    """A real audio phrase (safe island in the ocean)"""

    island_id: str
    vector: Vector17D


# =============================================================================
# Navigation Engine with Interpolation/Extrapolation
# =============================================================================


class NavigationEngine:
    """
    Enhanced navigation engine supporting both interpolation and extrapolation
    with delta clamping for safety.
    """

    def __init__(self, max_safe_warp: float = 0.2):
        """
        Initialize navigation engine.

        Args:
            max_safe_warp: Maximum safe warp distance (default 0.2 = 20%)
        """
        self.max_safe_warp = max_safe_warp

    def interpolate(self, start: Vector17D, end: Vector17D, alpha: float) -> Vector17D:
        """
        Interpolate between two vectors (Bridge Builder).

        Formula: result = start * (1 - alpha) + end * alpha

        Args:
            start: Starting vector
            end: Ending vector
            alpha: Interpolation factor (0.0 to 1.0)

        Returns:
            Interpolated vector
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")

        return (start * (1.0 - alpha)) + (end * alpha)

    def extrapolate(self, origin: Vector17D, direction: Vector17D, factor: float) -> Vector17D:
        """
        Extrapolate beyond a vector (Ocean Explorer).

        Formula: result = origin + (direction * factor)

        Args:
            origin: Starting point
            direction: Direction vector (usually end - start)
            factor: Extrapolation factor (> 1.0 goes beyond direction)

        Returns:
            Extrapolated vector
        """
        if factor < 0.0:
            raise ValueError(f"Factor must be >= 0, got {factor}")

        delta = direction * factor
        return origin + delta

    def clamp_to_safe_distance(self, target: Vector17D, anchor: Vector17D) -> NavigationWaypoint:
        """
        Apply "The Leash" - clamp target to safe warp distance.

        If target is too far from anchor, move it closer.

        Args:
            target: Desired target vector
            anchor: Nearest real island (anchor point)

        Returns:
            NavigationWaypoint with clamped target if needed
        """
        distance = target.distance_to(anchor)

        if distance <= self.max_safe_warp:
            # Safe! No clamping needed
            return NavigationWaypoint(
                target=target,
                mode="interpolation" if distance < self.max_safe_warp * 0.5 else "extrapolation",
                anchor_island=None,  # Will be filled by caller
                distance_to_anchor=distance,
                was_clamped=False,
            )
        else:
            # Too far! Apply clamping
            direction = target - anchor
            normalized_direction = direction.normalized()

            # Move to maximum safe distance
            safe_target = anchor + (normalized_direction * self.max_safe_warp)

            return NavigationWaypoint(
                target=safe_target,
                mode="extrapolation_clamped",
                anchor_island=None,  # Will be filled by caller
                distance_to_anchor=self.max_safe_warp,
                was_clamped=True,
                original_target=target,
            )

    def plan_interpolated_route(
        self, start_island: AudioIsland, end_island: AudioIsland, num_waypoints: int
    ) -> List[NavigationWaypoint]:
        """
        Plan an interpolated route between two islands (Bridge Builder).

        This is SAFE - both endpoints are real recordings.

        Args:
            start_island: Starting island
            end_island: Ending island
            num_waypoints: Number of waypoints to generate

        Returns:
            List of waypoints (all interpolation mode)
        """
        waypoints = []

        for i in range(num_waypoints):
            alpha = i / max(1, num_waypoints - 1)
            target = self.interpolate(start_island.vector, end_island.vector, alpha)

            # Find nearest anchor
            dist_to_start = target.distance_to(start_island.vector)
            dist_to_end = target.distance_to(end_island.vector)

            anchor = start_island if dist_to_start < dist_to_end else end_island
            anchor_id = anchor.island_id
            distance = min(dist_to_start, dist_to_end)

            waypoints.append(
                NavigationWaypoint(
                    target=target,
                    mode="interpolation",
                    anchor_island=anchor_id,
                    distance_to_anchor=distance,
                    was_clamped=False,
                )
            )

        return waypoints

    def plan_extrapolated_route(
        self,
        origin_island: AudioIsland,
        direction_island: AudioIsland,
        num_steps: int,
        extrapolation_factor: float,
    ) -> List[NavigationWaypoint]:
        """
        Plan an extrapolated route beyond known islands (Ocean Explorer).

        This is RISKY - only one anchor, can enter Uncanny Valley.

        Args:
            origin_island: Starting island
            direction_island: Island defining direction
            num_steps: Number of steps to take
            extrapolation_factor: How far to go (1.0 = reach direction, >1.0 = beyond)

        Returns:
            List of waypoints (extrapolation mode, possibly clamped)
        """
        waypoints = []

        # Calculate direction vector
        direction = direction_island.vector - origin_island.vector

        for i in range(1, num_steps + 1):
            # Step further into the ocean
            step_factor = extrapolation_factor * (i / num_steps)
            target = self.extrapolate(origin_island.vector, direction, step_factor)

            # Apply clamping (The Leash)
            waypoint = self.clamp_to_safe_distance(target, origin_island.vector)
            waypoint.anchor_island = origin_island.island_id

            waypoints.append(waypoint)

        return waypoints


# =============================================================================
# Phase 1 Tests: Interpolation (Bridge Builder)
# =============================================================================


class TestInterpolation(unittest.TestCase):
    """Test safe interpolation between known islands"""

    def setUp(self):
        self.engine = NavigationEngine(max_safe_warp=0.2)

        # Create test islands
        self.neutral_island = AudioIsland(
            island_id="neutral",
            vector=Vector17D(
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
            ),
        )

        self.aggressive_island = AudioIsland(
            island_id="aggressive",
            vector=Vector17D(
                mean_f0_hz=8000.0,
                duration_ms=40.0,
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
                spectral_contrast=5.0,
            ),
        )

    def test_interpolate_at_alpha_zero_returns_start(self):
        """Test that alpha=0 returns the start vector"""
        result = self.engine.interpolate(
            self.neutral_island.vector, self.aggressive_island.vector, 0.0
        )

        # Should equal neutral exactly
        self.assertAlmostEqual(result.mean_f0_hz, self.neutral_island.vector.mean_f0_hz)
        self.assertAlmostEqual(result.duration_ms, self.neutral_island.vector.duration_ms)

    def test_interpolate_at_alpha_one_returns_end(self):
        """Test that alpha=1 returns the end vector"""
        result = self.engine.interpolate(
            self.neutral_island.vector, self.aggressive_island.vector, 1.0
        )

        # Should equal aggressive exactly
        self.assertAlmostEqual(result.mean_f0_hz, self.aggressive_island.vector.mean_f0_hz)
        self.assertAlmostEqual(result.duration_ms, self.aggressive_island.vector.duration_ms)

    def test_interpolate_at_alpha_half_returns_midpoint(self):
        """Test that alpha=0.5 returns the midpoint"""
        result = self.engine.interpolate(
            self.neutral_island.vector, self.aggressive_island.vector, 0.5
        )

        # Should be halfway between
        expected_f0 = (7000.0 + 8000.0) / 2.0
        self.assertAlmostEqual(result.mean_f0_hz, expected_f0)

    def test_interpolate_rejects_invalid_alpha(self):
        """Test that invalid alpha values are rejected"""
        with self.assertRaises(ValueError):
            self.engine.interpolate(self.neutral_island.vector, self.aggressive_island.vector, -0.1)

        with self.assertRaises(ValueError):
            self.engine.interpolate(self.neutral_island.vector, self.aggressive_island.vector, 1.5)

    def test_interpolated_route_generates_waypoints(self):
        """Test that interpolated route generates correct waypoints"""
        waypoints = self.engine.plan_interpolated_route(
            self.neutral_island, self.aggressive_island, num_waypoints=5
        )

        self.assertEqual(len(waypoints), 5)

        # All should be interpolation mode
        for wp in waypoints:
            self.assertEqual(wp.mode, "interpolation")
            self.assertFalse(wp.was_clamped)

    def test_interpolated_route_waypoints_have_anchors(self):
        """Test that interpolated waypoints have anchor islands"""
        waypoints = self.engine.plan_interpolated_route(
            self.neutral_island, self.aggressive_island, num_waypoints=5
        )

        # All waypoints should have anchors
        for wp in waypoints:
            self.assertIsNotNone(wp.anchor_island)
            self.assertIn(wp.anchor_island, ["neutral", "aggressive"])


# =============================================================================
# Phase 2 Tests: Extrapolation (Ocean Explorer)
# =============================================================================


class TestExtrapolation(unittest.TestCase):
    """Test risky extrapolation beyond known islands"""

    def setUp(self):
        self.engine = NavigationEngine(max_safe_warp=0.2)

        self.neutral_island = AudioIsland(
            island_id="neutral",
            vector=Vector17D(
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
            ),
        )

        self.aggressive_island = AudioIsland(
            island_id="aggressive",
            vector=Vector17D(
                mean_f0_hz=8000.0,
                duration_ms=40.0,
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
                spectral_contrast=5.0,
            ),
        )

    def test_extrapolate_at_factor_zero_returns_origin(self):
        """Test that factor=0 returns the origin"""
        direction = self.aggressive_island.vector - self.neutral_island.vector
        result = self.engine.extrapolate(self.neutral_island.vector, direction, 0.0)

        # Should equal neutral exactly
        self.assertAlmostEqual(result.mean_f0_hz, self.neutral_island.vector.mean_f0_hz)

    def test_extrapolate_at_factor_one_reaches_target(self):
        """Test that factor=1 reaches the target (direction endpoint)"""
        direction = self.aggressive_island.vector - self.neutral_island.vector
        result = self.engine.extrapolate(self.neutral_island.vector, direction, 1.0)

        # Should reach aggressive
        self.assertAlmostEqual(
            result.mean_f0_hz, self.aggressive_island.vector.mean_f0_hz, places=1
        )

    def test_extrapolate_at_factor_two_goes_beyond(self):
        """Test that factor > 1 goes beyond the target"""
        direction = self.aggressive_island.vector - self.neutral_island.vector
        result = self.engine.extrapolate(self.neutral_island.vector, direction, 2.0)

        # Should go beyond aggressive (higher F0)
        self.assertGreater(result.mean_f0_hz, self.aggressive_island.vector.mean_f0_hz)

    def test_extrapolate_rejects_negative_factor(self):
        """Test that negative factors are rejected"""
        direction = self.aggressive_island.vector - self.neutral_island.vector

        with self.assertRaises(ValueError):
            self.engine.extrapolate(self.neutral_island.vector, direction, -0.5)

    def test_extrapolated_route_generates_waypoints(self):
        """Test that extrapolated route generates waypoints"""
        waypoints = self.engine.plan_extrapolated_route(
            self.neutral_island, self.aggressive_island, num_steps=3, extrapolation_factor=1.5
        )

        self.assertEqual(len(waypoints), 3)

    def test_extrapolated_waypoints_have_single_anchor(self):
        """Test that extrapolated waypoints only have one anchor"""
        waypoints = self.engine.plan_extrapolated_route(
            self.neutral_island, self.aggressive_island, num_steps=3, extrapolation_factor=1.5
        )

        # All should anchor to origin (neutral)
        for wp in waypoints:
            self.assertEqual(wp.anchor_island, "neutral")


# =============================================================================
# Phase 3 Tests: Delta Clamping (The Leash)
# =============================================================================


class TestDeltaClamping(unittest.TestCase):
    """Test delta clamping safety mechanism"""

    def setUp(self):
        # Strict clamp for testing (10% max)
        self.engine_strict = NavigationEngine(max_safe_warp=0.1)

        # Lenient clamp (50% max)
        self.engine_lenient = NavigationEngine(max_safe_warp=0.5)

        self.anchor = Vector17D(
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

    def test_safe_target_not_clamped(self):
        """Test that targets within safe distance are not clamped"""
        # Target 5% away (within 10% limit)
        target = Vector17D(
            mean_f0_hz=7100.0,  # +100Hz (small change)
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

        waypoint = self.engine_strict.clamp_to_safe_distance(target, self.anchor)

        self.assertFalse(waypoint.was_clamped)
        self.assertIsNone(waypoint.original_target)

    def test_dangerous_target_is_clamped(self):
        """Test that targets beyond safe distance are clamped"""
        # Target 20% away (beyond 10% limit)
        # Create a target that's significantly different
        target = Vector17D(
            mean_f0_hz=7500.0,  # +500Hz (large change)
            duration_ms=45.0,
            f0_range_hz=500.0,
            harmonic_to_noise_ratio=15.0,
            spectral_flatness=0.3,
            attack_time_ms=12.0,
            decay_time_ms=18.0,
            sustain_level=0.55,
            vibrato_rate_hz=5.0,
            vibrato_depth=0.015,
            jitter=0.05,
            shimmer=0.03,
            mfcc_1=1.3,
            mfcc_2=0.85,
            mfcc_3=-0.1,
            mfcc_4=0.25,
            spectral_contrast=12.0,
        )

        waypoint = self.engine_strict.clamp_to_safe_distance(target, self.anchor)

        self.assertTrue(waypoint.was_clamped)
        self.assertIsNotNone(waypoint.original_target)

        # Clamped target should be closer to anchor
        clamped_distance = waypoint.target.distance_to(self.anchor)
        self.assertLessEqual(clamped_distance, 0.1)

    def test_clamped_target_preserves_direction(self):
        """Test that clamping preserves the direction from anchor"""
        target = Vector17D(
            mean_f0_hz=7500.0,  # Direction: +500Hz
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

        waypoint = self.engine_strict.clamp_to_safe_distance(target, self.anchor)

        # Clamped target should have same sign for F0 change
        self.assertGreater(waypoint.target.mean_f0_hz, self.anchor.mean_f0_hz)

    def test_extrapolation_triggers_clamping(self):
        """Test that extrapolation beyond safe limit triggers clamping"""
        aggressive = Vector17D(
            mean_f0_hz=8000.0,
            duration_ms=40.0,
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
            spectral_contrast=5.0,
        )

        # Distance from neutral to aggressive is > 10%
        distance = self.anchor.distance_to(aggressive)
        self.assertGreater(distance, 0.1)

        waypoint = self.engine_strict.clamp_to_safe_distance(aggressive, self.anchor)

        # Should be clamped
        self.assertTrue(waypoint.was_clamped)


# =============================================================================
# Phase 4 Tests: Integration
# =============================================================================


class TestInterpolationExtrapolationIntegration(unittest.TestCase):
    """Test complete interpolation/extrapolation workflows"""

    def setUp(self):
        self.engine = NavigationEngine(max_safe_warp=0.2)

        self.neutral = AudioIsland(
            island_id="neutral",
            vector=Vector17D(
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
            ),
        )

        self.aggressive = AudioIsland(
            island_id="aggressive",
            vector=Vector17D(
                mean_f0_hz=8000.0,
                duration_ms=40.0,
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
                spectral_contrast=5.0,
            ),
        )

    def test_interpolation_route_is_safe(self):
        """Test that interpolation routes don't trigger clamping"""
        waypoints = self.engine.plan_interpolated_route(
            self.neutral, self.aggressive, num_waypoints=10
        )

        # None should be clamped (interpolation between islands is safe)
        clamped_count = sum(1 for wp in waypoints if wp.was_clamped)
        self.assertEqual(clamped_count, 0)

    def test_extrapolation_route_may_trigger_clamping(self):
        """Test that extrapolation routes may trigger clamping"""
        # Extrapolate 50% beyond aggressive (risky!)
        waypoints = self.engine.plan_extrapolated_route(
            self.neutral, self.aggressive, num_steps=5, extrapolation_factor=1.5
        )

        # Some waypoints should be clamped
        clamped_count = sum(1 for wp in waypoints if wp.was_clamped)
        self.assertGreater(clamped_count, 0)

    def test_extrapolation_respects_safety_limit(self):
        """Test that clamped waypoints respect the safety limit"""
        waypoints = self.engine.plan_extrapolated_route(
            self.neutral,
            self.aggressive,
            num_steps=5,
            extrapolation_factor=2.0,  # Very far!
        )

        for wp in waypoints:
            # All should be within safe distance
            self.assertLessEqual(wp.distance_to_anchor, 0.2)

    def test_clamping_preserves_mode_information(self):
        """Test that clamping preserves mode information"""
        # Create a target that will be clamped
        far_target = Vector17D(
            mean_f0_hz=9000.0,  # Very far!
            duration_ms=30.0,
            f0_range_hz=800.0,
            harmonic_to_noise_ratio=0.0,
            spectral_flatness=1.0,
            attack_time_ms=0.0,
            decay_time_ms=5.0,
            sustain_level=0.2,
            vibrato_rate_hz=0.0,
            vibrato_depth=0.0,
            jitter=0.2,
            shimmer=0.15,
            mfcc_1=2.5,
            mfcc_2=2.0,
            mfcc_3=1.0,
            mfcc_4=0.0,
            spectral_contrast=0.0,
        )

        waypoint = self.engine.clamp_to_safe_distance(far_target, self.neutral.vector)

        # Should be marked as extrapolation (was risky)
        self.assertIn("extrapolation", waypoint.mode)

        # Original target should be preserved
        self.assertIsNotNone(waypoint.original_target)
        self.assertAlmostEqual(waypoint.original_target.mean_f0_hz, 9000.0)

    def test_interpolation_vs_extrapolation_distances(self):
        """Test that extrapolation triggers clamping while interpolation doesn't"""
        interpolated = self.engine.plan_interpolated_route(
            self.neutral, self.aggressive, num_waypoints=5
        )

        extrapolated = self.engine.plan_extrapolated_route(
            self.neutral, self.aggressive, num_steps=5, extrapolation_factor=1.5
        )

        # Interpolation should not trigger clamping (safe mode)
        interp_clamped = any(wp.was_clamped for wp in interpolated)
        self.assertFalse(interp_clamped, "Interpolation should not trigger clamping")

        # Extrapolation should trigger clamping (risky mode)
        extrap_clamped = any(wp.was_clamped for wp in extrapolated)
        self.assertTrue(extrap_clamped, "Extrapolation should trigger clamping")

        # Clamped extrapolation waypoints should all be at max distance
        # (because they hit the safety limit)
        max_dist = max(wp.distance_to_anchor for wp in extrapolated)
        self.assertLessEqual(max_dist, 0.2, "Clamped extrapolation should respect safety limit")


if __name__ == "__main__":
    unittest.main()
