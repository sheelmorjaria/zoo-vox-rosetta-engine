"""
Python Integration Tests for Rust Island Hopping Navigation

This test file verifies that the PyO3 bindings for the Rust Island Hopping
module work correctly from Python.

Test Coverage:
- Vector17D operations (distance, interpolation, arithmetic)
- NavigationEngine (clamp, nearest neighbor lookup)
- NavigationWaypoint results
- PyO3 bridge functionality
"""

import pytest
import numpy as np


@pytest.mark.skip(reason="Requires maturin build of Rust module")
class TestVector17D:
    """Test Vector17D operations through PyO3 bindings"""

    def test_vector17d_creation(self):
        """Test creating a Vector17D with all 17 dimensions"""
        from technical_architecture import Vector17D

        v = Vector17D(
            mean_f0_hz=8000.0,
            duration_ms=60.0,
            f0_range_hz=500.0,
            harmonic_to_noise_ratio=25.0,
            spectral_flatness=0.4,
            attack_time_ms=10.0,
            decay_time_ms=25.0,
            sustain_level=0.8,
            vibrato_rate_hz=8.0,
            vibrato_depth=0.03,
            jitter=0.02,
            mfcc_1=-12.0,
            mfcc_2=-6.0,
            mfcc_3=-3.0,
            mfcc_4=-1.5,
            spectral_contrast=25.0,
            median_ici_ms=180.0,
            onset_rate_hz=10.0,
            ici_coefficient_of_variation=0.4,
        )

        assert v.get_mean_f0_hz() == 8000.0
        assert v.get_duration_ms() == 60.0
        assert v.get_f0_range_hz() == 500.0

    def test_vector17d_default(self):
        """Test creating a Vector17D with default values"""
        from technical_architecture import Vector17D

        v = Vector17D.default()

        assert v.get_mean_f0_hz() == 7000.0
        assert v.get_duration_ms() == 50.0

    def test_vector17d_distance_to(self):
        """Test normalized distance calculation between two vectors"""
        from technical_architecture import Vector17D

        v1 = Vector17D.default()
        v2 = Vector17D.default()

        # Same vectors should have zero distance
        assert v1.distance_to(v2) == 0.0

        # Create a different vector
        v2 = Vector17D(
            mean_f0_hz=8000.0,  # 1000 Hz higher
            duration_ms=50.0,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.3,
            attack_time_ms=5.0,
            decay_time_ms=20.0,
            sustain_level=0.7,
            vibrato_rate_hz=7.0,
            vibrato_depth=0.02,
            jitter=0.01,
            mfcc_1=-10.0,
            mfcc_2=-5.0,
            mfcc_3=-2.0,
            mfcc_4=-1.0,
            spectral_contrast=20.0,
            median_ici_ms=150.0,
            onset_rate_hz=8.0,
            ici_coefficient_of_variation=0.3,
        )

        # Distance should be positive
        distance = v1.distance_to(v2)
        assert distance > 0.0
        # With 1000 Hz diff and 2000 Hz range, contribution is 0.5
        # sqrt(0.5^2) = 0.5 (other dimensions match)
        assert abs(distance - 0.5) < 0.01

    def test_vector17d_interpolate(self):
        """Test interpolation between two vectors (Bridge Builder)"""
        from technical_architecture import Vector17D

        v1 = Vector17D.default()
        v2 = Vector17D(
            mean_f0_hz=8000.0,
            duration_ms=60.0,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.3,
            attack_time_ms=5.0,
            decay_time_ms=20.0,
            sustain_level=0.7,
            vibrato_rate_hz=7.0,
            vibrato_depth=0.02,
            jitter=0.01,
            mfcc_1=-10.0,
            mfcc_2=-5.0,
            mfcc_3=-2.0,
            mfcc_4=-1.0,
            spectral_contrast=20.0,
            median_ici_ms=150.0,
            onset_rate_hz=8.0,
            ici_coefficient_of_variation=0.3,
        )

        # Alpha = 0.5 should give midpoint
        result = v1.interpolate(v2, 0.5)
        assert result.get_mean_f0_hz() == 7500.0
        assert result.get_duration_ms() == 55.0

        # Alpha = 0.0 should return v1
        result = v1.interpolate(v2, 0.0)
        assert result.get_mean_f0_hz() == v1.get_mean_f0_hz()

        # Alpha = 1.0 should return v2
        result = v1.interpolate(v2, 1.0)
        assert result.get_mean_f0_hz() == v2.get_mean_f0_hz()

    def test_vector17d_add(self):
        """Test vector addition"""
        from technical_architecture import Vector17D

        v1 = Vector17D.default()
        v2 = Vector17D(
            mean_f0_hz=1000.0,
            duration_ms=10.0,
            f0_range_hz=100.0,
            harmonic_to_noise_ratio=5.0,
            spectral_flatness=0.1,
            attack_time_ms=5.0,
            decay_time_ms=10.0,
            sustain_level=0.3,
            vibrato_rate_hz=1.0,
            vibrato_depth=0.01,
            jitter=0.01,
            mfcc_1=-2.0,
            mfcc_2=-1.0,
            mfcc_3=-1.0,
            mfcc_4=-1.0,
            spectral_contrast=5.0,
            median_ici_ms=50.0,
            onset_rate_hz=2.0,
            ici_coefficient_of_variation=0.1,
        )

        result = v1.add(v2)
        assert result.get_mean_f0_hz() == 8000.0
        assert result.get_duration_ms() == 60.0

    def test_vector17d_sub(self):
        """Test vector subtraction"""
        from technical_architecture import Vector17D

        v1 = Vector17D.default()
        v2 = Vector17D(
            mean_f0_hz=6000.0,
            duration_ms=40.0,
            f0_range_hz=300.0,
            harmonic_to_noise_ratio=15.0,
            spectral_flatness=0.2,
            attack_time_ms=0.0,
            decay_time_ms=10.0,
            sustain_level=0.4,
            vibrato_rate_hz=6.0,
            vibrato_depth=0.01,
            jitter=0.0,
            mfcc_1=-12.0,
            mfcc_2=-6.0,
            mfcc_3=-3.0,
            mfcc_4=-2.0,
            spectral_contrast=15.0,
            median_ici_ms=100.0,
            onset_rate_hz=6.0,
            ici_coefficient_of_variation=0.2,
        )

        result = v1.sub(v2)
        assert result.get_mean_f0_hz() == 1000.0
        assert result.get_duration_ms() == 10.0

    def test_vector17d_scale(self):
        """Test vector scaling"""
        from technical_architecture import Vector17D

        v1 = Vector17D.default()
        result = v1.scale(2.0)
        assert result.get_mean_f0_hz() == 14000.0
        assert result.get_duration_ms() == 100.0

    def test_vector17d_magnitude(self):
        """Test vector magnitude calculation"""
        from technical_architecture import Vector17D

        v1 = Vector17D.default()
        mag = v1.magnitude()
        assert mag > 0.0

    def test_vector17d_normalized(self):
        """Test vector normalization"""
        from technical_architecture import Vector17D

        v1 = Vector17D.default()
        normalized = v1.normalized()
        mag = normalized.magnitude()

        # Normalized vector should have magnitude ~1.0
        assert abs(mag - 1.0) < 0.01


@pytest.mark.skip(reason="Requires maturin build of Rust module")
class TestNavigationEngine:
    """Test NavigationEngine through PyO3 bindings"""

    def test_navigation_engine_creation(self):
        """Test creating a NavigationEngine"""
        from technical_architecture import NavigationEngine

        engine = NavigationEngine()
        assert engine is not None

    def test_navigation_engine_with_max_warp(self):
        """Test creating a NavigationEngine with custom max warp"""
        from technical_architecture import NavigationEngine

        engine = NavigationEngine.with_max_warp(0.3)
        assert engine is not None

    def test_navigation_engine_interpolate(self):
        """Test interpolation through NavigationEngine"""
        from technical_architecture import NavigationEngine, Vector17D

        engine = NavigationEngine()
        v1 = Vector17D.default()
        v2 = Vector17D(
            mean_f0_hz=8000.0,
            duration_ms=60.0,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.3,
            attack_time_ms=5.0,
            decay_time_ms=20.0,
            sustain_level=0.7,
            vibrato_rate_hz=7.0,
            vibrato_depth=0.02,
            jitter=0.01,
            mfcc_1=-10.0,
            mfcc_2=-5.0,
            mfcc_3=-2.0,
            mfcc_4=-1.0,
            spectral_contrast=20.0,
            median_ici_ms=150.0,
            onset_rate_hz=8.0,
            ici_coefficient_of_variation=0.3,
        )

        result = engine.interpolate(v1, v2, 0.5)
        assert result.get_mean_f0_hz() == 7500.0

    def test_navigation_engine_clamp_safe(self):
        """Test clamping with safe distance"""
        from technical_architecture import NavigationEngine, Vector17D

        engine = NavigationEngine.with_max_warp(0.3)

        anchor = Vector17D.default()
        target = Vector17D(
            mean_f0_hz=7100.0,  # Close to anchor
            duration_ms=50.5,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.3,
            attack_time_ms=5.0,
            decay_time_ms=20.0,
            sustain_level=0.7,
            vibrato_rate_hz=7.0,
            vibrato_depth=0.02,
            jitter=0.01,
            mfcc_1=-10.0,
            mfcc_2=-5.0,
            mfcc_3=-2.0,
            mfcc_4=-1.0,
            spectral_contrast=20.0,
            median_ici_ms=150.0,
            onset_rate_hz=8.0,
            ici_coefficient_of_variation=0.3,
        )

        waypoint = engine.clamp_to_safe_distance(target, anchor, "island1")
        assert not waypoint.was_clamped()

    def test_navigation_engine_clamp_unsafe(self):
        """Test clamping with unsafe distance"""
        from technical_architecture import NavigationEngine, Vector17D

        engine = NavigationEngine.with_max_warp(0.2)

        anchor = Vector17D.default()
        target = Vector17D(
            mean_f0_hz=9000.0,  # Far from anchor
            duration_ms=70.0,
            f0_range_hz=600.0,
            harmonic_to_noise_ratio=25.0,
            spectral_flatness=0.5,
            attack_time_ms=15.0,
            decay_time_ms=30.0,
            sustain_level=0.9,
            vibrato_rate_hz=10.0,
            vibrato_depth=0.05,
            jitter=0.03,
            mfcc_1=-8.0,
            mfcc_2=-3.0,
            mfcc_3=-1.0,
            mfcc_4=0.0,
            spectral_contrast=30.0,
            median_ici_ms=200.0,
            onset_rate_hz=12.0,
            ici_coefficient_of_variation=0.5,
        )

        waypoint = engine.clamp_to_safe_distance(target, anchor, "island1")
        assert waypoint.was_clamped()

    def test_navigation_engine_find_nearest(self):
        """Test finding nearest island"""
        from technical_architecture import NavigationEngine, Vector17D

        engine = NavigationEngine()

        # Add some islands
        engine.add_island(
            "island1",
            Vector17D(
                mean_f0_hz=7000.0,
                duration_ms=50.0,
                f0_range_hz=400.0,
                harmonic_to_noise_ratio=20.0,
                spectral_flatness=0.3,
                attack_time_ms=5.0,
                decay_time_ms=20.0,
                sustain_level=0.7,
                vibrato_rate_hz=7.0,
                vibrato_depth=0.02,
                jitter=0.01,
                mfcc_1=-10.0,
                mfcc_2=-5.0,
                mfcc_3=-2.0,
                mfcc_4=-1.0,
                spectral_contrast=20.0,
                median_ici_ms=150.0,
                onset_rate_hz=8.0,
                ici_coefficient_of_variation=0.3,
            ),
            "marmoset",
        )

        engine.add_island(
            "island2",
            Vector17D(
                mean_f0_hz=8000.0,
                duration_ms=60.0,
                f0_range_hz=500.0,
                harmonic_to_noise_ratio=25.0,
                spectral_flatness=0.4,
                attack_time_ms=10.0,
                decay_time_ms=25.0,
                sustain_level=0.8,
                vibrato_rate_hz=8.0,
                vibrato_depth=0.03,
                jitter=0.02,
                mfcc_1=-8.0,
                mfcc_2=-3.0,
                mfcc_3=-1.0,
                mfcc_4=0.0,
                spectral_contrast=25.0,
                median_ici_ms=180.0,
                onset_rate_hz=10.0,
                ici_coefficient_of_variation=0.4,
            ),
            "marmoset",
        )

        target = Vector17D.default()
        nearest = engine.find_nearest_island(target)

        assert nearest is not None
        assert nearest.key == "island1"


@pytest.mark.skip(reason="Requires maturin build of Rust module")
class TestNavigationWaypoint:
    """Test NavigationWaypoint results"""

    def test_waypoint_get_target(self):
        """Test getting target from waypoint"""
        from technical_architecture import NavigationEngine, Vector17D

        engine = NavigationEngine()
        anchor = Vector17D.default()
        target = Vector17D(
            mean_f0_hz=7100.0,
            duration_ms=50.5,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.3,
            attack_time_ms=5.0,
            decay_time_ms=20.0,
            sustain_level=0.7,
            vibrato_rate_hz=7.0,
            vibrato_depth=0.02,
            jitter=0.01,
            mfcc_1=-10.0,
            mfcc_2=-5.0,
            mfcc_3=-2.0,
            mfcc_4=-1.0,
            spectral_contrast=20.0,
            median_ici_ms=150.0,
            onset_rate_hz=8.0,
            ici_coefficient_of_variation=0.3,
        )

        waypoint = engine.clamp_to_safe_distance(target, anchor, "island1")
        result_target = waypoint.get_target()

        assert result_target.get_mean_f0_hz() == target.get_mean_f0_hz()

    def test_waypoint_get_mode(self):
        """Test getting mode from waypoint"""
        from technical_architecture import NavigationEngine, Vector17D

        engine = NavigationEngine()
        anchor = Vector17D.default()
        target = Vector17D(
            mean_f0_hz=7100.0,
            duration_ms=50.5,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.3,
            attack_time_ms=5.0,
            decay_time_ms=20.0,
            sustain_level=0.7,
            vibrato_rate_hz=7.0,
            vibrato_depth=0.02,
            jitter=0.01,
            mfcc_1=-10.0,
            mfcc_2=-5.0,
            mfcc_3=-2.0,
            mfcc_4=-1.0,
            spectral_contrast=20.0,
            median_ici_ms=150.0,
            onset_rate_hz=8.0,
            ici_coefficient_of_variation=0.3,
        )

        waypoint = engine.clamp_to_safe_distance(target, anchor, "island1")
        mode = waypoint.get_mode()

        # Should be one of: Interpolation, Extrapolation, ExtrapolationClamped
        assert mode in ["Interpolation", "Extrapolation", "ExtrapolationClamped"]

    def test_waypoint_get_distance(self):
        """Test getting distance from waypoint"""
        from technical_architecture import NavigationEngine, Vector17D

        engine = NavigationEngine()
        anchor = Vector17D.default()
        target = Vector17D(
            mean_f0_hz=7100.0,
            duration_ms=50.5,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.3,
            attack_time_ms=5.0,
            decay_time_ms=20.0,
            sustain_level=0.7,
            vibrato_rate_hz=7.0,
            vibrato_depth=0.02,
            jitter=0.01,
            mfcc_1=-10.0,
            mfcc_2=-5.0,
            mfcc_3=-2.0,
            mfcc_4=-1.0,
            spectral_contrast=20.0,
            median_ici_ms=150.0,
            onset_rate_hz=8.0,
            ici_coefficient_of_variation=0.3,
        )

        waypoint = engine.clamp_to_safe_distance(target, anchor, "island1")
        distance = waypoint.get_distance_to_anchor()

        assert distance >= 0.0


@pytest.mark.skip(reason="Requires maturin build of Rust module")
class TestAudioIsland:
    """Test AudioIsland through PyO3 bindings"""

    def test_audio_island_creation(self):
        """Test creating an AudioIsland"""
        from technical_architecture import AudioIsland, Vector17D

        features = Vector17D.default()
        island = AudioIsland("test_key", features, "marmoset")

        assert island.key == "test_key"
        assert island.species == "marmoset"
        assert island.features.get_mean_f0_hz() == 7000.0


@pytest.mark.skip(reason="Requires maturin build of Rust module")
class TestIntegration:
    """Integration tests for the complete workflow"""

    def test_complete_navigation_workflow(self):
        """Test a complete navigation workflow from Python to Rust"""
        from technical_architecture import NavigationEngine, Vector17D

        # Create navigation engine
        engine = NavigationEngine.with_max_warp(0.2)

        # Create some islands
        marmoset_phee = Vector17D(
            mean_f0_hz=7000.0,
            duration_ms=50.0,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.3,
            attack_time_ms=5.0,
            decay_time_ms=20.0,
            sustain_level=0.7,
            vibrato_rate_hz=7.0,
            vibrato_depth=0.02,
            jitter=0.01,
            mfcc_1=-10.0,
            mfcc_2=-5.0,
            mfcc_3=-2.0,
            mfcc_4=-1.0,
            spectral_contrast=20.0,
            median_ici_ms=150.0,
            onset_rate_hz=8.0,
            ici_coefficient_of_variation=0.3,
        )

        aggressive_phee = Vector17D(
            mean_f0_hz=8000.0,
            duration_ms=30.0,
            f0_range_hz=600.0,
            harmonic_to_noise_ratio=25.0,
            spectral_flatness=0.5,
            attack_time_ms=3.0,
            decay_time_ms=15.0,
            sustain_level=0.9,
            vibrato_rate_hz=10.0,
            vibrato_depth=0.05,
            jitter=0.03,
            mfcc_1=-8.0,
            mfcc_2=-3.0,
            mfcc_3=-1.0,
            mfcc_4=0.0,
            spectral_contrast=30.0,
            median_ici_ms=100.0,
            onset_rate_hz=12.0,
            ici_coefficient_of_variation=0.5,
        )

        # Add islands to database
        engine.add_island("marmoset_phee", marmoset_phee, "marmoset")
        engine.add_island("aggressive_phee", aggressive_phee, "marmoset")

        # Find nearest island to a virtual target
        virtual_target = Vector17D(
            mean_f0_hz=7500.0,  # Between the two
            duration_ms=40.0,
            f0_range_hz=500.0,
            harmonic_to_noise_ratio=22.5,
            spectral_flatness=0.4,
            attack_time_ms=4.0,
            decay_time_ms=17.5,
            sustain_level=0.8,
            vibrato_rate_hz=8.5,
            vibrato_depth=0.035,
            jitter=0.02,
            mfcc_1=-9.0,
            mfcc_2=-4.0,
            mfcc_3=-1.5,
            mfcc_4=-0.5,
            spectral_contrast=25.0,
            median_ici_ms=125.0,
            onset_rate_hz=10.0,
            ici_coefficient_of_variation=0.4,
        )

        nearest = engine.find_nearest_island(virtual_target)
        assert nearest is not None

        # Calculate delta
        anchor = nearest.features
        delta_f0 = virtual_target.get_mean_f0_hz() - anchor.get_mean_f0_hz()
        delta_dur = virtual_target.get_duration_ms() - anchor.get_duration_ms()

        # Interpolate (Bridge Builder - SAFE)
        interpolated = engine.interpolate(anchor, virtual_target, 0.5)
        assert interpolated.get_mean_f0_hz() > anchor.get_mean_f0_hz()
        assert interpolated.get_mean_f0_hz() < virtual_target.get_mean_f0_hz()

        # Apply safety clamping
        waypoint = engine.clamp_to_safe_distance(virtual_target, anchor, nearest.key)

        assert waypoint.get_distance_to_anchor() >= 0.0
        assert waypoint.get_mode() in ["Interpolation", "Extrapolation", "ExtrapolationClamped"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
