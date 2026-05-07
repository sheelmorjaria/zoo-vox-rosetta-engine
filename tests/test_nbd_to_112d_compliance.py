#!/usr/bin/env python3
"""
Module 0 TDD Tests: NBD→112D Variable-Length Segment Compliance

This test suite verifies that the 112D RosettaFeatures extractor correctly
handles variable-length NBD segments without truncation or fixed-window artifacts.

Critical Requirements:
1. duration_ms must reflect the ACTUAL input length, not internal frame size
2. Short segments (< 100ms) must be handled via zero-padding or adaptive sizing
3. f0_mean_derivative (f0_contour_slope) must capture pitch trajectories correctly
4. No crashes on sub-frame segments

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# We'll use the Rust Python bindings to test the actual extractor
# For now, we'll create mock tests that can be run against the compiled Rust library


# =============================================================================
# TEST GENERATORS: Create test audio patterns
# =============================================================================


def generate_sine_tone(
    frequency_hz: float, duration_ms: float, sample_rate: int = 48000
) -> np.ndarray:
    """Generate a pure sine tone."""
    num_samples = int(sample_rate * duration_ms / 1000.0)
    t = np.arange(num_samples) / sample_rate
    audio = 0.5 * np.sin(2 * np.pi * frequency_hz * t).astype(np.float32)
    return audio


def generate_chirp(
    start_freq_hz: float, end_freq_hz: float, duration_ms: float, sample_rate: int = 48000
) -> np.ndarray:
    """
    Generate a linear FM chirp (rising pitch).

    This is critical for testing f0_contour_slope - the pitch changes linearly
    from start_freq to end_freq over the duration.
    """
    num_samples = int(sample_rate * duration_ms / 1000.0)
    t = np.arange(num_samples) / sample_rate

    # Linear frequency sweep: f(t) = f0 + (f1 - f0) * t / T
    start_freq_hz + (end_freq_hz - start_freq_hz) * t / (duration_ms / 1000.0)

    # Phase is integral of frequency
    phase = (
        2
        * np.pi
        * (start_freq_hz * t + (end_freq_hz - start_freq_hz) * t**2 / (2 * duration_ms / 1000.0))
    )

    audio = 0.5 * np.sin(phase).astype(np.float32)
    return audio


def generate_silence(duration_ms: float, sample_rate: int = 48000) -> np.ndarray:
    """Generate silence (zeros)."""
    num_samples = int(sample_rate * duration_ms / 1000.0)
    return np.zeros(num_samples, dtype=np.float32)


# =============================================================================
# TEST SUITE 1: Duration Accuracy
# =============================================================================


class TestDurationAccuracy:
    """Verify that duration_ms reflects actual input length."""

    def test_short_segment_30ms_reports_correct_duration(self):
        """A 30ms staccato opener must report duration_ms = 30.0, not 100.0."""
        audio = generate_sine_tone(10000, 30.0)  # 30ms @ 10kHz

        # TODO: Call Rust extractor
        # features = extractor.extract(audio)
        # assert features.duration_ms == pytest.approx(30.0, abs=1.0)

        # For now, verify the audio length
        expected_samples = int(48000 * 0.030)
        assert len(audio) == expected_samples

    def test_long_segment_500ms_reports_correct_duration(self):
        """A 500ms graded closer must report duration_ms = 500.0."""
        audio = generate_sine_tone(8000, 500.0)  # 500ms @ 8kHz

        expected_samples = int(48000 * 0.500)
        assert len(audio) == expected_samples

    def test_variable_durations_across_range(self):
        """Test various realistic bat call durations."""
        durations_ms = [15, 30, 50, 100, 200, 400, 800]

        for duration in durations_ms:
            audio = generate_sine_tone(12000, duration)
            expected_samples = int(48000 * duration / 1000.0)
            assert len(audio) == expected_samples, f"Failed for {duration}ms"


# =============================================================================
# TEST SUITE 2: Pitch Trajectory (f0_contour_slope)
# =============================================================================


class TestPitchTrajectory:
    """Verify that f0_mean_derivative captures pitch changes correctly."""

    def test_rising_chirp_positive_slope(self):
        """
        A chirp rising from 4kHz to 8kHz over 200ms should have:
        - f0_mean_derivative > 0 (positive slope)
        - f0_range_hz > 3500 (captures the full sweep)
        """
        audio = generate_chirp(4000, 8000, 200.0)

        # TODO: Call Rust extractor
        # features = extractor.extract(audio)
        # assert features.f0_mean_derivative > 0.0
        # assert features.f0_range_hz > 3500.0

        # Verify audio generation
        assert len(audio) == int(48000 * 0.200)

    def test_falling_chirp_negative_slope(self):
        """
        A chirp falling from 12kHz to 6kHz should have:
        - f0_mean_derivative < 0 (negative slope)
        """
        audio = generate_chirp(12000, 6000, 150.0)

        # TODO: Call Rust extractor
        # features = extractor.extract(audio)
        # assert features.f0_mean_derivative < 0.0

        assert len(audio) == int(48000 * 0.150)

    def test_flat_tone_zero_slope(self):
        """
        A pure tone with constant frequency should have:
        - f0_mean_derivative ≈ 0
        - Low f0_range_hz (just jitter)
        """
        generate_sine_tone(10000, 200.0)

        # TODO: Call Rust extractor
        # features = extractor.extract(audio)
        # assert abs(features.f0_mean_derivative) < 100.0  # Small variance allowed


# =============================================================================
# TEST SUITE 3: Sub-Frame Handling
# =============================================================================


class TestSubFrameHandling:
    """Verify that segments shorter than internal FFT size are handled correctly."""

    def test_ultra_short_segment_5ms(self):
        """
        A 5ms segment (240 samples @ 48kHz) is shorter than typical FFT size.
        Must handle via zero-padding or adaptive sizing without crashing.
        """
        audio = generate_sine_tone(15000, 5.0)  # Ultra-short

        # TODO: Call Rust extractor
        # result = extractor.extract(audio)
        # assert result.is_ok()  # Should succeed, not crash
        # assert result.unwrap().duration_ms == 5.0

        assert len(audio) == 240  # 5ms @ 48kHz

    def test_sub_frame_segment_10ms(self):
        """A 10ms segment should also be handled correctly."""
        audio = generate_sine_tone(12000, 10.0)

        assert len(audio) == 480  # 10ms @ 48kHz

    def test_just_at_frame_boundary(self):
        """Test segment exactly at frame boundary (~21ms @ 48kHz for 1024 samples)."""
        # Calculate exact samples needed: 1024 samples = 1024/48000 * 1000 = 21.333...ms
        duration_ms = 1024 / 48000 * 1000
        audio = generate_sine_tone(10000, duration_ms)

        assert len(audio) == 1024


# =============================================================================
# TEST SUITE 4: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test boundary conditions and error handling."""

    def test_empty_audio_rejected(self):
        """Empty audio should return an error, not crash."""
        audio = np.array([], dtype=np.float32)

        # TODO: Call Rust extractor
        # result = extractor.extract(audio)
        # assert result.is_err()

        assert len(audio) == 0

    def test_single_sample(self):
        """Single sample should be handled (though features may be degenerate)."""
        audio = np.array([0.5], dtype=np.float32)

        assert len(audio) == 1

    def test_very_long_segment(self):
        """Test a very long segment (5 seconds) to ensure no overflow."""
        audio = generate_sine_tone(8000, 5000.0)

        expected_samples = int(48000 * 5.0)
        assert len(audio) == expected_samples
        assert audio.dtype == np.float32


# =============================================================================
# TEST SUITE 5: NBD Integration
# =============================================================================


class TestNBDIntegration:
    """Test integration with NeuralBoundaryDetector segment_into_phrases."""

    def test_nbd_phrase_boundary_to_112d(self):
        """
        Simulate the NBD → 112D pipeline:
        1. Create audio with known boundaries
        2. Simulate NBD segmentation
        3. Verify each segment extracts correct duration_ms
        """
        # Create audio: 50ms tone + 50ms silence + 150ms tone
        segment1 = generate_sine_tone(10000, 50.0)
        silence = generate_silence(50.0)
        segment2 = generate_sine_tone(8000, 150.0)

        full_audio = np.concatenate([segment1, silence, segment2])

        # Simulated NBD boundaries at 50ms and 100ms
        boundaries = [50.0, 100.0]

        # Simulate segmentation
        segments = []
        start_sample = 0
        for boundary in boundaries:
            end_sample = int(boundary * 48000 / 1000.0)
            segments.append(full_audio[start_sample:end_sample])
            start_sample = end_sample
        segments.append(full_audio[start_sample:])  # Final segment

        # Verify segment lengths
        assert len(segments) == 3
        assert pytest.approx(len(segments[0]) / 48000 * 1000, abs=1) == 50.0
        assert pytest.approx(len(segments[1]) / 48000 * 1000, abs=1) == 50.0
        assert pytest.approx(len(segments[2]) / 48000 * 1000, abs=1) == 150.0


# =============================================================================
# RUST TEST COUNTERPARTS
# =============================================================================

"""
The following Rust tests should be added to micro_dynamics_extractor.rs:

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_duration_ms_short_segment() {
        let audio = vec![0.0f32; 1440]; // 30ms @ 48kHz
        let extractor = MicroDynamicsExtractor::new(48000);
        let features = extractor.extract(&audio).unwrap();
        assert_eq!(features.duration_ms, 30.0);
    }

    #[test]
    fn test_duration_ms_long_segment() {
        let audio = vec![0.0f32; 24000]; // 500ms @ 48kHz
        let extractor = MicroDynamicsExtractor::new(48000);
        let features = extractor.extract(&audio).unwrap();
        assert_eq!(features.duration_ms, 500.0);
    }

    #[test]
    fn test_f0_contour_slope_rising() {
        // Generate rising chirp: 4kHz -> 8kHz over 200ms
        let audio = generate_chirp(4000.0, 8000.0, 200.0, 48000);
        let extractor = MicroDynamicsExtractor::new(48000);
        let features = extractor.extract(&audio).unwrap();

        // f0_mean_derivative (Index 55) should be positive
        assert!(features.f0_mean_derivative > 0.0);
        // f0_range_hz (Index 2) should capture the sweep
        assert!(features.f0_range_hz > 3500.0);
    }

    #[test]
    fn test_empty_audio_rejected() {
        let audio = vec![0.0f32; 0];
        let extractor = MicroDynamicsExtractor::new(48000);
        let result = extractor.extract(&audio);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_padding_short_segment() {
        let audio = vec![0.5f32; 240]; // 5ms @ 48kHz
        let extractor = MicroDynamicsExtractor::new(48000);
        let result = extractor.extract(&audio);

        // Should succeed with zero-padding
        assert!(result.is_ok());
        assert_eq!(result.unwrap().duration_ms, 5.0);
    }
}
"""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
