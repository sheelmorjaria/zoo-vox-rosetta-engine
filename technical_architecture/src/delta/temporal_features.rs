//! Temporal delta features for various acoustic features
//!
//! This module computes delta features (Δ and ΔΔ) for temporal sequences
//! such as F0 contours, amplitude envelopes, and spectral flux.

use crate::delta::DeltaWidth;

/// Type of temporal feature
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalFeatureType {
    /// Fundamental frequency (F0)
    F0,
    /// Amplitude envelope
    Amplitude,
    /// Spectral flux
    SpectralFlux,
}

/// Temporal delta computer for 1D feature sequences
///
/// Computes first and second derivatives of temporal features.
///
/// # Example
///
/// ```rust
/// use technical_architecture::delta::{TemporalDeltaComputer, DeltaWidth};
///
/// let computer = TemporalDeltaComputer::new(DeltaWidth::N2);
/// let f0_contour = vec![100.0, 105.0, 110.0, 115.0];
///
/// let (delta, delta_delta) = computer.compute(&f0_contour).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TemporalDeltaComputer {
    width: DeltaWidth,
}

impl TemporalDeltaComputer {
    /// Create a new temporal delta computer
    pub fn new(width: DeltaWidth) -> Self {
        Self { width }
    }

    /// Compute delta and delta-delta for temporal sequence
    ///
    /// # Arguments
    ///
    /// * `sequence` - Temporal sequence of feature values
    ///
    /// # Returns
    ///
    /// * `(delta, delta_delta)` - Both same length as input
    pub fn compute(&self, sequence: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
        if sequence.is_empty() {
            return Ok((vec![], vec![]));
        }

        let len = sequence.len();

        // Validate finite values
        for &val in sequence {
            if !val.is_finite() {
                return Err(format!("Sequence contains non-finite value: {}", val));
            }
        }

        // Handle single element
        if len == 1 {
            return Ok((vec![0.0], vec![0.0]));
        }

        // Compute delta
        let delta = self.compute_delta(sequence);

        // Compute delta-delta
        let delta_delta = self.compute_delta_delta(&delta);

        Ok((delta, delta_delta))
    }

    /// Compute delta (first derivative)
    fn compute_delta(&self, seq: &[f32]) -> Vec<f32> {
        let len = seq.len();
        let mut delta = vec![0.0; len];

        match self.width {
            DeltaWidth::N1 => {
                // Δ[t] = x[t+1] - x[t]
                for t in 0..len.saturating_sub(1) {
                    delta[t] = seq[t + 1] - seq[t];
                }
                // Last frame: copy previous
                if len > 1 {
                    delta[len - 1] = delta[len - 2];
                }
            }
            DeltaWidth::N2 => {
                // Δ[t] = (x[t+1] - x[t-1]) / 2
                for t in 1..len.saturating_sub(1) {
                    delta[t] = (seq[t + 1] - seq[t - 1]) / 2.0;
                }
                // Edge frames
                if len > 1 {
                    delta[0] = seq[1] - seq[0];
                    delta[len - 1] = seq[len - 1] - seq[len - 2];
                }
            }
        }

        delta
    }

    /// Compute delta-delta (second derivative)
    fn compute_delta_delta(&self, delta: &[f32]) -> Vec<f32> {
        let len = delta.len();
        let mut delta_delta = vec![0.0; len];

        match self.width {
            DeltaWidth::N1 => {
                for t in 0..len.saturating_sub(1) {
                    delta_delta[t] = delta[t + 1] - delta[t];
                }
                if len > 1 {
                    delta_delta[len - 1] = delta_delta[len - 2];
                }
            }
            DeltaWidth::N2 => {
                for t in 1..len.saturating_sub(1) {
                    delta_delta[t] = (delta[t + 1] - delta[t - 1]) / 2.0;
                }
                if len > 1 {
                    delta_delta[0] = delta[1] - delta[0];
                    delta_delta[len - 1] = delta[len - 1] - delta[len - 2];
                }
            }
        }

        delta_delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // F0 Delta Tests (6 tests)
    // =========================================================================

    #[test]
    fn test_f0_constant_pitch() {
        let computer = TemporalDeltaComputer::new(DeltaWidth::N2);

        let f0_contour = vec![1000.0; 10];

        let (delta, delta_delta) = computer.compute(&f0_contour).unwrap();

        // All deltas should be zero
        for d in &delta {
            assert_eq!(*d, 0.0);
        }
        for dd in &delta_delta {
            assert_eq!(*dd, 0.0);
        }
    }

    #[test]
    fn test_f0_linear_ramp() {
        let computer = TemporalDeltaComputer::new(DeltaWidth::N2);

        // Linear pitch ramp
        let f0_contour: Vec<f32> = (0..10).map(|i| 1000.0 + i as f32 * 10.0).collect();

        let (delta, delta_delta) = computer.compute(&f0_contour).unwrap();

        // Delta should be ~10 Hz
        for i in 1..delta.len() - 1 {
            assert!((delta[i] - 10.0).abs() < 0.1, "Delta should be ~10 Hz");
        }

        // Delta-delta should be ~0
        for dd in &delta_delta {
            assert!(dd.abs() < 0.1, "Delta-delta should be ~0");
        }
    }

    #[test]
    fn test_f0_vibrato() {
        let computer = TemporalDeltaComputer::new(DeltaWidth::N2);

        // Simulated vibrato: sine wave modulation
        let f0_contour: Vec<f32> = (0..20)
            .map(|i| {
                1000.0 + 50.0 * (2.0 * std::f32::consts::PI * i as f32 / 10.0).sin()
            })
            .collect();

        let (delta, delta_delta) = computer.compute(&f0_contour).unwrap();

        // Delta should oscillate (vibrato rate)
        let delta_variance: f32 = delta.iter().map(|d| d * d).sum::<f32>() / delta.len() as f32;
        assert!(delta_variance > 0.0, "Delta should capture vibrato modulation");
    }

    #[test]
    fn test_f0_pitch_transition() {
        let computer = TemporalDeltaComputer::new(DeltaWidth::N2);

        // Pitch transition: 1kHz → 2kHz
        let f0_contour: Vec<f32> = (0..10)
            .map(|i| {
                let t = i as f32 / 10.0;
                1000.0 + t * 1000.0
            })
            .collect();

        let (delta, _) = computer.compute(&f0_contour).unwrap();

        // Delta should capture the transition
        let max_delta = delta.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(max_delta >= 100.0, "Delta should detect pitch transition");
    }

    #[test]
    fn test_f0_glissando() {
        let computer = TemporalDeltaComputer::new(DeltaWidth::N2);

        // Glissando: smooth frequency sweep
        let f0_contour: Vec<f32> = (0..20)
            .map(|i| 1000.0 + (i as f32).powi(2) * 5.0)
            .collect();

        let (delta, delta_delta) = computer.compute(&f0_contour).unwrap();

        // Should detect acceleration
        let dd_sum: f32 = delta_delta.iter().sum();
        assert!(dd_sum > 0.0, "Delta-delta should detect glissando acceleration");
    }

    #[test]
    fn test_f0_single_value() {
        let computer = TemporalDeltaComputer::new(DeltaWidth::N2);

        let f0_contour = vec![1000.0];

        let (delta, delta_delta) = computer.compute(&f0_contour).unwrap();

        assert_eq!(delta.len(), 1);
        assert_eq!(delta_delta.len(), 1);
        assert_eq!(delta[0], 0.0);
        assert_eq!(delta_delta[0], 0.0);
    }

    // =========================================================================
    // Amplitude Delta Tests (6 tests)
    // =========================================================================

    #[test]
    fn test_amplitude_constant() {
        let computer = TemporalDeltaComputer::new(DeltaWidth::N2);

        let envelope = vec![0.5; 10];

        let (delta, delta_delta) = computer.compute(&envelope).unwrap();

        // All deltas should be zero
        for d in &delta {
            assert_eq!(*d, 0.0);
        }
        for dd in &delta_delta {
            assert_eq!(*dd, 0.0);
        }
    }

    #[test]
    fn test_amplitude_attack() {
        let computer = TemporalDeltaComputer::new(DeltaWidth::N2);

        // Simulated attack: exponential rise
        let envelope: Vec<f32> = (0..10)
            .map(|i| 1.0 - (-0.5 * i as f32).exp())
            .collect();

        let (delta, delta_delta) = computer.compute(&envelope).unwrap();

        // Delta should be positive during attack
        let positive_delta: usize = delta.iter().filter(|&&d| d > 0.0).count();
        assert!(positive_delta > delta.len() / 2, "Delta should detect attack");
    }

    #[test]
    fn test_amplitude_decay() {
        let computer = TemporalDeltaComputer::new(DeltaWidth::N2);

        // Simulated decay: exponential decay
        let envelope: Vec<f32> = (0..10).map(|i| (-0.3 * i as f32).exp()).collect();

        let (delta, delta_delta) = computer.compute(&envelope).unwrap();

        // Delta should be negative during decay
        let negative_delta: usize = delta.iter().filter(|&&d| d < 0.0).count();
        assert!(negative_delta > delta.len() / 2, "Delta should detect decay");
    }

    #[test]
    fn test_amplitude_adsr() {
        let computer = TemporalDeltaComputer::new(DeltaWidth::N2);

        // ADSR envelope
        let mut envelope = Vec::new();

        // Attack
        for i in 0..3 {
            envelope.push(i as f32 / 3.0);
        }
        // Decay
        for i in 0..2 {
            envelope.push(1.0 - 0.2 * (i as f32 / 2.0));
        }
        // Sustain
        for _ in 0..3 {
            envelope.push(0.8);
        }
        // Release
        for i in 0..2 {
            envelope.push(0.8 * (1.0 - i as f32 / 2.0));
        }

        let (delta, delta_delta) = computer.compute(&envelope).unwrap();

        // Should detect all phases
        let max_delta = delta.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(max_delta > 0.1, "Delta should detect ADSR changes");
    }

    #[test]
    fn test_amplitude_tremolo() {
        let computer = TemporalDeltaComputer::new(DeltaWidth::N2);

        // Tremolo: amplitude modulation
        let envelope: Vec<f32> = (0..20)
            .map(|i| 0.5 + 0.3 * (2.0 * std::f32::consts::PI * i as f32 / 8.0).sin())
            .collect();

        let (delta, delta_delta) = computer.compute(&envelope).unwrap();

        // Delta should oscillate
        let delta_std: f32 = (delta.iter().map(|d| d * d).sum::<f32>() / delta.len() as f32).sqrt();
        assert!(delta_std > 0.0, "Delta should capture tremolo");
    }

    #[test]
    fn test_amplitude_silence_to_sound() {
        let computer = TemporalDeltaComputer::new(DeltaWidth::N2);

        let envelope = vec![0.0, 0.0, 0.0, 0.5, 0.8, 1.0];

        let (delta, _) = computer.compute(&envelope).unwrap();

        // Should detect onset
        let max_delta = delta.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(max_delta > 0.3, "Delta should detect onset");
    }

    // =========================================================================
    // Spectral Flux Delta Tests (6 tests)
    // =========================================================================

    #[test]
    fn test_spectral_flux_constant() {
        let computer = TemporalDeltaComputer::new(DeltaWidth::N2);

        let flux = vec![10.0; 10];

        let (delta, delta_delta) = computer.compute(&flux).unwrap();

        for d in &delta {
            assert_eq!(*d, 0.0);
        }
        for dd in &delta_delta {
            assert_eq!(*dd, 0.0);
        }
    }

    #[test]
    fn test_spectral_flux_onset() {
        let computer = TemporalDeltaComputer::new(DeltaWidth::N2);

        // Onset: flux spike
        let flux = vec![1.0, 1.5, 50.0, 30.0, 10.0, 5.0];

        let (delta, delta_delta) = computer.compute(&flux).unwrap();

        // Should detect onset
        let max_delta_idx = delta
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert!(max_delta_idx > 0, "Delta should detect onset location");
    }

    #[test]
    fn test_spectral_flux_changing_spectrum() {
        let computer = TemporalDeltaComputer::new(DeltaWidth::N2);

        // Gradually changing spectral flux
        let flux: Vec<f32> = (0..10).map(|i| 10.0 + i as f32 * 2.0).collect();

        let (delta, delta_delta) = computer.compute(&flux).unwrap();

        // Delta should be positive
        for d in &delta {
            assert!(*d >= 0.0, "Delta should be positive for increasing flux");
        }
    }

    #[test]
    fn test_spectral_flux_transient() {
        let computer = TemporalDeltaComputer::new(DeltaWidth::N2);

        // Transient event
        let flux = vec![5.0, 5.0, 5.0, 100.0, 20.0, 5.0, 5.0, 5.0];

        let (delta, delta_delta) = computer.compute(&flux).unwrap();

        // Delta should detect transient
        let max_delta = delta.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(max_delta > 40.0, "Delta should detect transient");
    }

    #[test]
    fn test_spectral_flux_rhythmic() {
        let computer = TemporalDeltaComputer::new(DeltaWidth::N2);

        // Rhythmic pattern
        let flux = vec![5.0, 50.0, 5.0, 50.0, 5.0, 50.0];

        let (delta, delta_delta) = computer.compute(&flux).unwrap();

        // Delta should capture rhythm
        let delta_changes: usize = delta
            .windows(2)
            .filter(|w| (w[1] - w[0]).abs() > 10.0)
            .count();
        assert!(delta_changes >= 2, "Delta should capture rhythmic changes");
    }

    #[test]
    fn test_spectral_flux_noise_robustness() {
        let computer = TemporalDeltaComputer::new(DeltaWidth::N2);

        // Noisy flux values
        let flux: Vec<f32> = (0..10)
            .map(|i| 10.0 + (i as f32 * 2.0) + 0.5 * (i as f32 % 3.0) as f32)
            .collect();

        let (delta, _) = computer.compute(&flux).unwrap();

        // Should still detect trend
        let delta_mean: f32 = delta.iter().sum::<f32>() / delta.len() as f32;
        assert!((delta_mean - 2.0).abs() < 1.0, "Delta should be robust to noise");
    }
}
