#!/usr/bin/env python3
"""
advanced_harmonic_extensions.py

Copyright (c) 2025 Sheel Morjaria
License: CC BY-ND 4.0 International
Author: Sheel Morjaria (sheelmorjaria@gmail.com)
Last Updated: December 27, 2025
"""

"""
Advanced Harmonic Analysis Extensions

Implements critical missing components identified in framework review:
1. Adaptive windowing based on F0
2. Harmonic phase coupling analysis (bispectrum)
3. Harmonic deviation patterns (formant pulling)
4. Cross-harmonic modulation
5. Formant-harmonic interaction modeling
6. Probabilistic harmonic fields (KDE)
7. Dynamic harmonic trajectories

These extensions address the deeper physics of harmonic encoding and reveal
the mechanisms by which semantic information is carried in vocal tract resonances.

Author: Animal Communication Analysis Framework
Date: October 2025
"""

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import butter, hilbert, sosfilt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KernelDensity

warnings.filterwarnings("ignore")


class AdaptiveHarmonicAnalyzer:
    """
    Adaptive windowing for optimal harmonic resolution.

    Key insight: Window length should adapt to local F0 to capture
    multiple periods while maintaining temporal resolution.
    """

    def __init__(self, sr: int = 250000):
        self.sr = sr
        self.min_periods = 3  # Minimum periods per window
        self.max_periods = 6  # Maximum periods per window

    def adaptive_window_harmonic_analysis(
        self, audio: np.ndarray, f0_track: np.ndarray, hop_length: int = 128
    ) -> List[Dict]:
        """
        Extract harmonics with adaptive windowing based on local F0.

        Args:
            audio: Audio signal
            f0_track: F0 track (Hz)
            hop_length: Hop between analysis frames

        Returns:
            List of harmonic features per frame
        """
        segments = []

        for i, f0 in enumerate(f0_track):
            if f0 <= 0:
                continue

            # Adaptive window: 4 periods of current F0
            periods_to_capture = 4
            window_length = int(periods_to_capture * self.sr / f0)

            # Ensure window doesn't exceed signal length
            start_idx = i * hop_length
            end_idx = min(start_idx + window_length, len(audio))

            if end_idx - start_idx < self.sr / f0:  # Less than 1 period
                continue

            segment = audio[start_idx:end_idx]

            # Window the segment
            windowed = segment * np.hanning(len(segment))

            # Extract harmonics with period-matched resolution
            harmonics = self._extract_harmonics_adaptive(windowed, f0)

            harmonics["frame"] = i
            harmonics["f0"] = f0
            harmonics["window_length"] = window_length

            segments.append(harmonics)

        return segments

    def _extract_harmonics_adaptive(self, segment: np.ndarray, f0: float) -> Dict:
        """Extract harmonic features from adaptively windowed segment."""

        # FFT
        spectrum = np.abs(fft(segment))
        freqs = fftfreq(len(segment), 1 / self.sr)

        # Keep positive frequencies
        pos_mask = freqs > 0
        spectrum = spectrum[pos_mask]
        freqs = freqs[pos_mask]

        # Extract harmonic amplitudes (H1-H10)
        harmonic_amps = []
        harmonic_freqs = []

        for h in range(1, 11):
            target_freq = f0 * h

            # Search window around expected harmonic
            window = 0.1 * f0  # ±10% of F0
            mask = (freqs >= target_freq - window) & (freqs <= target_freq + window)

            if np.any(mask):
                # Find peak in window
                peak_idx = np.argmax(spectrum[mask])
                peak_freq = freqs[mask][peak_idx]
                peak_amp = spectrum[mask][peak_idx]

                harmonic_amps.append(peak_amp)
                harmonic_freqs.append(peak_freq)
            else:
                harmonic_amps.append(0)
                harmonic_freqs.append(target_freq)

        return {f"h{i}_amp": amp for i, amp in enumerate(harmonic_amps, 1)}


class HarmonicPhaseAnalyzer:
    """
    Analyze phase relationships between harmonics using bispectral analysis.

    Phase coupling reveals nonlinear production mechanisms and source-filter
    interactions that pure amplitude analysis misses.
    """

    def __init__(self, sr: int = 250000):
        self.sr = sr

    def extract_harmonic_phase_coupling(self, audio: np.ndarray, f0: float) -> Dict:
        """
        Calculate phase coupling between harmonics using bispectrum.

        The bispectrum B(f1, f2) measures phase coupling at f1 + f2.
        High bicoherence indicates nonlinear coupling (e.g., H3 = H1 + H2).
        """
        features = {}

        # Complex FFT (preserves phase)
        spectrum_complex = fft(audio)
        freqs = fftfreq(len(audio), 1 / self.sr)

        # Positive frequencies only
        pos_mask = freqs > 0
        spectrum_complex = spectrum_complex[pos_mask]
        freqs = freqs[pos_mask]

        # Extract complex harmonic components
        harmonics_complex = []

        for h in range(1, 11):
            target_freq = f0 * h
            idx = np.argmin(np.abs(freqs - target_freq))
            harmonics_complex.append(spectrum_complex[idx])

        # Calculate bicoherence for key harmonic pairs
        # H3 = H1 + H2 coupling
        if len(harmonics_complex) >= 3:
            H1, H2, H3 = harmonics_complex[0], harmonics_complex[1], harmonics_complex[2]

            # Bispectrum: B(f1, f2) = E[X(f1) * X(f2) * X*(f1+f2)]
            bispectrum = H1 * H2 * np.conj(H3)

            # Bicoherence: normalized bispectrum
            power_H1 = np.abs(H1) ** 2
            power_H2 = np.abs(H2) ** 2
            power_H3 = np.abs(H3) ** 2

            bicoherence = np.abs(bispectrum) ** 2 / (power_H1 * power_H2 * power_H3 + 1e-10)

            features["bicoherence_h1_h2_h3"] = bicoherence

        # Phase coherence between adjacent harmonics
        phase_coherences = []

        for i in range(len(harmonics_complex) - 1):
            H_n = harmonics_complex[i]
            H_n1 = harmonics_complex[i + 1]

            # Phase difference
            phase_diff = np.angle(H_n1) - np.angle(H_n)

            # Wrap to [-π, π]
            phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

            phase_coherences.append(np.abs(np.exp(1j * phase_diff)))

        features["mean_phase_coherence"] = np.mean(phase_coherences) if phase_coherences else 0
        features["std_phase_coherence"] = np.std(phase_coherences) if phase_coherences else 0

        # Group delay (phase derivative)
        if len(harmonics_complex) >= 2:
            phases = [np.angle(h) for h in harmonics_complex]
            group_delay = np.diff(phases)

            features["group_delay_mean"] = np.mean(group_delay)
            features["group_delay_std"] = np.std(group_delay)

        return features


class HarmonicDeviationAnalyzer:
    """
    Measure deviations of harmonics from perfect integer ratios.

    Systematic deviations indicate formant pulling; random deviations
    indicate instability or noise. This reveals vocal tract coupling.
    """

    def __init__(self, sr: int = 250000):
        self.sr = sr

    def measure_harmonic_deviations(self, audio: np.ndarray, f0: float) -> Dict:
        """
        Measure how much harmonics deviate from expected frequencies.

        Returns deviations in cents (1200 cents = 1 octave).
        """
        features = {}

        # Get spectrum
        spectrum = np.abs(fft(audio))
        freqs = fftfreq(len(audio), 1 / self.sr)

        pos_mask = freqs > 0
        spectrum = spectrum[pos_mask]
        freqs = freqs[pos_mask]

        # Expected vs actual harmonic frequencies
        deviations_cents = []

        for h in range(1, 11):
            expected_freq = f0 * h

            # Search window
            window = 0.15 * f0
            mask = (freqs >= expected_freq - window) & (freqs <= expected_freq + window)

            if np.any(mask):
                # Find actual peak
                peak_idx = np.argmax(spectrum[mask])
                actual_freq = freqs[mask][peak_idx]

                # Deviation in cents
                if actual_freq > 0 and expected_freq > 0:
                    deviation_cents = 1200 * np.log2(actual_freq / expected_freq)
                    deviations_cents.append(deviation_cents)

        if deviations_cents:
            features["harmonic_deviation_mean_cents"] = np.mean(deviations_cents)
            features["harmonic_deviation_std_cents"] = np.std(deviations_cents)
            features["harmonic_deviation_max_cents"] = np.max(np.abs(deviations_cents))

            # Systematic vs random deviations
            # Systematic: deviations correlated with harmonic number
            if len(deviations_cents) > 3:
                harmonic_numbers = np.arange(1, len(deviations_cents) + 1)
                correlation = np.corrcoef(harmonic_numbers, deviations_cents)[0, 1]
                features["harmonic_deviation_systematicity"] = correlation
        else:
            features.update(
                {
                    "harmonic_deviation_mean_cents": 0,
                    "harmonic_deviation_std_cents": 0,
                    "harmonic_deviation_max_cents": 0,
                    "harmonic_deviation_systematicity": 0,
                }
            )

        return features


class CrossHarmonicModulationAnalyzer:
    """
    Analyze interactions between harmonics through vocal tract.

    When harmonics pass through formants, they modulate each other,
    creating sidebands that encode tract configuration.
    """

    def __init__(self, sr: int = 250000):
        self.sr = sr

    def extract_cross_harmonic_modulation(self, audio: np.ndarray, f0: float) -> Dict:
        """
        Measure cross-modulation between harmonic envelopes.
        """
        features = {}

        # Extract harmonic envelopes using Hilbert transform
        envelopes = {}

        for h in range(1, 10):
            harmonic_freq = f0 * h

            # Bandpass filter around harmonic
            bandwidth = 0.1 * f0
            low = max(harmonic_freq - bandwidth, 1)
            high = min(harmonic_freq + bandwidth, self.sr / 2 - 1)

            try:
                sos = butter(4, [low, high], btype="band", fs=self.sr, output="sos")
                filtered = sosfilt(sos, audio)

                # Hilbert envelope
                analytic_signal = hilbert(filtered)
                envelope = np.abs(analytic_signal)

                envelopes[h] = envelope
            except Exception:
                continue

        if len(envelopes) < 2:
            return {"cross_modulation_mean": 0, "cross_modulation_max": 0}

        # Cross-correlation between envelopes
        modulation_matrix = []

        for h1 in sorted(envelopes.keys()):
            for h2 in sorted(envelopes.keys()):
                if h2 <= h1:
                    continue

                env1 = envelopes[h1]
                env2 = envelopes[h2]

                # Ensure same length
                min_len = min(len(env1), len(env2))
                env1 = env1[:min_len]
                env2 = env2[:min_len]

                if np.std(env1) > 0 and np.std(env2) > 0:
                    correlation = np.corrcoef(env1, env2)[0, 1]
                    modulation_matrix.append(np.abs(correlation))

        if modulation_matrix:
            features["cross_modulation_mean"] = np.mean(modulation_matrix)
            features["cross_modulation_max"] = np.max(modulation_matrix)
            features["cross_modulation_std"] = np.std(modulation_matrix)
        else:
            features.update(
                {"cross_modulation_mean": 0, "cross_modulation_max": 0, "cross_modulation_std": 0}
            )

        return features


class FormantHarmonicInteractionModeler:
    """
    Model how formants shape harmonic amplitudes.

    This IS the semantic encoding mechanism: formants (vocal tract shape)
    selectively amplify/attenuate harmonics, creating the spectral pattern
    that carries meaning.
    """

    def __init__(self, sr: int = 250000):
        self.sr = sr

    def model_formant_harmonic_interaction(
        self, audio: np.ndarray, f0: float, formants: List[float], bandwidths: List[float]
    ) -> Dict:
        """
        Predict harmonic amplitudes from formants and compare to actual.

        Residual reveals what formants don't explain (source characteristics).
        """
        features = {}

        # Get actual harmonic amplitudes
        spectrum = np.abs(fft(audio))
        freqs = fftfreq(len(audio), 1 / self.sr)

        pos_mask = freqs > 0
        spectrum = spectrum[pos_mask]
        freqs = freqs[pos_mask]

        actual_harmonics = []
        for h in range(1, 11):
            target_freq = f0 * h
            idx = np.argmin(np.abs(freqs - target_freq))
            actual_harmonics.append(spectrum[idx])

        actual_harmonics = np.array(actual_harmonics)

        # Predict harmonic amplitudes from formant transfer function
        predicted_harmonics = []

        for h in range(1, 11):
            harmonic_freq = f0 * h

            # Vocal tract transfer function (sum of resonances)
            transfer = self._vocal_tract_transfer_function(harmonic_freq, formants, bandwidths)

            # Source spectrum (assume -12 dB/octave rolloff)
            source_amp = 1.0 / (h**1.0)  # Simplified source model

            predicted_amp = source_amp * transfer
            predicted_harmonics.append(predicted_amp)

        predicted_harmonics = np.array(predicted_harmonics)

        # Normalize for comparison
        if np.max(actual_harmonics) > 0:
            actual_harmonics = actual_harmonics / np.max(actual_harmonics)
        if np.max(predicted_harmonics) > 0:
            predicted_harmonics = predicted_harmonics / np.max(predicted_harmonics)

        # Calculate residual
        residual = actual_harmonics - predicted_harmonics

        features["formant_prediction_rmse"] = np.sqrt(np.mean(residual**2))
        features["formant_prediction_correlation"] = (
            np.corrcoef(actual_harmonics, predicted_harmonics)[0, 1]
            if len(actual_harmonics) > 1
            else 0
        )
        features["residual_energy"] = np.sum(residual**2)
        features["residual_mean"] = np.mean(residual)
        features["residual_std"] = np.std(residual)

        # Source quality from residual pattern
        # Positive residual = excess energy (harsh voice)
        # Negative residual = deficit (breathy voice)
        features["voice_harshness"] = np.mean(residual[residual > 0]) if np.any(residual > 0) else 0
        features["voice_breathiness"] = (
            -np.mean(residual[residual < 0]) if np.any(residual < 0) else 0
        )

        return features

    def _vocal_tract_transfer_function(
        self, freq: float, formants: List[float], bandwidths: List[float]
    ) -> float:
        """
        Calculate vocal tract transfer function at given frequency.

        Each formant is a resonator (pole in transfer function).
        """
        transfer = 0.0

        for f_formant, bw in zip(formants, bandwidths):
            # Resonator response (simplified)
            freq_diff = freq - f_formant

            # Gaussian approximation of resonance peak
            response = np.exp(-(freq_diff**2) / (2 * bw**2))

            transfer += response

        return transfer


class ProbabilisticHarmonicFieldModeler:
    """
    Model contexts as probability fields in harmonic space.

    Not discrete clusters, but overlapping probability densities.
    Classification by maximum likelihood across fields.
    """

    def __init__(self, bandwidth: float = 0.5):
        self.bandwidth = bandwidth
        self.harmonic_fields = {}

    def fit_harmonic_fields(
        self, df: pd.DataFrame, harmonic_features: List[str], context_col: str = "context"
    ) -> Dict:
        """
        Fit kernel density estimates for each context.
        """
        contexts = df[context_col].unique()

        for context in contexts:
            context_data = df[df[context_col] == context][harmonic_features].dropna()

            if len(context_data) < 10:
                continue

            # Fit KDE
            kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth)
            kde.fit(context_data.values)

            self.harmonic_fields[context] = {
                "kde": kde,
                "n_samples": len(context_data),
                "mean": context_data.mean().values,
                "cov": np.cov(context_data.T),
            }

        return self.harmonic_fields

    def classify_by_likelihood(self, harmonic_vector: np.ndarray) -> Tuple[int, Dict]:
        """
        Classify by maximum likelihood across probability fields.
        """
        likelihoods = {}

        for context, field in self.harmonic_fields.items():
            log_likelihood = field["kde"].score(harmonic_vector.reshape(1, -1))
            likelihoods[context] = np.exp(log_likelihood)

        best_context = max(likelihoods, key=likelihoods.get)

        return best_context, likelihoods

    def calculate_field_entropy(self, context: int) -> float:
        """
        Calculate entropy of probability field (uncertainty measure).
        """
        if context not in self.harmonic_fields:
            return 0

        # Estimate entropy from covariance
        cov = self.harmonic_fields[context]["cov"]

        # Differential entropy for multivariate Gaussian
        k = cov.shape[0]  # Dimensionality
        det_cov = np.linalg.det(cov + np.eye(k) * 1e-6)  # Regularize

        entropy_val = 0.5 * np.log((2 * np.pi * np.e) ** k * det_cov)

        return entropy_val


class DynamicHarmonicTrajectoryAnalyzer:
    """
    Model harmonic evolution through time as dynamical system.

    Trajectories encode gestures and transitions.
    Eigenanalysis reveals stable/unstable modes.
    """

    def __init__(self, sr: int = 250000, hop_length: int = 256):
        self.sr = sr
        self.hop_length = hop_length

    def model_harmonic_trajectories(
        self, audio: np.ndarray, f0_track: np.ndarray, window_size: int = 2048
    ) -> Dict:
        """
        Extract and model harmonic trajectories.
        """
        # Extract harmonic time series
        trajectory = []

        for i in range(0, len(audio) - window_size, self.hop_length):
            window = audio[i : i + window_size]

            # Get F0 for this window
            f0_idx = i // self.hop_length
            if f0_idx >= len(f0_track):
                break

            f0 = f0_track[f0_idx]

            if f0 <= 0:
                continue

            # Extract harmonics
            harmonics = self._extract_harmonic_vector(window, f0)
            trajectory.append(harmonics)

        if len(trajectory) < 10:
            return {"trajectory_length": 0}

        trajectory = np.array(trajectory)  # Time × Harmonics

        # Fit linear dynamical system: h(t+1) = A * h(t) + b
        X = trajectory[:-1]
        y = trajectory[1:]

        model = LinearRegression()
        model.fit(X, y)

        A = model.coef_  # Transition matrix

        # Eigenanalysis
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Classify modes by stability
        stable_modes = np.abs(eigenvalues) < 1
        unstable_modes = np.abs(eigenvalues) >= 1

        features = {
            "trajectory_length": len(trajectory),
            "n_stable_modes": np.sum(stable_modes),
            "n_unstable_modes": np.sum(unstable_modes),
            "max_eigenvalue": np.max(np.abs(eigenvalues)),
            "spectral_radius": np.max(np.abs(eigenvalues)),  # System stability
            "trajectory_complexity": np.linalg.matrix_rank(A),  # Effective dimensionality
        }

        # Trajectory statistics
        features["trajectory_mean_velocity"] = np.mean(
            np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        )
        features["trajectory_smoothness"] = np.mean(
            np.linalg.norm(np.diff(trajectory, n=2, axis=0), axis=1)
        )

        return features

    def _extract_harmonic_vector(self, audio: np.ndarray, f0: float) -> np.ndarray:
        """Extract harmonic amplitude vector."""
        spectrum = np.abs(fft(audio))
        freqs = fftfreq(len(audio), 1 / self.sr)

        pos_mask = freqs > 0
        spectrum = spectrum[pos_mask]
        freqs = freqs[pos_mask]

        harmonics = []
        for h in range(1, 11):
            target_freq = f0 * h
            idx = np.argmin(np.abs(freqs - target_freq))
            harmonics.append(spectrum[idx])

        return np.array(harmonics)


def main():
    """Demonstrate advanced harmonic extensions."""
    print("=" * 80)
    print("ADVANCED HARMONIC ANALYSIS EXTENSIONS")
    print("=" * 80)

    print("\nThis module implements 7 critical extensions:")
    print("  1. Adaptive windowing (F0-dependent)")
    print("  2. Phase coupling analysis (bispectrum)")
    print("  3. Harmonic deviation patterns (formant pulling)")
    print("  4. Cross-harmonic modulation")
    print("  5. Formant-harmonic interaction modeling")
    print("  6. Probabilistic harmonic fields (KDE)")
    print("  7. Dynamic harmonic trajectories")

    print("\nThese extensions reveal the PHYSICS of semantic encoding:")
    print("  - How formants shape harmonics → meaning")
    print("  - Phase relationships → production mechanisms")
    print("  - Trajectories → gestures and transitions")
    print("  - Probability fields → context boundaries")

    print("\n" + "=" * 80)
    print("Framework ready for deployment on vocalization data")
    print("=" * 80)


if __name__ == "__main__":
    main()
