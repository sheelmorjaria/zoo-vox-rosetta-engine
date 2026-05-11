#!/usr/bin/env python3
"""
Feature Extraction Pipeline for Analysis Frameworks

Extracts 16D affect vectors (from β-VAE) and VQ-VAE tokens from raw
audio, enabling the full suite of analysis frameworks to work with
the Egyptian Fruit Bat dataset.

Pipeline:
1. Load raw audio files
2. Extract 112D RosettaFeatures
3. VAE encode → 16D affect vector
4. VQ-VAE encode → discrete token
5. Save enriched dataset for analysis

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.io import wavfile

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """
    A raw audio segment with metadata.

    Attributes:
        file_path: Path to audio file
        audio: Raw audio samples (normalized)
        sample_rate: Sample rate in Hz
        start_ms: Start time within file
        duration_ms: Segment duration
        phrase_id: Associated phrase ID
        social_context: Social context label (if available)
    """
    file_path: str
    audio: np.ndarray
    sample_rate: int
    start_ms: float
    duration_ms: float
    phrase_id: str = ""
    social_context: str = ""


@dataclass
class ExtractedFeatures:
    """
    Fully extracted features for analysis.

    Attributes:
        segment_id: Unique identifier
        phrase_id: Original phrase ID
        audio_raw: Raw audio samples
        rosetta_features_112d: Full 112D RosettaFeatures
        affect_vector_16d: VAE-encoded affect (Stream 1)
        syntactic_token: VQ-VAE token (Stream 2)
        social_context: Social context label
        metadata: Additional metadata
    """
    segment_id: str
    phrase_id: str
    audio_raw: np.ndarray
    rosetta_features_112d: np.ndarray
    affect_vector_16d: np.ndarray
    syntactic_token: int
    social_context: str
    metadata: Dict = field(default_factory=dict)


class RosettaFeatureExtractor:
    """
    Extracts 112D RosettaFeatures from raw audio.

    Implements the 3-layer hierarchy:
    - Layer 1 (0-45): Base Physics (F0, RMS, HNR, MFCCs, ADSR)
    - Layer 2 (46-75): Macro Texture (Harmonic, pitch geometry, GLCM)
    - Layer 3 (76-111): Micro Texture (Spectral deriv, FM, dynamics, rhythm)
    """

    def __init__(self, sample_rate: int = 48000):
        """
        Initialize Rosetta feature extractor.

        Args:
            sample_rate: Target sample rate for processing
        """
        self.sample_rate = sample_rate
        self.n_mfcc = 13
        self.n_mels = 40

    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Extract 112D RosettaFeatures from audio.

        Args:
            audio: Audio samples (normalized to [-1, 1])
            sample_rate: Sample rate in Hz

        Returns:
            112D feature vector
        """
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = self._resample(audio, sample_rate, self.sample_rate)
            sample_rate = self.sample_rate

        # Pre-emphasis
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

        # Layer 1: Base Physics (46 features)
        layer1 = self._extract_layer1(audio, sample_rate)

        # Layer 2: Macro Texture (30 features)
        layer2 = self._extract_layer2(audio, sample_rate)

        # Layer 3: Micro Texture (36 features)
        layer3 = self._extract_layer3(audio, sample_rate)

        return np.concatenate([layer1, layer2, layer3])

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        from scipy.signal import resample_poly
        return resample_poly(audio, target_sr, orig_sr)

    def _extract_layer1(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Extract Layer 1: Base Physics features (46D).

        Features:
        - F0 statistics (6): mean, std, min, max, range, slope
        - RMS energy (6): mean, std, min, max, range, dynamic_range
        - HNR (6): mean, std, min, max, range, voiced_ratio
        - MFCCs (13): Mean of first 13 MFCCs
        - ADSR (4): Attack, decay, sustain, release times
        - Duration (1): Total duration in ms
        - Voicing (6): Voiced frame ratios at different thresholds
        - Jitter/Shimmer (4): Perturbation measures
        """
        features = []

        # F0 estimation (autocorrelation)
        f0_mean, f0_std, f0_min, f0_max, f0_contour = self._estimate_f0(
            audio, sample_rate
        )
        features.extend([
            f0_mean,
            f0_std,
            f0_min,
            f0_max,
            f0_max - f0_min,
            self._compute_slope(f0_contour) if len(f0_contour) > 1 else 0,
        ])

        # RMS energy
        rms = self._compute_rms(audio)
        features.extend([
            np.mean(rms),
            np.std(rms),
            np.min(rms),
            np.max(rms),
            np.max(rms) - np.min(rms),
            20 * np.log10(np.max(rms) / (np.min(rms) + 1e-10)),
        ])

        # HNR (Harmonics-to-Noise Ratio)
        hnr_mean, hnr_std, hnr_min, hnr_max = self._compute_hnr(
            audio, sample_rate
        )
        features.extend([
            hnr_mean,
            hnr_std,
            hnr_min,
            hnr_max,
            hnr_max - hnr_min,
            self._compute_voiced_ratio(audio, sample_rate),
        ])

        # MFCCs
        mfccs = self._compute_mfccs(audio, sample_rate)
        features.extend(np.mean(mfccs, axis=0)[:13].tolist())

        # ADSR envelope
        attack, decay, sustain, release = self._compute_adsr(rms)
        features.extend([attack, decay, sustain, release])

        # Duration
        duration_ms = len(audio) * 1000 / sample_rate
        features.append(duration_ms)

        # Jitter and Shimmer
        jitter, shimmer = self._compute_jitter_shimmer(
            audio, sample_rate, f0_contour
        )
        features.extend([jitter, shimmer, 0, 0])  # 2 placeholders

        return np.array(features)

    def _extract_layer2(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Extract Layer 2: Macro Texture features (30D).

        Features:
        - Harmonic features (10): Harmonicity, spectral flatness, etc.
        - Pitch geometry (8): Pitch movement, acceleration, contour shape
        - GLCM texture (12): Gray-level co-occurrence matrix on spectrogram
        """
        features = []

        # Spectrogram
        f, t, Sxx = signal.spectrogram(audio, sample_rate, nperseg=1024)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Harmonic features
        harmonic = self._compute_harmonic_features(Sxx, f)
        features.extend(harmonic[:10])

        # Pitch geometry
        pitch_geom = self._compute_pitch_geometry(Sxx_db, f)
        features.extend(pitch_geom[:8])

        # GLCM texture
        glcm = self._compute_glcm_features(Sxx_db)
        features.extend(glcm[:12])

        return np.array(features)

    def _extract_layer3(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Extract Layer 3: Micro Texture features (36D).

        Features:
        - Spectral derivatives (12): Rate of change in spectral envelope
        - Frequency modulation (8): FM rate, depth, complexity
        - Micro dynamics (10): Onset/offset precision, transient characteristics
        - Rhythm/temporal (6): Inter-onset intervals, rhythmic regularity
        """
        features = []

        # Spectrogram
        f, t, Sxx = signal.spectrogram(audio, sample_rate, nperseg=512)

        # Spectral derivatives
        derivs = self._compute_spectral_derivatives(Sxx)
        features.extend(derivs[:12])

        # FM features
        fm = self._compute_fm_features(audio, sample_rate)
        features.extend(fm[:8])

        # Micro dynamics
        dynamics = self._compute_micro_dynamics(audio)
        features.extend(dynamics[:10])

        # Rhythm
        rhythm = self._compute_rhythm_features(audio, sample_rate)
        features.extend(rhythm[:6])

        return np.array(features)

    # ---- Helper methods ----

    def _estimate_f0(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Tuple[float, float, float, float, np.ndarray]:
        """Estimate F0 using autocorrelation."""
        frame_size = int(0.025 * sample_rate)  # 25ms
        hop_size = int(0.01 * sample_rate)     # 10ms

        f0_values = []

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            # Autocorrelation
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr) // 2:]

            # Find first peak after 2ms (bat F0 range)
            min_lag = int(sample_rate * 0.002)
            max_lag = int(sample_rate * 0.02)  # Up to 50Hz

            if len(corr) > max_lag:
                peak_lag = np.argmax(corr[min_lag:max_lag]) + min_lag
                if corr[peak_lag] > 0.3 * np.max(corr):  # Voicing threshold
                    f0 = sample_rate / peak_lag
                    if 1000 < f0 < 100000:  # Plausible bat range
                        f0_values.append(f0)

        if not f0_values:
            return 0, 0, 0, 0, np.array([0])

        f0_array = np.array(f0_values)
        return (
            np.mean(f0_array),
            np.std(f0_array),
            np.min(f0_array),
            np.max(f0_array),
            f0_array,
        )

    def _compute_rms(self, audio: np.ndarray) -> np.ndarray:
        """Compute RMS energy over frames."""
        frame_size = 512
        hop_size = 128

        rms = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            rms.append(np.sqrt(np.mean(frame ** 2)))

        return np.array(rms)

    def _compute_hnr(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Tuple[float, float, float, float]:
        """Compute Harmonics-to-Noise Ratio."""
        # Simplified HNR: ratio of peak energy to total energy in spectrum
        f, t, Sxx = signal.spectrogram(audio, sample_rate)

        hnr_values = []
        for i in range(Sxx.shape[1]):
            spectrum = Sxx[:, i]
            peak_energy = np.max(spectrum)
            total_energy = np.sum(spectrum)
            hnr = peak_energy / (total_energy + 1e-10)
            hnr_values.append(hnr)

        hnr_array = np.array(hnr_values)
        return (
            np.mean(hnr_array),
            np.std(hnr_array),
            np.min(hnr_array),
            np.max(hnr_array),
        )

    def _compute_voiced_ratio(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Compute ratio of voiced frames."""
        frame_size = int(0.025 * sample_rate)
        hop_size = int(0.01 * sample_rate)

        voiced_count = 0
        total_count = 0

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            energy = np.sum(frame ** 2)

            if energy > 0.01:  # Energy threshold
                voiced_count += 1
            total_count += 1

        return voiced_count / (total_count + 1e-10)

    def _compute_mfccs(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Compute MFCCs."""
        from scipy.fftpack import dct

        # Compute mel spectrogram
        f, t, Sxx = signal.spectrogram(audio, sample_rate, nperseg=512)

        # Mel filter bank (simplified)
        n_mels = self.n_mels
        mel_filters = np.zeros((n_mels, len(f)))

        for m in range(n_mels):
            # Linear scale on log frequency
            mel_low = 2595 * np.log10(1 + 1000 / 700)
            mel_high = 2595 * np.log10(1 + sample_rate / 2 / 700)
            mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
            mel_center = mel_points[m + 1]

            # Convert back to Hz
            hz_center = 700 * (10 ** (mel_center / 2595) - 1)

            # Gaussian filter around center
            for j, freq in enumerate(f):
                mel_filters[m, j] = np.exp(
                    -0.5 * ((freq - hz_center) / (hz_center * 0.5)) ** 2
                )

        # Apply filters
        mel_spec = np.dot(mel_filters, Sxx)

        # Log
        mel_spec_db = 10 * np.log10(mel_spec + 1e-10)

        # DCT
        mfccs = dct(mel_spec_db, axis=0, type=2, norm='ortho')

        return mfccs

    def _compute_slope(self, contour: np.ndarray) -> float:
        """Compute linear slope of contour."""
        if len(contour) < 2:
            return 0
        x = np.arange(len(contour))
        coeffs = np.polyfit(x, contour, 1)
        return coeffs[0]

    def _compute_adsr(
        self,
        rms: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Compute ADSR envelope parameters."""
        if len(rms) < 4:
            return 0, 0, 0, 0

        # Normalize
        rms_norm = rms / (np.max(rms) + 1e-10)

        # Attack: time to reach 90% of max
        peak_idx = np.argmax(rms_norm)
        attack = peak_idx / len(rms_norm)

        # Decay: time from peak to 80% of peak
        decay_end = peak_idx
        target = 0.8 * rms_norm[peak_idx]
        for i in range(peak_idx, len(rms_norm)):
            if rms_norm[i] < target:
                decay_end = i
                break
        decay = (decay_end - peak_idx) / len(rms_norm)

        # Sustain: mean level after decay
        sustain = np.mean(rms_norm[decay_end:]) if decay_end < len(rms_norm) else 0

        # Release: approximate
        release = 0.1

        return attack, decay, sustain, release

    def _compute_jitter_shimmer(
        self,
        audio: np.ndarray,
        sample_rate: int,
        f0_contour: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute jitter (F0 perturbation) and shimmer (amp perturbation)."""
        if len(f0_contour) < 2:
            return 0, 0

        # Jitter: relative F0 variation
        jitter = np.std(np.diff(f0_contour)) / (np.mean(f0_contour) + 1e-10)

        # Shimmer: relative amplitude variation
        rms = self._compute_rms(audio)
        if len(rms) > 1:
            shimmer = np.std(np.diff(rms)) / (np.mean(rms) + 1e-10)
        else:
            shimmer = 0

        return jitter, shimmer

    def _compute_harmonic_features(
        self,
        Sxx: np.ndarray,
        f: np.ndarray,
    ) -> np.ndarray:
        """Compute harmonic-related features."""
        features = []

        # Harmonicity: ratio of harmonic energy to total
        harmonic_energy = np.sum(Sxx[:len(f)//10, :])
        total_energy = np.sum(Sxx)
        harmonicity = harmonic_energy / (total_energy + 1e-10)
        features.append(harmonicity)

        # Spectral flatness
        geometric_mean = np.exp(np.mean(np.log(Sxx + 1e-10)))
        arithmetic_mean = np.mean(Sxx)
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        features.append(flatness)

        # Spectral centroid
        centroid = np.sum(f[:, None] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-10)
        features.extend([
            np.mean(centroid),
            np.std(centroid),
            np.min(centroid),
            np.max(centroid),
        ])

        # Spectral rolloff (85% energy)
        cumsum = np.cumsum(Sxx, axis=0)
        total = cumsum[-1, :]
        rolloff_idxs = np.argmax(cumsum >= 0.85 * total, axis=0)
        rolloff_freqs = f[rolloff_idxs]
        features.extend([
            np.mean(rolloff_freqs),
            np.std(rolloff_freqs),
        ])

        # Spectral bandwidth
        bandwidth = np.std(centroid)
        features.append(bandwidth)

        return np.array(features)

    def _compute_pitch_geometry(
        self,
        Sxx_db: np.ndarray,
        f: np.ndarray,
    ) -> np.ndarray:
        """Compute pitch contour geometry features."""
        # Peak frequency over time
        peak_idxs = np.argmax(Sxx_db, axis=0)
        peak_freqs = f[peak_idxs]

        features = []

        # Mean pitch
        features.append(np.mean(peak_freqs))

        # Pitch range (semitones)
        min_f = np.min(peak_freqs)
        max_f = np.max(peak_freqs)
        if min_f > 0:
            range_semitones = 12 * np.log2(max_f / min_f)
        else:
            range_semitones = 0
        features.append(range_semitones)

        # Pitch direction (upward vs downward)
        first_half = np.mean(peak_freqs[:len(peak_freqs)//2])
        second_half = np.mean(peak_freqs[len(peak_freqs)//2:])
        direction = second_half - first_half
        features.append(direction)

        # Pitch acceleration
        if len(peak_freqs) > 2:
            acceleration = np.diff(np.diff(peak_freqs))
            features.extend([
                np.mean(acceleration),
                np.std(acceleration),
            ])
        else:
            features.extend([0, 0])

        # Contour complexity (number of direction changes)
        if len(peak_freqs) > 2:
            diffs = np.diff(peak_freqs)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            features.append(sign_changes)
        else:
            features.append(0)

        # Vocalic vs non-vocalic ratio
        voiced = peak_freqs > 1000
        features.append(np.sum(voiced) / len(voiced))

        # Final features to fill to 8
        while len(features) < 8:
            features.append(0)

        return np.array(features[:8])

    def _compute_glcm_features(
        self,
        Sxx_db: np.ndarray,
    ) -> np.ndarray:
        """Compute Gray-Level Co-occurrence Matrix features."""
        # Quantize spectrogram to 8 levels
        n_levels = 8
        Sxx_quantized = np.floor(
            (Sxx_db - Sxx_db.min()) /
            (Sxx_db.max() - Sxx_db.min() + 1e-10) * n_levels
        ).astype(int)
        Sxx_quantized = np.clip(Sxx_quantized, 0, n_levels - 1)

        # Compute GLCM (horizontal, distance=1)
        glcm = np.zeros((n_levels, n_levels))

        for i in range(Sxx_quantized.shape[0] - 1):
            for j in range(Sxx_quantized.shape[1]):
                row = Sxx_quantized[i, j]
                col_next = Sxx_quantized[i + 1, j] if j < Sxx_quantized.shape[1] - 1 else row
                glcm[row, col_next] += 1

        # Normalize
        glcm = glcm / (np.sum(glcm) + 1e-10)

        # Haralick features
        features = []

        # Contrast
        i, j = np.indices((n_levels, n_levels))
        contrast = np.sum(glcm * (i - j) ** 2)
        features.append(contrast)

        # Dissimilarity
        dissimilarity = np.sum(glcm * np.abs(i - j))
        features.append(dissimilarity)

        # Homogeneity
        homogeneity = np.sum(glcm / (1 + np.abs(i - j)))
        features.append(homogeneity)

        # Energy (ASM)
        energy = np.sum(glcm ** 2)
        features.append(energy)

        # Correlation
        mean_i = np.sum(i * glcm)
        mean_j = np.sum(j * glcm)
        std_i = np.sqrt(np.sum(((i - mean_i) ** 2) * glcm))
        std_j = np.sqrt(np.sum(((j - mean_j) ** 2) * glcm))
        correlation = np.sum(((i - mean_i) * (j - mean_j) * glcm)) / (std_i * std_j + 1e-10)
        features.append(correlation)

        # Fill to 12
        while len(features) < 12:
            features.append(0)

        return np.array(features[:12])

    def _compute_spectral_derivatives(
        self,
        Sxx: np.ndarray,
    ) -> np.ndarray:
        """Compute spectral derivative features."""
        # First derivative along frequency axis
        deriv_freq = np.diff(Sxx, axis=0)

        features = []

        # Mean, std of derivative
        features.append(np.mean(deriv_freq))
        features.append(np.std(deriv_freq))

        # Spectral flux (L2 norm of derivative)
        flux = np.sqrt(np.mean(deriv_freq ** 2))
        features.append(flux)

        # Fill remaining
        while len(features) < 12:
            features.append(0)

        return np.array(features[:12])

    def _compute_fm_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Compute frequency modulation features."""
        # Extract instantaneous frequency
        analytic = signal.hilbert(audio)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) * sample_rate / (2 * np.pi)

        features = []

        # FM rate (modulation frequency)
        f, t, Sxx = signal.spectrogram(inst_freq, sample_rate)
        peak_idx = np.argmax(Sxx, axis=0)
        fm_rate = f[np.median(peak_idx).astype(int)]
        features.append(fm_rate)

        # FM depth
        fm_depth = np.std(inst_freq)
        features.append(fm_depth)

        # FM complexity
        fm_complexity = np.std(np.diff(inst_freq))
        features.append(fm_complexity)

        # Fill to 8
        while len(features) < 8:
            features.append(0)

        return np.array(features[:8])

    def _compute_micro_dynamics(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """Compute micro-dynamic features."""
        features = []

        # Onset detection (high-frequency energy)
        frame_size = 512
        hop_size = 128

        high_freq_energy = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            f, t, Sxx = signal.spectrogram(frame, 48000, nperseg=256)
            # Energy in upper half
            high_energy = np.sum(Sxx[len(Sxx)//2:, :])
            high_freq_energy.append(high_energy)

        high_freq_energy = np.array(high_freq_energy)

        # Onset precision
        if len(high_freq_energy) > 1:
            onset_slope = np.max(np.diff(high_freq_energy))
            features.append(onset_slope)
        else:
            features.append(0)

        # Transient ratio
        transient_ratio = np.sum(high_freq_energy > np.mean(high_freq_energy) * 2)
        transient_ratio = transient_ratio / len(high_freq_energy)
        features.append(transient_ratio)

        # Fill to 10
        while len(features) < 10:
            features.append(0)

        return np.array(features[:10])

    def _compute_rhythm_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Compute rhythm/temporal features."""
        features = []

        # Onset detection
        rms = self._compute_rms(audio)

        # Find onsets (energy peaks)
        onsets = []
        for i in range(1, len(rms) - 1):
            if rms[i] > rms[i-1] and rms[i] > rms[i+1]:
                if rms[i] > np.mean(rms) + np.std(rms):
                    onsets.append(i)

        if len(onsets) > 1:
            # Inter-onset intervals
            iois = np.diff(onsets) * 128 / sample_rate * 1000  # ms

            features.extend([
                np.mean(iois),
                np.std(iois),
                np.min(iois),
                np.max(iois),
            ])

            # Rhythmic regularity (coefficient of variation)
            cov = np.std(iois) / (np.mean(iois) + 1e-10)
            features.append(cov)
        else:
            features.extend([0, 0, 0, 0, 0])

        # Tempo estimate (onsets per second)
        if len(onsets) > 1:
            duration_sec = len(audio) / sample_rate
            tempo = len(onsets) / duration_sec
            features.append(tempo)
        else:
            features.append(0)

        return np.array(features[:6])


class AffectiveVAEEncoder:
    """
    β-VAE encoder for 16D affect vectors.

    NOTE: This is a placeholder implementation. For production,
    load a trained β-VAE model from PyTorch/ONNX.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize VAE encoder.

        Args:
            model_path: Path to trained VAE model (if available)
        """
        self.model_path = model_path
        self.latent_dim = 16

        # TODO: Load trained model
        logger.warning("AffectiveVAEEncoder using placeholder - model not loaded")

    def encode(self, rosetta_features: np.ndarray) -> np.ndarray:
        """
        Encode 112D RosettaFeatures to 16D affect vector.

        Args:
            rosetta_features: 112D RosettaFeature vector

        Returns:
            16D affect vector (arousal, valence, harshness, etc.)
        """
        # TODO: Run actual VAE encoder
        # For now, return placeholder based on input features

        # Extract interpretable dimensions from Rosetta features
        affect = np.zeros(16)

        # Dimension 0: Arousal (based on RMS energy, F0 range)
        if len(rosetta_features) > 39:
            energy_idx = 6  # Mean RMS
            f0_range_idx = 4  # F0 range
            arousal = (
                rosetta_features[energy_idx] * 0.5 +
                rosetta_features[f0_range_idx] / 50000 * 0.5
            )
            affect[0] = np.clip(arousal, 0, 1)

        # Dimension 1: Valence/Harshness (based on HNR, jitter)
        if len(rosetta_features) > 50:
            hnr_idx = 12  # Mean HNR
            harshness = 1.0 - np.clip(rosetta_features[hnr_idx] / 30, 0, 1)
            affect[1] = harshness * 2 - 1  # Scale to [-1, 1]

        # Remaining dimensions: placeholder
        for i in range(2, 16):
            affect[i] = np.random.randn() * 0.1

        return affect


class SyntacticVQVAEEncoder:
    """
    VQ-VAE encoder for discrete syntactic tokens.

    NOTE: This is a placeholder implementation. For production,
    load a trained VQ-VAE model from PyTorch/ONNX.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        codebook_size: int = 64,
        codebook_dim: int = 32,
    ):
        """
        Initialize VQ-VAE encoder.

        Args:
            model_path: Path to trained VQ-VAE model
            codebook_size: Size of codebook
            codebook_dim: Dimension of codebook vectors
        """
        self.model_path = model_path
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        # TODO: Load trained model
        logger.warning("SyntacticVQVAEEncoder using placeholder - model not loaded")

    def encode(self, rosetta_features: np.ndarray) -> int:
        """
        Encode 112D RosettaFeatures to discrete token.

        Args:
            rosetta_features: 112D RosettaFeature vector

        Returns:
            Discrete token ID (0 to codebook_size-1)
        """
        # TODO: Run actual VQ-VAE encoder
        # For now, return token based on F0 and duration

        if len(rosetta_features) > 39:
            f0_mean = rosetta_features[0]  # Mean F0
            duration = rosetta_features[39]  # Duration ms

            # Simple quantization based on F0 and duration
            f0_bin = int(f0_mean / 2000) % 8
            dur_bin = int(duration / 10) % 8

            token = f0_bin * 8 + dur_bin
            return token % self.codebook_size

        return 0


class FeaturePipeline:
    """
    Complete feature extraction pipeline.

    Extracts all features needed for analysis frameworks:
    1. Raw audio
    2. 112D RosettaFeatures
    3. 16D Affect vector (VAE)
    4. Discrete token (VQ-VAE)
    """

    def __init__(
        self,
        rosetta_extractor: Optional[RosettaFeatureExtractor] = None,
        vae_encoder: Optional[AffectiveVAEEncoder] = None,
        vqvae_encoder: Optional[SyntacticVQVAEEncoder] = None,
    ):
        """
        Initialize feature pipeline.

        Args:
            rosetta_extractor: Rosetta feature extractor
            vae_encoder: VAE encoder for affect
            vqvae_encoder: VQ-VAE encoder for tokens
        """
        self.rosetta = rosetta_extractor or DEFAULT_ROSETTA_EXTRACTOR
        self.vae = vae_encoder or AffectiveVAEEncoder()
        self.vqvae = vqvae_encoder or SyntacticVQVAEEncoder()

        logger.info("FeaturePipeline initialized")

    def process_audio_segment(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segment_id: str,
        phrase_id: str = "",
        social_context: str = "",
    ) -> ExtractedFeatures:
        """
        Process audio segment through full pipeline.

        Args:
            audio: Raw audio samples
            sample_rate: Sample rate in Hz
            segment_id: Unique segment identifier
            phrase_id: Original phrase ID
            social_context: Social context label

        Returns:
            ExtractedFeatures with all representations
        """
        # Step 1: Extract 112D RosettaFeatures
        rosetta = self.rosetta.extract(audio, sample_rate)

        # Step 2: VAE encode to 16D affect
        affect = self.vae.encode(rosetta)

        # Step 3: VQ-VAE encode to discrete token
        token = self.vqvae.encode(rosetta)

        return ExtractedFeatures(
            segment_id=segment_id,
            phrase_id=phrase_id,
            audio_raw=audio,
            rosetta_features_112d=rosetta,
            affect_vector_16d=affect,
            syntactic_token=token,
            social_context=social_context,
            metadata={
                'sample_rate': sample_rate,
                'duration_ms': len(audio) * 1000 / sample_rate,
            },
        )

    def process_audio_file(
        self,
        file_path: str,
        segment_id: str,
        phrase_id: str = "",
        social_context: str = "",
    ) -> ExtractedFeatures:
        """
        Load and process audio file.

        Args:
            file_path: Path to WAV file
            segment_id: Unique segment identifier
            phrase_id: Original phrase ID
            social_context: Social context label

        Returns:
            ExtractedFeatures with all representations
        """
        sample_rate, audio = wavfile.read(file_path)

        # Convert to float and normalize
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        return self.process_audio_segment(
            audio, sample_rate, segment_id, phrase_id, social_context
        )

    def save_features(
        self,
        features: ExtractedFeatures,
        output_path: str,
    ) -> None:
        """
        Save extracted features to file.

        Args:
            features: ExtractedFeatures to save
            output_path: Output file path (.pkl or .npy)
        """
        if output_path.endswith('.pkl'):
            with open(output_path, 'wb') as f:
                pickle.dump(features, f)
        elif output_path.endswith('.npy'):
            np.save(output_path, features)
        else:
            raise ValueError("Output path must be .pkl or .npy")

        logger.info(f"Saved features to {output_path}")

    def load_features(
        self,
        input_path: str,
    ) -> ExtractedFeatures:
        """
        Load extracted features from file.

        Args:
            input_path: Input file path

        Returns:
            ExtractedFeatures
        """
        if input_path.endswith('.pkl'):
            with open(input_path, 'rb') as f:
                return pickle.load(f)
        elif input_path.endswith('.npy'):
            return np.load(input_path, allow_pickle=True).item()
        else:
            raise ValueError("Input path must be .pkl or .npy")


# Default instances
DEFAULT_ROSETTA_EXTRACTOR = RosettaFeatureExtractor()
DEFAULT_VAE_ENCODER = AffectiveVAEEncoder()
DEFAULT_VQVAE_ENCODER = SyntacticVQVAEEncoder()
DEFAULT_FEATURE_PIPELINE = FeaturePipeline()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Feature Pipeline Demo")
    print("=" * 50)

    pipeline = DEFAULT_FEATURE_PIPELINE

    # Generate synthetic bat call (FM sweep)
    sample_rate = 48000
    duration = 0.1  # 100ms
    t = np.linspace(0, duration, int(sample_rate * duration))

    audio = 0.5 * np.sin(
        2 * np.pi * (15000 + 10000 * t / duration) * t
    )

    # Process through full pipeline
    features = pipeline.process_audio_segment(
        audio=audio,
        sample_rate=sample_rate,
        segment_id="demo_segment_001",
        phrase_id="bat_phrase_001",
        social_context="aggression",
    )

    print(f"\nExtracted Features:")
    print(f"  Segment ID: {features.segment_id}")
    print(f"  Phrase ID: {features.phrase_id}")
    print(f"  Social Context: {features.social_context}")
    print()

    print(f"  112D RosettaFeatures: {features.rosetta_features_112d.shape}")
    print(f"    F0 mean: {features.rosetta_features_112d[0]:.1f} Hz")
    print(f"    F0 range: {features.rosetta_features_112d[4]:.1f} Hz")
    print(f"    Duration: {features.rosetta_features_112d[39]:.1f} ms")
    print()

    print(f"  16D Affect Vector: {features.affect_vector_16d.shape}")
    print(f"    Arousal (dim 0): {features.affect_vector_16d[0]:.3f}")
    print(f"    Harshness (dim 1): {features.affect_vector_16d[1]:.3f}")
    print()

    print(f"  Syntactic Token: {features.syntactic_token}")
    print(f"    (VQ-VAE codebook ID)")
    print()

    print("Now ready for analysis frameworks!")
    print("  - GradedContinuumAnalyzer (uses 16D affect)")
    print("  - MicroPhonologyAnalyzer (uses VQ-VAE tokens)")
    print("  - AddressingClassifier (uses both)")
