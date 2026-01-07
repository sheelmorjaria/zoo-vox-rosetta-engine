"""
Universal Rosetta Stone - Species-Agnostic Analysis Engine

This module implements a physics-based approach to animal vocalization analysis
that works across species and acoustic modalities.

Architecture:
- Modality Router: Detects harmonic, FM sweep, transient, or rhythmic patterns
- Universal Vocabulary: Clusters similar phrases regardless of species
- Syntax Engine: Discovers grammatical rules and sentence structure

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import warnings
from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

# Import acoustic persona definitions for hybrid architecture
try:
    from analysis.rosetta_stone.acoustic_similarity_for_atomic_phrase_candidates import (
        ACOUSTIC_PERSONAS,
        compute_persona_score,
    )

    HAS_PERSONA_SUPPORT = True
except ImportError:
    HAS_PERSONA_SUPPORT = False
    ACOUSTIC_PERSONAS = {}

    def compute_persona_score(*args, **kwargs):
        return 0.0


# Try to import advanced segmentation libraries
try:
    import ruptures as rpt

    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    print("Warning: ruptures library not available. Install with: pip install ruptures")

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa library not available. Install with: pip install librosa")

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class Modality(Enum):
    """Acoustic modality types based on physical properties."""

    HARMONIC = 1  # Flat tones, pitch stable (marmosets, some birds)
    FM_SWEEP = 2  # Pitch changes over time (bats, dolphins)
    TRANSIENT = 3  # Clicks/pulses (whales, insects)
    RHYTHMIC = 4  # Temporal patterns (crickets, frogs)


class PhraseSignature:
    """
    Represents the acoustic signature of a phrase.

    Each phrase is characterized by modality-specific features that capture
    its essential physical properties.
    """

    def __init__(
        self,
        modality: Modality,
        data: np.ndarray,
        timestamp: Optional[float] = None,
        sample_rate: int = 48000,
    ):
        self.modality = modality
        self.data = data
        self.timestamp = timestamp
        self.sample_rate = sample_rate
        self.features = self._extract_features()

    def _extract_features(self) -> Dict[str, float]:
        """Extract modality-specific features from audio data."""
        features = {}

        if self.modality == Modality.HARMONIC:
            # Extract pitch-related features for harmonic signals
            features.update(self._extract_harmonic_features())

        elif self.modality == Modality.FM_SWEEP:
            # Extract sweep-related features for FM signals
            features.update(self._extract_fm_features())

        elif self.modality == Modality.TRANSIENT:
            # Extract energy and timing features for transients
            features.update(self._extract_transient_features())

        elif self.modality == Modality.RHYTHMIC:
            # Extract temporal pattern features
            features.update(self._extract_rhythmic_features())

        # Common features for all modalities
        features.update(self._extract_common_features())

        return features

    def _extract_harmonic_features(self) -> Dict[str, float]:
        """Extract features specific to harmonic signals."""
        # Fundamental frequency tracking using autocorrelation
        frame_size = min(2048, len(self.data))
        if frame_size < 256:
            return {"f0_mean": 0, "f0_std": 0, "harmonicity": 0}

        # Use PYIN-style approach for F0 estimation
        f0_values = []
        for i in range(0, len(self.data) - frame_size, frame_size // 4):
            frame = self.data[i : i + frame_size]
            acf = np.correlate(frame, frame, mode="full")
            acf = acf[len(acf) // 2 :]

            # Find first peak after zero lag
            if len(acf) > 10:
                peak_indices = np.where(acf[1:10] > acf[:9])[0]
                if len(peak_indices) > 0:
                    period = peak_indices[0] + 1
                    f0 = 48000 / period if period > 0 else 0
                    f0_values.append(f0)

        if f0_values:
            return {
                "f0_mean": np.mean(f0_values),
                "f0_std": np.std(f0_values),
                "f0_range": np.max(f0_values) - np.min(f0_values) if f0_values else 0,
                "harmonicity": 1.0 / (1.0 + np.std(f0_values)) if f0_values else 0,
            }
        else:
            return {"f0_mean": 0, "f0_std": 0, "f0_range": 0, "harmonicity": 0}

    def _extract_fm_features(self) -> Dict[str, float]:
        """Extract features specific to FM sweep signals."""
        # Instantaneous frequency estimation
        inst_freq = self._estimate_instantaneous_frequency()

        if len(inst_freq) > 0:
            return {
                "start_freq": inst_freq[0],
                "end_freq": inst_freq[-1],
                "freq_range": np.max(inst_freq) - np.min(inst_freq),
                "mean_freq": np.mean(inst_freq),
                "freq_slope": np.polyfit(range(len(inst_freq)), inst_freq, 1)[0]
                if len(inst_freq) > 1
                else 0,
                "curve_linearity": 1.0 / (1.0 + np.std(np.diff(inst_freq))),
            }
        else:
            return {
                "start_freq": 0,
                "end_freq": 0,
                "freq_range": 0,
                "mean_freq": 0,
                "freq_slope": 0,
                "curve_linearity": 0,
            }

    def _extract_transient_features(self) -> Dict[str, float]:
        """Extract features specific to transient signals."""
        # Energy envelope
        energy = np.sum(self.data**2)
        rms = np.sqrt(energy / len(self.data))

        # Spectral centroid (brightness)
        spectrum = np.abs(np.fft.rfft(self.data))
        freqs = np.fft.rfftfreq(len(self.data), 1 / 48000)
        spectral_centroid = (
            np.sum(freqs * spectrum) / np.sum(spectrum) if np.sum(spectrum) > 0 else 0
        )

        # Zero crossing rate (spikiness)
        zcr = np.mean(np.abs(np.diff(np.sign(self.data))))

        # Kurtosis (peakiness) - manual calculation
        len(self.data)
        mean = np.mean(self.data)
        std = np.std(self.data)
        if std > 0:
            kurtosis = np.mean(((self.data - mean) / std) ** 4) - 3
        else:
            kurtosis = 0

        return {
            "rms": rms,
            "energy": energy,
            "spectral_centroid": spectral_centroid,
            "zero_crossing_rate": zcr,
            "kurtosis": kurtosis,
            "peak_amplitude": np.max(np.abs(self.data)),
        }

    def _extract_rhythmic_features(self) -> Dict[str, float]:
        """Extract features specific to rhythmic patterns."""
        # Autocorrelation for rhythmicity detection
        if len(self.data) < 256:
            return {"rhythmic_strength": 0, "tempo": 0}

        acf = np.correlate(self.data, self.data, mode="full")
        acf = acf[len(acf) // 2 :]

        # Find periodicities
        periodicities = []
        for lag in range(20, min(len(acf) // 2, 500)):
            if acf[lag] > acf[lag - 1] and acf[lag] > acf[lag + 1]:
                if acf[lag] > 0.1 * acf[0]:
                    periodicities.append(lag)

        if periodicities:
            dominant_period = periodicities[0]
            tempo = 60000 / (dominant_period * 2) if dominant_period > 0 else 0  # BPM estimation
            rhythmic_strength = acf[dominant_period] / acf[0]
            return {"rhythmic_strength": rhythmic_strength, "tempo": tempo}
        else:
            return {"rhythmic_strength": 0, "tempo": 0}

    def _extract_common_features(self) -> Dict[str, float]:
        """
        Extract features common to all modalities.

        Now includes MICRO-DYNAMICS for atomic phrase discovery:
        1. Harmonic-to-Noise Ratio (HNR) - Clarity
        2. Amplitude Envelope Statistics - Dynamics (Attack/Decay)
        3. Vibrato/Jitter - Micro-pitch variation
        4. MFCCs (1-4) - Spectral fingerprint
        5. Spectral Contrast - Formant structure
        6. Inter-Click Interval - Rhythm for transients
        """
        # Duration in milliseconds
        duration = len(self.data) / self.sample_rate * 1000

        # ===== BASIC TIMBRE FEATURES =====
        # Spectral flatness (Wiener entropy)
        spectrum = np.abs(np.fft.rfft(self.data))
        spectrum_nonzero = spectrum[spectrum > 1e-10]  # Avoid log(0)
        if len(spectrum_nonzero) > 0:
            spectral_flatness = np.exp(np.mean(np.log(spectrum_nonzero))) / np.mean(
                spectrum_nonzero
            )
        else:
            spectral_flatness = 1.0

        freqs = np.fft.rfftfreq(len(self.data), 1 / self.sample_rate)
        freqs_for_spectrum = freqs[: len(spectrum)]

        # Spectral centroid (brightness/"center of mass")
        spectral_centroid_hz = (
            np.sum(freqs_for_spectrum * spectrum) / np.sum(spectrum) if np.sum(spectrum) > 0 else 0
        )

        # Spectral bandwidth (spread around centroid)
        if np.sum(spectrum) > 0:
            spectral_bandwidth = np.sqrt(
                np.sum(((freqs_for_spectrum - spectral_centroid_hz) ** 2) * spectrum)
                / np.sum(spectrum)
            )
        else:
            spectral_bandwidth = 0

        # Spectral slope (linear regression fit to log spectrum)
        if len(spectrum_nonzero) > 1:
            log_spectrum = np.log(spectrum_nonzero + 1e-10)
            freq_for_log = freqs_for_spectrum[: len(log_spectrum)]
            freq_mean = np.mean(freq_for_log)
            log_spectrum_mean = np.mean(log_spectrum)
            numerator = np.sum((freq_for_log - freq_mean) * (log_spectrum - log_spectrum_mean))
            denominator = np.sum((freq_for_log - freq_mean) ** 2)
            spectral_slope = numerator / denominator if denominator > 0 else 0
        else:
            spectral_slope = 0

        # Spectral rolloff (frequency below which 85% of energy is contained)
        if np.sum(spectrum) > 0:
            cumulative_energy = np.cumsum(spectrum)
            total_energy = cumulative_energy[-1]
            rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
            if len(rolloff_idx) > 0:
                spectral_rolloff = freqs_for_spectrum[rolloff_idx[0]]
            else:
                spectral_rolloff = freqs_for_spectrum[-1]
        else:
            spectral_rolloff = 0

        # ===== CATEGORY 1: HARMONIC CLARITY (GRIT FACTORS) =====
        # Harmonic-to-Noise Ratio (HNR) - "The Grit Factor"
        try:
            # Use autocorrelation to estimate HNR
            acf = np.correlate(self.data, self.data, mode="full")
            acf = acf[len(acf) // 2 :]

            # Peak height (periodic component)
            peak_height = np.max(acf[1 : len(acf) // 10]) if len(acf) > 10 else 0

            # Mean of remaining (noise floor)
            noise_floor = np.mean(acf[len(acf) // 10 :]) if len(acf) > 10 else 1

            # HNR as ratio
            harmonic_to_noise_ratio = peak_height / noise_floor if noise_floor > 0 else 0
        except:
            harmonic_to_noise_ratio = 0

        # ===== CATEGORY 2: TEMPORAL DYNAMICS (MOTION FACTORS) =====
        # Amplitude Envelope Statistics
        envelope = np.abs(signal.hilbert(self.data))

        # Smooth envelope for cleaner attack/decay detection
        from scipy.ndimage import gaussian_filter1d

        smoothed_envelope = gaussian_filter1d(envelope, sigma=min(5, len(envelope) // 10))

        # Normalize envelope
        if np.max(smoothed_envelope) > 0:
            smoothed_envelope = smoothed_envelope / np.max(smoothed_envelope)

        # Attack time: time to reach 90% of max
        max_idx = np.argmax(smoothed_envelope)
        threshold = 0.1
        attack_start = np.where(smoothed_envelope[:max_idx] > threshold)[0]
        attack_time_ms = (
            (max_idx - attack_start[0]) / self.sample_rate * 1000 if len(attack_start) > 0 else 0
        )

        # Decay time: time from max to 10%
        decay_start = max_idx
        decay_end = np.where(smoothed_envelope[decay_start:] < threshold)[0]
        decay_time_ms = (
            (decay_end[0] if len(decay_end) > 0 else len(smoothed_envelope) - decay_start)
            / self.sample_rate
            * 1000
        )

        # Sustain level: mean amplitude during sustained portion
        sustain_portion = (
            smoothed_envelope[max_idx : max_idx + len(self.data) // 4]
            if max_idx + len(self.data) // 4 < len(smoothed_envelope)
            else smoothed_envelope[max_idx:]
        )
        sustain_level = np.mean(sustain_portion) if len(sustain_portion) > 0 else 0

        # Vibrato / Jitter (Micro-pitch variation)
        try:
            # Estimate F0 contour
            frame_size = min(2048, len(self.data))
            hop_size = frame_size // 4

            f0_contour = []
            for i in range(0, len(self.data) - frame_size, hop_size):
                frame = self.data[i : i + frame_size]
                # Use autocorrelation for pitch tracking
                acf = np.correlate(frame, frame, mode="full")
                acf = acf[len(acf) // 2 :]

                # Find first peak
                if len(acf) > 20:
                    peak_idx = np.where(acf[1:20] > acf[:19])[0]
                    if len(peak_idx) > 0:
                        period = peak_idx[0] + 1
                        f0 = self.sample_rate / period
                        f0_contour.append(f0)

            if len(f0_contour) > 1:
                # Vibrato: periodic modulation
                f0_contour = np.array(f0_contour)
                f0_detrended = f0_contour - np.mean(f0_contour)

                # Autocorrelation of F0 contour to detect periodicity
                f0_acf = np.correlate(f0_detrended, f0_detrended, mode="full")
                f0_acf = f0_acf[len(f0_acf) // 2 :]

                # Find peaks
                from scipy.signal import find_peaks

                peaks, _ = find_peaks(f0_acf[1 : len(f0_acf) // 10], height=0.3 * np.max(f0_acf))

                if len(peaks) > 0:
                    vibrato_rate_hz = len(peaks) / (len(f0_contour) * hop_size / self.sample_rate)
                    vibrato_depth = (
                        np.std(f0_detrended) / np.mean(f0_contour) if np.mean(f0_contour) > 0 else 0
                    )
                else:
                    vibrato_rate_hz = 0
                    vibrato_depth = 0

                # Jitter: random variation (CV of F0)
                jitter = np.std(f0_contour) / np.mean(f0_contour) if np.mean(f0_contour) > 0 else 0
            else:
                vibrato_rate_hz = 0
                vibrato_depth = 0
                jitter = 0
        except:
            vibrato_rate_hz = 0
            vibrato_depth = 0
            jitter = 0

        # ===== CATEGORY 3: FINGERPRINT FACTORS =====
        # MFCCs (1-4) - Spectral fingerprint
        try:
            # Compute mel-frequency cepstral coefficients
            # Use librosa if available, otherwise manual computation
            n_mfcc = 13
            n_fft = min(2048, len(self.data))
            n_fft // 4

            # Power spectrum
            power_spectrum = spectrum[: n_fft // 2 + 1] ** 2

            # Mel filterbank (simplified)
            n_mels = 40
            mel_filters = np.zeros((n_mels, len(power_spectrum)))

            # Create mel-scale filterbank
            mel_min = 0
            mel_max = 1127 * np.log(1 + self.sample_rate / 700)  # Mel scale formula
            mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
            bin_points = np.floor((n_fft + 1) * mel_points / mel_max).astype(int)

            for i in range(n_mels):
                left, right = bin_points[i], bin_points[i + 1]
                mel_filters[i, left:right] = np.ones(right - left)
                mel_filters[i, left:right] /= right - left

            # Apply mel filters
            mel_spectrum = np.dot(mel_filters, power_spectrum)
            log_mel_spectrum = np.log(mel_spectrum + 1e-10)

            # DCT to get MFCCs
            from scipy.fft import dct

            mfccs = dct(log_mel_spectrum, type=2, norm="ortho")[:n_mfcc]

            # Store first 4 MFCCs
            mfcc_1 = mfccs[0] if len(mfccs) > 0 else 0
            mfcc_2 = mfccs[1] if len(mfccs) > 1 else 0
            mfcc_3 = mfccs[2] if len(mfccs) > 2 else 0
            mfcc_4 = mfccs[3] if len(mfccs) > 3 else 0

            # MFCC Delta (temporal derivative)
            mfcc_delta = np.diff(mfccs[:5], axis=0) if len(mfccs) >= 5 else np.zeros(4)
            mfcc_delta_mean = np.mean(np.abs(mfcc_delta)) if len(mfcc_delta) > 0 else 0

        except:
            mfcc_1 = 0
            mfcc_2 = 0
            mfcc_3 = 0
            mfcc_4 = 0
            mfcc_delta_mean = 0

        # Spectral Contrast
        try:
            # Divide spectrum into subbands
            n_subbands = 6
            subband_size = len(spectrum) // n_subbands

            spectral_contrast = []
            for i in range(n_subbands):
                start = i * subband_size
                end = start + subband_size if i < n_subbands - 1 else len(spectrum)
                subband = spectrum[start:end]

                if len(subband) > 0:
                    peak_val = np.max(subband)
                    valley_val = np.min(subband)
                    contrast = peak_val - valley_val if valley_val > 0 else peak_val
                    spectral_contrast.append(contrast)

            spectral_contrast_mean = np.mean(spectral_contrast) if spectral_contrast else 0
        except:
            spectral_contrast_mean = 0

        # ===== CATEGORY 4: RHYTHM FACTORS =====
        # Inter-Click Interval (ICI) / Onset Rate
        try:
            # Detect onsets in envelope
            from scipy.signal import find_peaks

            onset_threshold = np.mean(envelope) + 0.5 * np.std(envelope)
            min_distance = int(0.005 * self.sample_rate)  # 5ms minimum

            peaks, _ = find_peaks(envelope, height=onset_threshold, distance=min_distance)

            if len(peaks) > 1:
                # Calculate inter-click intervals
                icis_samples = np.diff(peaks)
                icis_ms = icis_samples / self.sample_rate * 1000

                median_ici_ms = np.median(icis_ms)
                onset_rate_hz = len(peaks) / (len(self.data) / self.sample_rate)
                ici_cv = np.std(icis_ms) / median_ici_ms if median_ici_ms > 0 else 0
            else:
                median_ici_ms = 0
                onset_rate_hz = 0
                ici_cv = 0
        except:
            median_ici_ms = 0
            onset_rate_hz = 0
            ici_cv = 0

        # Amplitude statistics
        amplitude = np.abs(self.data)

        return {
            # ===== BASIC FEATURES =====
            "duration_ms": duration,
            "spectral_flatness": spectral_flatness,
            "mean_amplitude": np.mean(amplitude),
            "max_amplitude": np.max(amplitude),
            "dynamic_range": np.max(amplitude) - np.min(amplitude),
            # ===== TIMBRE FEATURES =====
            "spectral_centroid_hz": spectral_centroid_hz,
            "spectral_slope": spectral_slope,
            "spectral_bandwidth_hz": spectral_bandwidth,
            "spectral_rolloff_hz": spectral_rolloff,
            # ===== CATEGORY 1: GRIT FACTORS (HARMONIC CLARITY) =====
            "harmonic_to_noise_ratio": harmonic_to_noise_ratio,
            # ===== CATEGORY 2: MOTION FACTORS (TEMPORAL DYNAMICS) =====
            "attack_time_ms": attack_time_ms,
            "decay_time_ms": decay_time_ms,
            "sustain_level": sustain_level,
            "vibrato_rate_hz": vibrato_rate_hz,
            "vibrato_depth": vibrato_depth,
            "jitter": jitter,
            # ===== CATEGORY 3: FINGERPRINT FACTORS =====
            "mfcc_1": mfcc_1,
            "mfcc_2": mfcc_2,
            "mfcc_3": mfcc_3,
            "mfcc_4": mfcc_4,
            "mfcc_delta_mean": mfcc_delta_mean,
            "spectral_contrast": spectral_contrast_mean,
            # ===== CATEGORY 4: RHYTHM FACTORS =====
            "median_ici_ms": median_ici_ms,
            "onset_rate_hz": onset_rate_hz,
            "ici_coefficient_of_variation": ici_cv,
        }

    def _estimate_instantaneous_frequency(self) -> np.ndarray:
        """Estimate instantaneous frequency using phase differentiation."""
        if len(self.data) < 3:
            return np.array([])

        # Compute analytic signal
        analytic = signal.hilbert(self.data)

        # Differentiate phase to get instantaneous frequency
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) / (2 * np.pi) * 48000

        return inst_freq

    def distance_to(self, other: "PhraseSignature") -> float:
        """
        Calculate distance to another phrase signature.

        Uses normalized Euclidean distance on feature space.
        """
        if self.modality != other.modality:
            return float("inf")  # Different modalities are maximally distant

        # Get common features
        common_features = set(self.features.keys()) & set(other.features.keys())
        if not common_features:
            return float("inf")

        # Extract feature values, ensuring they're numeric
        self_vals = []
        other_vals = []

        for f in common_features:
            self_val = self.features[f]
            other_val = other.features[f]

            # Convert to float if numeric, skip if string
            if isinstance(self_val, (int, float)) and isinstance(other_val, (int, float)):
                self_vals.append(float(self_val))
                other_vals.append(float(other_val))

        if not self_vals:
            return float("inf")

        self_vals = np.array(self_vals)
        other_vals = np.array(other_vals)

        # Normalize by feature ranges to avoid scale bias
        ranges = np.maximum(np.abs(self_vals), np.abs(other_vals))
        ranges[ranges == 0] = 1  # Avoid division by zero

        normalized_diff = (self_vals - other_vals) / ranges

        return np.sqrt(np.sum(normalized_diff**2))

    def __repr__(self) -> str:
        return f"PhraseSignature(modality={self.modality.name}, duration={self.features.get('duration_ms', 0):.1f}ms)"


class Sentence:
    """
    Represents a single vocalization/recording event containing phrases.

    In the METHODOLOGY_SUMMARY.md, a "sentence" is an individual vocalization
    that contains multiple "atomic phrases".
    """

    def __init__(self, sentence_id: int, audio: np.ndarray, timestamp: float, sample_rate: int):
        self.sentence_id = sentence_id
        self.audio = audio
        self.timestamp = timestamp
        self.sample_rate = sample_rate
        self.phrases = []
        self.atomic_units = {}  # Discovered atomic units (vocabulary)
        self.logger = logging.getLogger(f"Sentence_{sentence_id}")

    def add_phrase(self, phrase: PhraseSignature):
        """Add a phrase to this sentence."""
        self.phrases.append(phrase)

    def discover_atomic_units(
        self,
        f0_bin_size: float = 200.0,
        duration_bin_size: float = 25.0,
        range_bin_size: float = 100.0,
    ) -> Dict[str, List[PhraseSignature]]:
        """
        Discover atomic units within this sentence using feature binning.

        Implements the METHODOLOGY_SUMMARY.md binning approach:
        - Bin F0, duration, and range features
        - Create phrase keys like "F0_6400_DUR_50_RANGE_0"
        - Group phrases with similar features

        Args:
            f0_bin_size: Size of F0 bins (Hz)
            duration_bin_size: Size of duration bins (ms)
            range_bin_size: Size of range bins (Hz)

        Returns:
            Dictionary mapping phrase keys to lists of phrases
        """
        vocabulary = defaultdict(list)

        for phrase in self.phrases:
            # Extract features
            f0_mean = phrase.features.get("f0_mean", 0)
            duration_ms = phrase.features.get("duration_ms", 0)
            f0_range = phrase.features.get("f0_range", 0)

            # Apply binning (METHODOLOGY_SUMMARY.md approach)
            f0_bin = round(f0_mean / f0_bin_size) * f0_bin_size
            dur_bin = round(duration_ms / duration_bin_size) * duration_bin_size
            range_bin = round(f0_range / range_bin_size) * range_bin_size

            # Create phrase key (METHODOLOGY_SUMMARY.md format)
            phrase_key = f"F0_{f0_bin:.0f}_DUR_{dur_bin:.0f}_RANGE_{range_bin:.0f}"

            # Add to vocabulary
            vocabulary[phrase_key].append(phrase)

        # Filter out single-instance phrases (rare combinations)
        filtered_vocabulary = {k: v for k, v in vocabulary.items() if len(v) > 1}

        self.atomic_units = filtered_vocabulary
        self.logger.info(
            f"Discovered {len(filtered_vocabulary)} atomic units in sentence {self.sentence_id}"
        )
        return filtered_vocabulary

    def microharmonic_similarity(
        self, phrase1: PhraseSignature, phrase2: PhraseSignature, similarity_threshold: float = 0.7
    ) -> float:
        """
        Calculate microharmonic similarity between two phrases.

        This confirms atomicity by checking if phrases belong to the same
        harmonic category based on their F0 relationship.

        Args:
            phrase1: First phrase
            phrase2: Second phrase
            similarity_threshold: Threshold for harmonic similarity

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Only compare harmonic phrases
        if phrase1.modality != Modality.HARMONIC or phrase2.modality != Modality.HARMONIC:
            return 0.0

        f1 = phrase1.features.get("f0_mean", 0)
        f2 = phrase2.features.get("f0_mean", 0)

        if f1 == 0 or f2 == 0:
            return 0.0

        # Check for harmonic relationship (octave, fifth, etc.)
        # Check both ratios (f2/f1 and f1/f2)
        ratio1 = f2 / f1 if f1 > 0 else 0
        ratio2 = f1 / f2 if f2 > 0 else 0

        # Perfect unison (1:1)
        if abs(ratio1 - 1.0) < 0.05 or abs(ratio2 - 1.0) < 0.05:
            return 1.0
        # Octave (2:1 or 1:2)
        elif abs(ratio1 - 2.0) < 0.1 or abs(ratio2 - 2.0) < 0.1:
            return 0.9
        # Fifth (3:2 or 2:3)
        elif abs(ratio1 - 1.5) < 0.1 or abs(ratio2 - 1.5) < 0.1:
            return 0.8
        # Fourth (4:3 or 3:4)
        elif abs(ratio1 - 1.33) < 0.1 or abs(ratio2 - 1.33) < 0.1:
            return 0.7
        # Minor third (6:5 or 5:6)
        elif abs(ratio1 - 1.2) < 0.1 or abs(ratio2 - 1.2) < 0.1:
            return 0.6
        # Major third (5:4 or 4:5)
        elif abs(ratio1 - 1.25) < 0.1 or abs(ratio2 - 1.25) < 0.1:
            return 0.65

        # General harmonic proximity
        harmonic_diff = abs(f1 - f2) / min(f1, f2)
        if harmonic_diff < 0.1:
            return 0.8
        elif harmonic_diff < 0.2:
            return 0.6
        elif harmonic_diff < 0.3:
            return 0.4

        return 0.0

    def validate_atomic_units(
        self, similarity_threshold: float = 0.7
    ) -> Dict[str, List[PhraseSignature]]:
        """
        Validate atomic units using microharmonic similarity.

        Args:
            similarity_threshold: Threshold for similarity validation

        Returns:
            Validated atomic units
        """
        validated_units = {}

        for key, phrases in self.atomic_units.items():
            # Group phrases by microharmonic similarity
            validated_group = []

            for phrase in phrases:
                # Find most similar phrase in current group
                max_similarity = 0.0

                for existing_phrase in validated_group:
                    similarity = self.microharmonic_similarity(
                        phrase, existing_phrase, similarity_threshold
                    )
                    if similarity > max_similarity:
                        max_similarity = similarity

                # Add to group if sufficiently similar
                if max_similarity >= similarity_threshold or not validated_group:
                    validated_group.append(phrase)

            validated_units[key] = validated_group

        return validated_units


class UniversalRosettaStone:
    """
    Universal Rosetta Stone - Species-Agnostic Analysis Engine.

    This system analyzes animal vocalizations without species-specific knowledge,
    focusing instead on physics-based modality detection and universal pattern discovery.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.vocabulary: Dict[int, PhraseSignature] = {}
        self.grammar: Counter = Counter()
        self.clustering_model = None
        self.feature_scaler = MinMaxScaler()
        self.logger = logging.getLogger(__name__)

        # Configure logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def detect_modality(self, audio: np.ndarray) -> Modality:
        """
        Detect acoustic modality using physics-based analysis.

        Args:
            audio: Audio signal (1D numpy array)

        Returns:
            Detected modality type
        """
        if len(audio) < 256:
            raise ValueError("Audio must be at least 256 samples (5.3ms at 48kHz)")

        # Calculate physics-based features
        features = self._extract_modality_features(audio)

        # Apply threshold-based classification
        if self._is_harmonic(features):
            return Modality.HARMONIC
        elif self._is_fm_sweep(features):
            return Modality.FM_SWEEP
        elif self._is_transient(features):
            return Modality.TRANSIENT
        elif self._is_rhythmic(features):
            return Modality.RHYTHMIC
        else:
            # Default to TRANSIENT if unclear
            return Modality.TRANSIENT

    def _extract_modality_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract features for modality classification."""
        features = {}

        # Zero-crossing rate
        signs = np.sign(audio)
        crossings = np.where(np.diff(signs))[0]
        zcr = len(crossings) / len(audio)
        features["zcr"] = zcr

        # Spectral flatness
        spectrum = np.abs(np.fft.rfft(audio))
        spectrum = spectrum[spectrum > 1e-10]
        if len(spectrum) > 0:
            features["spectral_flatness"] = np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum)
        else:
            features["spectral_flatness"] = 1.0

        # Energy distribution
        envelope = np.abs(signal.hilbert(audio))
        features["envelope_std"] = np.std(envelope)
        features["envelope_mean"] = np.mean(envelope)
        features["envelope_cv"] = features["envelope_std"] / (features["envelope_mean"] + 1e-10)

        # Timbre features (Category 1, Item 1: Spectral Centroid & Slope)
        freqs = np.fft.rfftfreq(len(audio), 1 / self.sample_rate)
        freqs = freqs[: len(spectrum)]  # Ensure matching lengths

        # Spectral centroid (already computed, but store in Hz)
        spectral_centroid_hz = (
            np.sum(freqs * spectrum) / np.sum(spectrum) if np.sum(spectrum) > 0 else 0
        )
        features["spectral_centroid"] = spectral_centroid_hz
        features["spectral_centroid_hz"] = spectral_centroid_hz

        # Spectral bandwidth (spread around centroid)
        if np.sum(spectrum) > 0:
            spectral_bandwidth = np.sqrt(
                np.sum(((freqs - spectral_centroid_hz) ** 2) * spectrum) / np.sum(spectrum)
            )
            features["spectral_bandwidth_hz"] = spectral_bandwidth
        else:
            features["spectral_bandwidth_hz"] = 0.0

        # Spectral slope (linear regression fit to log spectrum)
        # Positive slope = brighter, negative slope = darker/muffled
        if len(spectrum) > 1:
            log_spectrum = np.log(spectrum + 1e-10)
            # Simple linear regression: slope = covariance(freq, log_spectrum) / variance(freq)
            freq_mean = np.mean(freqs)
            log_spectrum_mean = np.mean(log_spectrum)
            numerator = np.sum((freqs - freq_mean) * (log_spectrum - log_spectrum_mean))
            denominator = np.sum((freqs - freq_mean) ** 2)
            spectral_slope = numerator / denominator if denominator > 0 else 0
            features["spectral_slope"] = spectral_slope
        else:
            features["spectral_slope"] = 0.0

        # Spectral rolloff (frequency below which 85% of energy is contained)
        if np.sum(spectrum) > 0:
            cumulative_energy = np.cumsum(spectrum)
            total_energy = cumulative_energy[-1]
            rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
            if len(rolloff_idx) > 0:
                spectral_rolloff = freqs[rolloff_idx[0]]
                features["spectral_rolloff_hz"] = spectral_rolloff
            else:
                features["spectral_rolloff_hz"] = freqs[-1]
        else:
            features["spectral_rolloff_hz"] = 0.0

        # Peak detection
        peaks, _ = signal.find_peaks(envelope, height=np.mean(envelope))
        features["num_peaks"] = len(peaks)
        features["peak_density"] = len(peaks) / len(audio)

        return features

    def _is_harmonic(self, features: Dict[str, float]) -> bool:
        """
        Check if features indicate harmonic signal with frequency-aware thresholds.

        High-frequency harmonic signals (like marmoset calls at 5-12 kHz) naturally have
        higher zero-crossing rates than low-frequency signals. We adjust the ZCR threshold
        based on the spectral centroid to account for this.
        """
        # Determine frequency range based on spectral centroid
        spectral_centroid = features.get("spectral_centroid", 0)

        # Frequency-aware ZCR threshold
        # Low frequency (< 2 kHz): strict threshold
        # Mid frequency (2-5 kHz): moderate threshold
        # High frequency (5-12 kHz): relaxed threshold (for marmosets, etc.)
        # Very high frequency (> 12 kHz): most relaxed threshold
        if spectral_centroid < 2000:
            zcr_threshold = 0.1  # Strict for low frequencies
        elif spectral_centroid < 5000:
            zcr_threshold = 0.15  # Moderate for mid frequencies
        elif spectral_centroid < 12000:
            zcr_threshold = 0.25  # Relaxed for marmoset range (5-12 kHz)
        else:
            zcr_threshold = 0.35  # Most relaxed for very high frequencies

        # Low zero-crossing rate (stable pitch) - frequency-aware
        # Low spectral flatness indicates harmonic structure
        # Stable envelope (low coefficient of variation)
        return (
            features["zcr"] < zcr_threshold
            and features["spectral_flatness"] < 0.3
            and features["envelope_cv"] < 0.5
        )

    def _is_fm_sweep(self, features: Dict[str, float]) -> bool:
        """
        Check if features indicate FM sweep signal with frequency-aware thresholds.

        FM sweeps have ZCR above the harmonic threshold (which varies by frequency)
        but not extremely high (which would indicate noise or transients).
        """
        # Calculate frequency-aware ZCR threshold (same as _is_harmonic)
        spectral_centroid = features.get("spectral_centroid", 0)
        if spectral_centroid < 2000:
            zcr_min = 0.1  # Above low-frequency harmonic threshold
        elif spectral_centroid < 5000:
            zcr_min = 0.15  # Above mid-frequency harmonic threshold
        elif spectral_centroid < 12000:
            zcr_min = 0.25  # Above marmoset-range harmonic threshold
        else:
            zcr_min = 0.35  # Above very-high-frequency harmonic threshold

        # FM sweep: ZCR above harmonic threshold but not extremely high
        # Moderate spectral flatness
        return (
            features["zcr"] > zcr_min
            and features["zcr"] < 0.6
            and features["spectral_flatness"] < 0.6
        )

    def _is_transient(self, features: Dict[str, float]) -> bool:
        """Check if features indicate transient signal."""
        # High peak density and energy variations
        return (
            features["peak_density"] > 0.01
            and features["envelope_cv"] > 1.0
            and features["spectral_flatness"] > 0.5
        )

    def _is_rhythmic(self, features: Dict[str, float]) -> bool:
        """Check if features indicate rhythmic signal."""
        # Regular peak pattern
        return (
            features["peak_density"] > 0.005
            and features["peak_density"] < 0.02
            and features["envelope_cv"] < 0.3
        )

    def get_modality_probabilities(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Get probability scores for each modality (useful for mixed-modality detection).

        Instead of returning a single modality, this method returns probability-like scores
        for each modality type. This allows detection of mixed-modality signals.

        Args:
            audio: Audio signal (1D numpy array)

        Returns:
            Dictionary mapping modality names to probability scores (0-1)
        """
        if len(audio) < 256:
            raise ValueError("Audio must be at least 256 samples (5.3ms at 48kHz)")

        # Calculate physics-based features
        features = self._extract_modality_features(audio)

        # Calculate score for each modality
        scores = {}

        # Determine frequency-aware ZCR threshold (same logic as _is_harmonic)
        spectral_centroid = features.get("spectral_centroid", 0)
        if spectral_centroid < 2000:
            zcr_threshold = 0.1
        elif spectral_centroid < 5000:
            zcr_threshold = 0.15
        elif spectral_centroid < 12000:
            zcr_threshold = 0.25  # Marmoset range
        else:
            zcr_threshold = 0.35

        # Harmonic score: frequency-aware ZCR, low spectral flatness, stable envelope
        harmonic_score = 0.0
        if features["zcr"] < zcr_threshold:
            # Score based on how far below threshold we are
            zcr_margin = (zcr_threshold - features["zcr"]) / zcr_threshold
            harmonic_score += 0.4 * (1 + zcr_margin)  # Bonus for being well below threshold
        if features["spectral_flatness"] < 0.3:
            harmonic_score += 0.4
        if features["envelope_cv"] < 0.5:
            harmonic_score += 0.2
        scores["harmonic"] = min(harmonic_score, 1.0)

        # FM sweep score: moderate ZCR (above harmonic threshold), moderate spectral flatness
        fm_score = 0.0
        # Use frequency-aware lower bound (above harmonic threshold but not too high)
        if zcr_threshold < features["zcr"] < 0.6:
            fm_score += 0.5
        if features["spectral_flatness"] < 0.6:
            fm_score += 0.3
        if features["envelope_cv"] > 0.3:
            fm_score += 0.2  # FM sweeps have varying amplitude
        scores["fm_sweep"] = min(fm_score, 1.0)

        # Transient score: high peak density, high envelope variation, high spectral flatness
        transient_score = 0.0
        if features["peak_density"] > 0.01:
            transient_score += 0.3
        if features["envelope_cv"] > 0.8:
            transient_score += 0.4
        if features["spectral_flatness"] > 0.5:
            transient_score += 0.3
        scores["transient"] = min(transient_score, 1.0)

        # Rhythmic score: moderate peak density, low envelope variation
        rhythmic_score = 0.0
        if 0.005 < features["peak_density"] < 0.02:
            rhythmic_score += 0.5
        if features["envelope_cv"] < 0.3:
            rhythmic_score += 0.3
        if features["num_peaks"] > 3:  # Multiple peaks indicate rhythm
            rhythmic_score += 0.2
        scores["rhythmic"] = min(rhythmic_score, 1.0)

        # Normalize to sum to 1.0
        total = sum(scores.values())
        if total > 0:
            for key in scores:
                scores[key] = scores[key] / total

        return scores

    def _detect_overall_modality(self, audio: np.ndarray) -> Modality:
        """
        Quickly detect the overall modality of audio without detailed analysis.

        This uses a lightweight feature check to determine if the audio is primarily
        TRANSIENT/RHYTHMIC (event-based) or HARMONIC/FM_SWEEP (tone-based).

        Args:
            audio: Input audio signal

        Returns:
            Detected modality (simplified to 4 categories)
        """
        features = self._extract_modality_features(audio)

        # Quick classification using existing methods
        if self._is_harmonic(features):
            return Modality.HARMONIC
        elif self._is_fm_sweep(features):
            return Modality.FM_SWEEP
        elif self._is_transient(features):
            return Modality.TRANSIENT
        elif self._is_rhythmic(features):
            return Modality.RHYTHMIC
        else:
            # Default to TRANSIENT for event-based signals
            return Modality.TRANSIENT

    def _calculate_adaptive_gap_threshold(
        self, audio: np.ndarray, percentile: float = 99.0
    ) -> float:
        """
        Calculate adaptive gap threshold based on inter-event interval distribution.

        For TRANSIENT and RHYTHMIC modalities, this analyzes the envelope to find
        the natural gaps between events (clicks, pulses, beats). The 99th percentile
        of inter-event intervals provides a good threshold for detecting phrase boundaries.

        Args:
            audio: Input audio signal
            percentile: Percentile to use for threshold (default: 99.0)

        Returns:
            Adaptive gap threshold in milliseconds
        """
        # Compute analytic signal (envelope)
        envelope = np.abs(signal.hilbert(audio))

        # Set threshold for event detection (2 SD above mean)
        event_threshold = np.mean(envelope) + 2.0 * np.std(envelope)

        # Find events (peaks above threshold)
        from scipy.signal import find_peaks

        min_interval_samples = int(0.005 * self.sample_rate)  # Minimum 5ms between events
        peaks, _ = find_peaks(envelope, height=event_threshold, distance=min_interval_samples)

        if len(peaks) < 2:
            # Not enough events to calculate intervals, return default
            return 50.0

        # Calculate inter-event intervals in milliseconds
        intervals_samples = np.diff(peaks)
        intervals_ms = intervals_samples / self.sample_rate * 1000.0

        # Use percentile as threshold (99th percentile captures natural gaps)
        adaptive_threshold_ms = np.percentile(intervals_ms, percentile)

        # Clamp to reasonable range [5ms, 500ms]
        adaptive_threshold_ms = max(5.0, min(adaptive_threshold_ms, 500.0))

        return adaptive_threshold_ms

    def segment_phrases(
        self,
        audio: np.ndarray,
        min_gap_ms: float = 50.0,
        min_phrase_duration_ms: float = 20.0,
        use_adaptive_gap: bool = True,
    ) -> List[PhraseSignature]:
        """
        Segment audio into individual phrases using harmonic similarity.

        This method first performs energy-based segmentation to get candidate phrases,
        then merges phrases with high harmonic similarity while preserving gaps
        between dissimilar phrases.

        For TRANSIENT and RHYTHMIC modalities, automatically adapts the gap threshold
        based on inter-event interval distribution when use_adaptive_gap=True.

        Args:
            audio: Input audio signal
            min_gap_ms: Minimum silence gap between phrases (ms) - maximum allowed
            min_phrase_duration_ms: Minimum duration of a phrase (ms)
            use_adaptive_gap: Enable adaptive gap threshold for TRANSIENT/RHYTHMIC

        Returns:
            List of PhraseSignature objects
        """
        # Detect overall modality to determine if adaptive gap should be used
        if use_adaptive_gap:
            overall_modality = self._detect_overall_modality(audio)
            if overall_modality in [Modality.TRANSIENT, Modality.RHYTHMIC]:
                adaptive_gap_ms = self._calculate_adaptive_gap_threshold(audio)
                # Use minimum of adaptive and user-specified gap
                effective_gap_ms = min(adaptive_gap_ms, min_gap_ms)
                self.logger.debug(
                    f"Adaptive gap threshold: {adaptive_gap_ms:.2f}ms "
                    f"(using: {effective_gap_ms:.2f}ms)"
                )
                min_gap_ms = effective_gap_ms

        min_gap_samples = int(min_gap_ms * self.sample_rate / 1000)
        min_duration_samples = int(min_phrase_duration_ms * self.sample_rate / 1000)

        # Step 1: Initial segmentation (choose method based on modality)
        overall_modality = (
            self._detect_overall_modality(audio) if use_adaptive_gap else Modality.HARMONIC
        )

        if overall_modality in [Modality.TRANSIENT, Modality.RHYTHMIC]:
            # Use event-based segmentation for click/pulse-based signals
            phrases = self._event_based_segmentation(audio, min_gap_samples, min_duration_samples)
        else:
            # Use energy-based segmentation for harmonic/FM signals
            phrases = self._energy_based_segmentation(audio, min_duration_samples)

        if len(phrases) <= 1:
            # If we have 0 or 1 phrases, no merging needed
            return phrases

        # Step 2: Harmonic similarity-based merging
        merged_phrases = self._harmonic_similarity_merging(phrases, min_gap_samples)

        self.logger.info(
            f"Segmented {len(merged_phrases)} phrases from {len(audio) / self.sample_rate:.2f}s audio "
            f"(initial: {len(phrases)}, merged: {len(merged_phrases)})"
        )
        return merged_phrases

    def _energy_based_segmentation(
        self, audio: np.ndarray, min_duration_samples: int
    ) -> List[PhraseSignature]:
        """Perform initial energy-based segmentation to get candidate phrases."""
        envelope = np.abs(signal.hilbert(audio))
        energy_threshold = np.median(envelope) * 0.5

        # Find speech/silence regions
        is_speech = envelope > energy_threshold

        # Find phrase boundaries
        speech_onsets = np.where(np.diff(is_speech.astype(int)) == 1)[0]
        speech_offsets = np.where(np.diff(is_speech.astype(int)) == -1)[0]

        # Handle edge cases
        if len(speech_onsets) == 0 and len(speech_offsets) == 0:
            if np.mean(envelope) > energy_threshold:
                speech_onsets = [0]
                speech_offsets = [len(audio) - 1]
            else:
                return []
        elif len(speech_onsets) == 0:
            speech_onsets = [0]
        elif len(speech_offsets) == 0:
            speech_offsets = [len(audio) - 1]

        # Pair onset and offset
        paired_segments = list(zip(speech_onsets, speech_offsets))

        # Create phrase candidates
        phrases = []
        for onset, offset in paired_segments:
            duration = offset - onset + 1
            if duration >= min_duration_samples:
                phrase_data = audio[onset : offset + 1]
                try:
                    modality = self.detect_modality(phrase_data)
                    phrase = PhraseSignature(
                        modality=modality, data=phrase_data, timestamp=onset / self.sample_rate
                    )
                    phrases.append(phrase)
                except ValueError:
                    continue

        return phrases

    def _event_based_segmentation(
        self, audio: np.ndarray, min_gap_samples: int, min_duration_samples: int
    ) -> List[PhraseSignature]:
        """
        Perform event-based segmentation for TRANSIENT/RHYTHMIC signals.

        This method detects individual events (clicks, pulses) and groups them into
        phrases based on inter-event intervals. Events closer than min_gap_samples
        are grouped into the same phrase.

        Args:
            audio: Input audio signal
            min_gap_samples: Minimum gap between phrases (in samples)
            min_duration_samples: Minimum duration of a phrase (in samples)

        Returns:
            List of PhraseSignature objects
        """
        # Compute envelope
        envelope = np.abs(signal.hilbert(audio))

        # Detect events (peaks in envelope)
        event_threshold = np.mean(envelope) + 2.0 * np.std(envelope)
        from scipy.signal import find_peaks

        min_event_distance = int(0.005 * self.sample_rate)  # Minimum 5ms between events
        peaks, properties = find_peaks(
            envelope, height=event_threshold, distance=min_event_distance, width=10
        )

        if len(peaks) < 1:
            return []

        # Group events into phrases based on gaps
        phrases = []
        current_phrase_events = [peaks[0]]

        for i in range(1, len(peaks)):
            gap_samples = peaks[i] - peaks[i - 1]

            if gap_samples <= min_gap_samples:
                # Part of same phrase
                current_phrase_events.append(peaks[i])
            else:
                # Gap too large - finalize current phrase and start new one
                phrase_start = current_phrase_events[0]
                phrase_end = current_phrase_events[-1]

                # Add padding around events
                padding = int(0.010 * self.sample_rate)  # 10ms padding
                phrase_start = max(0, phrase_start - padding)
                phrase_end = min(len(audio), phrase_end + padding)

                phrase_duration = phrase_end - phrase_start

                if phrase_duration >= min_duration_samples:
                    phrase_data = audio[phrase_start:phrase_end]
                    try:
                        modality = self.detect_modality(phrase_data)
                        phrase = PhraseSignature(
                            modality=modality,
                            data=phrase_data,
                            timestamp=phrase_start / self.sample_rate,
                        )
                        phrases.append(phrase)
                    except ValueError:
                        pass

                # Start new phrase
                current_phrase_events = [peaks[i]]

        # Don't forget the last phrase
        if len(current_phrase_events) >= 1:
            phrase_start = current_phrase_events[0]
            phrase_end = current_phrase_events[-1]

            padding = int(0.010 * self.sample_rate)
            phrase_start = max(0, phrase_start - padding)
            phrase_end = min(len(audio), phrase_end + padding)

            phrase_duration = phrase_end - phrase_start

            if phrase_duration >= min_duration_samples:
                phrase_data = audio[phrase_start:phrase_end]
                try:
                    modality = self.detect_modality(phrase_data)
                    phrase = PhraseSignature(
                        modality=modality,
                        data=phrase_data,
                        timestamp=phrase_start / self.sample_rate,
                    )
                    phrases.append(phrase)
                except ValueError:
                    pass

        return phrases

    def _harmonic_similarity_merging(
        self, phrases: List[PhraseSignature], min_gap_samples: int
    ) -> List[PhraseSignature]:
        """Merge phrases with high harmonic similarity while preserving gaps."""
        if len(phrases) <= 1:
            return phrases

        merged_phrases = []
        current_group = [phrases[0]]

        for i in range(1, len(phrases)):
            prev_phrase = phrases[i - 1]
            curr_phrase = phrases[i]

            # Check if phrases should be merged based on harmonic similarity
            gap_samples = int(curr_phrase.timestamp * self.sample_rate) - int(
                prev_phrase.timestamp * self.sample_rate + prev_phrase.data.shape[0]
            )

            # Only merge if gap is small AND harmonic similarity is high
            if gap_samples <= min_gap_samples and (
                prev_phrase.modality == Modality.HARMONIC
                and curr_phrase.modality == Modality.HARMONIC
            ):
                # Calculate harmonic similarity
                similarity = self._calculate_harmonic_similarity(prev_phrase, curr_phrase)

                if similarity >= 0.7:  # High similarity threshold
                    current_group.append(curr_phrase)
                    continue

            # If we shouldn't merge, finalize the current group
            if len(current_group) == 1:
                merged_phrases.append(current_group[0])
            else:
                # Merge the group into a single phrase
                merged_phrase = self._merge_phrase_group(current_group)
                merged_phrases.append(merged_phrase)

            # Start new group
            current_group = [curr_phrase]

        # Don't forget the last group
        if len(current_group) == 1:
            merged_phrases.append(current_group[0])
        else:
            merged_phrase = self._merge_phrase_group(current_group)
            merged_phrases.append(merged_phrase)

        return merged_phrases

    def _calculate_harmonic_similarity(
        self, phrase1: PhraseSignature, phrase2: PhraseSignature
    ) -> float:
        """Calculate harmonic similarity between two phrases."""
        # Only compare harmonic phrases
        if phrase1.modality != Modality.HARMONIC or phrase2.modality != Modality.HARMONIC:
            return 0.0

        f1 = phrase1.features.get("f0_mean", 0)
        f2 = phrase2.features.get("f0_mean", 0)

        if f1 == 0 or f2 == 0:
            return 0.0

        # Check for harmonic relationship
        ratio = f2 / f1 if f1 > f2 else f1 / f2

        # Perfect unison (1:1)
        if abs(ratio - 1.0) < 0.05:
            return 1.0
        # Octave (2:1)
        elif abs(ratio - 2.0) < 0.1:
            return 0.9
        # Fifth (3:2)
        elif abs(ratio - 1.5) < 0.1:
            return 0.8
        # Fourth (4:3)
        elif abs(ratio - 1.33) < 0.1:
            return 0.7
        # Minor third (6:5)
        elif abs(ratio - 1.2) < 0.1:
            return 0.6
        # Major third (5:4)
        elif abs(ratio - 1.25) < 0.1:
            return 0.65

        # General harmonic proximity
        harmonic_diff = abs(f1 - f2) / min(f1, f2)
        return max(0.0, 1.0 - harmonic_diff)

    def _merge_phrase_group(self, phrase_group: List[PhraseSignature]) -> PhraseSignature:
        """Merge a group of similar phrases into a single phrase."""
        if len(phrase_group) == 1:
            return phrase_group[0]

        # Concatenate audio data
        merged_audio = np.concatenate([phrase.data for phrase in phrase_group])

        # Keep the modality of the first phrase (they should all be the same)
        modality = phrase_group[0].modality

        # Calculate average timestamp
        avg_timestamp = np.mean([phrase.timestamp for phrase in phrase_group])

        # Create merged phrase
        merged_phrase = PhraseSignature(
            modality=modality, data=merged_audio, timestamp=avg_timestamp
        )

        # Calculate average features
        features = {}
        for feature_name in ["f0_mean", "f0_std", "duration_ms", "f0_range"]:
            values = [phrase.features.get(feature_name, 0) for phrase in phrase_group]
            features[feature_name] = np.mean(values)

        merged_phrase.features = features

        return merged_phrase

    def build_vocabulary(
        self, phrases: List[PhraseSignature], eps: float = 0.3, min_samples: int = 2
    ) -> Dict[int, List[PhraseSignature]]:
        """
        Build vocabulary by clustering similar phrases.

        Args:
            phrases: List of phrase signatures
            eps: DBSCAN epsilon parameter (maximum feature distance)
            min_samples: Minimum phrases to form a cluster

        Returns:
            Dictionary mapping cluster IDs to lists of phrases
        """
        if len(phrases) < min_samples:
            self.logger.warning(f"Insufficient phrases ({len(phrases)}) for clustering")
            return {}

        # Group by modality first
        modality_groups = defaultdict(list)
        for phrase in phrases:
            modality_groups[phrase.modality].append(phrase)

        vocabulary_clusters = {}

        # Cluster within each modality group
        cluster_id = 0
        for modality, modality_phrases in modality_groups.items():
            if len(modality_phrases) < min_samples:
                continue

            # Extract feature matrix
            feature_names = list(modality_phrases[0].features.keys())
            feature_matrix = np.array(
                [[phrase.features[name] for name in feature_names] for phrase in modality_phrases]
            )

            # Handle NaN/inf values
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize features
            if len(feature_matrix) > 1:
                normalized_features = self.feature_scaler.fit_transform(feature_matrix)
            else:
                normalized_features = feature_matrix

            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(normalized_features)

            # Store clusters
            for i, label in enumerate(clustering.labels_):
                if label == -1:  # Noise
                    continue

                if label not in vocabulary_clusters:
                    vocabulary_clusters[label] = []
                    vocabulary_clusters[label].append((cluster_id, modality_phrases[i]))
                    cluster_id += 1
                else:
                    vocabulary_clusters[label].append((cluster_id, modality_phrases[i]))
                    cluster_id += 1

        # Update vocabulary with cluster representatives
        self.vocabulary = {}
        for cluster_label, cluster_phrases in vocabulary_clusters.items():
            # Use the first phrase as representative
            cluster_id, phrase = cluster_phrases[0]
            self.vocabulary[cluster_id] = phrase

        self.logger.info(f"Built vocabulary with {len(self.vocabulary)} unique phrases")
        return vocabulary_clusters

    def compute_cluster_persona_score(
        self, phrases: List[PhraseSignature], persona_name: str
    ) -> float:
        """
        Compute the average persona match score for a cluster of phrases.

        Tier 2 of hybrid architecture: Post-hoc persona mapping for semantic interpretability.

        Args:
            phrases: List of phrases in the cluster
            persona_name: Name of the acoustic persona ('gritty', 'pure', etc.)

        Returns:
            Average persona score between 0.0 (no match) and 1.0 (perfect match)
        """
        if not HAS_PERSONA_SUPPORT:
            self.logger.warning("Persona support not available")
            return 0.0

        if persona_name not in ACOUSTIC_PERSONAS:
            self.logger.warning(f"Unknown persona: {persona_name}")
            return 0.0

        if not phrases:
            return 0.0

        persona = ACOUSTIC_PERSONAS[persona_name]
        scores = []

        for phrase in phrases:
            # Convert phrase features to persona-compatible format
            features = dict(phrase.features)
            score = compute_persona_score(features, persona)
            if score > 0:
                scores.append(score)

        if not scores:
            return 0.0

        return np.mean(scores)

    def build_vocabulary_with_personas(
        self,
        phrases: List[PhraseSignature],
        eps: float = 0.3,
        min_samples: int = 2,
        enable_persona_mapping: bool = True,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Build vocabulary using hybrid architecture: DBSCAN + persona mapping.

        Tier 1: Unsupervised DBSCAN clustering (data-driven discovery)
        Tier 2: Acoustic persona mapping (semantic interpretation)
        Tier 3: Contextual validation (deferred to external validation)

        Args:
            phrases: List of phrase signatures
            eps: DBSCAN epsilon parameter
            min_samples: Minimum phrases to form a cluster
            enable_persona_mapping: Whether to enable Tier 2 persona mapping

        Returns:
            Dictionary mapping cluster IDs to cluster metadata:
            {
                cluster_id: {
                    'phrases': [phrase1, phrase2, ...],
                    'dominant_persona': 'pure' | 'gritty' | ... | 'unclassified',
                    'persona_scores': {'gritty': 0.2, 'pure': 0.8, ...},
                    'cluster_size': int,
                    'mean_features': {...}
                }
            }
        """
        if not HAS_PERSONA_SUPPORT:
            self.logger.warning("Persona support not available, using basic clustering")
            enable_persona_mapping = False

        # Tier 1: Unsupervised DBSCAN clustering
        vocabulary_clusters = self.build_vocabulary(phrases, eps, min_samples)

        # Transform into hybrid format
        hybrid_clusters = {}

        for cluster_label, cluster_data in vocabulary_clusters.items():
            # Extract phrases from cluster_data
            phrase_list = [item[1] for item in cluster_data]

            cluster_id = cluster_label
            cluster_size = len(phrase_list)

            # Compute mean features for the cluster
            feature_names = list(phrase_list[0].features.keys()) if phrase_list else []
            mean_features = {}
            for fname in feature_names:
                values = [p.features.get(fname, 0) for p in phrase_list]
                mean_features[fname] = np.mean(values) if values else 0

            # Tier 2: Acoustic persona mapping (post-hoc)
            persona_scores = {}
            dominant_persona = "unclassified"
            dominant_score = 0.0

            if enable_persona_mapping and HAS_PERSONA_SUPPORT:
                for persona_name in ACOUSTIC_PERSONAS.keys():
                    score = self.compute_cluster_persona_score(phrase_list, persona_name)
                    persona_scores[persona_name] = score

                    if score > dominant_score:
                        dominant_score = score
                        dominant_persona = persona_name

                # Only assign persona if score exceeds threshold
                # Lower threshold (0.15) for limited feature sets
                if dominant_score < 0.15:
                    dominant_persona = "unclassified"

            hybrid_clusters[cluster_id] = {
                "phrases": phrase_list,
                "dominant_persona": dominant_persona,
                "persona_scores": persona_scores,
                "cluster_size": cluster_size,
                "mean_features": mean_features,
            }

        self.logger.info(
            f"Built hybrid vocabulary with {len(hybrid_clusters)} clusters "
            f"(persona mapping: {'enabled' if enable_persona_mapping else 'disabled'})"
        )

        return hybrid_clusters

    def find_phrases_by_persona(
        self, clusters: Dict[int, Dict[str, Any]], persona_name: str, min_score: float = 0.3
    ) -> List[Tuple[int, List[PhraseSignature], float]]:
        """
        Find vocabulary clusters matching a specific acoustic persona.

        Enables semantic phrase search: "Find all aggressive alert phrases" -> persona='gritty'

        Args:
            clusters: Hybrid vocabulary from build_vocabulary_with_personas()
            persona_name: Acoustic persona to search for
            min_score: Minimum persona score threshold

        Returns:
            List of (cluster_id, phrases, score) tuples matching the persona
        """
        if not HAS_PERSONA_SUPPORT:
            self.logger.warning("Persona support not available")
            return []

        matches = []

        for cluster_id, cluster_data in clusters.items():
            cluster_data.get("dominant_persona", "unclassified")
            persona_scores = cluster_data.get("persona_scores", {})

            # Check if cluster matches the requested persona
            score = persona_scores.get(persona_name, 0.0)

            if score >= min_score:
                phrases = cluster_data["phrases"]
                matches.append((cluster_id, phrases, score))

        # Sort by score (descending)
        matches.sort(key=lambda x: x[2], reverse=True)

        self.logger.info(
            f"Found {len(matches)} clusters matching persona '{persona_name}' "
            f"(min_score: {min_score})"
        )

        return matches

    def get_persona_summary(self, clusters: Dict[int, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Generate a summary of persona distribution across the vocabulary.

        Useful for understanding the semantic composition of a species' vocalizations.

        Args:
            clusters: Hybrid vocabulary from build_vocabulary_with_personas()

        Returns:
            Dictionary summarizing each persona:
            {
                'pure': {'cluster_count': 5, 'total_phrases': 42, 'avg_score': 0.75},
                'gritty': {'cluster_count': 3, 'total_phrases': 18, 'avg_score': 0.62},
                ...
            }
        """
        if not HAS_PERSONA_SUPPORT:
            return {}

        summary = {}

        # Initialize summary for all personas
        for persona_name in ACOUSTIC_PERSONAS.keys():
            summary[persona_name] = {"cluster_count": 0, "total_phrases": 0, "scores": []}
        summary["unclassified"] = {"cluster_count": 0, "total_phrases": 0, "scores": []}

        # Aggregate cluster data
        for cluster_data in clusters.values():
            dominant = cluster_data.get("dominant_persona", "unclassified")
            cluster_size = cluster_data.get("cluster_size", 0)
            persona_scores = cluster_data.get("persona_scores", {})

            if dominant in summary:
                summary[dominant]["cluster_count"] += 1
                summary[dominant]["total_phrases"] += cluster_size

                # Track scores for averaging
                if dominant in persona_scores:
                    summary[dominant]["scores"].append(persona_scores[dominant])

        # Compute average scores
        for persona_name, data in summary.items():
            if data["scores"]:
                data["avg_score"] = np.mean(data["scores"])
            else:
                data["avg_score"] = 0.0
            del data["scores"]  # Remove raw scores from output

        return summary

    def discover_grammar(
        self, audio: np.ndarray, min_gap_ms: float = 50.0, min_phrase_duration_ms: float = 20.0
    ) -> Tuple[Dict[int, PhraseSignature], Counter]:
        """
        Discover syntactic rules from audio.

        Args:
            audio: Input audio signal
            min_gap_ms: Minimum silence gap between phrases
            min_phrase_duration_ms: Minimum duration of a phrase

        Returns:
            Tuple of (vocabulary, grammar)
        """
        # Segment audio into phrases
        phrases = self.segment_phrases(audio, min_gap_ms, min_phrase_duration_ms)

        # Build vocabulary
        self.build_vocabulary(phrases)

        # If we have phrases but no vocabulary clustering, create individual entries
        if not self.vocabulary and phrases:
            for i, phrase in enumerate(phrases):
                self.vocabulary[i] = phrase

        # Build grammar from phrase sequence
        sequence = []
        for phrase in phrases:
            # Find nearest cluster
            min_distance = float("inf")
            nearest_cluster = None

            for cluster_id, vocab_phrase in self.vocabulary.items():
                distance = phrase.distance_to(vocab_phrase)
                if distance < min_distance:
                    min_distance = distance
                    nearest_cluster = cluster_id

            if nearest_cluster is not None and min_distance < 1.0:  # Threshold for matching
                sequence.append(nearest_cluster)

        # Build transition matrix
        self.grammar = Counter()
        for i in range(len(sequence) - 1):
            transition = (sequence[i], sequence[i + 1])
            self.grammar[transition] += 1

        self.logger.info(f"Discovered grammar with {len(self.grammar)} unique transitions")
        return self.vocabulary, self.grammar

    def get_phrase_statistics(self) -> Dict[str, Any]:
        """Get statistics about discovered phrases."""
        if not self.vocabulary:
            return {}

        stats = {
            "total_phrases": len(self.vocabulary),
            "modality_distribution": defaultdict(int),
            "feature_statistics": defaultdict(dict),
        }

        # Count by modality
        for phrase in self.vocabulary.values():
            stats["modality_distribution"][phrase.modality.name] += 1

        # Feature statistics
        for feature_name in self.vocabulary.values()[0].features.keys():
            values = [phrase.features[feature_name] for phrase in self.vocabulary.values()]
            stats["feature_statistics"][feature_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }

        return dict(stats)

    def _pyin_phrase_segmentation(
        self,
        audio: np.ndarray,
        window_size_ms: float = 30.0,
        hop_size_ms: float = 10.0,
        f0_threshold: float = 50.0,
    ) -> List[PhraseSignature]:
        """
        Segment audio into phrases using sliding window PYIN analysis.

        This method:
        1. Uses sliding window PYIN to get F0 contours
        2. Identifies phrase boundaries based on F0 stability
        3. Groups stable regions into phrases

        Args:
            audio: Input audio signal
            window_size_ms: Size of sliding window for PYIN analysis (ms)
            hop_size_ms: Hop size between windows (ms)
            f0_threshold: Minimum F0 for harmonic detection (Hz)

        Returns:
            List of PhraseSignature objects
        """
        if not HAS_LIBROSA:
            # Fallback to energy-based segmentation if librosa not available
            return self._energy_based_segmentation(
                audio,
                int(20 * self.sample_rate / 1000),  # 20ms min duration
            )

        window_size_samples = int(window_size_ms * self.sample_rate / 1000)
        hop_size_samples = int(hop_size_ms * self.sample_rate / 1000)

        # Pad audio to handle edge cases
        padded_audio = np.pad(
            audio, (window_size_samples // 2, window_size_samples // 2), mode="reflect"
        )

        phrases = []
        current_phrase_start = 0
        current_phrase_f0s = []

        for i in range(0, len(audio) - window_size_samples + 1, hop_size_samples):
            window_start = i + window_size_samples // 2
            window_end = window_start + window_size_samples
            window_audio = padded_audio[window_start:window_end]

            # Use librosa's pyin for F0 estimation
            try:
                f0, voiced_flag, voiced_prob = librosa.pyin(
                    window_audio,
                    fmin=500,  # Lower min for marmosets
                    fmax=12000,  # Higher max for marmosets
                    frame_length=window_size_samples,
                    hop_length=hop_size_samples,
                    win_length=window_size_samples,
                )

                # Use the median F0 of voiced frames
                voiced_f0s = f0[voiced_flag > 0.5]
                if len(voiced_f0s) > 0:
                    current_f0 = np.median(voiced_f0s)
                else:
                    current_f0 = 0

                current_phrase_f0s.append(current_f0)

                # Check for phrase boundary
                if i > 0:  # Not the first window
                    prev_f0 = current_phrase_f0s[-2] if len(current_phrase_f0s) >= 2 else current_f0

                    # Boundary conditions:
                    # 1. F0 drops to zero (end of harmonic phrase)
                    # 2. F0 changes abruptly (> 20% change)
                    # 3. Energy drops significantly
                    if (current_f0 < f0_threshold and prev_f0 >= f0_threshold) or (
                        prev_f0 > f0_threshold
                        and current_f0 > 0
                        and abs(current_f0 - prev_f0) / prev_f0 > 0.2
                    ):
                        # Create phrase from stable region
                        if i - current_phrase_start > window_size_samples:  # Minimum phrase length
                            phrase_data = audio[current_phrase_start:i]

                            # Detect modality
                            try:
                                modality = self.detect_modality(phrase_data)

                                # Calculate phrase features
                                if modality == Modality.HARMONIC and len(current_phrase_f0s) > 1:
                                    f0_mean = np.mean(
                                        [f for f in current_phrase_f0s[:-1] if f > f0_threshold]
                                    )
                                    f0_std = np.std(
                                        [f for f in current_phrase_f0s[:-1] if f > f0_threshold]
                                    )
                                else:
                                    f0_mean = 0
                                    f0_std = 0

                                phrase = PhraseSignature(
                                    modality=modality,
                                    data=phrase_data,
                                    timestamp=current_phrase_start / self.sample_rate,
                                )
                                phrase.features = {
                                    "f0_mean": f0_mean,
                                    "f0_std": f0_std,
                                    "duration_ms": len(phrase_data) / self.sample_rate * 1000,
                                    "f0_range": 0,  # Will be calculated later
                                }

                                phrases.append(phrase)
                            except ValueError:
                                pass

                        # Start new phrase
                        current_phrase_start = i
                        current_phrase_f0s = [current_f0]

            except Exception:
                # If PYIN fails, skip this window
                continue

        # Add the last phrase
        if len(audio) - current_phrase_start > window_size_samples:
            phrase_data = audio[current_phrase_start:]
            try:
                modality = self.detect_modality(phrase_data)
                phrase = PhraseSignature(
                    modality=modality,
                    data=phrase_data,
                    timestamp=current_phrase_start / self.sample_rate,
                )
                phrase.features = {
                    "f0_mean": 0,
                    "f0_std": 0,
                    "duration_ms": len(phrase_data) / self.sample_rate * 1000,
                    "f0_range": 0,
                }
                phrases.append(phrase)
            except ValueError:
                pass

        return phrases

    def detect_sentences_pelt(
        self, audio: np.ndarray, max_sentences: int = 10, penalty: float = 10.0
    ) -> List[int]:
        """
        Detect sentence boundaries using PELT change point detection.

        Args:
            audio: Input audio signal
            max_sentences: Maximum number of sentences to detect
            penalty: Penalty parameter for PELT algorithm

        Returns:
            List of sentence boundary indices
        """
        if not HAS_RUPTURES:
            # Fallback to simple energy-based detection
            envelope = np.abs(signal.hilbert(audio))
            threshold = np.median(envelope) * 2
            boundaries = np.where(envelope > threshold)[0]
            return boundaries[:max_sentences].tolist()

        # Use spectral energy as cost function
        # Compute STFT
        n_fft = 1024
        hop_length = 512
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        spectral_energy = np.sum(np.abs(stft) ** 2, axis=0)

        # PELT change point detection
        algo = rpt.Pelt(model="rbf").fit(spectral_energy)
        change_points = algo.predict(pen=penalty)

        # Convert to sample indices and limit to max_sentences
        if len(change_points) > 1:
            sentence_boundaries = []
            for cp in change_points[1:]:  # Skip first point (start of audio)
                boundary_sample = cp * hop_length
                if boundary_sample < len(audio):
                    sentence_boundaries.append(int(boundary_sample))
                if len(sentence_boundaries) >= max_sentences:
                    break
            return sentence_boundaries

        return []

    def create_sentences_from_vocalizations(
        self, audio_segments: List[np.ndarray], segment_timestamps: List[float]
    ) -> List["Sentence"]:
        """
        Create sentences from individual vocalizations (recordings).

        Each sentence represents a single vocalization event/recording.
        Within each sentence, phrases will be segmented.

        Args:
            audio_segments: List of audio segments (each is a vocalization)
            segment_timestamps: List of timestamps for each segment

        Returns:
            List of Sentence objects
        """
        sentences = []

        for i, (audio, timestamp) in enumerate(zip(audio_segments, segment_timestamps)):
            # Create a sentence for each vocalization
            sentence = Sentence(
                sentence_id=i, audio=audio, timestamp=timestamp, sample_rate=self.sample_rate
            )

            # Segment the vocalization into phrases using PYIN-based approach
            phrases = self._pyin_phrase_segmentation(audio)

            # Add phrases to sentence - their timestamps are already correctly set by segment_phrases
            for phrase in phrases:
                sentence.add_phrase(phrase)

            sentences.append(sentence)

            self.logger.info(f"Created sentence {i} with {len(phrases)} phrases")

        return sentences

    def discover_sentences(
        self, phrases: List["PhraseSignature"], gap_threshold_ms: float = 500.0
    ) -> List[List["PhraseSignature"]]:
        """
        Legacy method - group phrases into sentences based on temporal proximity.
        Kept for backward compatibility.
        """
        # For backward compatibility, use timestamps to group phrases
        if not phrases:
            return []

        # Sort phrases by timestamp
        sorted_phrases = sorted(
            phrases, key=lambda p: p.timestamp if p.timestamp is not None else 0
        )

        sentences = []
        current_sentence = [sorted_phrases[0]]

        for i in range(1, len(sorted_phrases)):
            gap = (
                sorted_phrases[i].timestamp - current_sentence[-1].timestamp
                if sorted_phrases[i].timestamp and current_sentence[-1].timestamp
                else float("inf")
            )

            if gap > gap_threshold_ms:
                if len(current_sentence) >= 2:
                    sentences.append(current_sentence)
                current_sentence = [sorted_phrases[i]]
            else:
                current_sentence.append(sorted_phrases[i])

        if len(current_sentence) >= 2:
            sentences.append(current_sentence)

        return sentences

    def _get_phrase_start_time(self, phrase: "PhraseSignature") -> float:
        """Get the start time of a phrase in samples."""
        return phrase.timestamp if phrase.timestamp is not None else 0

    def _get_phrase_end_time(self, phrase: "PhraseSignature") -> float:
        """Get the end time of a phrase in samples."""
        start_time = self._get_phrase_start_time(phrase)
        duration_samples = int(phrase.features.get("duration_ms", 0) * self.sample_rate / 1000)
        return start_time + duration_samples

    def detect_superposition(
        self,
        phrases: List["PhraseSignature"],
        min_overlap_ratio: float = 0.3,
        same_recording_only: bool = True,
    ) -> List[List[str]]:
        """
        Detect superposition (temporal overlap) between phrases.

        Args:
            phrases: List of phrases to analyze
            min_overlap_ratio: Minimum overlap ratio (0.3 = 30% overlap)
            same_recording_only: Whether to only consider phrases from same recording

        Returns:
            List of superposition groups, where each group contains phrase keys
        """
        superposition_groups = []

        # Group phrases by recording if needed
        if same_recording_only:
            recording_groups = defaultdict(list)
            for phrase in phrases:
                # Use timestamp as pseudo-recording ID
                recording_id = str(phrase.timestamp // 10000)  # Group by 10-second windows
                recording_groups[recording_id].append(phrase)
        else:
            recording_groups = {"all": phrases}

        for recording_id, recording_phrases in recording_groups.items():
            n_phrases = len(recording_phrases)

            for i in range(n_phrases):
                for j in range(i + 1, n_phrases):
                    phrase_a = recording_phrases[i]
                    phrase_b = recording_phrases[j]

                    # Check temporal overlap
                    overlap_start = max(
                        self._get_phrase_start_time(phrase_a), self._get_phrase_start_time(phrase_b)
                    )
                    overlap_end = min(
                        self._get_phrase_end_time(phrase_a), self._get_phrase_end_time(phrase_b)
                    )

                    overlap_duration = max(0, overlap_end - overlap_start)

                    if overlap_duration > 0:
                        # Calculate overlap ratio
                        min_duration = min(
                            phrase_a.features.get("duration_ms", 0),
                            phrase_b.features.get("duration_ms", 0),
                        )
                        overlap_ratio = overlap_duration / (min_duration * self.sample_rate / 1000)

                        if overlap_ratio >= min_overlap_ratio:
                            # Check harmonic compatibility (optional for same modality)
                            if phrase_a.modality == phrase_b.modality:
                                superposition_groups.append(
                                    [phrase_a.__repr__(), phrase_b.__repr__()]
                                )

        self.logger.info(f"Detected {len(superposition_groups)} superposition groups")
        return superposition_groups

    def compute_network_metrics(self, grammar: Counter) -> Dict[str, float]:
        """
        Compute advanced network metrics for phrase transition network.

        Args:
            grammar: Counter of transition probabilities

        Returns:
            Dictionary of network metrics
        """
        if not grammar:
            return {}

        import networkx as nx

        # Build directed graph
        G = nx.DiGraph()
        for (from_phrase, to_phrase), count in grammar.items():
            G.add_edge(from_phrase, to_phrase, weight=count)

        metrics = {}

        if len(G.nodes) > 0:
            try:
                # Basic metrics
                metrics["num_nodes"] = len(G.nodes)
                metrics["num_edges"] = len(G.edges)
                metrics["density"] = nx.density(G)

                # Clustering coefficient
                if len(G.nodes) > 2:
                    undirected_G = G.to_undirected()
                    metrics["avg_clustering"] = nx.average_clustering(undirected_G)

                    # Path length
                    if nx.is_connected(undirected_G):
                        metrics["avg_path_length"] = nx.average_shortest_path_length(undirected_G)
                    else:
                        # Handle disconnected components
                        components = list(nx.connected_components(undirected_G))
                        largest_component = max(components, key=len)
                        largest_subgraph = undirected_G.subgraph(largest_component)
                        metrics["avg_path_length"] = nx.average_shortest_path_length(
                            largest_subgraph
                        )

                    # Small-world coefficient
                    random_graph = nx.gnm_random_graph(len(G.nodes), len(G.edges))
                    random_clustering = nx.average_clustering(random_graph)
                    random_path_length = nx.average_shortest_path_length(
                        random_graph.to_undirected()
                    )

                    if random_path_length > 0:
                        metrics["small_world_sigma"] = (
                            metrics["avg_clustering"] / random_clustering
                        ) * (random_path_length / metrics["avg_path_length"])
                    else:
                        metrics["small_world_sigma"] = 0

                    # Modularity
                    try:
                        communities = nx.community.greedy_modularity_communities(undirected_G)
                        metrics["num_communities"] = len(communities)
                        metrics["modularity"] = nx.community.modularity(undirected_G, communities)
                    except:
                        metrics["num_communities"] = 0
                        metrics["modularity"] = 0

            except Exception as e:
                self.logger.warning(f"Error computing network metrics: {e}")
                metrics = {"num_nodes": len(G.nodes), "num_edges": len(G.edges)}

        return metrics

    def species_specific_validation(
        self, phrases: List["PhraseSignature"], species_type: str = "harmonic"
    ) -> Dict[str, Any]:
        """
        Perform species-specific validation including harmonic affirmation and compositional validation.

        Args:
            phrases: List of phrases to validate
            species_type: Type of species ('harmonic', 'fm_sweep', 'transient', 'rhythmic')

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "species_type": species_type,
            "harmonic_affirmation": {},
            "compositional_validation": {},
        }

        if species_type == "harmonic" and phrases:
            # Harmonic affirmation
            harmonic_phrases = [p for p in phrases if p.modality == Modality.HARMONIC]

            if harmonic_phrases:
                # Check F0 similarity within harmonic series
                f0_values = [
                    p.features.get("f0_mean", 0)
                    for p in harmonic_phrases
                    if p.features.get("f0_mean", 0) > 0
                ]

                if f0_values:
                    # Group by harmonic similarity (threshold: 20% of fundamental)
                    fundamental_freq = np.median(f0_values)
                    harmonic_threshold = fundamental_freq * 0.2

                    harmonic_groups = defaultdict(list)
                    for i, f0 in enumerate(f0_values):
                        if abs(f0 - fundamental_freq) <= harmonic_threshold:
                            harmonic_groups["fundamental"].append(i)
                        elif abs(f0 - 2 * fundamental_freq) <= harmonic_threshold:
                            harmonic_groups["second_harmonic"].append(i)
                        elif abs(f0 - 3 * fundamental_freq) <= harmonic_threshold:
                            harmonic_groups["third_harmonic"].append(i)
                        else:
                            harmonic_groups["non_harmonic"].append(i)

                    validation_results["harmonic_affirmation"] = {
                        "total_harmonic_phrases": len(harmonic_phrases),
                        "fundamental_freq": fundamental_freq,
                        "harmonic_groups": dict(harmonic_groups),
                        "harmonic_ratio": len([g for g in harmonic_groups.values() if g])
                        / len(f0_values),
                        "threshold": harmonic_threshold,
                    }

        # Compositional validation (Chi-squared test for sequential dependence)
        if self.grammar and len(self.grammar) > 0:
            # Build contingency table
            transitions = defaultdict(lambda: defaultdict(int))
            total_transitions = sum(self.grammar.values())

            for (from_phrase, to_phrase), count in self.grammar.items():
                transitions[from_phrase][to_phrase] = count

            # Perform chi-squared test
            from_phrase_types = list(transitions.keys())
            if len(from_phrase_types) > 1:
                # Simple chi-squared test for uniformity
                observed_counts = [sum(transitions[fp].values()) for fp in from_phrase_types]
                expected_count = total_transitions / len(from_phrase_types)

                chi_squared = sum(
                    (obs - expected_count) ** 2 / expected_count for obs in observed_counts
                )
                degrees_of_freedom = len(from_phrase_types) - 1
                p_value = 1 - 0.5 * (
                    1
                    + np.sign(chi_squared - degrees_of_freedom)
                    * (1 - np.exp(-0.5 * chi_squared / degrees_of_freedom))
                )

                validation_results["compositional_validation"] = {
                    "chi_squared": chi_squared,
                    "degrees_of_freedom": degrees_of_freedom,
                    "p_value": max(0, min(1, p_value)),  # Clamp to [0,1]
                    "total_transitions": total_transitions,
                    "uniform_p_value": p_value < 0.05,  # Significant if < 0.05
                }

        return validation_results

    def mixed_structure_analysis(
        self, phrases: List["PhraseSignature"], grammar: Counter, gap_threshold_ms: float = 500.0
    ) -> Dict[str, Any]:
        """
        Analyze mixed structure combining sequential and superpositional elements.

        Args:
            phrases: List of phrases
            grammar: Counter of sequential transitions
            gap_threshold_ms: Threshold for sentence detection

        Returns:
            Dictionary with mixed structure analysis results
        """
        results = {}

        # Sequential structure analysis
        sequential_phrases = len(phrases)
        total_possible_pairs = sequential_phrases * (sequential_phrases - 1) / 2

        # Sentence discovery for sequential structure
        sentences = self.discover_sentences(phrases, gap_threshold_ms)
        sequential_ratio = len(sentences) / max(
            1, len(phrases) / 2
        )  # Normalize by expected sentences

        # Superposition analysis
        superposition_groups = self.detect_superposition(phrases)
        superpositional_ratio = len(superposition_groups) / max(1, total_possible_pairs)

        # Mixed structure score (METHODOLOGY_SUMMARY.md formula)
        mixed_score = sequential_ratio * superpositional_ratio

        # Modality distribution
        modality_counts = defaultdict(int)
        for phrase in phrases:
            modality_counts[phrase.modality.name] += 1

        results.update(
            {
                "sequential_phrases": sequential_phrases,
                "sentences_discovered": len(sentences),
                "sequential_ratio": sequential_ratio,
                "superposition_groups": len(superposition_groups),
                "superpositional_ratio": superpositional_ratio,
                "mixed_structure_score": mixed_score,
                "modality_distribution": dict(modality_counts),
                "total_phrase_pairs": total_possible_pairs,
            }
        )

        # Compare with species from METHODOLOGY_SUMMARY.md
        if mixed_score > 0.5:
            results["structure_category"] = "High Mixed Complexity"
        elif mixed_score > 0.1:
            results["structure_category"] = "Moderate Mixed Complexity"
        else:
            results["structure_category"] = "Low/Sequential Only"

        # Compare with known species (approximate)
        if mixed_score >= 0.8:
            results["comparison"] = "Similar to Zebra Finch (0.90) or Chimpanzee (0.84)"
        elif mixed_score >= 0.3:
            results["comparison"] = "Similar to Human (0.36) - moderate complexity"
        else:
            results["comparison"] = "Similar to Marmoset (0.00) - pure sequential"

        self.logger.info(
            f"Mixed structure score: {mixed_score:.3f} ({results['structure_category']})"
        )
        return results

    def comprehensive_analysis(
        self,
        audio_segments: List[np.ndarray],
        segment_timestamps: List[float],
        gap_threshold_ms: float = 500.0,
        min_phrase_duration_ms: float = 20.0,
        species_type: Optional[str] = None,
        species_config: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis including all missing components.

        Args:
            audio_segments: List of audio segments (each is a vocalization/recording)
            segment_timestamps: List of timestamps for each vocalization
            gap_threshold_ms: Gap threshold for sentence discovery
            min_phrase_duration_ms: Minimum phrase duration
            species_type: Optional species type for validation
            species_config: Optional species-specific configuration (bin sizes, etc.)

        Returns:
            Comprehensive analysis results
        """
        if species_config is None:
            # Default configuration for different species (from METHODOLOGY_SUMMARY.md)
            species_config = {
                "marmoset": {"f0_bin": 200, "duration_bin": 25, "range_bin": 100},
                "zebra_finch": {"f0_bin": 50, "duration_bin": 50, "range_bin": 100},
                "chimpanzee": {"f0_bin": 100, "duration_bin": 50, "range_bin": 200},
                "sperm_whale": {"f0_bin": 100, "duration_bin": 100, "range_bin": 0},
                "egyptian_bat": {"f0_bin": 500, "duration_bin": 25, "range_bin": 1000},
            }

        # Create sentences from vocalizations
        sentences = self.create_sentences_from_vocalizations(audio_segments, segment_timestamps)

        results = {
            "basic_analysis": {
                "total_sentences": len(sentences),
                "total_phrases": sum(len(s.phrases) for s in sentences),
                "atomic_units_discovered": 0,
            }
        }

        # Process each sentence
        all_phrases = []
        all_atomic_units = {}

        for sentence in sentences:
            # Segment the vocalization into phrases
            phrases = self.segment_phrases(sentence.audio, 50.0, min_phrase_duration_ms)
            sentence.phrases = phrases
            all_phrases.extend(phrases)

            # Discover atomic units using species-specific binning
            if species_type and species_type in species_config:
                config = species_config[species_type]
                sentence.discover_atomic_units(
                    f0_bin_size=config["f0_bin"],
                    duration_bin_size=config["duration_bin"],
                    range_bin_size=config["range_bin"],
                )
            else:
                # Default configuration
                sentence.discover_atomic_units()

            # Validate atomic units using microharmonic similarity
            validated_units = sentence.validate_atomic_units()
            all_atomic_units.update(validated_units)
            sentence.atomic_units = validated_units

        # Discover grammar across sentences
        vocabulary, grammar = self.discover_grammar_from_sentences(sentences)

        results["basic_analysis"].update(
            {
                "vocabulary_size": len(vocabulary),
                "grammar_rules": len(grammar),
                "atomic_units_discovered": len(all_atomic_units),
            }
        )

        # Sentence-level analysis
        results["sentence_analysis"] = []
        for sentence in sentences:
            sentence_result = {
                "sentence_id": sentence.sentence_id,
                "phrases": len(sentence.phrases),
                "atomic_units": len(sentence.atomic_units),
                "top_atomic_units": dict(list(sentence.atomic_units.items())[:5]),  # Top 5
            }
            results["sentence_analysis"].append(sentence_result)

        # Cross-sentence grammar discovery
        if sentences:
            # Superposition detection within each sentence
            all_superpositions = []
            for sentence in sentences:
                sentence_superpositions = self.detect_superposition(
                    sentence.phrases, same_recording_only=True
                )
                all_superpositions.extend(sentence_superpositions)

            # Network metrics
            results["network_metrics"] = self.compute_network_metrics(grammar)

            # Species-specific validation
            if species_type:
                results["species_validation"] = self.species_specific_validation(
                    all_phrases, species_type
                )

            # Mixed structure analysis
            results["mixed_structure"] = self.mixed_structure_analysis(
                all_phrases, grammar, gap_threshold_ms
            )

            # Cross-sentence superposition analysis
            results["superposition_detection"] = {
                "total_groups": len(all_superpositions),
                "groups_per_sentence": [
                    len(self.detect_superposition(s.phrases)) for s in sentences
                ],
                "average_superposition_rate": len(all_superpositions) / max(1, len(all_phrases)),
            }

        # Overall statistics
        results["summary"] = {
            "species_type": species_type,
            "total_vocalizations": len(sentences),
            "total_phrases": len(all_phrases),
            "unique_atomic_units": len(all_atomic_units),
            "grammar_transitions": sum(grammar.values()),
            "average_phrases_per_vocalization": len(all_phrases) / len(sentences)
            if sentences
            else 0,
        }

        return results

    def discover_grammar_from_sentences(
        self, sentences: List[Sentence]
    ) -> Tuple[Dict[int, PhraseSignature], Counter]:
        """
        Discover grammar patterns across sentences.

        Args:
            sentences: List of Sentence objects

        Returns:
            Tuple of (vocabulary, grammar)
        """
        # Collect all phrases from all sentences
        all_phrases = []
        for sentence in sentences:
            all_phrases.extend(sentence.phrases)

        # Build vocabulary from atomic units across sentences
        vocabulary = {}
        phrase_counter = 0

        for sentence in sentences:
            for atomic_key, phrases in sentence.atomic_units.items():
                if phrases:  # Non-empty atomic unit
                    vocabulary[phrase_counter] = phrases[0]  # Use first phrase as representative
                    phrase_counter += 1

        # If no atomic units found, fall back to individual phrases
        if not vocabulary and all_phrases:
            for i, phrase in enumerate(all_phrases):
                vocabulary[i] = phrase

        # Discover grammar from phrase sequences
        grammar = Counter()

        for sentence in sentences:
            # Create sequence of phrase IDs within this sentence
            sequence = []
            for phrase in sentence.phrases:
                # Find the nearest atomic unit/vocabulary entry
                min_distance = float("inf")
                nearest_phrase_id = None

                for vocab_id, vocab_phrase in vocabulary.items():
                    distance = phrase.distance_to(vocab_phrase)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_phrase_id = vocab_id

                if nearest_phrase_id is not None:
                    sequence.append(nearest_phrase_id)

            # Build transitions from sequence
            for i in range(len(sequence) - 1):
                transition = (sequence[i], sequence[i + 1])
                grammar[transition] += 1

        return vocabulary, grammar

    def analyze_modality_sequences(self, phrases: List[PhraseSignature]) -> Dict[str, Any]:
        """
        Analyze modality transition patterns (Texture Grammar).

        Category 1, Item 3: Modality Sequence Graphing

        This method analyzes the transition patterns between modalities within
        a sequence of phrases. It computes:
        - Transition probability matrix (e.g., P(H|T) = 0.8)
        - Transition counts
        - Sequence statistics (runs, alternations, entropy)
        - Most common sequences

        Args:
            phrases: List of PhraseSignature objects with detected modalities

        Returns:
            Dictionary containing:
                - transition_matrix: Dict of (from_modality, to_modality) -> probability
                - transition_counts: Dict of (from_modality, to_modality) -> count
                - sequence_stats: Statistics about modality patterns
                - common_sequences: Most common modality sequences
        """
        if not phrases:
            return {
                "transition_matrix": {},
                "transition_counts": {},
                "sequence_stats": {"total_phrases": 0},
                "common_sequences": [],
            }

        # Extract modality sequence
        modality_sequence = [phrase.modality for phrase in phrases]
        n_phrases = len(modality_sequence)

        # Build transition counts
        transition_counts = {}
        for i in range(n_phrases - 1):
            from_mod = modality_sequence[i]
            to_mod = modality_sequence[i + 1]
            key = (from_mod.name, to_mod.name)
            transition_counts[key] = transition_counts.get(key, 0) + 1

        # Convert counts to probabilities (row-normalized)
        transition_matrix = {}
        for from_mod in Modality:
            from_name = from_mod.name
            # Count total transitions from this modality
            total_from = sum(
                count for (frm, to), count in transition_counts.items() if frm == from_name
            )

            if total_from > 0:
                for to_mod in Modality:
                    to_name = to_mod.name
                    key = (from_name, to_name)
                    count = transition_counts.get(key, 0)
                    probability = count / total_from
                    transition_matrix[key] = probability

        # Sequence statistics
        modality_counts = Counter([m.name for m in modality_sequence])

        # Count runs (consecutive same modality)
        runs = []
        current_run = [modality_sequence[0]]
        for i in range(1, n_phrases):
            if modality_sequence[i] == modality_sequence[i - 1]:
                current_run.append(modality_sequence[i])
            else:
                runs.append(current_run)
                current_run = [modality_sequence[i]]
        if current_run:
            runs.append(current_run)

        run_lengths = [len(run) for run in runs]
        avg_run_length = np.mean(run_lengths) if run_lengths else 0

        # Count alternations (modality changes)
        alternations = sum(
            1 for i in range(n_phrases - 1) if modality_sequence[i] != modality_sequence[i + 1]
        )

        # Compute entropy of modality distribution
        modality_probs = {m: count / n_phrases for m, count in modality_counts.items()}
        entropy = -sum(p * np.log2(p) for p in modality_probs.values() if p > 0)

        # Find most common n-gram sequences
        common_sequences = []
        for n in range(2, min(5, n_phrases + 1)):  # 2-grams to 4-grams
            ngram_counts = Counter()
            for i in range(n_phrases - n + 1):
                ngram = tuple(modality_sequence[i : i + n])
                ngram_counts[ngram] += 1

            for ngram, count in ngram_counts.most_common(3):
                common_sequences.append(
                    {
                        "sequence": [m.name for m in ngram],
                        "length": n,
                        "count": count,
                        "frequency": count / (n_phrases - n + 1),
                    }
                )

        # Compile results
        results = {
            "transition_matrix": transition_matrix,
            "transition_counts": {
                (frm, to): count for (frm, to), count in transition_counts.items()
            },
            "sequence_stats": {
                "total_phrases": n_phrases,
                "unique_modalities": len(modality_counts),
                "modality_distribution": dict(modality_counts),
                "avg_run_length": avg_run_length,
                "total_alternations": alternations,
                "alternation_rate": alternations / (n_phrases - 1) if n_phrases > 1 else 0,
                "entropy": entropy,
                "max_entropy": np.log2(len(modality_counts)) if len(modality_counts) > 0 else 0,
                "normalized_entropy": entropy / np.log2(len(modality_counts))
                if len(modality_counts) > 1
                else 0,
            },
            "common_sequences": common_sequences,
        }

        return results

    def __repr__(self) -> str:
        return (
            f"UniversalRosettaStone(vocabulary_size={len(self.vocabulary)}, "
            f"grammar_rules={len(self.grammar)})"
        )
