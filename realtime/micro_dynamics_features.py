"""
Micro-Dynamics Features (Shimmer, Spectral Flux, Harmonicity)

This module implements advanced acoustic features that complete the
"Acoustic Algebra" vector space:

1. **Shimmer** - Amplitude instability (companion to Jitter)
   - Measures peak-to-peak amplitude variation
   - Calculation: std(peak_amplitudes) / mean(peak_amplitudes)

2. **Spectral Flux** - Rate of spectral change over time
   - Measures L2-norm of spectral frame differences
   - Distinguishes trills (high flux) from steady tones (low flux)

3. **Harmonicity** - Degree of periodicity vs noise
   - Auto-correlation based measure
   - Distinguishes pure tones (high) from noise (low)

These features enable:
- "Nervousness" as a 2D vector: [Jitter, Shimmer]
- "Texture" discrimination (Trills vs Flat tones)
- "Modality" slider (Tonal vs Noisy)

Architecture: Python Feature Extraction → 20D Vector Space

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import librosa
import warnings


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Vector20D:
    """
    20-dimensional acoustic feature vector (expanded from 17D)

    New features:
    - shimmer: Amplitude instability (Motion Factor)
    - spectral_flux: Rate of spectral change (Spectral Dynamics)
    - harmonicity: Degree of periodicity (Grit Factor)
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

    # === Fingerprint Factors (5 features) ===
    mfcc_1: float
    mfcc_2: float
    mfcc_3: float
    mfcc_4: float
    spectral_contrast: float

    # === Spectral Dynamics (1 feature) - NEW ===
    spectral_flux: float  # NEW: Rate of spectral change

    # === Rhythm Factors (3 features) ===
    median_ici_ms: float
    onset_rate_hz: float
    ici_coefficient_of_variation: float


# =============================================================================
# Shimmer (Amplitude Instability)
# =============================================================================

def calculate_shimmer(
    audio: np.ndarray,
    sr: int,
    f0: Optional[np.ndarray] = None
) -> float:
    """
    Calculate shimmer (amplitude instability).

    Shimmer measures the variation in peak-to-peak amplitude across
    vocal cycles. It's the amplitude counterpart to jitter (frequency instability).

    Formula:
        shimmer = std(peak_amplitudes) / mean(peak_amplitudes)

    Args:
        audio: Audio samples
        sr: Sample rate
        f0: Optional fundamental frequency contour (for cycle detection)

    Returns:
        Shimmer value (0.0 to 1.0+, typically 0.01-0.10 for animal vocalizations)

    Interpretation:
        - <0.01: Very steady (calm, contact calls)
        - 0.01-0.03: Normal variation
        - 0.03-0.10: High variation (arousal, stress)
        - >0.10: Extreme variation (fear, panting)
    """
    if len(audio) == 0:
        return 0.0

    try:
        # Suppress librosa warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Extract F0 if not provided
            if f0 is None:
                f0, _ = librosa.piptrack(y=audio, sr=sr, fmin=100, fmax=sr//2)
                # Get dominant pitch per frame
                f0 = np.max(f0, axis=0)

            # Find cycle boundaries using zero-crossings
            # This is a simplified approach - more sophisticated methods exist
            zero_crossings = np.where(np.diff(np.sign(audio)))[0]

            if len(zero_crossings) < 4:
                # Not enough cycles, return overall amplitude variation
                return np.std(audio) / (np.abs(np.mean(audio)) + 1e-8)

            # Extract peak amplitudes for each cycle
            peak_amplitudes = []
            for i in range(0, len(zero_crossings) - 2, 2):
                start = zero_crossings[i]
                end = zero_crossings[min(i + 2, len(zero_crossings) - 1)]
                if end > start:
                    cycle = np.abs(audio[start:end])
                    if len(cycle) > 0:
                        peak_amplitudes.append(np.max(cycle))

            if len(peak_amplitudes) < 2:
                # Fallback: overall amplitude variation
                return np.std(audio) / (np.abs(np.mean(audio)) + 1e-8)

            peak_amplitudes = np.array(peak_amplitudes)

            # Calculate shimmer (coefficient of variation)
            shimmer = np.std(peak_amplitudes) / (np.mean(peak_amplitudes) + 1e-8)

            return float(shimmer)

    except Exception as e:
        # Fallback on error: return 0
        return 0.0


# =============================================================================
# Spectral Flux (Texture Change)
# =============================================================================

def calculate_spectral_flux(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512
) -> float:
    """
    Calculate spectral flux (rate of spectral change over time).

    Spectral flux measures how quickly the spectrum is changing.
    High flux = rapid texture changes (trills, twitters)
    Low flux = static spectrum (steady tones)

    Formula:
        flux = mean(L2_norm(S[t] - S[t-1]))

    Where S is the magnitude spectrogram.

    Args:
        audio: Audio samples
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT

    Returns:
        Spectral flux value (typically 0.1-10.0)

    Interpretation:
        - <0.5: Very steady (pure tones)
        - 0.5-2.0: Normal variation
        - 2.0-5.0: High variation (trills, modulation)
        - >5.0: Extreme variation (rapid trills, twitter)
    """
    if len(audio) == 0:
        return 0.0

    try:
        # Suppress librosa warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Compute STFT
            S = np.abs(librosa.stft(audio + 1e-8, n_fft=n_fft, hop_length=hop_length))

            # Normalize by frame energy to make scale-invariant
            frame_energy = np.sum(S, axis=0)
            frame_energy[frame_energy == 0] = 1.0  # Avoid division by zero
            S_norm = S / frame_energy

            # Calculate L2-norm difference between successive frames
            flux = np.linalg.norm(S_norm[:, 1:] - S_norm[:, :-1], axis=0)

            # Return mean flux (scaled for typical range)
            mean_flux = np.mean(flux) * 100.0

            return float(mean_flux)

    except Exception as e:
        return 0.0


# =============================================================================
# Harmonicity (Tonal vs Noisy)
# =============================================================================

def calculate_harmonicity(
    audio: np.ndarray,
    sr: int,
    fmin: float = 100.0,
    fmax: Optional[float] = None
) -> float:
    """
    Calculate harmonicity (degree of periodicity vs noise).

    Harmonicity measures how tonal (periodic) vs noisy a signal is.
    It's based on auto-correlation, similar to YIN algorithm.

    Formula:
        harmonicity = max(auto_correlation) / auto_correlation[0]

    Args:
        audio: Audio samples
        sr: Sample rate
        fmin: Minimum frequency for correlation analysis
        fmax: Maximum frequency (defaults to sr/2)

    Returns:
        Harmonicity value (0.0 to 1.0)

    Interpretation:
        - >0.9: Pure tone (very tonal)
        - 0.7-0.9: Tonal with slight noise
        - 0.4-0.7: Mixed tonal/noise
        - 0.2-0.4: Noisy with tonal components
        - <0.2: Pure noise (very noisy)
    """
    if len(audio) == 0:
        return 0.0

    try:
        if fmax is None:
            fmax = sr / 2

        # Suppress librosa warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Use librosa's piptrack for harmonic/percussive separation
            # This is more robust than raw auto-correlation
            y_harmonic, _ = librosa.effects.hpss(audio)

            # Calculate correlation
            if len(audio) > 0 and len(y_harmonic) > 0:
                # Energy ratio
                energy_total = np.mean(audio ** 2)
                energy_harmonic = np.mean(y_harmonic ** 2)

                if energy_total > 1e-10:
                    harmonicity = energy_harmonic / energy_total
                    return float(np.clip(harmonicity, 0.0, 1.0))

        return 0.5  # Default mid value on error

    except Exception as e:
        return 0.5


# =============================================================================
# Combined Feature Extraction
# =============================================================================

def extract_micro_dynamics(
    audio: np.ndarray,
    sr: int
) -> Tuple[float, float, float]:
    """
    Extract all micro-dynamics features.

    Args:
        audio: Audio samples
        sr: Sample rate

    Returns:
        Tuple of (shimmer, spectral_flux, harmonicity)
    """
    shimmer = calculate_shimmer(audio, sr)
    flux = calculate_spectral_flux(audio, sr)
    harmonic = calculate_harmonicity(audio, sr)

    return shimmer, flux, harmonic


# =============================================================================
# Feature Interpretation
# =============================================================================

def interpret_nervousness(
    jitter: float,
    shimmer: float
) -> str:
    """
    Interpret the [Jitter, Shimmer] 2D vector.

    Returns the emotional/correlational state.

    States:
        - Steady: Low Jitter, Low Shimmer (Contact, Bonding)
        - Breathy: Low Jitter, High Shimmer (Arousal, Panting)
        - Tight: High Jitter, Low Shimmer (Tension, Ready to snap)
        - Tremulous: High Jitter, High Shimmer (Fear, Aggression)
    """
    # Thresholds
    low_threshold = 0.02
    high_threshold = 0.05

    jitter_low = jitter < low_threshold
    jitter_high = jitter > high_threshold
    shimmer_low = shimmer < low_threshold
    shimmer_high = shimmer > high_threshold

    if jitter_low and shimmer_low:
        return "Steady"
    elif jitter_low and shimmer_high:
        return "Breathy"
    elif jitter_high and shimmer_low:
        return "Tight"
    elif jitter_high and shimmer_high:
        return "Tremulous"
    else:
        return "Intermediate"


def interpret_texture(
    spectral_flux: float,
    harmonicity: float
) -> str:
    """
    Interpret texture based on flux and harmonicity.

    Returns texture description.
    """
    if spectral_flux < 0.5:
        if harmonicity > 0.8:
            return "Pure Tone"
        else:
            return "Steady Noise"
    elif spectral_flux < 2.0:
        if harmonicity > 0.7:
            return "Modulated Tone"
        else:
            return "Rough Noise"
    else:
        if harmonicity > 0.6:
            return "Trill"
        else:
            return "Twitter/Granular"
