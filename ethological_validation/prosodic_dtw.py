#!/usr/bin/env python3
"""
Prosodic Similarity via Dynamic Time Warping (DTW)

Filters out the "Confusion Metric" by comparing temporal prosody of
animal responses against baseline natural conspecific conversations.

An aggressive response might have high acoustic convergence (matching F0)
but entirely wrong temporal prosody (staccato bursts vs. fluid sweeps).
DTW compares the temporal structure to differentiate conversation from
aggression/confusion.

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProsodicFeature:
    """Extracted prosodic features from a vocalization."""
    f0_contour: np.ndarray      # F0 trajectory (Hz)
    amplitude_envelope: np.ndarray  # Amplitude envelope
    duration_ms: float
    spectral_centroid: Optional[np.ndarray] = None  # Spectral centroid trajectory


@dataclass
class DTWResult:
    """Result of DTW-based prosodic comparison."""
    similarity_score: float     # [0, 1], higher = more similar
    dtw_distance: float         # Raw DTW distance
    normalized_distance: float  # Normalized by path length
    warping_path: np.ndarray    # Optimal warping path
    best_match_idx: int         # Index of best matching baseline


class FastDTW:
    """
    Fast Dynamic Time Warping implementation.

    Uses the Sakoe-Chiba algorithm with a window constraint for
    efficient DTW distance computation.
    """

    def __init__(self, window_size: Optional[int] = None):
        """
        Initialize DTW calculator.

        Args:
            window_size: Sakoe-Chiba window constraint (None = full matrix)
        """
        self.window_size = window_size

    def compute_distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Compute DTW distance between two sequences.

        Args:
            x: First sequence (1D array)
            y: Second sequence (1D array)

        Returns:
            DTW distance
        """
        n, m = len(x), len(y)

        # Initialize cost matrix
        # Use float32 for memory efficiency
        cost = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
        cost[0, 0] = 0

        # Window constraint (Sakoe-Chiba band)
        if self.window_size is not None:
            window = max(n, m) // self.window_size
        else:
            window = max(n, m)

        # Fill cost matrix
        for i in range(1, n + 1):
            j_start = max(1, i - window)
            j_end = min(m + 1, i + window + 1)

            for j in range(j_start, j_end):
                # Local cost (squared difference)
                local_cost = (x[i - 1] - y[j - 1]) ** 2

                # Transition costs (three possible paths)
                cost[i, j] = local_cost + min(
                    cost[i - 1, j - 1],  # Match
                    cost[i - 1, j],      # Insertion
                    cost[i, j - 1],      # Deletion
                )

        return cost[n, m]

    def compute_distance_with_path(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute DTW distance and return optimal warping path.

        Args:
            x: First sequence
            y: Second sequence

        Returns:
            (DTW distance, warping path as (i, j) indices)
        """
        n, m = len(x), len(y)

        # Initialize
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0

        # Backpointers for path reconstruction
        traceback = np.zeros((n + 1, m + 1), dtype=int)

        if self.window_size is not None:
            window = max(n, m) // self.window_size
        else:
            window = max(n, m)

        # Fill cost matrix
        for i in range(1, n + 1):
            j_start = max(1, i - window)
            j_end = min(m + 1, i + window + 1)

            for j in range(j_start, j_end):
                local_cost = (x[i - 1] - y[j - 1]) ** 2

                min_cost = cost[i - 1, j - 1]
                traceback[i, j] = 1  # Diagonal

                if cost[i - 1, j] < min_cost:
                    min_cost = cost[i - 1, j]
                    traceback[i, j] = 2  # Up

                if cost[i, j - 1] < min_cost:
                    min_cost = cost[i, j - 1]
                    traceback[i, j] = 3  # Left

                cost[i, j] = local_cost + min_cost

        # Reconstruct path
        path = []
        i, j = n, m
        while i > 0 or j > 0:
            path.append((i - 1, j - 1))
            tb = traceback[i, j]
            if tb == 1:
                i, j = i - 1, j - 1
            elif tb == 2:
                i = i - 1
            elif tb == 3:
                j = j - 1
            else:
                break

        path.reverse()
        path_array = np.array(path)

        return cost[n, m], path_array


class ProsodicDTW:
    """
    Compares temporal prosody against baseline natural conversations
    to differentiate conversation from aggression/confusion.

    Uses DTW on F0 contours and amplitude envelopes to measure
    temporal similarity.
    """

    def __init__(
        self,
        baseline_contours: Optional[List[np.ndarray]] = None,
        sigma: float = 5.0,
        dtw_window: Optional[int] = None,
    ):
        """
        Initialize prosodic DTW engine.

        Args:
            baseline_contours: List of F0 contours from natural conversations
            sigma: Scaling factor for distance-to-similarity conversion
            dtw_window: DTW window constraint (None = full matrix)
        """
        self.baselines = baseline_contours or []
        self.sigma = sigma
        self.dtw = FastDTW(window_size=dtw_window)

        logger.info(f"ProsodicDTW initialized with {len(self.baselines)} baselines")

    def add_baseline(self, contour: np.ndarray):
        """Add a baseline contour from natural conversation."""
        self.baselines.append(contour)

    def set_baselines(self, contours: List[np.ndarray]):
        """Replace all baselines."""
        self.baselines = contours.copy()

    def score_response(
        self,
        f0_contour: np.ndarray,
        amplitude_envelope: Optional[np.ndarray] = None,
    ) -> DTWResult:
        """
        Score animal response against natural conversational baselines.

        Returns similarity score [0, 1] where 1.0 is perfectly natural prosody.

        Args:
            f0_contour: F0 trajectory (Hz) of animal response
            amplitude_envelope: Optional amplitude envelope for joint analysis

        Returns:
            DTWResult with similarity score
        """
        if not self.baselines:
            logger.warning("No baselines available, returning default score")
            return DTWResult(
                similarity_score=0.5,
                dtw_distance=0.0,
                normalized_distance=0.0,
                warping_path=np.array([]),
                best_match_idx=-1,
            )

        # Compute DTW to each baseline
        distances = []
        paths = []

        for baseline in self.baselines:
            # Ensure same length by interpolation
            f0_interp = self._interpolate_to_length(f0_contour, len(baseline))
            baseline_interp = baseline

            dist, path = self.dtw.compute_distance_with_path(f0_interp, baseline_interp)
            distances.append(dist)
            paths.append(path)

        # Find best match
        min_dist = min(distances)
        best_idx = distances.index(min_dist)
        best_path = paths[best_idx]

        # Normalize by path length
        normalized_dist = min_dist / len(best_path)

        # Convert to similarity score using exponential decay
        # Higher distance -> lower similarity
        similarity = np.exp(-normalized_dist / self.sigma)

        return DTWResult(
            similarity_score=float(similarity),
            dtw_distance=float(min_dist),
            normalized_distance=float(normalized_dist),
            warping_path=best_path,
            best_match_idx=best_idx,
        )

    def score_joint_prosody(
        self,
        f0_contour: np.ndarray,
        amplitude_envelope: np.ndarray,
        f0_weight: float = 0.7,
        amp_weight: float = 0.3,
    ) -> float:
        """
        Score using both F0 and amplitude envelope.

        Args:
            f0_contour: F0 trajectory
            amplitude_envelope: Amplitude envelope
            f0_weight: Weight for F0 similarity
            amp_weight: Weight for amplitude similarity

        Returns:
            Combined similarity score [0, 1]
        """
        # Score F0
        f0_result = self.score_response(f0_contour)
        f0_sim = f0_result.similarity_score

        # Score amplitude (use same baselines but with amp envelopes)
        # For now, use simple correlation as proxy
        if self.baselines:
            # Compare against mean baseline shape
            mean_baseline_len = int(np.mean([len(b) for b in self.baselines]))
            amp_interp = self._interpolate_to_length(amplitude_envelope, mean_baseline_len)

            # Compute correlation with each baseline (simplified)
            amp_similarities = []
            for baseline in self.baselines:
                baseline_amp = np.ones_like(baseline)  # Placeholder
                if len(baseline_amp) != len(amp_interp):
                    baseline_amp = self._interpolate_to_length(baseline_amp, len(amp_interp))

                # Normalize
                amp_interp_norm = (amp_interp - amp_interp.mean()) / (amp_interp.std() + 1e-8)
                baseline_amp_norm = (baseline_amp - baseline_amp.mean()) / (baseline_amp.std() + 1e-8)

                # Correlation
                corr = np.corrcoef(amp_interp_norm, baseline_amp_norm)[0, 1]
                amp_similarities.append(max(0, corr))  # Clip negative

            amp_sim = np.mean(amp_similarities)
        else:
            amp_sim = 0.5

        # Weighted combination
        combined = f0_weight * f0_sim + amp_weight * amp_sim

        return float(combined)

    def _interpolate_to_length(
        self,
        array: np.ndarray,
        target_length: int,
    ) -> np.ndarray:
        """Interpolate array to target length."""
        if len(array) == target_length:
            return array

        # Linear interpolation
        old_indices = np.linspace(0, len(array) - 1, len(array))
        new_indices = np.linspace(0, len(array) - 1, target_length)

        return np.interp(new_indices, old_indices, array)


class ProsodicFeatureExtractor:
    """
    Extract prosodic features from audio for DTW comparison.

    Extracts F0 contours, amplitude envelopes, and other temporal features.
    """

    def __init__(self, sample_rate: int = 48000, frame_size: int = 512):
        """
        Initialize feature extractor.

        Args:
            sample_rate: Audio sample rate
            frame_size: Analysis frame size in samples
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = frame_size // 4  # 75% overlap

    def extract_from_audio(
        self,
        audio: np.ndarray,
    ) -> ProsodicFeature:
        """
        Extract prosodic features from audio.

        Args:
            audio: Audio samples (float32, normalized [-1, 1])

        Returns:
            ProsodicFeature with extracted contours
        """
        import scipy.signal as signal

        # Duration
        duration_ms = len(audio) / self.sample_rate * 1000

        # Amplitude envelope (RMS)
        envelope = self._extract_amplitude_envelope(audio)

        # F0 contour (using autocorrelation)
        f0_contour = self._extract_f0_contour(audio)

        # Spectral centroid (optional, for future use)
        spectral_centroid = self._extract_spectral_centroid(audio)

        return ProsodicFeature(
            f0_contour=f0_contour,
            amplitude_envelope=envelope,
            duration_ms=duration_ms,
            spectral_centroid=spectral_centroid,
        )

    def _extract_amplitude_envelope(self, audio: np.ndarray) -> np.ndarray:
        """Extract RMS amplitude envelope."""
        # Frame the audio
        n_frames = 1 + (len(audio) - self.frame_size) // self.hop_size
        envelope = []

        for i in range(n_frames):
            start = i * self.hop_size
            end = min(start + self.frame_size, len(audio))

            if end > start:
                frame = audio[start:end]
                rms = np.sqrt(np.mean(frame ** 2)) + 1e-8
                envelope.append(20 * np.log10(rms))  # dB

        return np.array(envelope, dtype=np.float32)

    def _extract_f0_contour(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract F0 contour using autocorrelation.

        Returns F0 in Hz (0 for unvoiced frames).
        """
        n_frames = 1 + (len(audio) - self.frame_size) // self.hop_size
        f0_values = []

        for i in range(n_frames):
            start = i * self.hop_size
            end = min(start + self.frame_size, len(audio))

            frame = audio[start:end] * np.hanning(end - start)

            # Autocorrelation
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr) // 2:]

            # Find fundamental period
            min_period = int(self.sample_rate / 10000)  # Max 10 kHz
            max_period = int(self.sample_rate / 50)      # Min 50 Hz

            if len(corr) > max_period:
                peak_region = corr[min_period:max_period]
                if len(peak_region) > 0:
                    peak_idx = np.argmax(peak_region) + min_period
                    f0 = self.sample_rate / peak_idx

                    # Voicing detection (check peak strength)
                    if corr[peak_idx] > 0.3 * np.max(corr):
                        f0_values.append(f0)
                    else:
                        f0_values.append(0.0)
                else:
                    f0_values.append(0.0)
            else:
                f0_values.append(0.0)

        return np.array(f0_values, dtype=np.float32)

    def _extract_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral centroid trajectory."""
        n_frames = 1 + (len(audio) - self.frame_size) // self.hop_size
        centroids = []

        for i in range(n_frames):
            start = i * self.hop_size
            end = min(start + self.frame_size, len(audio))

            frame = audio[start:end] * np.hanning(end - start)

            # FFT
            fft = np.fft.rfft(frame)
            mag = np.abs(fft)
            freqs = np.fft.rfftfreq(len(frame), 1 / self.sample_rate)

            # Spectral centroid
            centroid = np.sum(freqs * mag) / (np.sum(mag) + 1e-8)
            centroids.append(centroid)

        return np.array(centroids, dtype=np.float32)


def create_baseline_database(
    audio_files: List[str],
    species: str,
) -> List[np.ndarray]:
    """
    Create baseline prosody database from natural conversation recordings.

    Args:
        audio_files: List of paths to natural conversation audio files
        species: Species identifier (for feature extractor config)

    Returns:
        List of F0 contours for DTW comparison
    """
    extractor = ProsodicFeatureExtractor(sample_rate=48000)
    baselines = []

    for audio_path in audio_files:
        try:
            # Load audio (simplified - in practice use librosa or soundfile)
            # audio = load_audio(audio_path)
            # features = extractor.extract_from_audio(audio)
            # baselines.append(features.f0_contour)
            pass  # Placeholder
        except Exception as e:
            logger.warning(f"Failed to process {audio_path}: {e}")

    logger.info(f"Created baseline database with {len(baselines)} contours")

    return baselines


# =============================================================================
# Preset Configurations
# =============================================================================

# Default prosodic DTW engine
DEFAULT_PROSODIC_DTW = ProsodicDTW(
    sigma=5.0,
    dtw_window=None,
)

# Feature extractor
DEFAULT_FEATURE_EXTRACTOR = ProsodicFeatureExtractor()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Prosodic DTW Demo")
    print("=" * 50)

    # Create synthetic baseline contours (simulating natural conversations)
    np.random.seed(42)

    # Baseline 1: Rising F0 (contact call)
    baseline1 = np.linspace(5000, 7000, 50) + np.random.randn(50) * 100

    # Baseline 2: Flat F0 (social contact)
    baseline2 = np.ones(50) * 6000 + np.random.randn(50) * 50

    # Baseline 3: Modulated F0 (courtship)
    baseline3 = 6000 + 1000 * np.sin(np.linspace(0, 2*np.pi, 50)) + np.random.randn(50) * 50

    baselines = [baseline1, baseline2, baseline3]

    # Create DTW engine with baselines
    dtw_engine = ProsodicDTW(baselines=baselines, sigma=5.0)

    # Test responses
    print("\nTest 1: Natural-like response (should score high)")
    natural_response = np.linspace(5000, 7000, 45) + np.random.randn(45) * 100
    result1 = dtw_engine.score_response(natural_response)
    print(f"  Similarity: {result1.similarity_score:.3f}")
    print(f"  Best match: baseline {result1.best_match_idx}")

    print("\nTest 2: Aggressive staccato (should score low)")
    aggressive = np.concatenate([
        np.ones(20) * 9000,  # High pitch burst
        np.zeros(15),        # Silence
        np.ones(15) * 9000,  # Another burst
    ])
    result2 = dtw_engine.score_response(aggressive)
    print(f"  Similarity: {result2.similarity_score:.3f}")
    print(f"  Best match: baseline {result2.best_match_idx}")
