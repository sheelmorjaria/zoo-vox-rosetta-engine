#!/usr/bin/env python3
"""
Ethological "Turing Test" via Prosodic DTW

Compares the Dynamic Time Warping distance between AI-bat conversations
and natural bat-bat conversations.

Hypothesis: If bats respond similarly to AI as to conspecifics, the
prosodic DTW distance will be comparable. Significant deviation indicates
the AI is not passing the "Turing Test" of naturalistic interaction.

Uses multi-dimensional DTW across:
- F0 contours (pitch prosody)
- RMS energy (amplitude prosody)
- Spectral centroid (timbral prosody)
- Affect vector trajectories (cognitive prosody)

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class ConversationType(Enum):
    """Type of conversation for comparison."""
    AI_BAT = "ai_bat"        # AI interacting with bat
    BAT_BAT = "bat_bat"      # Natural bat-bat interaction
    BAT_BAT_CONTROL = "bat_bat_control"  # Control (unrelated dyad)


@dataclass
class ProsodicTrajectory:
    """
    Multi-dimensional prosodic trajectory for a conversation.

    Attributes:
        conversation_id: Unique identifier
        conversation_type: AI_BAT or BAT_BAT
        participant_ids: IDs of participants
        f0_contours: List of F0 contours (one per vocalization)
        rms_contours: List of RMS contours
        centroid_contours: List of spectral centroid contours
        affect_trajectories: List of 16D affect vectors
        timestamps_ms: Timestamp of each vocalization
        duration_ms: Total conversation duration
    """
    conversation_id: str
    conversation_type: ConversationType
    participant_ids: Tuple[int, int]
    f0_contours: List[np.ndarray]
    rms_contours: List[np.ndarray]
    centroid_contours: List[np.ndarray]
    affect_trajectories: List[np.ndarray]
    timestamps_ms: List[float]
    duration_ms: float

    @property
    def vocalization_count(self) -> int:
        """Number of vocalizations in conversation."""
        return len(self.timestamps_ms)


@dataclass
class DTWResult:
    """
    Result of DTW comparison between two conversations.

    Attributes:
        conversation_a_id: First conversation ID
        conversation_b_id: Second conversation ID
        dtw_distance: Overall DTW distance
        dtw_f0: F0-specific DTW distance
        dtw_rms: RMS-specific DTW distance
        dtw_centroid: Centroid-specific DTW distance
        dtw_affect: Affect-specific DTW distance
        normalized_distance: Distance normalized by duration
        path_warp: Warping path (list of indices)
        similarity_score: 1 - normalized distance (0-1)
    """
    conversation_a_id: str
    conversation_b_id: str
    dtw_distance: float
    dtw_f0: float
    dtw_rms: float
    dtw_centroid: float
    dtw_affect: float
    normalized_distance: float
    path_warp: List[Tuple[int, int]]
    similarity_score: float


@dataclass
class TuringTestResult:
    """
    Result of ethological Turing Test.

    Compares AI-bat conversation similarity to bat-bat baseline.
    """
    ai_bat_similarity: float  # Mean similarity to bat-bat conversations
    bat_bat_baseline: float   # Mean similarity within bat-bat pairs
    turing_score: float       # Ratio: ai_bat / bat_bat
    passed: bool              # True if score > threshold
    interpretation: str


def dtw_distance(
    seq1: np.ndarray,
    seq2: np.ndarray,
    window: Optional[int] = None,
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Compute Dynamic Time Warping distance between two sequences.

    Uses Sakoe-Chiba band for efficiency if window specified.

    Args:
        seq1: First sequence (T1, D)
        seq2: Second sequence (T2, D)
        window: Sakoe-Chiba window size (None = full matrix)

    Returns:
        (distance, warping_path) tuple
    """
    T1, D = seq1.shape
    T2 = seq2.shape

    # Initialize cost matrix
    dtw_matrix = np.full((T1 + 1, T2 + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Compute costs
    if window is None:
        window = max(T1, T2)

    for i in range(1, T1 + 1):
        j_start = max(1, i - window)
        j_end = min(T2, i + window)

        for j in range(j_start, j_end + 1):
            # Euclidean distance between feature vectors
            cost = euclidean(seq1[i - 1], seq2[j - 1])

            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # Insertion
                dtw_matrix[i, j - 1],      # Deletion
                dtw_matrix[i - 1, j - 1],  # Match
            )

    distance = dtw_matrix[T1, T2]

    # Backtrack to find warping path
    path = []
    i, j = T1, T2
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([
            dtw_matrix[i - 1, j],
            dtw_matrix[i, j - 1],
            dtw_matrix[i - 1, j - 1],
        ])
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

    path.reverse()

    return distance, path


def interpolate_contour(
    contour: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """
    Interpolate contour to target length.

    Args:
        contour: Original contour (T,)
        target_length: Desired length

    Returns:
        Interpolated contour (target_length,)
    """
    if len(contour) == 0:
        return np.zeros(target_length)

    x_orig = np.linspace(0, 1, len(contour))
    x_new = np.linspace(0, 1, target_length)

    interpolator = interp1d(
        x_orig,
        contour,
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )

    return interpolator(x_new)


class EthologicalTuringTest:
    """
    Runs ethological Turing Test using DTW comparison.

    Tests whether AI-bat conversations are prosodically similar
    to natural bat-bat conversations.
    """

    def __init__(
        self,
        passing_threshold: float = 0.7,  # Similarity ratio threshold
        prosodic_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize Turing Test analyzer.

        Args:
            passing_threshold: Min turing_score to pass test
            prosodic_weights: Weights for each prosodic dimension
        """
        self.passing_threshold = passing_threshold

        # Default prosodic weights
        self.prosodic_weights = prosodic_weights or {
            'f0': 0.3,        # Pitch prosody
            'rms': 0.2,       # Amplitude prosody
            'centroid': 0.2,  # Timbral prosody
            'affect': 0.3,    # Cognitive prosody
        }

        self.conversations: Dict[str, ProsodicTrajectory] = {}

        logger.info("EthologicalTuringTest initialized")

    def add_conversation(
        self,
        trajectory: ProsodicTrajectory,
    ) -> None:
        """Add a conversation to the database."""
        self.conversations[trajectory.conversation_id] = trajectory

    def compare_conversations(
        self,
        conv_a_id: str,
        conv_b_id: str,
    ) -> DTWResult:
        """
        Compare two conversations using multi-dimensional DTW.

        Args:
            conv_a_id: First conversation ID
            conv_b_id: Second conversation ID

        Returns:
            DTWResult with detailed comparison
        """
        conv_a = self.conversations.get(conv_a_id)
        conv_b = self.conversations.get(conv_b_id)

        if conv_a is None or conv_b is None:
            raise ValueError("Conversation not found")

        # Normalize to same length for comparison
        target_len = max(
            conv_a.vocalization_count,
            conv_b.vocalization_count
        )

        # Build feature matrices
        features_a = self._build_feature_matrix(conv_a, target_len)
        features_b = self._build_feature_matrix(conv_b, target_len)

        # Compute per-dimension DTW
        dtw_f0, _ = dtw_distance(
            features_a[:, :1], features_b[:, :1], window=10
        )
        dtw_rms, _ = dtw_distance(
            features_a[:, 1:2], features_b[:, 1:2], window=10
        )
        dtw_centroid, _ = dtw_distance(
            features_a[:, 2:3], features_b[:, 2:3], window=10
        )
        dtw_affect, path = dtw_distance(
            features_a[:, 3:], features_b[:, 3:], window=10
        )

        # Weighted total distance
        dtw_total = (
            self.prosodic_weights['f0'] * dtw_f0 +
            self.prosodic_weights['rms'] * dtw_rms +
            self.prosodic_weights['centroid'] * dtw_centroid +
            self.prosodic_weights['affect'] * dtw_affect
        )

        # Normalize by duration
        max_duration = max(conv_a.duration_ms, conv_b.duration_ms)
        normalized = dtw_total / (max_duration / 1000)

        # Similarity score (0-1, higher = more similar)
        similarity = 1.0 / (1.0 + normalized)

        return DTWResult(
            conversation_a_id=conv_a_id,
            conversation_b_id=conv_b_id,
            dtw_distance=dtw_total,
            dtw_f0=dtw_f0,
            dtw_rms=dtw_rms,
            dtw_centroid=dtw_centroid,
            dtw_affect=dtw_affect,
            normalized_distance=normalized,
            path_warp=path,
            similarity_score=similarity,
        )

    def _build_feature_matrix(
        self,
        conv: ProsodicTrajectory,
        target_length: int,
    ) -> np.ndarray:
        """
        Build feature matrix for DTW comparison.

        Each vocalization summarized by mean prosodic values.

        Args:
            conv: Conversation trajectory
            target_length: Target number of time steps

        Returns:
            Feature matrix (target_length, D)
        """
        # Summarize each vocalization
        n_voc = conv.vocalization_count
        features = []

        for i in range(n_voc):
            # Mean F0
            f0_mean = np.mean(conv.f0_contours[i]) if len(conv.f0_contours[i]) > 0 else 0

            # Mean RMS
            rms_mean = np.mean(conv.rms_contours[i]) if len(conv.rms_contours[i]) > 0 else 0

            # Mean centroid
            cent_mean = np.mean(conv.centroid_contours[i]) if len(conv.centroid_contours[i]) > 0 else 0

            # Affect vector (already 16D)
            affect = conv.affect_trajectories[i] if i < len(conv.affect_trajectories) else np.zeros(16)

            feature_vec = np.concatenate([
                [f0_mean],
                [rms_mean],
                [cent_mean],
                affect,
            ])
            features.append(feature_vec)

        features = np.array(features)

        # Interpolate to target length
        if len(features) < target_length:
            # Pad with zeros
            padded = np.zeros((target_length, features.shape[1]))
            padded[:len(features)] = features
            features = padded
        elif len(features) > target_length:
            # Downsample (simple stride)
            features = features[:target_length]

        return features

    def run_turing_test(
        self,
        ai_bat_conv_ids: List[str],
        bat_bat_conv_ids: List[str],
    ) -> TuringTestResult:
        """
        Run full Turing Test comparison.

        Compares AI-bat conversations against bat-bat baseline.

        Args:
            ai_bat_conv_ids: List of AI-bat conversation IDs
            bat_bat_conv_ids: List of bat-bat conversation IDs

        Returns:
            TuringTestResult with pass/fail determination
        """
        # Compute AI-bat to bat-bat similarities
        cross_similarities = []

        for ai_id in ai_bat_conv_ids:
            for bat_id in bat_bat_conv_ids:
                result = self.compare_conversations(ai_id, bat_id)
                cross_similarities.append(result.similarity_score)

        # Compute bat-bat baseline (within-bat similarity)
        baseline_similarities = []

        for i, id1 in enumerate(bat_bat_conv_ids):
            for id2 in bat_bat_conv_ids[i + 1:]:
                result = self.compare_conversations(id1, id2)
                baseline_similarities.append(result.similarity_score)

        # Compute statistics
        ai_bat_sim = np.mean(cross_similarities) if cross_similarities else 0
        bat_bat_sim = np.mean(baseline_similarities) if baseline_similarities else 1.0

        # Turing score: ratio of AI similarity to baseline
        turing_score = ai_bat_sim / (bat_bat_sim + 1e-10)

        # Determine pass/fail
        passed = turing_score >= self.passing_threshold

        if passed:
            interpretation = (
                f"PASSED Turing Test (score={turing_score:.2f}). "
                f"AI-bat conversations are {turing_score*100:.0f}% "
                f"as prosodically similar as natural bat-bat conversations."
            )
        else:
            interpretation = (
                f"FAILED Turing Test (score={turing_score:.2f}). "
                f"AI-bat conversations are only {turing_score*100:.0f}% "
                f"as prosodically similar as natural bat-bat conversations. "
                f"Target: {self.passing_threshold*100:.0f}%."
            )

        return TuringTestResult(
            ai_bat_similarity=ai_bat_sim,
            bat_bat_baseline=bat_bat_sim,
            turing_score=turing_score,
            passed=passed,
            interpretation=interpretation,
        )

    def analyze_failures(
        self,
        ai_bat_conv_ids: List[str],
        bat_bat_conv_ids: List[str],
    ) -> Dict[str, float]:
        """
        Analyze which prosodic dimensions contribute to failure.

        Compares each dimension separately to identify weak points.

        Args:
            ai_bat_conv_ids: AI-bat conversation IDs
            bat_bat_conv_ids: Bat-bat conversation IDs

        Returns:
            Dictionary mapping dimension to failure contribution
        """
        # Compare per-dimension distances
        contributions = {}

        for dim in ['f0', 'rms', 'centroid', 'affect']:
            # Would need to isolate each dimension
            contributions[dim] = 0.0  # Placeholder

        return contributions


# Preset configurations

# Default Turing Test
DEFAULT_TURING_TEST = EthologicalTuringTest()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Ethological Turing Test Demo")
    print("=" * 50)

    test = DEFAULT_TURING_TEST

    # Create mock AI-bat conversation
    ai_conv = ProsodicTrajectory(
        conversation_id="ai_bat_001",
        conversation_type=ConversationType.AI_BAT,
        participant_ids=(0, 1),  # AI, Bat 1
        f0_contours=[
            np.array([8000, 8200, 8000]),
            np.array([7500, 7800, 7500]),
            np.array([8100, 8300, 8100]),
        ],
        rms_contours=[
            np.array([0.5, 0.6, 0.5]),
            np.array([0.4, 0.5, 0.4]),
            np.array([0.5, 0.6, 0.5]),
        ],
        centroid_contours=[
            np.array([10000, 10500, 10000]),
            np.array([9500, 10000, 9500]),
            np.array([10200, 10700, 10200]),
        ],
        affect_trajectories=[
            np.array([0.5, 0.3] + [0.0] * 14),
            np.array([0.4, 0.2] + [0.0] * 14),
            np.array([0.5, 0.3] + [0.0] * 14),
        ],
        timestamps_ms=[0, 200, 400],
        duration_ms=600,
    )

    # Create mock bat-bat conversation
    bat_conv = ProsodicTrajectory(
        conversation_id="bat_bat_001",
        conversation_type=ConversationType.BAT_BAT,
        participant_ids=(1, 2),
        f0_contours=[
            np.array([8100, 8300, 8100]),
            np.array([7600, 7900, 7600]),
            np.array([8200, 8400, 8200]),
        ],
        rms_contours=[
            np.array([0.5, 0.6, 0.5]),
            np.array([0.4, 0.5, 0.4]),
            np.array([0.5, 0.6, 0.5]),
        ],
        centroid_contours=[
            np.array([10100, 10600, 10100]),
            np.array([9600, 10100, 9600]),
            np.array([10300, 10800, 10300]),
        ],
        affect_trajectories=[
            np.array([0.5, 0.3] + [0.0] * 14),
            np.array([0.4, 0.2] + [0.0] * 14),
            np.array([0.5, 0.3] + [0.0] * 14),
        ],
        timestamps_ms=[0, 200, 400],
        duration_ms=600,
    )

    # Add conversations
    test.add_conversation(ai_conv)
    test.add_conversation(bat_conv)

    # Compare
    result = test.compare_conversations("ai_bat_001", "bat_bat_001")

    print(f"\nDTW Comparison:")
    print(f"  Total Distance: {result.dtw_distance:.2f}")
    print(f"  Normalized: {result.normalized_distance:.2f}")
    print(f"  Similarity: {result.similarity_score:.2%}")
    print(f"  F0 DTW: {result.dtw_f0:.2f}")
    print(f"  RMS DTW: {result.dtw_rms:.2f}")
    print(f"  Affect DTW: {result.dtw_affect:.2f}")

    # Run full Turing Test
    turing_result = test.run_turing_test(
        ai_bat_conv_ids=["ai_bat_001"],
        bat_bat_conv_ids=["bat_bat_001"],
    )

    print(f"\n{TuringResult.interpretation}")
