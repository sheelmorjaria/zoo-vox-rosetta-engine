#!/usr/bin/env python3
"""
Graded Continuum Analysis: Mapping Bat Dispute Trajectories

The old pipeline forced disputes into discrete buckets (Cluster A vs B).
The new VAE affect space allows mapping the continuous trajectory of
conflict escalation from low-level grumbling to physical fighting.

This module analyzes dispute trajectories in 16D affect latent space,
identifying the "tipping point" where spatial disputes turn physical.

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


@dataclass
class DisputeSegment:
    """
    A segment of a dispute with homogeneous affect characteristics.

    Attributes:
        start_time_ms: Start of segment
        end_time_ms: End of segment
        mean_affect: Mean affect vector (16D)
        arousal: Mean arousal level (dimension 0)
        harshness: Mean harshness level (dimension 1)
        vocal_intensity: RMS energy of vocalizations
        segment_type: Type of segment
    """
    start_time_ms: float
    end_time_ms: float
    mean_affect: np.ndarray
    arousal: float
    harshness: float
    vocal_intensity: float
    segment_type: str


@dataclass
class DisputeTrajectory:
    """
    Complete trajectory of a bat dispute in affect space.

    Attributes:
        dispute_id: Unique identifier
        participants: List of bat IDs involved
        start_time_ms: Dispute start time
        end_time_ms: Dispute end time
        affect_trajectory: (T, 16) array of affect vectors over time
        arousal_trajectory: Arousal level over time
        segments: Discovered segments
        tipping_point: Index of transition to physical fight
        became_physical: Whether dispute escalated to physical
    """
    dispute_id: str
    participants: List[int]
    start_time_ms: float
    end_time_ms: float
    affect_trajectory: np.ndarray  # (T, 16)
    arousal_trajectory: np.ndarray  # (T,)
    segments: List[DisputeSegment]
    tipping_point: Optional[int]
    became_physical: bool


@dataclass
class TippingPointAnalysis:
    """
    Analysis of dispute tipping points.

    Identifies the acoustic/affective threshold where
    spatial disputes escalate to physical fights.
    """
    mean_arousal_at_tip: float
    std_arousal_at_tip: float
    mean_harshness_at_tip: float
    mean_duration_to_tip_ms: float
    typical_trajectory: List[str]  # Sequence of segment types


class GradedContinuumAnalyzer:
    """
    Analyzes continuous dispute trajectories in VAE affect space.

    Replaces discrete categorization with continuous trajectory
    analysis, enabling identification of:
    1. Escalation gradients
    2. De-escalation patterns
    3. Tipping points (dispute → fight)
    4. Intervention opportunities
    """

    def __init__(
        self,
        arousal_dim: int = 0,
        harshness_dim: int = 1,
        physical_threshold_arousal: float = 0.8,
    ):
        """
        Initialize graded continuum analyzer.

        Args:
            arousal_dim: Index of arousal dimension in affect vector
            harshness_dim: Index of harshness dimension
            physical_threshold_arousal: Arousal level indicating physical fight
        """
        self.arousal_dim = arousal_dim
        self.harshness_dim = harshness_dim
        self.physical_threshold = physical_threshold_arousal

        logger.info("GradedContinuumAnalyzer initialized")

    def analyze_dispute(
        self,
        affect_trajectory: np.ndarray,
        timestamps_ms: np.ndarray,
        participants: List[int],
        dispute_id: str,
        physical_label: Optional[bool] = None,
    ) -> DisputeTrajectory:
        """
        Analyze a complete dispute trajectory.

        Args:
            affect_trajectory: (T, 16) array of affect vectors
            timestamps_ms: (T,) array of timestamps
            participants: List of bat IDs involved
            dispute_id: Unique identifier
            physical_label: Optional label if dispute became physical

        Returns:
            DisputeTrajectory with full analysis
        """
        # Extract arousal trajectory
        arousal = affect_trajectory[:, self.arousal_dim]
        harshness = affect_trajectory[:, self.harshness_dim]

        # Segment the trajectory
        segments = self._segment_trajectory(
            affect_trajectory,
            timestamps_ms,
            arousal,
            harshness,
        )

        # Find tipping point
        tipping_point = self._find_tipping_point(arousal, segments)

        # Determine if physical (from label or arousal threshold)
        became_physical = physical_label
        if physical_label is None:
            became_physical = np.max(arousal) > self.physical_threshold

        return DisputeTrajectory(
            dispute_id=dispute_id,
            participants=participants,
            start_time_ms=timestamps_ms[0],
            end_time_ms=timestamps_ms[-1],
            affect_trajectory=affect_trajectory,
            arousal_trajectory=arousal,
            segments=segments,
            tipping_point=tipping_point,
            became_physical=became_physical,
        )

    def _segment_trajectory(
        self,
        affect_trajectory: np.ndarray,
        timestamps_ms: np.ndarray,
        arousal: np.ndarray,
        harshness: np.ndarray,
    ) -> List[DisputeSegment]:
        """
        Segment trajectory into homogeneous phases.

        Uses change point detection on arousal to find
        transition points between dispute phases.
        """
        segments = []

        # Smooth arousal for change point detection
        window = min(10, len(arousal) // 4)
        if window > 1:
            arousal_smooth = savgol_filter(arousal, window, 2)
        else:
            arousal_smooth = arousal

        # Find change points (large gradients)
        gradients = np.diff(arousal_smooth)
        threshold = np.std(gradients) * 2

        change_points = [0]
        for i, grad in enumerate(gradients):
            if abs(grad) > threshold:
                change_points.append(i + 1)
        change_points.append(len(arousal))

        # Create segments
        for i in range(len(change_points) - 1):
            start_idx = change_points[i]
            end_idx = change_points[i + 1]

            if end_idx <= start_idx:
                continue

            start_ms = timestamps_ms[start_idx]
            end_ms = timestamps_ms[end_idx - 1]

            # Mean affect in segment
            segment_affect = np.mean(
                affect_trajectory[start_idx:end_idx],
                axis=0
            )
            segment_arousal = np.mean(arousal[start_idx:end_idx])
            segment_harshness = np.mean(harshness[start_idx:end_idx])

            # Determine segment type
            segment_type = self._classify_segment(
                segment_arousal,
                segment_harshness,
            )

            segments.append(DisputeSegment(
                start_time_ms=start_ms,
                end_time_ms=end_ms,
                mean_affect=segment_affect,
                arousal=segment_arousal,
                harshness=segment_harshness,
                vocal_intensity=np.mean(np.linalg.norm(
                    affect_trajectory[start_idx:end_idx], axis=1
                )),
                segment_type=segment_type,
            ))

        return segments

    def _classify_segment(
        self,
        arousal: float,
        harshness: float,
    ) -> str:
        """Classify a dispute segment based on affect."""
        if arousal < 0.3:
            return "grumbling"
        elif arousal < 0.5:
            return "squabbling"
        elif arousal < 0.7:
            return "aggressive_vocal"
        elif harshness > 0.5:
            return "harsh_aggression"
        else:
            return "escalated"

    def _find_tipping_point(
        self,
        arousal: np.ndarray,
        segments: List[DisputeSegment],
    ) -> Optional[int]:
        """
        Find the tipping point where dispute escalates to physical.

        Looks for rapid transition in arousal level combined
        with harshness increase.
        """
        if len(segments) < 2:
            return None

        # Look for transition from "aggressive_vocal" to "escalated"
        for i, segment in enumerate(segments):
            if segment.segment_type in ["escalated", "harsh_aggression"]:
                # Found high-arousal segment
                if i > 0:
                    # Check if previous segment was much lower
                    prev_arousal = segments[i - 1].arousal
                    if segment.arousal - prev_arousal > 0.2:
                        # Rapid escalation found
                        # Return index in original arousal array
                        return self._find_closest_index(
                            segments, segment.start_time_ms
                        )

        # Alternative: find max gradient in arousal
        gradients = np.diff(arousal)
        if len(gradients) > 0:
            max_grad_idx = np.argmax(gradients)
            if gradients[max_grad_idx] > 0.05:  # Significant increase
                return max_grad_idx

        return None

    def _find_closest_index(
        self,
        segments: List[DisputeSegment],
        timestamp_ms: float,
    ) -> int:
        """Find the index in trajectory closest to timestamp."""
        # This would need the original timestamps
        # Simplified: return approximate position
        return len(segments) * 10  # Placeholder

    def analyze_tipping_points(
        self,
        disputes: List[DisputeTrajectory],
    ) -> TippingPointAnalysis:
        """
        Analyze tipping points across multiple disputes.

        Args:
            disputes: List of analyzed dispute trajectories

        Returns:
            TippingPointAnalysis with aggregated statistics
        """
        tip_arousals = []
        tip_harshness = []
        durations = []
        trajectories = []

        for dispute in disputes:
            if dispute.tipping_point is not None:
                idx = dispute.tipping_point
                tip_arousals.append(dispute.arousal_trajectory[idx])
                tip_harshness.append(
                    dispute.affect_trajectory[idx, self.harshness_dim]
                )
                durations.append(
                    dispute.timestamps_ms[idx] - dispute.start_time_ms
                    if hasattr(dispute, 'timestamps_ms')
                    else idx * 100  # Placeholder
                )

        if not tip_arousals:
            return TippingPointAnalysis(
                mean_arousal_at_tip=0,
                std_arousal_at_tip=0,
                mean_harshness_at_tip=0,
                mean_duration_to_tip_ms=0,
                typical_trajectory=[],
            )

        return TippingPointAnalysis(
            mean_arousal_at_tip=np.mean(tip_arousals),
            std_arousal_at_tip=np.std(tip_arousals),
            mean_harshness_at_tip=np.mean(tip_harshness),
            mean_duration_to_tip_ms=np.mean(durations),
            typical_trajectory=self._get_typical_trajectory(disputes),
        )

    def _get_typical_trajectory(
        self,
        disputes: List[DisputeTrajectory],
    ) -> List[str]:
        """Extract the most common segment sequence."""
        sequences = []
        for dispute in disputes:
            seq = [s.segment_type for s in dispute.segments]
            sequences.append(seq)

        # Find most common sequence
        from collections import Counter
        seq_tuples = [tuple(s) for s in sequences]
        most_common = Counter(seq_tuples).most_common(1)

        if most_common:
            return list(most_common[0][0])
        return []


def visualize_dispute_trajectory(
    dispute: DisputeTrajectory,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize dispute trajectory in affect space.

    Args:
        dispute: DisputeTrajectory to visualize
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Arousal over time
    axes[0, 0].plot(dispute.arousal_trajectory)
    axes[0, 0].set_title("Arousal Trajectory")
    axes[0, 0].set_xlabel("Frame")
    axes[0, 0].set_ylabel("Arousal")
    axes[0, 0].axhline(y=0.8, color='r', linestyle='--', label='Physical threshold')
    axes[0, 0].legend()

    # 2D projection (Arousal vs Harshness)
    harshness = dispute.affect_trajectory[:, 1]
    axes[0, 1].scatter(
        dispute.arousal_trajectory,
        harshness,
        c=range(len(dispute.arousal_trajectory)),
        cmap='viridis',
        s=20
    )
    axes[0, 1].set_title("Affect Space Trajectory")
    axes[0, 1].set_xlabel("Arousal")
    axes[0, 1].set_ylabel("Harshness")
    axes[0, 1].plot(
        dispute.arousal_trajectory[0],
        harshness[0],
        'go', markersize=10, label='Start'
    )
    axes[0, 1].plot(
        dispute.arousal_trajectory[-1],
        harshness[-1],
        'ro', markersize=10, label='End'
    )
    axes[0, 1].legend()

    # Segment types
    segments = dispute.segments
    segment_starts = [s.start_time_ms for s in segments]
    segment_types = [s.segment_type for s in segments]
    segment_arousals = [s.arousal for s in segments]

    colors = {
        "grumbling": "green",
        "squabbling": "yellow",
        "aggressive_vocal": "orange",
        "harsh_aggression": "red",
        "escalated": "darkred",
    }

    axes[1, 0].barh(range(len(segments)), segment_arousals, color=[colors.get(t, "gray") for t in segment_types])
    axes[1, 0].set_yticks(range(len(segments)))
    axes[1, 0].set_yticklabels(segment_types)
    axes[1, 0].set_title("Segment Types by Arousal")
    axes[1, 0].set_xlabel("Arousal")

    # Tipping point
    if dispute.tipping_point is not None:
        axes[1, 1].axvline(x=dispute.tipping_point, color='r', linestyle='--', linewidth=2)
        axes[1, 1].text(
            dispute.tipping_point,
            np.max(dispute.arousal_trajectory) * 0.9,
            "TIPPING POINT",
            ha='center',
            color='red',
            fontweight='bold'
        )

    axes[1, 1].plot(dispute.arousal_trajectory, 'b-', linewidth=2)
    axes[1, 1].set_title("Dispute with Tipping Point")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylabel("Arousal")

    plt.suptitle(f"Dispute Trajectory: {dispute.dispute_id}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved trajectory plot to {save_path}")
    else:
        plt.show()


# Preset configurations

# Default graded continuum analyzer for bats
DEFAULT_GRADED_CONTINUUM = GradedContinuumAnalyzer()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Graded Continuum Analysis Demo")
    print("=" * 50)

    analyzer = DEFAULT_GRADED_CONTINUUM

    # Simulate a dispute trajectory
    # Start low, escalate, then de-escalate
    t = np.linspace(0, 100, 100)
    arousal = (
        0.2 +  # Base grumbling
        0.5 * (1 / (1 + np.exp(-0.1 * (t - 50)))) -  # Sigmoid escalation
        0.1 * np.sin(t / 10)  # Oscillation
    )

    # Create affect trajectory (16D)
    affect_trajectory = np.zeros((100, 16))
    affect_trajectory[:, 0] = arousal  # Arousal
    affect_trajectory[:, 1] = arousal * 0.8  # Harshness
    # Other dimensions are noise
    affect_trajectory[:, 2:] = np.random.randn(100, 14) * 0.1

    timestamps = t * 100  # ms

    # Analyze
    dispute = analyzer.analyze_dispute(
        affect_trajectory=affect_trajectory,
        timestamps_ms=timestamps,
        participants=[1, 2],
        dispute_id="sim_dispute_001",
        physical_label=False,
    )

    print(f"\nDispute Analysis: {dispute.dispute_id}")
    print(f"  Participants: {dispute.participants}")
    print(f"  Duration: {dispute.end_time_ms - dispute.start_time_ms:.0f}ms")
    print(f"  Max Arousal: {np.max(dispute.arousal_trajectory):.2f}")
    print(f"  Segments: {len(dispute.segments)}")
    print(f"  Tipping Point: {dispute.tipping_point}")
    print(f"  Became Physical: {dispute.became_physical}")

    print(f"\nSegment Sequence:")
    for i, segment in enumerate(dispute.segments):
        print(f"  {i+1}. {segment.segment_type} "
              f"(arousal={segment.arousal:.2f})")
