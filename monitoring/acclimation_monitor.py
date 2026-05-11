#!/usr/bin/env python3
"""
Acclimation Phase Monitoring Dashboard

Monitors colony response during the acclimation phase (Week 7)
when the AI begins broadcasting but not yet engaging in
closed-loop interaction.

Tracks alarm rates, response patterns, MFAS scores, and
colony-wide agitation to ensure the synthesized audio is
biologically acceptable.

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional

import numpy as np

from ethological_validation import (
    InteractionEvent,
    MultiFactorAcceptanceScore,
    MFASResult,
    TaxaTemporalProfile,
    get_temporal_gate,
)

logger = logging.getLogger(__name__)


@dataclass
class PlaybackEvent:
    """Record of an AI playback event."""
    timestamp_ms: float
    affect_vector: np.ndarray
    syntactic_token: int
    emitter_id: int
    target_bat_id: Optional[int] = None
    duration_ms: float = 200


@dataclass
class ColonyResponse:
    """Colony response to a playback event."""
    timestamp_ms: float
    playback_event: PlaybackEvent

    # Individual responses
    responding_bat_ids: List[int] = field(default_factory=list)
    alarm_bat_ids: List[int] = field(default_factory=list)

    # Colony-level metrics
    colony_agitation: float = 0.0  # 0-1, colony-wide arousal
    mean_arousal: float = 0.0
    flight_to_exits: bool = False

    # MFAS scores
    mfas_results: List[MFASResult] = field(default_factory=list)

    def add_response(
        self,
        bat_id: int,
        is_alarm: bool,
        arousal: float,
        interaction: Optional[InteractionEvent] = None,
        mfas_result: Optional[MFASResult] = None,
    ) -> None:
        """Add an individual bat's response."""
        self.responding_bat_ids.append(bat_id)
        if is_alarm:
            self.alarm_bat_ids.append(bat_id)

        if interaction is not None and mfas_result is not None:
            self.mfas_results.append(mfas_result)

    def compute_metrics(self) -> Dict:
        """Compute response metrics."""
        response_count = len(self.responding_bat_ids)
        alarm_count = len(self.alarm_bat_ids)

        mfas_scores = [r.mfas_score for r in self.mfas_results]

        return {
            "response_count": response_count,
            "alarm_count": alarm_count,
            "alarm_rate": alarm_count / max(response_count, 1),
            "mean_mfas": np.mean(mfas_scores) if mfas_scores else 0.0,
            "colony_agitation": self.colony_agitation,
            "flight_to_exits": self.flight_to_exits,
        }


@dataclass
class AcclimationAlert:
    """Alert generated during acclimation monitoring."""
    timestamp_ms: float
    severity: str  # "warning", "critical"
    metric_name: str
    current_value: float
    threshold: float
    message: str
    recommended_action: str


class AcclimationMonitor:
    """
    Monitors colony response during acclimation phase.

    Responsibilities:
    1. Track alarm rates vs baseline
    2. Monitor MFAS scores of AI vs natural
    3. Detect colony-wide agitation
    4. Generate alerts for safety violations
    5. Recommend parameter adjustments
    """

    def __init__(
        self,
        species: str,
        baseline_alarm_rate: float = 0.05,
        estimated_colony_size: int = 50,
        alert_window: int = 100,
    ):
        """
        Initialize acclimation monitor.

        Args:
            species: Species identifier for temporal gating
            baseline_alarm_rate: Natural alarm rate (no AI)
            estimated_colony_size: Approximate colony size
            alert_window: Number of recent events to analyze
        """
        self.species = species
        self.baseline_alarm_rate = baseline_alarm_rate
        self.colony_size = estimated_colony_size
        self.alert_window = alert_window

        # Create MFAS calculator
        from ethological_validation import create_mfas_for_species
        self.mfas = create_mfas_for_species(species)

        # History
        self.playback_history: Deque[PlaybackEvent] = deque(maxlen=alert_window)
        self.response_history: Deque[ColonyResponse] = deque(maxlen=alert_window)
        self.alerts: Deque[AcclimationAlert] = deque(maxlen=1000)

        # Thresholds
        self.thresholds = {
            "alarm_rate_warning": 0.10,  # 2x baseline
            "alarm_rate_critical": 0.15,  # 3x baseline
            "agitation_warning": 0.5,
            "agitation_critical": 0.7,
            "mfas_warning": 0.4,  # Below this = poor acceptance
            "mfas_critical": 0.3,
        }

        logger.info(
            f"AcclimationMonitor initialized for {species}, "
            f"baseline alarm rate: {baseline_alarm_rate:.1%}"
        )

    def record_playback(
        self,
        affect_vector: np.ndarray,
        syntactic_token: int,
        emitter_id: int,
        target_bat_id: Optional[int] = None,
        duration_ms: float = 200,
    ) -> PlaybackEvent:
        """Record an AI playback event."""
        import time

        event = PlaybackEvent(
            timestamp_ms=time.time() * 1000,
            affect_vector=affect_vector,
            syntactic_token=syntactic_token,
            emitter_id=emitter_id,
            target_bat_id=target_bat_id,
            duration_ms=duration_ms,
        )

        self.playback_history.append(event)
        return event

    def start_response_window(
        self,
        playback_event: Optional[PlaybackEvent] = None,
    ) -> "ResponseWindow":
        """
        Start a new response collection window.

        Args:
            playback_event: The playback event (uses latest if None)

        Returns:
            ResponseWindow for collecting responses
        """
        if playback_event is None and self.playback_history:
            playback_event = self.playback_history[-1]

        return ResponseWindow(
            monitor=self,
            playback_event=playback_event,
        )

    def add_response(
        self,
        response: ColonyResponse,
    ) -> None:
        """Add a colony response to history."""
        self.response_history.append(response)

        # Check for alerts
        self._check_alerts(response)

    def _check_alerts(self, response: ColonyResponse) -> None:
        """Check response against thresholds and generate alerts."""
        metrics = response.compute_metrics()

        # Check alarm rate
        alarm_rate = metrics["alarm_rate"]
        if alarm_rate > self.thresholds["alarm_rate_critical"]:
            self._create_alert(
                severity="critical",
                metric_name="alarm_rate",
                current_value=alarm_rate,
                threshold=self.thresholds["alarm_rate_critical"],
                message=f"Alarm rate {alarm_rate:.1%} exceeds critical threshold",
                recommended_action="Pause playback immediately, check synthesis artifacts",
            )
        elif alarm_rate > self.thresholds["alarm_rate_warning"]:
            self._create_alert(
                severity="warning",
                metric_name="alarm_rate",
                current_value=alarm_rate,
                threshold=self.thresholds["alarm_rate_warning"],
                message=f"Alarm rate {alarm_rate:.1%} elevated",
                recommended_action="Reduce playback volume or adjust affect parameters",
            )

        # Check agitation
        agitation = metrics["colony_agitation"]
        if agitation > self.thresholds["agitation_critical"]:
            self._create_alert(
                severity="critical",
                metric_name="agitation",
                current_value=agitation,
                threshold=self.thresholds["agitation_critical"],
                message=f"Colony agitation {agitation:.2f} critically high",
                recommended_action="Pause playback, allow colony to settle",
            )
        elif agitation > self.thresholds["agitation_warning"]:
            self._create_alert(
                severity="warning",
                metric_name="agitation",
                current_value=agitation,
                threshold=self.thresholds["agitation_warning"],
                message=f"Colony agitation {agitation:.2f} elevated",
                recommended_action="Monitor closely, consider reducing arousal in output",
            )

        # Check MFAS
        mean_mfas = metrics["mean_mfas"]
        if mean_mfas < self.thresholds["mfas_critical"] and metrics["response_count"] > 0:
            self._create_alert(
                severity="warning",
                metric_name="mfas",
                current_value=mean_mfas,
                threshold=self.thresholds["mfas_critical"],
                message=f"MFAS {mean_mfas:.2f} indicates poor acceptance",
                recommended_action="Review DDSP synthesis quality, check for phase artifacts",
            )

        # Check for flight response
        if metrics["flight_to_exits"]:
            self._create_alert(
                severity="critical",
                metric_name="flight_response",
                current_value=1.0,
                threshold=0.0,
                message="Flight to exits detected",
                recommended_action="Emergency stop, colony perceives threat",
            )

    def _create_alert(
        self,
        severity: str,
        metric_name: str,
        current_value: float,
        threshold: float,
        message: str,
        recommended_action: str,
    ) -> None:
        """Create and store an alert."""
        import time

        alert = AcclimationAlert(
            timestamp_ms=time.time() * 1000,
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            message=message,
            recommended_action=recommended_action,
        )

        self.alerts.append(alert)

        logger.warning(
            f"[{severity.upper()}] {message} (value={current_value:.2f}, "
            f"threshold={threshold:.2f})"
        )

    def get_summary(self) -> Dict:
        """Get summary of acclimation progress."""
        if not self.response_history:
            return {
                "playback_count": 0,
                "response_count": 0,
                "mean_alarm_rate": 0.0,
                "mean_mfas": 0.0,
                "alert_count": 0,
            }

        responses = list(self.response_history)

        # Aggregate metrics
        total_responses = sum(len(r.responding_bat_ids) for r in responses)
        total_alarms = sum(len(r.alarm_bat_ids) for r in responses)

        all_mfas = []
        for r in responses:
            all_mfas.extend([m.mfas_score for m in r.mfas_results])

        # Compute alarm rate (alarms / total responses)
        alarm_rate = total_alarms / max(total_responses, 1)

        # Compute agitation trend
        recent_agitation = [r.colony_agitation for r in responses[-10:]]

        return {
            "playback_count": len(self.playback_history),
            "response_count": total_responses,
            "mean_alarm_rate": alarm_rate,
            "baseline_alarm_rate": self.baseline_alarm_rate,
            "alarm_rate_ratio": alarm_rate / self.baseline_alarm_rate,
            "mean_mfas": np.mean(all_mfas) if all_mfas else 0.0,
            "std_mfas": np.std(all_mfas) if all_mfas else 0.0,
            "mean_agitation": np.mean(recent_agitation) if recent_agitation else 0.0,
            "alert_count": len(self.alerts),
            "critical_alert_count": sum(
                1 for a in self.alerts if a.severity == "critical"
            ),
        }

    def is_ready_for_closed_loop(self) -> Tuple[bool, str]:
        """
        Check if colony is ready for closed-loop interaction.

        Returns:
            (ready, reason) tuple
        """
        summary = self.get_summary()

        # Must have enough data
        if summary["playback_count"] < 50:
            return False, f"Need more playbacks ({summary['playback_count']}/50)"

        # Alarm rate must be close to baseline
        if summary["alarm_rate_ratio"] > 1.5:
            return False, f"Alarm rate too high ({summary['mean_alarm_rate']:.1%})"

        # Mean MFAS must be acceptable
        if summary["mean_mfas"] < 0.5:
            return False, f"MFAS too low ({summary['mean_mfas']:.2f})"

        # No recent critical alerts
        recent_critical = sum(
            1 for a in self.alerts
            if a.severity == "critical" and
            (self.alerts[-1].timestamp_ms - a.timestamp_ms) < 300000  # 5 min
        ) if self.alerts else 0

        if recent_critical > 0:
            return False, f"Recent critical alerts ({recent_critical})"

        # Agitation must be low
        if summary["mean_agitation"] > 0.4:
            return False, f"Agitation too high ({summary['mean_agitation']:.2f})"

        return True, "Colony ready for closed-loop interaction"

    def save_report(self, filepath: str) -> None:
        """Save acclimation report to file."""
        summary = self.get_summary()

        with open(filepath, "w") as f:
            f.write("Acclimation Phase Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Species: {self.species}\n")
            f.write(f"Colony Size: ~{self.colony_size}\n\n")

            f.write("Summary:\n")
            f.write(f"  Playbacks: {summary['playback_count']}\n")
            f.write(f"  Responses: {summary['response_count']}\n")
            f.write(f"  Mean Alarm Rate: {summary['mean_alarm_rate']:.1%}\n")
            f.write(f"  Baseline Alarm Rate: {summary['baseline_alarm_rate']:.1%}\n")
            f.write(f"  Alarm Rate Ratio: {summary['alarm_rate_ratio']:.2f}x\n")
            f.write(f"  Mean MFAS: {summary['mean_mfas']:.2f} ± {summary['std_mfas']:.2f}\n")
            f.write(f"  Mean Agitation: {summary['mean_agitation']:.2f}\n\n")

            f.write(f"Alerts: {summary['alert_count']}\n")
            f.write(f"  Critical: {summary['critical_alert_count']}\n\n")

            ready, reason = self.is_ready_for_closed_loop()
            f.write(f"Ready for Closed-Loop: {ready}\n")
            f.write(f"  Reason: {reason}\n")

            if len(self.alerts) > 0:
                f.write("\nRecent Alerts:\n")
                for alert in list(self.alerts)[-10:]:
                    f.write(
                        f"  [{alert.severity.upper()}] {alert.message}\n"
                    )

        logger.info(f"Report saved to {filepath}")


class ResponseWindow:
    """
    Context manager for collecting colony responses to a playback.

    Usage:
        with monitor.start_response_window() as window:
            # Collect individual bat responses
            window.add_bat_response(bat_id, arousal, is_alarm, interaction)
    """

    def __init__(
        self,
        monitor: AcclimationMonitor,
        playback_event: Optional[PlaybackEvent] = None,
        window_duration_ms: float = 5000,
    ):
        """
        Initialize response window.

        Args:
            monitor: AcclimationMonitor instance
            playback_event: Associated playback event
            window_duration_ms: How long to collect responses (ms)
        """
        self.monitor = monitor
        self.playback_event = playback_event
        self.window_duration = window_duration_ms

        self.start_time_ms: Optional[float] = None
        self.response = ColonyResponse(
            timestamp_ms=0,
            playback_event=playback_event or PlaybackEvent(
                timestamp_ms=0,
                affect_vector=np.zeros(16),
                syntactic_token=0,
                emitter_id=0,
            ),
        )

    def __enter__(self) -> "ResponseWindow":
        """Start response collection window."""
        import time
        self.start_time_ms = time.time() * 1000
        self.response.timestamp_ms = self.start_time_ms
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End response collection and add to monitor."""
        self.monitor.add_response(self.response)

    def add_bat_response(
        self,
        bat_id: int,
        arousal: float,
        is_alarm: bool,
        interaction: Optional[InteractionEvent] = None,
    ) -> None:
        """
        Add an individual bat's response.

        Args:
            bat_id: Responding bat ID
            arousal: Arousal level (0-1) from VAE
            is_alarm: Whether this was an alarm call
            interaction: Optional InteractionEvent for MFAS scoring
        """
        # Score interaction if provided
        mfas_result = None
        if interaction is not None:
            mfas_result = self.monitor.mfas.evaluate_interaction(interaction)

        self.response.add_response(
            bat_id=bat_id,
            is_alarm=is_alarm,
            arousal=arousal,
            interaction=interaction,
            mfas_result=mfas_result,
        )

        # Update colony agitation (weighted by arousal)
        self.response.colony_agitation += arousal * (1.5 if is_alarm else 1.0)

        # Normalize agitation
        if len(self.response.responding_bat_ids) > 0:
            self.response.colony_agitation /= len(self.response.responding_bat_ids)


class DeploymentDashboard:
    """
    Real-time monitoring dashboard for closed-loop deployment.

    Extends acclimation monitoring with additional metrics for
    interactive engagement.
    """

    def __init__(self, acclimation_monitor: AcclimationMonitor):
        """
        Initialize deployment dashboard.

        Args:
            acclimation_monitor: Base acclimation monitor
        """
        self.monitor = acclimation_monitor

        # Additional metrics for closed-loop
        self.conversation_stats: Dict[int, Dict] = {}  # Per-bat stats
        self.interaction_count = 0
        self.successful_interactions = 0  # MFAS > 0.7

    def record_interaction(
        self,
        bat_id: int,
        turn_count: int,
        final_mfas: float,
    ) -> None:
        """Record a completed conversation."""
        self.interaction_count += 1
        if final_mfas > 0.7:
            self.successful_interactions += 1

        # Update per-bat stats
        if bat_id not in self.conversation_stats:
            self.conversation_stats[bat_id] = {
                "conversations": 0,
                "total_turns": 0,
                "mean_mfas": [],
            }

        self.conversation_stats[bat_id]["conversations"] += 1
        self.conversation_stats[bat_id]["total_turns"] += turn_count
        self.conversation_stats[bat_id]["mean_mfas"].append(final_mfas)

    def get_dashboard_metrics(self) -> Dict:
        """Get all dashboard metrics."""
        acclimation_summary = self.monitor.get_summary()

        # Compute conversation success rate
        success_rate = (
            self.successful_interactions / max(self.interaction_count, 1)
            if self.interaction_count > 0 else 0
        )

        # Per-bat statistics
        bat_stats = {}
        for bat_id, stats in self.conversation_stats.items():
            bat_stats[bat_id] = {
                "conversations": stats["conversations"],
                "mean_turns": (
                    stats["total_turns"] / max(stats["conversations"], 1)
                ),
                "mean_mfas": (
                    np.mean(stats["mean_mfas"])
                    if stats["mean_mfas"] else 0
                ),
            }

        return {
            **acclimation_summary,
            "interaction_count": self.interaction_count,
            "successful_interactions": self.successful_interactions,
            "success_rate": success_rate,
            "active_bats": len(self.conversation_stats),
            "bat_stats": bat_stats,
        }


# Preset configurations

# Egyptian Fruit Bat acclimation monitor
BAT_ACCLIMATION_MONITOR = AcclimationMonitor(
    species="rousettus_aegyptiacus",
    baseline_alarm_rate=0.05,
    estimated_colony_size=50,
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Acclimation Monitor Demo")
    print("=" * 50)

    monitor = BAT_ACCLIMATION_MONITOR

    # Simulate some playbacks and responses
    for i in range(20):
        # Record playback
        affect = np.random.randn(16) * 0.1  # Low arousal
        monitor.record_playback(
            affect_vector=affect,
            syntactic_token=5,
            emitter_id=0,
        )

        # Simulate responses
        with monitor.start_response_window() as window:
            # Most bats respond normally
            for bat_id in range(5):
                window.add_bat_response(
                    bat_id=bat_id,
                    arousal=np.random.uniform(0.1, 0.3),
                    is_alarm=False,
                )

            # Occasional alarm
            if np.random.random() < 0.1:
                window.add_bat_response(
                    bat_id=99,
                    arousal=0.8,
                    is_alarm=True,
                )

    # Get summary
    summary = monitor.get_summary()

    print(f"\nAcclimation Summary:")
    print(f"  Playbacks: {summary['playback_count']}")
    print(f"  Responses: {summary['response_count']}")
    print(f"  Alarm Rate: {summary['mean_alarm_rate']:.1%} "
          f"(vs baseline {summary['baseline_alarm_rate']:.1%})")
    print(f"  Mean MFAS: {summary['mean_mfas']:.2f}")
    print(f"  Alerts: {summary['alert_count']}")
    print(f"  Critical: {summary['critical_alert_count']}")

    # Check readiness
    ready, reason = monitor.is_ready_for_closed_loop()
    print(f"\nReady for Closed-Loop: {ready}")
    print(f"  Reason: {reason}")
