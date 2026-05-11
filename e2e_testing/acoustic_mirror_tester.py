#!/usr/bin/env python3
"""
Acoustic Mirror Tester

Tests feedback loop resistance by routing AI output back to input via
digital loopback. Validates that the cognitive agent suppresses self-replies.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional
import time

logger = logging.getLogger(__name__)


class FeedbackLoopDetected(Exception):
    """Raised when interaction rate exceeds threshold (feedback loop)."""
    pass


@dataclass
class MirrorTestResult:
    """Result of acoustic mirror test."""
    passed: bool
    duration_seconds: float
    total_interactions: int
    interactions_per_minute: float
    max_ipm: float
    reason: Optional[str] = None


@dataclass
class InteractionEvent:
    """Record of a synthesized response."""
    timestamp_ms: float
    confidence: float
    was_self_reply: bool = False


class AcousticMirrorTester:
    """
    Tests feedback loop resistance during acoustic mirror test.

    Validates:
    - Interaction rate drops to near zero (agent suppresses self-replies)
    - Confidence-based suppression engages
    - No infinite feedback loops
    """

    def __init__(
        self,
        max_interactions_per_minute: int = 30,
        suppression_window_seconds: float = 60.0,
    ):
        """
        Initialize the acoustic mirror tester.

        Args:
            max_interactions_per_minute: Threshold for feedback loop detection
            suppression_window_seconds: Time window for IPM calculation
        """
        self.max_ipm = max_interactions_per_minute
        self.suppression_window_ms = suppression_window_seconds * 1000

        self.interaction_events: List[InteractionEvent] = []
        self.self_reply_count = 0
        self.start_time_ms: Optional[float] = None

        # Simple timestamp list for easier testing
        self.interaction_timestamps: List[float] = []

    def start_test(self) -> None:
        """Start the mirror test."""
        self.start_time_ms = time.time() * 1000
        self.interaction_events.clear()
        self.interaction_timestamps.clear()
        self.self_reply_count = 0
        logger.info("Acoustic mirror test started")

    def log_interaction(
        self,
        timestamp_ms: float,
        confidence: float = 1.0,
        is_self_reply: bool = False,
    ) -> None:
        """
        Log a synthesized response during mirror test.

        Args:
            timestamp_ms: Timestamp of the interaction in milliseconds
            confidence: Agent's confidence in this response
            is_self_reply: Whether this was a self-reply (should be suppressed)

        Raises:
            FeedbackLoopDetected: If interaction rate exceeds threshold
        """
        # Auto-start test if not already started
        if self.start_time_ms is None:
            self.start_time_ms = timestamp_ms

        # Add to simple timestamp list
        self.interaction_timestamps.append(timestamp_ms)

        event = InteractionEvent(
            timestamp_ms=timestamp_ms,
            confidence=confidence,
            was_self_reply=is_self_reply,
        )
        self.interaction_events.append(event)

        if is_self_reply:
            self.self_reply_count += 1

        # Clean old events outside suppression window
        cutoff_ms = timestamp_ms - self.suppression_window_ms
        self.interaction_events = [
            e for e in self.interaction_events
            if e.timestamp_ms > cutoff_ms
        ]
        self.interaction_timestamps = [
            ts for ts in self.interaction_timestamps
            if ts > cutoff_ms
        ]

        # Check IPM threshold
        current_ipm = len(self.interaction_events) / (self.suppression_window_ms / 60000)
        if current_ipm > self.max_ipm:
            raise FeedbackLoopDetected(
                f"Interaction rate exceeded: {current_ipm:.1f} IPM > {self.max_ipm} IPM"
            )

        logger.debug(
            f"Logged interaction (confidence={confidence:.2f}, IPM={current_ipm:.1f})"
        )

    def get_interaction_rate(self, current_time_ms: float) -> int:
        """
        Get number of interactions in the last 60 seconds.

        Args:
            current_time_ms: Current time in milliseconds

        Returns:
            Number of interactions in the last minute
        """
        cutoff_ms = current_time_ms - 60000
        return sum(1 for ts in self.interaction_timestamps if ts > cutoff_ms)

    def reset(self) -> None:
        """Reset all state."""
        self.interaction_events.clear()
        self.interaction_timestamps.clear()
        self.self_reply_count = 0
        self.start_time_ms = None

    def get_current_ipm(self) -> float:
        """
        Get current interactions per minute.

        Returns:
            IPM over the suppression window
        """
        if self.start_time_ms is None:
            return 0.0

        now_ms = time.time() * 1000
        cutoff_ms = now_ms - self.suppression_window_ms

        recent_events = [
            e for e in self.interaction_events
            if e.timestamp_ms > cutoff_ms
        ]

        return len(recent_events) / (self.suppression_window_ms / 60000)

    def get_self_reply_rate(self) -> float:
        """
        Get the rate of self-replies.

        Returns:
            Fraction of interactions that were self-replies
        """
        if not self.interaction_events:
            return 0.0

        return self.self_reply_count / len(self.interaction_events)

    def execute_mirror_test(
        self,
        duration_seconds: float = 300,
        target_ipm: float = 5.0,
    ) -> MirrorTestResult:
        """
        Run mirror test for specified duration.

        This is a stub - the actual test requires running the full pipeline.
        In production, this would coordinate with the Rust shadow mode pipeline.

        Args:
            duration_seconds: Test duration in seconds
            target_ipm: Target IPM after initial response period

        Returns:
            MirrorTestResult with test outcome
        """
        self.start_test()

        logger.info(
            f"Running mirror test for {duration_seconds}s "
            f"(target IPM: {target_ipm})"
        )

        # In production, this would:
        # 1. Enable digital loopback in Rust
        # 2. Stream corpus audio through pipeline
        # 3. Log each interaction via log_interaction()
        # 4. Monitor IPM and self-reply rate

        # For now, return a placeholder result
        elapsed = time.time() - (self.start_time_ms / 1000)

        return MirrorTestResult(
            passed=False,
            duration_seconds=elapsed,
            total_interactions=len(self.interaction_events),
            interactions_per_minute=self.get_current_ipm(),
            max_ipm=float(self.max_ipm),
            reason="Test stub - requires full pipeline integration",
        )

    def get_statistics(self) -> dict:
        """
        Get current test statistics.

        Returns:
            Dictionary with current statistics
        """
        return {
            "duration_seconds": (
                (time.time() * 1000 - self.start_time_ms) / 1000
                if self.start_time_ms
                else 0
            ),
            "total_interactions": len(self.interaction_events),
            "current_ipm": self.get_current_ipm(),
            "self_reply_count": self.self_reply_count,
            "self_reply_rate": self.get_self_reply_rate(),
            "max_ipm": self.max_ipm,
        }
