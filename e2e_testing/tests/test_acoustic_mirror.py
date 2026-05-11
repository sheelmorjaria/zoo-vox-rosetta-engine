# Acoustic Mirror Tests
# =====================
#
# Tests for feedback loop resistance validation via digital loopback.

import pytest
import time
from e2e_testing.acoustic_mirror_tester import AcousticMirrorTester, FeedbackLoopDetected


class TestAcousticMirrorTester:
    """Test feedback loop detection logic."""

    def test_tester_creation(self):
        """Test creating acoustic mirror tester."""
        tester = AcousticMirrorTester(max_interactions_per_minute=30)
        assert tester.max_ipm == 30
        assert len(tester.interaction_timestamps) == 0

    def test_log_interaction(self):
        """Test logging interactions."""
        tester = AcousticMirrorTester()
        timestamp_ms = 1000

        tester.log_interaction(timestamp_ms)
        assert len(tester.interaction_timestamps) == 1

    def test_old_timestamps_removed(self):
        """Test that old timestamps are removed."""
        tester = AcousticMirrorTester()
        now = 60000  # 60 seconds in ms

        # Add interactions over 70 seconds ago
        for i in range(10):
            tester.log_interaction(now - 70000 + i * 1000)

        # Log current interaction
        tester.log_interaction(now)

        # Old interactions should be removed
        assert len(tester.interaction_timestamps) == 1

    def test_feedback_loop_detected(self):
        """Test feedback loop detection."""
        tester = AcousticMirrorTester(max_interactions_per_minute=6)
        now = 60000

        # Log 6 interactions within 1 minute (at threshold)
        for i in range(6):
            tester.log_interaction(now + i * 1000)

        # Should detect feedback loop on 7th interaction
        with pytest.raises(FeedbackLoopDetected) as exc_info:
            tester.log_interaction(now + 6000)

        assert "interaction rate" in str(exc_info.value).lower()

    def test_interaction_rate_within_limit(self):
        """Test normal interaction rate."""
        tester = AcousticMirrorTester(max_interactions_per_minute=30)
        now = 60000

        # Log 20 interactions within 1 minute (under threshold)
        for i in range(20):
            tester.log_interaction(now + i * 1000)

        # Should not raise
        tester.log_interaction(now + 20000)

    def test_get_interaction_rate(self):
        """Test getting current interaction rate."""
        tester = AcousticMirrorTester()
        now = 60000

        # Add 10 interactions
        for i in range(10):
            tester.log_interaction(now + i * 1000)

        rate = tester.get_interaction_rate(now + 10000)
        assert rate == 10  # 10 interactions in last 60 seconds

    def test_reset(self):
        """Test resetting state."""
        tester = AcousticMirrorTester()
        tester.log_interaction(1000)
        tester.log_interaction(2000)

        tester.reset()
        assert len(tester.interaction_timestamps) == 0


class TestFeedbackLoopDetected:
    """Test FeedbackLoopDetected exception."""

    def test_exception_creation(self):
        """Test creating exception."""
        exc = FeedbackLoopDetected("Test message")
        assert "Test message" in str(exc)

    def test_exception_with_rate(self):
        """Test exception with rate information."""
        exc = FeedbackLoopDetected("35 interactions per minute")
        assert "35" in str(exc)
