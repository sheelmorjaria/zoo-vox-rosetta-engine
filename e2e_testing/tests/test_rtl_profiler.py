# RTL Profiler Tests
# ==================
#
# Tests for round-trip latency measurement with sync pulse correlation.

import pytest
from e2e_testing.rtl_profiler import RoundTripProfiler, RTLStatistics, SyncPulseRecord
from datetime import datetime


class TestSyncPulseRecord:
    """Test sync pulse record data structure."""

    def test_pulse_record_creation(self):
        """Test creating a pulse record."""
        record = SyncPulseRecord(
            pulse_id=1,
            ptp_timestamp_ns=1234567890,
            injection_time_ns=1234567890,
        )
        assert record.pulse_id == 1
        assert record.ptp_timestamp_ns == 1234567890


class TestRoundTripProfiler:
    """Test RTL measurement engine."""

    def test_profiler_creation(self):
        """Test creating RTL profiler."""
        profiler = RoundTripProfiler(target_rtl_ms=50.0)
        assert profiler.target_rtl_ms == 50.0
        assert len(profiler.rtl_history) == 0

    def test_record_injection(self):
        """Test recording sync pulse injection."""
        profiler = RoundTripProfiler()
        ptp_ts = 1234567890
        injection_time = 1234567890

        profiler.record_injection(ptp_ts, injection_time)
        assert ptp_ts in profiler.sync_pulse_injections
        assert profiler.sync_pulse_injections[ptp_ts] == injection_time

    def test_record_detection(self):
        """Test recording sync pulse detection."""
        profiler = RoundTripProfiler()
        profiler.record_injection(100, 100)
        profiler.record_detection(100, 50_000_100)  # 50ms RTL

        assert len(profiler.rtl_history) == 1
        assert profiler.rtl_history[0] == 50.0  # 50,000,000ns = 50.0ms

    def test_rtl_within_budget(self):
        """Test RTL within target budget."""
        profiler = RoundTripProfiler(target_rtl_ms=50.0)
        profiler.record_injection(100, 100)
        profiler.record_detection(100, 25_000_000)  # 25ms RTL

        stats = profiler.get_statistics()
        assert stats.mean_rtl_ms < 50.0

    def test_rtl_percentiles(self):
        """Test RTL percentile calculation."""
        profiler = RoundTripProfiler()
        profiler.record_injection(100, 100)
        profiler.record_detection(100, 10_000_000)  # 10ms
        profiler.record_injection(200, 200)
        profiler.record_detection(200, 20_000_000)  # 20ms
        profiler.record_injection(300, 300)
        profiler.record_detection(300, 30_000_000)  # 30ms

        stats = profiler.get_statistics()
        assert stats.p50_rtl_ms == pytest.approx(20.0, rel=0.1)
        assert stats.p95_rtl_ms == pytest.approx(30.0, rel=0.1)
        assert stats.max_rtl_ms == pytest.approx(30.0, rel=0.1)

    def test_latency_violation_alert(self):
        """Test latency violation detection."""
        profiler = RoundTripProfiler(target_rtl_ms=50.0)
        profiler.record_injection(100, 100)

        # This should raise an exception
        with pytest.raises(Exception) as exc_info:
            profiler.record_detection(100, 60_000_000)  # 60ms RTL

        assert "exceeded" in str(exc_info.value).lower()

    def test_nbd_confidence_tracking(self):
        """Test Predictive NBD confidence tracking."""
        profiler = RoundTripProfiler()

        # Record normal confidence levels
        profiler.record_nbd_confidence(0.8, "Phonetic")
        profiler.record_nbd_confidence(0.9, "Syllable")
        profiler.record_nbd_confidence(0.7, "Phrase")

        stats = profiler.get_statistics()
        assert stats.nbd_confidence_mean == pytest.approx(0.8, rel=0.1)

    def test_nbd_low_confidence_warning(self):
        """Test low NBD confidence tracking."""
        profiler = RoundTripProfiler()

        # Record many low confidence readings
        for _ in range(15):
            profiler.record_nbd_confidence(0.5, "Phonetic")

        stats = profiler.get_statistics()
        assert stats.low_confidence_count == 15

    def test_empty_statistics(self):
        """Test statistics with no data."""
        profiler = RoundTripProfiler()
        stats = profiler.get_statistics()

        assert stats.mean_rtl_ms == 0.0
        assert stats.p50_rtl_ms == 0.0
        assert stats.max_rtl_ms == 0.0


class TestRTLStatistics:
    """Test RTL statistics data structure."""

    def test_statistics_creation(self):
        """Test creating statistics."""
        stats = RTLStatistics(
            mean_rtl_ms=25.0,
            p50_rtl_ms=24.0,
            p95_rtl_ms=45.0,
            p99_rtl_ms=48.0,
            max_rtl_ms=50.0,
            sample_count=1000,
            nbd_confidence_mean=0.85,
            nbd_confidence_p5=0.7,
            low_confidence_rate=0.02,
        )
        assert stats.mean_rtl_ms == 25.0
        assert stats.sample_count == 1000
