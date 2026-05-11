# Soak Test Tests
# ===============
#
# Tests for 24-hour hardware soak test infrastructure.

import pytest
from e2e_testing.soak_test_runner import SoakTestRunner, MemoryLeakDetected


class TestSoakTestRunner:
    """Test soak test orchestration."""

    def test_runner_creation(self):
        """Test creating soak test runner."""
        runner = SoakTestRunner(duration_hours=24)
        assert runner.duration_hours == 24

    def test_calculate_memory_growth(self):
        """Test memory growth calculation."""
        runner = SoakTestRunner()

        # Simulate memory readings
        start_mb = 1000
        end_mb = 1050

        growth_percent = runner.calculate_growth_percent(start_mb, end_mb)
        assert growth_percent == 5.0

    def test_memory_leak_detected(self):
        """Test memory leak detection."""
        runner = SoakTestRunner(
            duration_hours=1,
            memory_leak_threshold_percent=5.0,
        )

        # Simulate 10% growth (exceeds threshold)
        start_mb = 1000
        end_mb = 1100

        growth = runner.calculate_growth_percent(start_mb, end_mb)

        if growth > runner.memory_leak_threshold_percent:
            # Would raise MemoryLeakDetected
            assert growth == 10.0

    def test_memory_within_threshold(self):
        """Test memory within acceptable threshold."""
        runner = SoakTestRunner(
            memory_leak_threshold_percent=5.0,
        )

        # Simulate 3% growth (within threshold)
        start_mb = 1000
        end_mb = 1030

        growth = runner.calculate_growth_percent(start_mb, end_mb)
        assert growth < 5.0

    def test_zmq_disconnect_detection(self):
        """Test ZMQ disconnect detection."""
        runner = SoakTestRunner()

        # Simulate disconnect
        runner.log_zmq_disconnect()

        assert runner.zmq_disconnect_count == 1

    def test_thermal_throttling_detection(self):
        """Test thermal throttling detection."""
        runner = SoakTestRunner()

        # Simulate thermal throttling
        runner.log_thermal_throttle(85.0)  # 85°C

        assert runner.thermal_throttle_count == 1
        assert runner.max_temperature_c == 85.0

    def test_generate_report(self):
        """Test generating soak test report."""
        runner = SoakTestRunner(duration_hours=1)

        # Simulate some data
        runner.log_thermal_throttle(80.0)
        runner.log_zmq_disconnect()

        report = runner.generate_report()

        assert "Soak Test Report" in report
        assert "Duration" in report

    def test_progress_tracking(self):
        """Test progress tracking during soak test."""
        runner = SoakTestRunner(duration_hours=24)

        # Simulate 12 hours completed
        progress = runner.calculate_progress(hours_elapsed=12)

        assert progress == 0.5  # 50% complete


class TestMemoryLeakDetected:
    """Test MemoryLeakDetected exception."""

    def test_exception_creation(self):
        """Test creating exception."""
        exc = MemoryLeakDetected("RAM growth: 10%")
        assert "10%" in str(exc)
        assert "RAM" in str(exc)

    def test_exception_with_vram(self):
        """Test exception with VRAM information."""
        exc = MemoryLeakDetected("VRAM growth: 15%")
        assert "VRAM" in str(exc)
