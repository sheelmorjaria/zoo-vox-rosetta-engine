#!/usr/bin/env python3
"""
TDD Tests for Self Heal (Phase 2: Rehydrator)
==============================================

Test-Driven Development tests for system self-healing and recovery.

Tests are written FIRST (Red phase), then implementation follows (Green phase).

Run with: pytest tests/test_self_heal.py -v

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import os
import signal
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Import modules to test (will fail initially - RED phase)
# Implementation comes later
from system.self_heal import HealthStatus, SelfHeal


class TestProcessHealthDetection(unittest.TestCase):
    """Test 2.1: Process Detection - Detect dead Python process"""

    def test_detect_alive_python_process(self):
        """
        RED TEST: Detect alive Python process

        Scenario:
        - Start a long-lived Python process
        - Get its PID
        - Action: self_heal.check_health(pid)

        Expected:
        - Returns HealthStatus.ALIVE
        - Process is running
        """
        # Create a simple long-lived Python process
        # Using sleep command as a mock process
        import subprocess
        proc = subprocess.Popen(
            ["python3", "-c", "import time; time.sleep(10)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        healer = SelfHeal(checkpoint_dir=Path(tempfile.gettempdir()))

        try:
            # Act - Check health of alive process
            status = healer.check_health(proc.pid)

            # Assert - Should be alive
            self.assertEqual(status, HealthStatus.ALIVE)
            self.assertTrue(healer.is_process_alive(proc.pid))

        finally:
            # Clean up
            proc.terminate()
            proc.wait(timeout=5)

    def test_detect_dead_python_process(self):
        """
        RED TEST: Detect dead Python process

        Scenario:
        - Start a Python process
        - Kill it
        - Action: self_heal.check_health(pid)

        Expected:
        - Returns HealthStatus.DEAD
        - Process is not running
        """
        # Create a short-lived process
        import subprocess
        proc = subprocess.Popen(
            ["python3", "-c", "pass"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Wait for it to finish
        proc.wait(timeout=5)

        healer = SelfHeal(checkpoint_dir=Path(tempfile.gettempdir()))

        # Act - Check health of dead process
        status = healer.check_health(proc.pid)

        # Assert - Should be dead
        self.assertEqual(status, HealthStatus.DEAD)
        self.assertFalse(healer.is_process_alive(proc.pid))

    def test_detect_nonexistent_process(self):
        """
        RED TEST: Detect nonexistent process

        Scenario:
        - Check a PID that never existed

        Expected:
        - Returns HealthStatus.DEAD
        """
        healer = SelfHeal(checkpoint_dir=Path(tempfile.gettempdir()))

        # Use a very high PID that likely doesn't exist
        fake_pid = 999999

        # Act - Check health of nonexistent process
        status = healer.check_health(fake_pid)

        # Assert - Should be dead
        self.assertEqual(status, HealthStatus.DEAD)


class TestStateRehydration(unittest.TestCase):
    """Test 2.2: State Rehydration - Load and inject state"""

    def test_rehydrate_agent_state(self):
        """
        RED TEST: Rehydrate ContextualAgent from checkpoint

        Scenario:
        - Checkpoint contains ContextualAgent state
        - Context='FOOD', History=['PheeA', 'PheeB']
        - Action: self_heal.rehydrate_agent(checkpoint_path)

        Expected:
        - Returns agent state dictionary
        - context='FOOD', history=['PheeA', 'PheeB']
        - Can be used to recreate agent
        """
        healer = SelfHeal(checkpoint_dir=Path(tempfile.gettempdir()))

        # Create a checkpoint file
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            # Create checkpoint data
            checkpoint_data = {
                "component": "contextual_agent",
                "context": "FOOD",
                "history": ["PheeA", "PheeB"],
                "dialogue_state": {"turn": 3, "initiator": "human"},
                "timestamp": "2025-01-07T12:00:00Z"
            }

            # Write checkpoint
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f)

            # Act - Rehydrate agent state
            agent_state = healer.rehydrate_agent(checkpoint_path)

            # Assert - State is correctly loaded
            self.assertIsNotNone(agent_state)
            self.assertEqual(agent_state["context"], "FOOD")
            self.assertEqual(agent_state["history"], ["PheeA", "PheeB"])
            self.assertEqual(agent_state["dialogue_state"]["turn"], 3)

    def test_rehydrate_from_latest_checkpoint(self):
        """
        RED TEST: Rehydrate from latest checkpoint automatically

        Scenario:
        - Multiple checkpoints exist in checkpoint directory
        - Action: self_heal.rehydrate_from_latest()

        Expected:
        - Loads most recent checkpoint
        - Returns complete system state
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            healer = SelfHeal(checkpoint_dir=checkpoint_dir)

            # Create multiple checkpoints with different timestamps
            for i in range(3):
                checkpoint_data = {
                    "component": "contextual_agent",
                    "context": f"CONTEXT_{i}",
                    "history": [],
                    "dialogue_state": {"turn": i},
                    "timestamp": f"2025-01-07T12:00:0{i}Z"
                }
                checkpoint_path = checkpoint_dir / f"checkpoint_0{i}.json"
                with open(checkpoint_path, "w") as f:
                    json.dump(checkpoint_data, f)

            # Give small delays to ensure different mtimes
            time.sleep(0.1)

            # Create latest checkpoint
            latest_data = {
                "component": "contextual_agent",
                "context": "LATEST",
                "history": ["PheeA"],
                "dialogue_state": {"turn": 5},
                "timestamp": "2025-01-07T12:00:05Z"
            }
            latest_path = checkpoint_dir / "checkpoint_latest.json"
            with open(latest_path, "w") as f:
                json.dump(latest_data, f)

            # Act - Rehydrate from latest checkpoint
            system_state = healer.rehydrate_from_latest()

            # Assert - Latest checkpoint loaded
            self.assertEqual(system_state["context"], "LATEST")
            self.assertEqual(system_state["dialogue_state"]["turn"], 5)

    def test_rehydrate_with_missing_checkpoint(self):
        """
        RED TEST: Handle missing checkpoint file gracefully

        Scenario:
        - Checkpoint file doesn't exist
        - Action: self_heal.rehydrate_agent(nonexistent_path)

        Expected:
        - Returns None
        - Logs error message
        """
        healer = SelfHeal(checkpoint_dir=Path(tempfile.gettempdir()))

        # Act - Try to load nonexistent checkpoint
        result = healer.rehydrate_agent(Path("/nonexistent/checkpoint.json"))

        # Assert - Returns None gracefully
        self.assertIsNone(result)


class TestRustCacheSynchronization(unittest.TestCase):
    """Test 2.3: Rust Cache Sync - Synchronize cache after recovery"""

    def test_sync_rust_cache_after_heal(self):
        """
        RED TEST: Synchronize Rust LRU cache after recovery

        Scenario:
        - Checkpoint contains rust_cache_keys=['clip_A', 'clip_B']
        - Action: self_heal.sync_rust_cache(checkpoint_path)

        Expected:
        - Extracts cache keys from checkpoint
        - Returns list of keys for preloading
        - Rust can use this to warm-start cache
        """
        healer = SelfHeal(checkpoint_dir=Path(tempfile.gettempdir()))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            # Create checkpoint with Rust cache state
            checkpoint_data = {
                "component": "rust_cache",
                "rust_cache_keys": ["clip_A", "clip_B", "clip_C"],
                "cache_size_mb": 12.5,
                "cache_hit_rate": 0.85,
                "timestamp": "2025-01-07T12:00:00Z"
            }

            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f)

            # Act - Sync Rust cache
            cache_keys = healer.sync_rust_cache(checkpoint_path)

            # Assert - Cache keys extracted
            self.assertEqual(len(cache_keys), 3)
            self.assertIn("clip_A", cache_keys)
            self.assertIn("clip_B", cache_keys)
            self.assertIn("clip_C", cache_keys)

    def test_sync_empty_rust_cache(self):
        """
        RED TEST: Handle empty Rust cache (cold start scenario)

        Scenario:
        - Checkpoint contains empty rust_cache_keys=[]
        - Action: self_heal.sync_rust_cache(checkpoint_path)

        Expected:
        - Returns empty list
        - Indicates cold start needed
        """
        healer = SelfHeal(checkpoint_dir=Path(tempfile.gettempdir()))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            # Create checkpoint with empty cache
            checkpoint_data = {
                "component": "rust_cache",
                "rust_cache_keys": [],
                "cache_size_mb": 0.0,
                "cache_hit_rate": 0.0,
                "timestamp": "2025-01-07T12:00:00Z"
            }

            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f)

            # Act - Sync Rust cache
            cache_keys = healer.sync_rust_cache(checkpoint_path)

            # Assert - Empty list returned
            self.assertEqual(len(cache_keys), 0)

    def test_sync_rust_cache_from_complete_checkpoint(self):
        """
        RED TEST: Extract Rust cache from complete system checkpoint

        Scenario:
        - Complete system checkpoint contains rust_cache section
        - Action: self_heal.sync_rust_cache(checkpoint_path)

        Expected:
        - Extracts rust_cache nested data
        - Returns cache keys
        """
        healer = SelfHeal(checkpoint_dir=Path(tempfile.gettempdir()))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            # Create complete system checkpoint
            checkpoint_data = {
                "contextual_agent": {
                    "context": "FOOD",
                    "history": []
                },
                "cognitive_engine": {
                    "current_intensity": 0.7
                },
                "rust_cache": {
                    "cache_keys": ["clip_X", "clip_Y"],
                    "cache_size_mb": 15.0
                },
                "semiotic_engine": {
                    "deception_accumulator": 0.0
                },
                "metadata": {
                    "timestamp": "2025-01-07T12:00:00Z",
                    "version": "1.0.0"
                }
            }

            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f)

            # Act - Sync Rust cache from complete checkpoint
            cache_keys = healer.sync_rust_cache(checkpoint_path)

            # Assert - Cache keys extracted from nested structure
            self.assertEqual(len(cache_keys), 2)
            self.assertIn("clip_X", cache_keys)
            self.assertIn("clip_Y", cache_keys)


class TestCompleteHealingWorkflow(unittest.TestCase):
    """Test 2.4: Complete Healing Workflow - End-to-end recovery"""

    @patch('subprocess.Popen')
    def test_full_recovery_workflow(self, mock_popen):
        """
        RED TEST: Complete healing workflow from detection to restart

        Scenario:
        - Python process is dead
        - Latest checkpoint exists
        - Action: self_heal.heal()

        Expected:
        - Detects dead process
        - Loads latest checkpoint
        - Restarts Python process
        - Returns success status
        """
        # Mock process that will be restarted
        mock_proc = Mock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            healer = SelfHeal(checkpoint_dir=checkpoint_dir)

            # Create a checkpoint
            checkpoint_data = {
                "component": "contextual_agent",
                "context": "FOOD",
                "history": ["PheeA"],
                "dialogue_state": {"turn": 1},
                "timestamp": "2025-01-07T12:00:00Z"
            }
            checkpoint_path = checkpoint_dir / "checkpoint_20250107_120000.json"
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f)

            # Act - Perform full healing
            success = healer.heal(pid=99999, restart_command=["python3", "-m", "cognitive_agent"])

            # Assert - Healing succeeded
            self.assertTrue(success)
            # Verify restart was attempted
            mock_popen.assert_called_once()

    def test_heal_without_checkpoint(self):
        """
        RED TEST: Handle healing when no checkpoint exists

        Scenario:
        - Python process is dead
        - No checkpoint available
        - Action: self_heal.heal()

        Expected:
        - Logs warning about missing checkpoint
        - Still restarts process (cold start)
        - Returns success status
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            healer = SelfHeal(checkpoint_dir=checkpoint_dir)

            # No checkpoint created

            # Act - Attempt healing without checkpoint
            # Use mock restart to avoid actually starting a process
            with patch('subprocess.Popen') as mock_popen:
                mock_proc = Mock()
                mock_proc.pid = 12345
                mock_popen.return_value = mock_proc

                success = healer.heal(
                    pid=99999,
                    restart_command=["python3", "-m", "cognitive_agent"]
                )

            # Assert - Still succeeds (cold start)
            self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
