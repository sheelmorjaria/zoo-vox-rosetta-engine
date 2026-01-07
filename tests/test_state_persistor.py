#!/usr/bin/env python3
"""
TDD Tests for State Persistor (Phase 1: Checkpointer)
====================================================

Test-Driven Development tests for system state checkpointing.

Tests are written FIRST (Red phase), then implementation follows (Green phase).

Run with: pytest tests/test_state_persistor.py -v

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

# Import modules to test (will fail initially - RED phase)
# Implementation comes later
from system.state_persistor import StatePersistor


class TestContextualAgentSerialization(unittest.TestCase):
    """Test 1.1: Context Serialization - Save conversational state"""

    def test_serialize_contextual_agent(self):
        """
        RED TEST: Save ContextualAgent state to checkpoint file

        Scenario:
        - ContextualAgent with Context='FOOD' and History=['PheeA', 'PheeB']
        - Action: persistor.save("checkpoint.json")

        Expected:
        - File exists
        - JSON contains context='FOOD' and history=['PheeA', 'PheeB']
        - File size is reasonable (<50KB)
        """
        # This will fail until we implement StatePersistor
        persistor = StatePersistor()

        # Create mock agent state
        agent_state = {
            "context": "FOOD",
            "history": ["PheeA", "PheeB"],
            "dialogue_state": {"turn": 3, "initiator": "human"}
        }

        # Act - Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            persistor.save_contextual_agent(agent_state, checkpoint_path)

            # Assert - File exists
            self.assertTrue(checkpoint_path.exists(), "Checkpoint file should exist")

            # Assert - JSON contains correct data
            with open(checkpoint_path) as f:
                loaded = json.load(f)

            self.assertEqual(loaded["context"], "FOOD")
            self.assertEqual(loaded["history"], ["PheeA", "PheeB"])
            self.assertEqual(loaded["dialogue_state"]["turn"], 3)

            # Assert - File size is reasonable (<50KB)
            file_size = checkpoint_path.stat().st_size
            self.assertLess(file_size, 50 * 1024, f"File too large: {file_size} bytes")

    def test_serialize_contextual_agent_with_empty_history(self):
        """
        RED TEST: Handle agent with no history (fresh start)

        Expected:
        - Checkpoint saves successfully
        - History is empty list
        """
        persistor = StatePersistor()

        agent_state = {
            "context": None,  # Fresh start
            "history": [],
            "dialogue_state": {"turn": 0, "initiator": None}
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            persistor.save_contextual_agent(agent_state, checkpoint_path)

            with open(checkpoint_path) as f:
                loaded = json.load(f)

            self.assertIsNone(loaded["context"])
            self.assertEqual(loaded["history"], [])
            self.assertEqual(loaded["dialogue_state"]["turn"], 0)


class TestRustCacheSerialization(unittest.TestCase):
    """Test 1.2: Granular Cache Serialization - Save Rust LRU state"""

    def test_serialize_rust_cache(self):
        """
        RED TEST: Save Rust LRU cache keys to checkpoint

        Scenario:
        - Mock Rust Cache with keys ['clip_A', 'clip_B']
        - Action: Save to checkpoint

        Expected:
        - Checkpoint contains rust_cache_keys
        - Keys can be loaded for preloading
        """
        persistor = StatePersistor()

        # Mock Rust cache state
        rust_state = {
            "rust_cache_keys": ["clip_A", "clip_B"],
            "cache_size_mb": 12.5,
            "cache_hit_rate": 0.85
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            persistor.save_rust_cache(rust_state, checkpoint_path)

            # Assert - Checkpoint contains cache keys
            with open(checkpoint_path) as f:
                loaded = json.load(f)

            self.assertIn("rust_cache_keys", loaded)
            self.assertEqual(loaded["rust_cache_keys"], ["clip_A", "clip_B"])
            self.assertEqual(loaded["cache_size_mb"], 12.5)
            self.assertEqual(loaded["cache_hit_rate"], 0.85)

    def test_serialize_empty_rust_cache(self):
        """
        RED TEST: Handle empty Rust cache (cold start)

        Expected:
        - Checkpoint saves with empty cache keys list
        - Can be distinguished from warm cache
        """
        persistor = StatePersistor()

        rust_state = {
            "rust_cache_keys": [],  # Cold start
            "cache_size_mb": 0.0,
            "cache_hit_rate": 0.0
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            persistor.save_rust_cache(rust_state, checkpoint_path)

            with open(checkpoint_path) as f:
                loaded = json.load(f)

            self.assertEqual(loaded["rust_cache_keys"], [])
            self.assertEqual(loaded["cache_size_mb"], 0.0)


class TestCompleteCheckpoint(unittest.TestCase):
    """Test 1.3: Full System Checkpoint - Combined state"""

    def test_save_complete_system_state(self):
        """
        RED TEST: Save complete system state in one operation

        Scenario:
        - ContextualAgent: Context='AGGRESSION', History=['ChirpA']
        - CognitiveEngine: intensity=0.7, persona_id='dominant'
        - Rust Cache: ['clip_X', 'clip_Y']

        Expected:
        - Single checkpoint file contains all state
        - Can be loaded by self_heal.py for recovery
        - All components represented in JSON
        """
        persistor = StatePersistor()

        # Complete system state
        system_state = {
            "contextual_agent": {
                "context": "AGGRESSION",
                "history": ["ChirpA"],
                "dialogue_state": {"turn": 1, "initiator": "marmoset"}
            },
            "cognitive_engine": {
                "current_intensity": 0.7,
                "active_persona_id": "dominant",
                "acoustic_vectors": {"f0_mean": 8500.0}
            },
            "rust_cache": {
                "cache_keys": ["clip_X", "clip_Y"],
                "cache_size_mb": 15.0
            },
            "semiotic_engine": {
                "deception_accumulator": 0.0,
                "innovation_accumulator": 1.2
            },
            "metadata": {
                "timestamp": "2025-01-07T12:00:00Z",
                "version": "1.0.0"
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            persistor.save_complete_state(system_state, checkpoint_path)

            # Assert - All components present
            with open(checkpoint_path) as f:
                loaded = json.load(f)

            self.assertIn("contextual_agent", loaded)
            self.assertIn("cognitive_engine", loaded)
            self.assertIn("rust_cache", loaded)
            self.assertIn("semiotic_engine", loaded)

            # Assert - Values preserved
            self.assertEqual(loaded["contextual_agent"]["context"], "AGGRESSION")
            self.assertEqual(loaded["cognitive_engine"]["current_intensity"], 0.7)
            self.assertEqual(loaded["rust_cache"]["cache_keys"], ["clip_X", "clip_Y"])


if __name__ == "__main__":
    unittest.main()
