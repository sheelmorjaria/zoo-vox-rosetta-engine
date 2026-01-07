"""
State Persistor - Phase 1: Checkpointer
======================================

This module is responsible for creating snapshots of system state for
crash recovery and self-healing in long-duration field experiments.

It handles:
- ContextualAgent serialization (context, conversation history)
- CognitiveEngine serialization (intensity, persona, vectors)
- Rust LRU cache serialization (for warm restarts)
- SemioticEngine serialization (learning metrics)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class StatePersistor:
    """
    Saves system state to checkpoint files for crash recovery.

    This is the "Persistor" component of the self-healing system.
    It creates JSON snapshots of all critical system components.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize the StatePersistor.

        Args:
            checkpoint_dir: Directory to save checkpoints. If None, uses default.
        """
        if checkpoint_dir is None:
            checkpoint_dir = Path.cwd() / "state"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_contextual_agent(
        self, agent_state: Dict[str, Any], checkpoint_path: Path
    ) -> Path:
        """
        Save ContextualAgent state to checkpoint file.

        Args:
            agent_state: Dictionary containing context, history, dialogue_state
            checkpoint_path: Path where checkpoint should be saved

        Returns:
            Path to the saved checkpoint file

        Raises:
            IOError: If file cannot be written
        """
        logger.info(f"Saving ContextualAgent state to {checkpoint_path}")

        checkpoint_data = {
            "component": "contextual_agent",
            "context": agent_state.get("context"),
            "history": agent_state.get("history", []),
            "dialogue_state": agent_state.get("dialogue_state", {}),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        # Ensure parent directory exists
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Saved ContextualAgent: context={checkpoint_data['context']}, "
                   f"history_length={len(checkpoint_data['history'])}")

        return checkpoint_path

    def save_rust_cache(
        self, rust_state: Dict[str, Any], checkpoint_path: Path
    ) -> Path:
        """
        Save Rust LRU cache state to checkpoint file.

        This allows "warm restart" - telling Rust what clips to preload
        into RAM after recovery.

        Args:
            rust_state: Dictionary containing cache_keys, cache_size_mb, etc.
            checkpoint_path: Path where checkpoint should be saved

        Returns:
            Path to the saved checkpoint file

        Raises:
            IOError: If file cannot be written
        """
        logger.info(f"Saving Rust cache state to {checkpoint_path}")

        checkpoint_data = {
            "component": "rust_cache",
            "rust_cache_keys": rust_state.get("rust_cache_keys", []),
            "cache_size_mb": rust_state.get("cache_size_mb", 0.0),
            "cache_hit_rate": rust_state.get("cache_hit_rate", 0.0),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        # Ensure parent directory exists
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Saved Rust cache: {len(checkpoint_data['rust_cache_keys'])} keys, "
                   f"{checkpoint_data['cache_size_mb']} MB")

        return checkpoint_path

    def save_complete_state(
        self, system_state: Dict[str, Any], checkpoint_path: Optional[Path] = None
    ) -> Path:
        """
        Save complete system state in one operation.

        This creates a unified checkpoint containing all components:
        - ContextualAgent (conversation state)
        - CognitiveEngine (intensity, persona)
        - Rust Cache (LRU keys for warm restart)
        - SemioticEngine (learning accumulators)

        Args:
            system_state: Dictionary containing all component states
            checkpoint_path: Path where checkpoint should be saved.
                          If None, generates timestamped filename.

        Returns:
            Path to the saved checkpoint file

        Raises:
            IOError: If file cannot be written
        """
        if checkpoint_path is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{timestamp}.json"

        logger.info(f"Saving complete system state to {checkpoint_path}")

        # Add metadata
        checkpoint_data = {
            **system_state,
            "metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "version": "1.0.0",
                "components_saved": list(system_state.keys())
            }
        }

        # Ensure parent directory exists
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Saved complete system state: {len(system_state)} components, "
                   f"size={checkpoint_path.stat().st_size} bytes")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Load system state from checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary containing all system state

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        logger.info(f"Loading system state from {checkpoint_path}")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(checkpoint_path) as f:
            system_state = json.load(f)

        logger.info(f"Loaded system state: {len(system_state)} components")
        return system_state

    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Find the most recent checkpoint file.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.json"))

        if not checkpoints:
            return None

        # Sort by modification time (most recent first)
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        logger.info(f"Latest checkpoint: {latest}")
        return latest
