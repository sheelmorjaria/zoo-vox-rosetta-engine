"""
Self Heal - Phase 2: Rehydrator
===============================

This module is responsible for detecting process crashes and recovering
system state from checkpoints for autonomous healing in long-duration
field experiments.

It handles:
- Process health monitoring (alive/dead detection)
- State rehydration from checkpoints
- Rust cache synchronization for warm restarts
- Complete healing workflow with process restart

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import logging
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Process health status."""

    ALIVE = "alive"
    DEAD = "dead"


class SelfHeal:
    """
    Autonomous system healer for crash recovery.

    This is the "Rehydrator" component of the self-healing system.
    It detects dead processes, loads state from checkpoints, and
    restarts processes with recovered state.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize the SelfHeal system.

        Args:
            checkpoint_dir: Directory containing checkpoint files.
                           If None, uses default 'state' directory.
        """
        if checkpoint_dir is None:
            checkpoint_dir = Path.cwd() / "state"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def check_health(self, pid: int) -> HealthStatus:
        """
        Check if a process is alive or dead.

        Args:
            pid: Process ID to check

        Returns:
            HealthStatus.ALIVE if process is running, HealthStatus.DEAD otherwise
        """
        if self.is_process_alive(pid):
            return HealthStatus.ALIVE
        return HealthStatus.DEAD

    def is_process_alive(self, pid: int) -> bool:
        """
        Check if process with given PID is alive.

        Args:
            pid: Process ID to check

        Returns:
            True if process exists and is running, False otherwise
        """
        try:
            # Send signal 0 to check if process exists
            # This doesn't actually kill the process
            import os

            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def rehydrate_agent(self, checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        """
        Rehydrate ContextualAgent state from checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary containing agent state, or None if checkpoint invalid
        """
        logger.info(f"Rehydrating agent state from {checkpoint_path}")

        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return None

        try:
            with open(checkpoint_path) as f:
                checkpoint_data = json.load(f)

            # Extract agent state
            agent_state = {
                "context": checkpoint_data.get("context"),
                "history": checkpoint_data.get("history", []),
                "dialogue_state": checkpoint_data.get("dialogue_state", {}),
            }

            logger.info(
                f"Rehydrated agent: context={agent_state['context']}, "
                f"history_length={len(agent_state['history'])}"
            )

            return agent_state

        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def rehydrate_from_latest(self) -> Optional[Dict[str, Any]]:
        """
        Rehydrate system state from the most recent checkpoint.

        Args:
            None

        Returns:
            Dictionary containing system state, or None if no checkpoint found
        """
        logger.info("Rehydrating from latest checkpoint")

        # Find latest checkpoint
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.json"))

        if not checkpoints:
            logger.warning("No checkpoints found")
            return None

        # Sort by modification time (most recent first)
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        logger.info(f"Latest checkpoint: {latest}")

        # Load and return the checkpoint
        return self.rehydrate_agent(latest)

    def sync_rust_cache(self, checkpoint_path: Path) -> List[str]:
        """
        Extract Rust cache keys from checkpoint for warm restart.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            List of cache keys to preload into Rust LRU cache
        """
        logger.info(f"Syncing Rust cache from {checkpoint_path}")

        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return []

        try:
            with open(checkpoint_path) as f:
                checkpoint_data = json.load(f)

            # Handle both standalone rust_cache checkpoints and complete checkpoints
            if "rust_cache_keys" in checkpoint_data:
                # Standalone rust cache checkpoint
                cache_keys = checkpoint_data.get("rust_cache_keys", [])
            elif "rust_cache" in checkpoint_data:
                # Complete system checkpoint with nested rust_cache
                rust_cache = checkpoint_data.get("rust_cache", {})
                cache_keys = rust_cache.get("cache_keys", [])
            else:
                logger.warning("No Rust cache data found in checkpoint")
                return []

            logger.info(f"Synced {len(cache_keys)} Rust cache keys")
            return cache_keys

        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load checkpoint for cache sync: {e}")
            return []

    def heal(
        self, pid: int, restart_command: List[str], checkpoint_path: Optional[Path] = None
    ) -> bool:
        """
        Perform complete healing workflow.

        1. Check if process is alive
        2. If dead, load latest checkpoint
        3. Restart process with recovered state

        Args:
            pid: Process ID to check
            restart_command: Command to restart process (e.g., ["python3", "-m", "agent"])
            checkpoint_path: Optional specific checkpoint to use. If None, uses latest.

        Returns:
            True if healing succeeded, False otherwise
        """
        logger.info(f"Starting healing workflow for PID {pid}")

        # Check process health
        status = self.check_health(pid)

        if status == HealthStatus.ALIVE:
            logger.info(f"Process {pid} is alive, no healing needed")
            return True

        logger.warning(f"Process {pid} is dead, initiating recovery")

        # Load checkpoint for recovery
        if checkpoint_path is None:
            system_state = self.rehydrate_from_latest()
        else:
            system_state = self.rehydrate_agent(checkpoint_path)

        if system_state is None:
            logger.warning("No checkpoint available, performing cold start")
        else:
            logger.info(f"Checkpoint loaded: context={system_state.get('context')}")

            # Sync Rust cache if available
            if checkpoint_path is None:
                # Find latest checkpoint for cache sync
                checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.json"))
                if checkpoints:
                    cache_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    cache_keys = self.sync_rust_cache(cache_checkpoint)
                    if cache_keys:
                        logger.info(f"Rust cache will be preloaded with {len(cache_keys)} keys")
            else:
                cache_keys = self.sync_rust_cache(checkpoint_path)
                if cache_keys:
                    logger.info(f"Rust cache will be preloaded with {len(cache_keys)} keys")

        # Restart the process
        try:
            logger.info(f"Restarting process with command: {' '.join(restart_command)}")
            process = subprocess.Popen(
                restart_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            logger.info(f"Process restarted with new PID {process.pid}")
            return True

        except (subprocess.SubprocessError, OSError) as e:
            logger.error(f"Failed to restart process: {e}")
            return False
