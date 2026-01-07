#!/usr/bin/env python3
"""
Production Python Cognitive Agent with Self-Healing
====================================================

This is the main entry point for the Python Cognitive Agent in production
deployment. It integrates:

1. Self-healing checkpoint recovery on startup
2. ZeroMQ heartbeat communication with Rust
3. Periodic checkpoint saving
4. Graceful shutdown handling

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import zmq

# Add system module to path (for production deployment)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from system import StatePersistor, SelfHeal, HealthStatus


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration
DEFAULT_HEARTBEAT_ENDPOINT = "ipc:///tmp/cognitive_heartbeat.ipc"
DEFAULT_HEARTBEAT_INTERVAL_MS = 20
DEFAULT_CHECKPOINT_DIR = "./state"
DEFAULT_CHECKPOINT_INTERVAL_SEC = 60


class CognitiveAgent:
    """
    Production Python Cognitive Agent with self-healing capabilities.

    This agent:
    1. Recovers state from checkpoints on startup
    2. Sends heartbeats to Rust Field Engine
    3. Periodically saves checkpoints
    4. Handles graceful shutdown
    """

    def __init__(
        self,
        heartbeat_endpoint: str = DEFAULT_HEARTBEAT_ENDPOINT,
        heartbeat_interval_ms: int = DEFAULT_HEARTBEAT_INTERVAL_MS,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        checkpoint_interval_sec: int = DEFAULT_CHECKPOINT_INTERVAL_SEC,
    ):
        """
        Initialize the Cognitive Agent.

        Args:
            heartbeat_endpoint: ZeroMQ endpoint for heartbeat socket
            heartbeat_interval_ms: Heartbeat interval in milliseconds
            checkpoint_dir: Directory for checkpoint files
            checkpoint_interval_sec: Interval between checkpoints
        """
        self.heartbeat_endpoint = heartbeat_endpoint
        self.heartbeat_interval_ms = heartbeat_interval_ms
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval_sec = checkpoint_interval_sec

        # ZeroMQ context and socket
        self.zmq_context: Optional[zmq.Context] = None
        self.heartbeat_socket: Optional[zmq.Socket] = None

        # State management
        self.running = False
        self.sequence = 0
        self.pid = os.getpid()

        # Agent state (conversation context, history, etc.)
        self.agent_state: Dict[str, Any] = {
            "context": None,
            "history": [],
            "dialogue_state": {"turn": 0, "initiator": None},
        }

        # Self-healing components
        self.persistor = StatePersistor(checkpoint_dir=self.checkpoint_dir)
        self.healer = SelfHeal(checkpoint_dir=self.checkpoint_dir)

        # Timing
        self.last_checkpoint_time = time.time()
        self.last_heartbeat_time = time.time()

    def recover_state(self) -> bool:
        """
        Attempt to recover state from latest checkpoint.

        Returns:
            True if state was recovered, False otherwise
        """
        logger.info("Attempting to recover state from checkpoint...")

        latest_state = self.healer.rehydrate_from_latest()

        if latest_state is None:
            logger.info("No checkpoint found, starting with fresh state")
            return False

        # Restore agent state
        self.agent_state = {
            "context": latest_state.get("context"),
            "history": latest_state.get("history", []),
            "dialogue_state": latest_state.get("dialogue_state", {"turn": 0, "initiator": None}),
        }

        logger.info(f"✓ Recovered from checkpoint:")
        logger.info(f"  Context: {self.agent_state['context']}")
        logger.info(f"  History length: {len(self.agent_state['history'])}")
        logger.info(f"  Turn: {self.agent_state['dialogue_state']['turn']}")

        return True

    def connect_to_rust(self) -> None:
        """Connect to Rust Field Engine via ZeroMQ"""
        logger.info(f"Connecting to Rust Field Engine: {self.heartbeat_endpoint}")

        self.zmq_context = zmq.Context()
        self.heartbeat_socket = self.zmq_context.socket(zmq.PUB)

        # Set socket options for reliability
        self.heartbeat_socket.setsockopt(zmq.LINGER, 1000)
        self.heartbeat_socket.setsockopt(zmq.SNDHWM, 10)

        # Connect to the Rust heartbeat endpoint
        self.heartbeat_socket.connect(self.heartbeat_endpoint)

        logger.info("✓ Connected to Rust Field Engine")
        logger.info(f"  PID: {self.pid}")
        logger.info(f"  Endpoint: {self.heartbeat_endpoint}")

    def send_heartbeat(self) -> None:
        """Send a heartbeat message to Rust Field Engine"""
        if not self.heartbeat_socket:
            raise RuntimeError("Not connected to Rust Field Engine")

        # Create heartbeat message
        heartbeat = {
            "timestamp": int(time.time() * 1000),  # milliseconds since epoch
            "sequence": self.sequence,
            "pid": self.pid,
            "state": "active",
        }

        # Serialize to JSON
        message = json.dumps(heartbeat).encode("utf-8")

        # Send heartbeat
        try:
            self.heartbeat_socket.send(message, flags=zmq.NOBLOCK)
            self.sequence += 1

            if self.sequence % 250 == 0:  # Log every 250 heartbeats (5 seconds)
                logger.info(f"Heartbeat: sequence={self.sequence}, "
                           f"context={self.agent_state['context']}, "
                           f"turn={self.agent_state['dialogue_state']['turn']}")

        except zmq.ZMQError as e:
            logger.warning(f"Failed to send heartbeat: {e}")

    def save_checkpoint(self) -> None:
        """Save current agent state to checkpoint"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{timestamp}.json"

        self.persistor.save_contextual_agent(self.agent_state, checkpoint_path)
        self.last_checkpoint_time = time.time()

        logger.info(f"✓ Checkpoint saved: {checkpoint_path.name}")

    def maybe_save_checkpoint(self) -> None:
        """Save checkpoint if interval has elapsed"""
        elapsed = time.time() - self.last_checkpoint_time

        if elapsed >= self.checkpoint_interval_sec:
            self.save_checkpoint()

    def process_intents(self) -> None:
        """
        Process cognitive intents (placeholder for actual implementation).

        This is where the actual cognitive intelligence would happen:
        - Context interpretation
        - Phrase selection
        - Learning updates
        - Intent generation
        """
        # Placeholder: In production, this would:
        # 1. Receive intent requests from Rust
        # 2. Process using cognitive intelligence
        # 3. Generate response intents
        # 4. Update agent state
        pass

    def run(self) -> None:
        """
        Main run loop

        This method:
        1. Sends heartbeats to Rust
        2. Processes cognitive intents
        3. Saves checkpoints periodically
        4. Handles graceful shutdown
        """
        self.running = True
        heartbeat_interval_sec = self.heartbeat_interval_ms / 1000.0

        logger.info(f"Starting main loop (heartbeat interval: {self.heartbeat_interval_ms}ms)")
        logger.info(f"Checkpoint interval: {self.checkpoint_interval_sec} seconds")

        try:
            while self.running:
                current_time = time.time()
                heartbeat_elapsed = current_time - self.last_heartbeat_time

                # Send heartbeat if interval has elapsed
                if heartbeat_elapsed >= heartbeat_interval_sec:
                    self.send_heartbeat()
                    self.last_heartbeat_time = current_time

                # Process cognitive intents
                self.process_intents()

                # Maybe save checkpoint
                self.maybe_save_checkpoint()

                # Small sleep to prevent busy-waiting
                time.sleep(0.001)  # 1ms

        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            raise

        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Graceful shutdown"""
        logger.info("Shutting down...")

        # Save final checkpoint
        try:
            self.save_checkpoint()
            logger.info("✓ Final checkpoint saved")
        except Exception as e:
            logger.error(f"Failed to save final checkpoint: {e}")

        # Disconnect from Rust
        if self.heartbeat_socket:
            self.heartbeat_socket.close()
        if self.zmq_context:
            self.zmq_context.term()

        logger.info("✓ Disconnected from Rust Field Engine")
        logger.info("Cognitive Agent stopped")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Production Python Cognitive Agent with Self-Healing"
    )
    parser.add_argument(
        "--heartbeat-endpoint",
        default=os.environ.get("RUST_HEARTBEAT_ENDPOINT", DEFAULT_HEARTBEAT_ENDPOINT),
        help="ZeroMQ endpoint for heartbeat socket"
    )
    parser.add_argument(
        "--heartbeat-interval-ms",
        type=int,
        default=DEFAULT_HEARTBEAT_INTERVAL_MS,
        help="Heartbeat interval in milliseconds"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory for checkpoint files"
    )
    parser.add_argument(
        "--checkpoint-interval-sec",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL_SEC,
        help="Interval between checkpoints in seconds"
    )

    args = parser.parse_args()

    # Print startup banner
    print("=" * 70)
    print("Python Cognitive Agent - Production Deployment")
    print("=" * 70)
    print(f"PID: {os.getpid()}")
    print(f"Heartbeat endpoint: {args.heartbeat_endpoint}")
    print(f"Heartbeat interval: {args.heartbeat_interval_ms}ms")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Checkpoint interval: {args.checkpoint_interval_sec}s")
    print("=" * 70)

    # Create agent
    agent = CognitiveAgent(
        heartbeat_endpoint=args.heartbeat_endpoint,
        heartbeat_interval_ms=args.heartbeat_interval_ms,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval_sec=args.checkpoint_interval_sec,
    )

    # Setup signal handlers
    signal.signal(signal.SIGINT, agent.signal_handler)
    signal.signal(signal.SIGTERM, agent.signal_handler)

    try:
        # Recover state from checkpoint
        agent.recover_state()

        # Connect to Rust
        agent.connect_to_rust()

        # Run main loop
        agent.run()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
