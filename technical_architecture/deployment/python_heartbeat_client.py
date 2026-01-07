#!/usr/bin/env python3
"""
Python Cognitive Agent - Heartbeat Client
==========================================

This script demonstrates how to connect to the Rust Field Engine
and send heartbeats to maintain the Interactive Mode.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import os
import signal
import sys
import time
from typing import Optional

import zmq

# Configuration
HEARTBEAT_ENDPOINT = os.environ.get(
    "RUST_HEARTBEAT_ENDPOINT",
    "ipc:///tmp/cognitive_heartbeat.ipc"
)
HEARTBEAT_INTERVAL_MS = 20  # Send heartbeat every 20ms


class HeartbeatClient:
    """Client for sending heartbeats to Rust Field Engine"""

    def __init__(self, endpoint: str = HEARTBEAT_ENDPOINT):
        """
        Initialize the heartbeat client

        Args:
            endpoint: ZeroMQ endpoint for heartbeat socket
        """
        self.endpoint = endpoint
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None
        self.sequence = 0
        self.running = False

        # Get current process ID
        self.pid = os.getpid()

    def connect(self) -> None:
        """Connect to the Rust Field Engine heartbeat endpoint"""
        print(f"Connecting to Rust Field Engine: {self.endpoint}")

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)

        # Set socket options for reliability
        self.socket.setsockopt(zmq.LINGER, 1000)  # 1 second linger
        self.socket.setsockopt(zmq.SNDHWM, 10)    # Send high water mark

        # Connect to the Rust heartbeat endpoint
        self.socket.connect(self.endpoint)

        print("✓ Connected to Rust Field Engine")
        print(f"  PID: {self.pid}")
        print(f"  Endpoint: {self.endpoint}")

    def disconnect(self) -> None:
        """Disconnect from the Rust Field Engine"""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()

        print("✓ Disconnected from Rust Field Engine")

    def send_heartbeat(self) -> None:
        """Send a heartbeat message to Rust Field Engine"""
        if not self.socket:
            raise RuntimeError("Not connected to Rust Field Engine")

        # Create heartbeat message
        heartbeat = {
            "timestamp": int(time.time() * 1000),  # milliseconds since epoch
            "sequence": self.sequence,
            "pid": self.pid,
            "state": "active"
        }

        # Serialize to JSON
        message = json.dumps(heartbeat).encode('utf-8')

        # Send heartbeat
        try:
            self.socket.send(message)
            self.sequence += 1

            if self.sequence % 50 == 0:  # Log every 50 heartbeats
                print(f"Heartbeat sent: sequence={self.sequence}")

        except zmq.ZMQError as e:
            print(f"✗ Failed to send heartbeat: {e}", file=sys.stderr)
            raise

    def run(self) -> None:
        """
        Run the heartbeat loop

        This method runs in a loop, sending heartbeats at regular intervals.
        It handles graceful shutdown on SIGINT and SIGTERM.
        """
        self.running = True
        HEARTBEAT_INTERVAL_MS / 1000.0

        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, shutting down...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print(f"Starting heartbeat loop (interval: {HEARTBEAT_INTERVAL_MS}ms)")
        print("Press Ctrl+C to stop")

        last_heartbeat = time.time()

        try:
            while self.running:
                current_time = time.time()
                elapsed = (current_time - last_heartbeat) * 1000  # ms

                if elapsed >= HEARTBEAT_INTERVAL_MS:
                    self.send_heartbeat()
                    last_heartbeat = current_time

                # Small sleep to prevent busy-waiting
                time.sleep(0.001)  # 1ms

        except Exception as e:
            print(f"✗ Error in heartbeat loop: {e}", file=sys.stderr)
            raise

        finally:
            self.disconnect()
            print("Heartbeat client stopped")


def main():
    """Main entry point"""
    print("=" * 60)
    print("Python Cognitive Agent - Heartbeat Client")
    print("=" * 60)

    client = HeartbeatClient()

    try:
        client.connect()
        client.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
