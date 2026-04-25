#!/usr/bin/env python3
"""
Config Client - REQ Client for Rust Config Server
==================================================

This module provides the Python Logic Layer with a REQ (request-reply)
client for loading configuration data from the Rust Execution Layer at
startup. This eliminates data drift by ensuring Python always loads
acoustic grammar data from the single source of truth (Rust).

The client is designed to fail gracefully:
- Returns None on any failure (timeout, connection refused, parse error)
- Falls back to hardcoded defaults in parsing_strategy.py

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CONFIG_ENDPOINT = os.environ.get("RUST_CONFIG_ENDPOINT", "ipc:///tmp/cognitive_config.ipc")


@dataclass
class RigidIdiomData:
    """Rigid idiom data loaded from Rust"""

    segments: List[int]
    meaning: str
    confidence: float


@dataclass
class AcousticProfileData:
    """Acoustic profile data loaded from Rust Execution Layer"""

    profile_name: str
    transition_strictness: float
    valid_bigrams: List[Tuple[int, int]]
    openers: List[int]
    closers: List[int]
    rigid_idioms: List[RigidIdiomData]
    position_weights: Dict[str, Any] = field(default_factory=dict)


class ConfigClient:
    """
    ZeroMQ REQ client for loading configuration from Rust.

    Usage:
        client = ConfigClient()
        profile = client.request_acoustic_profile("bat")
        if profile:
            print(f"Loaded {profile.profile_name} with {len(profile.valid_bigrams)} bigrams")
        else:
            print("Rust unavailable, using defaults")
    """

    def __init__(self, endpoint: str = CONFIG_ENDPOINT, timeout_ms: int = 2000):
        """
        Initialize config client.

        Args:
            endpoint: ZeroMQ REQ/REP endpoint
            timeout_ms: Request timeout in milliseconds
        """
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms

    def request_acoustic_profile(self, species: str) -> Optional[AcousticProfileData]:
        """
        Request acoustic profile data from Rust.

        Args:
            species: Species name (e.g., "bat", "egyptian fruit bat")

        Returns:
            AcousticProfileData on success, None on any failure
        """
        try:
            import zmq
        except ImportError:
            logger.debug("ZeroMQ not installed, cannot reach Rust config server")
            return None

        request_id = f"py-{os.getpid()}-{id(self)}"
        request = {
            "request_type": "acoustic_profile",
            "species": species,
            "request_id": request_id,
        }

        try:
            ctx = zmq.Context()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            sock.connect(self.endpoint)

            # Send request
            sock.send_string(json.dumps(request))

            # Receive response
            msg = sock.recv_string()
            response = json.loads(msg)

            sock.close()
            ctx.term()

            if not response.get("success", False):
                error = response.get("error", "Unknown error")
                logger.warning(f"Rust config server error: {error}")
                return None

            data = response.get("data")
            if not data:
                logger.warning("Rust config server returned empty data")
                return None

            # Parse into AcousticProfileData
            bigrams = [tuple(b) for b in data.get("valid_bigrams", [])]
            idioms = [
                RigidIdiomData(
                    segments=idiom["segments"],
                    meaning=idiom["meaning"],
                    confidence=idiom["confidence"],
                )
                for idiom in data.get("rigid_idioms", [])
            ]

            profile = AcousticProfileData(
                profile_name=data["profile_name"],
                transition_strictness=data["transition_strictness"],
                valid_bigrams=bigrams,
                openers=data.get("openers", []),
                closers=data.get("closers", []),
                rigid_idioms=idioms,
                position_weights=data.get("position_weights", {}),
            )

            logger.info(
                f"Loaded acoustic profile from Rust: {profile.profile_name} "
                f"({len(profile.valid_bigrams)} bigrams, "
                f"{len(profile.openers)} openers, "
                f"{len(profile.closers)} closers)"
            )
            return profile

        except Exception as e:
            logger.debug(f"Could not reach Rust config server: {e}")
            return None


__all__ = ["ConfigClient", "AcousticProfileData", "RigidIdiomData"]
