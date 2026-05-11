#!/usr/bin/env python3
"""
E2E Shadow Mode Test Configuration

Configuration dataclass for the E2E Shadow Mode Test Suite.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ShadowModeConfig:
    """Configuration for E2E Shadow Mode Testing Suite."""

    # === Test Selection ===
    run_rtl_test: bool = True
    run_mirror_test: bool = True
    run_chaos_test: bool = True
    run_soak_test: bool = False  # Default off (takes 24 hours)

    # === RTL Configuration ===
    target_rtl_ms: float = 50.0
    sync_pulse_interval_ms: int = 5000
    sync_pulse_frequency_hz: int = 80000  # Ultrasonic 80kHz
    sync_pulse_duration_ms: float = 1.0

    # === Mirror Test Configuration ===
    mirror_test_duration_seconds: int = 300
    max_interactions_per_minute: int = 30
    loopback_gain: float = 0.3
    loopback_delay_samples: int = 480  # 10ms @ 48kHz

    # === Chaos Test Configuration ===
    chaos_duration_seconds: int = 600
    chaos_overlap_count: int = 5
    max_gibberish_ratio: float = 0.05
    max_merge_rate: float = 0.20  # 20% merged segments threshold

    # === Soak Test Configuration ===
    soak_duration_hours: int = 24
    memory_leak_threshold_percent: float = 5.0
    telemetry_interval_seconds: int = 60

    # === Predictive NBD Configuration ===
    nbd_boundary_threshold: float = 2.5
    nbd_boundary_threshold_lower: float = 1.5
    nbd_slow_decay: float = 0.99
    nbd_fast_decay: float = 0.9
    nbd_min_confidence: float = 0.6

    # === ZMQ Configuration ===
    zmq_feature_port: int = 5555
    zmq_action_port: int = 5556
    zmq_timeout_ms: int = 100

    # === Audio Configuration ===
    sample_rate: int = 48000
    frame_size_ms: int = 10
    frame_size_samples: int = field(init=False)

    # === Corpus Paths ===
    corpus_dir: str = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio/"
    chaos_output_dir: str = "/tmp/e2e_chaos_corpus/"
    soak_output_dir: str = "/tmp/e2e_soak_results/"

    # === BioMAE Model Path ===
    biomae_path: Optional[str] = None

    def __post_init__(self):
        """Calculate derived values."""
        self.frame_size_samples = int(self.sample_rate * self.frame_size_ms / 1000)

    @property
    def corpus_path(self) -> Path:
        """Get corpus directory as Path object."""
        return Path(self.corpus_dir)

    @property
    def chaos_output_path(self) -> Path:
        """Get chaos output directory as Path object."""
        return Path(self.chaos_output_dir)

    @property
    def soak_output_path(self) -> Path:
        """Get soak output directory as Path object."""
        return Path(self.soak_output_dir)
