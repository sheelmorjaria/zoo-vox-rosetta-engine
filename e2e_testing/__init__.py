#!/usr/bin/env python3
"""
E2E Shadow Mode Testing Suite

End-to-end validation of the complete closed-loop Zoo Vox Rosetta Engine.
Tests the full Rust→Python→Rust pipeline with corpus audio streaming,
round-trip latency measurement, feedback loop resistance, and hardware soak testing.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from e2e_testing.config import ShadowModeConfig
from e2e_testing.rtl_profiler import RoundTripProfiler
from e2e_testing.acoustic_mirror_tester import AcousticMirrorTester
from e2e_testing.chaos_corpus_generator import ChaosCorpusGenerator
from e2e_testing.syntactic_coherence_tester import SyntacticCoherenceTester

__all__ = [
    "ShadowModeConfig",
    "RoundTripProfiler",
    "AcousticMirrorTester",
    "ChaosCorpusGenerator",
    "SyntacticCoherenceTester",
]

__version__ = "1.0.0"
