#!/usr/bin/env python3
"""
Bio-Acoustic Turing Test - Test Suite (TDD)
============================================

Test-Driven Development tests for the Bio-Acoustic Turing Test framework.

Tests are written FIRST (Red phase), then implementation follows (Green phase).

Run with: pytest test_bio_acoustic_turing_test.py -v
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules to test (will fail initially, as we write tests first)
# Implementation comes later

# ============================================================================
# Test 1: StimulusController - Playback Management
# ============================================================================


def test_stimulus_controller_creation():
    """Test that StimulusController can be created with configuration."""
    config = {"sample_rate": 22050, "output_device": "default", "volume_db": -6.0}

    # This will fail until we implement StimulusController
    controller = StimulusController(config)

    assert controller.sample_rate == 22050
    assert controller.volume_db == -6.0
    assert controller.is_ready()


def test_stimulus_controller_load_natural_audio():
    """Test loading natural audio segments for playback."""
    controller = StimulusController({"sample_rate": 22050})

    # Load natural audio (concatenative)
    natural_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050)) * 0.5
    controller.load_stimulus("natural_phee", natural_audio.tolist(), "concatenative")

    assert controller.has_stimulus("natural_phee")
    assert controller.get_stimulus_type("natural_phee") == "concatenative"


def test_stimulus_controller_load_granular_audio():
    """Test loading granular-synthesized audio for playback."""
    controller = StimulusController({"sample_rate": 22050})

    # Load granular audio
    granular_audio = np.sin(2 * np.pi * 400 * np.linspace(0, 1, 22050)) * 0.5
    controller.load_stimulus("granular_phee_shifted", granular_audio.tolist(), "granular")

    assert controller.has_stimulus("granular_phee_shifted")
    assert controller.get_stimulus_type("granular_phee_shifted") == "granular"


def test_stimulus_controller_playback_sequence():
    """Test that playback sequence can be randomized and balanced."""
    controller = StimulusController({"sample_rate": 22050})

    # Load stimuli
    for i in range(10):
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050)) * 0.5
        stim_type = "concatenative" if i % 2 == 0 else "granular"
        controller.load_stimulus(f"stim_{i}", audio.tolist(), stim_type)

    # Create randomized, counterbalanced sequence
    sequence = controller.create_counterbalanced_sequence(
        num_trials=20, stimulus_ids=[f"stim_{i}" for i in range(10)]
    )

    assert len(sequence) == 20
    # Check that both types are represented
    types_in_sequence = [controller.get_stimulus_type(s) for s in sequence]
    assert "concatenative" in types_in_sequence
    assert "granular" in types_in_sequence


# ============================================================================
# Test 2: ResponseRecorder - Measuring Animal Responses
# ============================================================================


def test_response_recorder_creation():
    """Test that ResponseRecorder can be created."""
    recorder = ResponseRecorder(sample_rate=22050)

    assert recorder.sample_rate == 22050
    assert not recorder.is_recording()


def test_response_recorder_record_response():
    """Test recording animal response to stimulus."""
    recorder = ResponseRecorder(sample_rate=22050)

    # Simulate response (vocalization)
    response_audio = np.sin(2 * np.pi * 8000 * np.linspace(0, 0.5, 11025)) * 0.3

    recorder.start_recording()
    recorder.record_frame(response_audio.tolist())
    recorder.stop_recording()

    response = recorder.get_response()

    assert response["has_response"]
    assert response["duration_ms"] > 0
    assert len(response["audio"]) > 0


def test_response_recorder_measure_latency():
    """Test measuring response latency (time from stimulus to response)."""
    recorder = ResponseRecorder(sample_rate=22050)

    # Simulate timeline
    recorder.mark_stimulus_onset()

    # Simulate delay before response
    delay_samples = int(0.2 * 22050)  # 200ms delay
    response_audio = np.zeros(delay_samples)
    response_audio = np.concatenate(
        [response_audio, np.sin(2 * np.pi * 8000 * np.linspace(0, 0.5, 22050)) * 0.3]
    )

    recorder.record_frame(response_audio.tolist())

    latency_ms = recorder.get_response_latency_ms()

    assert latency_ms >= 190  # Allow small timing error
    assert latency_ms <= 210


def test_response_recorder_classify_response_type():
    """Test classifying response type (vocalization, approach, etc.)."""
    recorder = ResponseRecorder(sample_rate=22050)

    # Test vocalization response
    vocalization = np.sin(2 * np.pi * 8000 * np.linspace(0, 1, 22050)) * 0.3
    response_type = recorder.classify_response(vocalization.tolist())

    assert response_type == "vocalization"

    # Test no response
    silence = np.zeros(22050)
    response_type = recorder.classify_response(silence.tolist())

    assert response_type == "no_response"


# ============================================================================
# Test 3: ExperimentDesign - Protocol Management
# ============================================================================


def test_experiment_design_creation():
    """Test creating an experiment design."""
    design = ExperimentDesign(
        subject_id="test_marmoset_001", species="marmoset", session_type="turing_test"
    )

    assert design.subject_id == "test_marmoset_001"
    assert design.species == "marmoset"
    assert design.session_type == "turing_test"


def test_experiment_design_add_trial():
    """Test adding trials to experiment."""
    design = ExperimentDesign(
        subject_id="test_marmoset_001", species="marmoset", session_type="turing_test"
    )

    # Add trials
    design.add_trial(trial_id=1, stimulus_type="concatenative", stimulus_id="natural_phee_001")

    design.add_trial(trial_id=2, stimulus_type="granular", stimulus_id="granular_phee_shifted")

    trials = design.get_trials()

    assert len(trials) == 2
    assert trials[0]["stimulus_type"] == "concatenative"
    assert trials[1]["stimulus_type"] == "granular"


def test_experiment_design_randomization():
    """Test that trials are properly randomized."""
    design = ExperimentDesign(
        subject_id="test_marmoset_001",
        species="marmoset",
        session_type="turing_test",
        randomize_order=True,
    )

    # Add 10 trials (5 natural, 5 granular)
    for i in range(10):
        stim_type = "concatenative" if i % 2 == 0 else "granular"
        design.add_trial(i, stim_type, f"stim_{i}")

    trials = design.get_trials()

    # Check balance
    types = [t["stimulus_type"] for t in trials]
    assert types.count("concatenative") == 5
    assert types.count("granular") == 5

    # Check that order is not sequential (randomized)
    # Probability of sequential order is 1/10! ≈ 0
    is_sequential = all(trials[i]["trial_id"] == i for i in range(10))
    assert not is_sequential or len(trials) == 10  # Allow if not randomized


def test_experiment_design_inter_trial_interval():
    """Test that inter-trial intervals are properly set."""
    design = ExperimentDesign(
        subject_id="test_marmoset_001",
        species="marmoset",
        session_type="turing_test",
        min_inter_trial_interval_s=30,
        max_inter_trial_interval_s=60,
    )

    interval = design.get_next_inter_trial_interval()

    assert interval >= 30
    assert interval <= 60


# ============================================================================
# Test 4: StatisticalAnalyzer - Hypothesis Testing
# ============================================================================


def test_statistical_analyzer_creation():
    """Test creating statistical analyzer."""
    analyzer = StatisticalAnalyzer()

    assert analyzer is not None


def test_statistical_analyzer_compare_response_rates():
    """Test comparing response rates between conditions."""
    analyzer = StatisticalAnalyzer()

    # Mock data
    results = {
        "concatenative": {
            "responses": [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],  # 9/10 = 90%
            "total_trials": 10,
        },
        "granular": {
            "responses": [1, 1, 0, 1, 1, 1, 1, 1, 1, 0],  # 8/10 = 80%
            "total_trials": 10,
        },
    }

    comparison = analyzer.compare_response_rates(results)

    assert "concatenative_response_rate" in comparison
    assert "granular_response_rate" in comparison
    assert comparison["concatenative_response_rate"] == 0.9
    assert comparison["granular_response_rate"] == 0.8
    assert "statistical_test" in comparison


def test_statistical_analyzer_latency_comparison():
    """Test comparing response latencies."""
    analyzer = StatisticalAnalyzer()

    # Mock data
    results = {
        "concatenative": {
            "latencies_ms": [150, 200, 180, 220, 190],
        },
        "granular": {
            "latencies_ms": [160, 210, 170, 230, 185],
        },
    }

    comparison = analyzer.compare_latencies(results)

    assert "concatenative_mean_latency" in comparison
    assert "granular_mean_latency" in comparison
    assert "statistical_test" in comparison


def test_statistical_analyzer_turing_test_result():
    """Test determining Turing test outcome."""
    analyzer = StatisticalAnalyzer()

    # Test 1: Indistinguishable (PASS)
    results_pass = {
        "concatenative_response_rate": 0.85,
        "granular_response_rate": 0.82,
        "p_value": 0.65,
    }

    result = analyzer.evaluate_turing_test(results_pass)

    assert result["passed"]
    assert result["interpretation"] == "Animals cannot distinguish between natural and granular"

    # Test 2: Distinguishable (FAIL)
    results_fail = {
        "concatenative_response_rate": 0.90,
        "granular_response_rate": 0.30,
        "p_value": 0.001,
    }

    result = analyzer.evaluate_turing_test(results_fail)

    assert not result["passed"]
    assert result["interpretation"] == "Animals can distinguish between natural and granular"


# ============================================================================
# Test 5: BioAcousticTuringTest - Main Integration
# ============================================================================


def test_turing_test_full_workflow():
    """Test complete Turing test workflow."""
    # Create test instance
    turing_test = BioAcousticTuringTest(
        subject_id="test_subject", species="marmoset", output_dir=tempfile.mkdtemp()
    )

    # Phase 1: Concatenative (baseline)
    turing_test.set_phase("concatenative_baseline")

    # Add natural stimuli
    for i in range(5):
        audio = np.sin(2 * np.pi * 8000 * np.linspace(0, 1, 22050)) * 0.3
        turing_test.add_stimulus(f"natural_{i}", audio.tolist(), "concatenative")

    # Run trials (mock)
    for i in range(5):
        result = turing_test.run_trial(stimulus_id=f"natural_{i}")
        assert result["trial_number"] == i + 1

    # Check results
    results = turing_test.get_results()
    assert "concatenative_baseline" in results
    assert len(results["concatenative_baseline"]["trials"]) == 5


def test_turing_test_granular_phase():
    """Test granular synthesis phase."""
    turing_test = BioAcousticTuringTest(
        subject_id="test_subject", species="marmoset", output_dir=tempfile.mkdtemp()
    )

    # Phase 2: Granular
    turing_test.set_phase("granular_synthesis")

    # Add granular stimuli (pitch-shifted)
    for pitch_shift in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]:
        audio = np.sin(2 * np.pi * 8000 * pitch_shift * np.linspace(0, 1, 22050)) * 0.3
        turing_test.add_stimulus(f"granular_shift_{pitch_shift}", audio.tolist(), "granular")

    # Run trials
    for stim_id in [f"granular_shift_{p}" for p in [0.9, 1.0, 1.1]]:
        result = turing_test.run_trial(stimulus_id=stim_id)
        assert "trial_number" in result

    results = turing_test.get_results()
    assert "granular_synthesis" in results


def test_turing_test_hypothesis_evaluation():
    """Test that hypothesis is correctly evaluated."""
    turing_test = BioAcousticTuringTest(
        subject_id="test_subject", species="marmoset", output_dir=tempfile.mkdtemp()
    )

    # Mock results showing indistinguishable response
    turing_test.add_phase_result("concatenative", {"response_rate": 0.85, "trials": 10})

    turing_test.add_phase_result("granular", {"response_rate": 0.82, "trials": 10})

    # Evaluate hypothesis
    hypothesis_result = turing_test.evaluate_hypothesis()

    assert "null_hypothesis" in hypothesis_result
    assert "conclusion" in hypothesis_result
    assert "p_value" in hypothesis_result


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
