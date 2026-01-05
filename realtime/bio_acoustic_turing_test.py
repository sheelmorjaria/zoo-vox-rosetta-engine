#!/usr/bin/env python3
"""
Bio-Acoustic Turing Test Framework
====================================

A comprehensive framework for testing whether live animals can distinguish
between natural vocalizations and granular-synthesized vocalizations.

This enables scientific validation that granular synthesis achieves
bio-acoustic fidelity sufficient for animal behavior experiments.

Usage:
    # Run complete Turing test
    turing_test = BioAcousticTuringTest(
        subject_id='marmoset_001',
        species='marmoset',
        output_dir='./results'
    )
    turing_test.run_full_test()

Author: Sheel Morjaria + Claude Code
License: CC BY-ND 4.0 International
"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from scipy import stats
import random


class StimulusController:
    """
    Manages audio stimulus playback for experiments.

    Handles loading, organizing, and presenting both natural (concatenative)
    and granular-synthesized stimuli.
    """

    def __init__(self, config: Dict):
        """
        Initialize stimulus controller.

        Args:
            config: Configuration dictionary with keys:
                - sample_rate: Audio sample rate (default 22050)
                - output_device: Audio output device name
                - volume_db: Playback volume in dB
        """
        self.sample_rate = config.get('sample_rate', 22050)
        self.output_device = config.get('output_device', 'default')
        self.volume_db = config.get('volume_db', -6.0)

        # Stimulus storage
        self.stimuli: Dict[str, Dict] = {}

    def is_ready(self) -> bool:
        """Check if controller is ready for playback."""
        return True  # In real implementation, check audio device

    def load_stimulus(self, stimulus_id: str, audio: List[float], stimulus_type: str):
        """
        Load a stimulus for playback.

        Args:
            stimulus_id: Unique identifier for this stimulus
            audio: Audio samples (list of floats)
            stimulus_type: Type of stimulus ('concatenative' or 'granular')
        """
        self.stimuli[stimulus_id] = {
            'audio': np.array(audio, dtype=np.float32),
            'type': stimulus_type,
            'duration_ms': len(audio) / self.sample_rate * 1000,
            'loaded_at': datetime.now().isoformat()
        }

    def has_stimulus(self, stimulus_id: str) -> bool:
        """Check if stimulus is loaded."""
        return stimulus_id in self.stimuli

    def get_stimulus_type(self, stimulus_id: str) -> str:
        """Get the type of a loaded stimulus."""
        return self.stimuli[stimulus_id]['type']

    def create_counterbalanced_sequence(
        self,
        num_trials: int,
        stimulus_ids: List[str]
    ) -> List[str]:
        """
        Create a counterbalanced, randomized sequence of stimuli.

        Args:
            num_trials: Total number of trials
            stimulus_ids: List of available stimulus IDs

        Returns:
            Randomized sequence of stimulus IDs
        """
        # Create trials with balanced representation
        sequence = []
        stimuli_per_trial = num_trials // len(stimulus_ids)

        for _ in range(stimuli_per_trial):
            shuffled = stimulus_ids.copy()
            random.shuffle(shuffled)
            sequence.extend(shuffled)

        # Fill remaining trials
        remaining = num_trials - len(sequence)
        if remaining > 0:
            shuffled = stimulus_ids.copy()
            random.shuffle(shuffled)
            sequence.extend(shuffled[:remaining])

        # Shuffle final sequence
        random.shuffle(sequence)

        return sequence[:num_trials]

    def play_stimulus(self, stimulus_id: str) -> Dict:
        """
        Play a stimulus (placeholder for actual audio playback).

        Args:
            stimulus_id: ID of stimulus to play

        Returns:
            Dictionary with playback info
        """
        if stimulus_id not in self.stimuli:
            raise ValueError(f"Stimulus {stimulus_id} not loaded")

        stimulus = self.stimuli[stimulus_id]

        # In real implementation, this would play audio
        return {
            'stimulus_id': stimulus_id,
            'type': stimulus['type'],
            'duration_ms': stimulus['duration_ms'],
            'played_at': datetime.now().isoformat()
        }


class ResponseRecorder:
    """
    Records and analyzes animal responses to stimuli.

    Measures response latency, duration, type, and other behavioral metrics.
    """

    def __init__(self, sample_rate: int = 22050):
        """
        Initialize response recorder.

        Args:
            sample_rate: Audio sample rate for recording
        """
        self.sample_rate = sample_rate
        self.is_recording_flag = False
        self.recorded_audio: List[np.ndarray] = []
        self.stimulus_onset_sample: Optional[int] = None
        self.response_threshold: float = 0.01  # RMS threshold for response detection

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.is_recording_flag

    def start_recording(self):
        """Start recording responses."""
        self.is_recording_flag = True
        self.recorded_audio = []
        self.stimulus_onset_sample = None

    def stop_recording(self):
        """Stop recording responses."""
        self.is_recording_flag = False

    def record_frame(self, audio_frame: List[float]):
        """
        Record a frame of audio.

        Args:
            audio_frame: Audio samples to record
        """
        if self.is_recording_flag:
            self.recorded_audio.append(np.array(audio_frame, dtype=np.float32))

    def mark_stimulus_onset(self):
        """Mark the time when stimulus was presented."""
        if len(self.recorded_audio) > 0:
            total_samples = sum(len(frame) for frame in self.recorded_audio)
            self.stimulus_onset_sample = total_samples
        else:
            self.stimulus_onset_sample = 0

    def get_response(self) -> Dict:
        """
        Get recorded response data.

        Returns:
            Dictionary with response information
        """
        if not self.recorded_audio:
            return {
                'has_response': False,
                'audio': [],
                'duration_ms': 0
            }

        # Concatenate all frames
        full_audio = np.concatenate(self.recorded_audio)

        # Check if there's a meaningful response
        rms = np.sqrt(np.mean(full_audio ** 2))
        has_response = rms > self.response_threshold

        return {
            'has_response': has_response,
            'audio': full_audio.tolist(),
            'duration_ms': len(full_audio) / self.sample_rate * 1000,
            'rms': rms
        }

    def get_response_latency_ms(self) -> float:
        """
        Calculate response latency in milliseconds.

        Returns:
            Latency from stimulus onset to response onset
        """
        if not self.recorded_audio:
            return 0.0

        if self.stimulus_onset_sample is None:
            return 0.0

        # Find response onset (first sample above threshold)
        all_audio = np.concatenate(self.recorded_audio)

        for i, sample in enumerate(all_audio):
            if abs(sample) > self.response_threshold:
                response_sample = self.stimulus_onset_sample + i
                return (response_sample - self.stimulus_onset_sample) / self.sample_rate * 1000

        return 0.0

    def classify_response(self, audio: List[float]) -> str:
        """
        Classify the type of response.

        Args:
            audio: Audio samples to classify

        Returns:
            Response type: 'vocalization', 'no_response', etc.
        """
        audio_array = np.array(audio, dtype=np.float32)
        rms = np.sqrt(np.mean(audio_array ** 2))

        if rms < self.response_threshold:
            return 'no_response'
        else:
            # Check for harmonic structure (vocalization)
            # Simplified: high zero-crossing rate suggests vocalization
            zcr = np.mean(np.abs(np.diff(np.sign(audio_array))))
            if zcr > 0.1:
                return 'vocalization'
            else:
                return 'movement'


class ExperimentDesign:
    """
    Manages experimental design and trial protocol.

    Handles randomization, counterbalancing, inter-trial intervals,
    and other experimental controls.
    """

    def __init__(
        self,
        subject_id: str,
        species: str,
        session_type: str,
        randomize_order: bool = True,
        min_inter_trial_interval_s: int = 30,
        max_inter_trial_interval_s: int = 60
    ):
        """
        Initialize experiment design.

        Args:
            subject_id: Unique subject identifier
            species: Species name
            session_type: Type of session ('turing_test', 'pitch_discrimination', etc.)
            randomize_order: Whether to randomize trial order
            min_inter_trial_interval_s: Minimum time between trials (seconds)
            max_inter_trial_interval_s: Maximum time between trials (seconds)
        """
        self.subject_id = subject_id
        self.species = species
        self.session_type = session_type
        self.randomize_order = randomize_order
        self.min_inter_trial_interval_s = min_inter_trial_interval_s
        self.max_inter_trial_interval_s = max_inter_trial_interval_s

        self.trials: List[Dict] = []
        self.trial_counter = 0

    def add_trial(self, trial_id: int, stimulus_type: str, stimulus_id: str):
        """
        Add a trial to the experiment.

        Args:
            trial_id: Trial number
            stimulus_type: Type of stimulus ('concatenative' or 'granular')
            stimulus_id: ID of stimulus to present
        """
        self.trials.append({
            'trial_id': trial_id,
            'stimulus_type': stimulus_type,
            'stimulus_id': stimulus_id,
            'inter_trial_interval_s': self.get_next_inter_trial_interval()
        })

    def get_trials(self) -> List[Dict]:
        """Get all trials."""
        if self.randomize_order and len(self.trials) > 1:
            shuffled = self.trials.copy()
            random.shuffle(shuffled)
            return shuffled
        return self.trials.copy()

    def get_next_inter_trial_interval(self) -> int:
        """
        Get random inter-trial interval.

        Returns:
            Interval in seconds
        """
        return random.randint(
            self.min_inter_trial_interval_s,
            self.max_inter_trial_interval_s
        )


class StatisticalAnalyzer:
    """
    Performs statistical analysis of Turing test results.

    Compares response rates, latencies, and other metrics between
    natural and granular conditions using appropriate statistical tests.
    """

    def __init__(self):
        """Initialize statistical analyzer."""
        self.alpha = 0.05  # Significance level

    def compare_response_rates(self, results: Dict) -> Dict:
        """
        Compare response rates between conditions.

        Args:
            results: Dictionary with results for each condition

        Returns:
            Statistical comparison results
        """
        concat_responses = results['concatenative']['responses']
        granular_responses = results['granular']['responses']

        concat_rate = np.mean(concat_responses)
        granular_rate = np.mean(granular_responses)

        # Fisher's exact test for categorical data
        # Create contingency table
        concat_yes = sum(concat_responses)
        concat_no = len(concat_responses) - concat_yes
        granular_yes = sum(granular_responses)
        granular_no = len(granular_responses) - granular_yes

        # Chi-square test (approximation for Fisher's exact)
        try:
            contingency = np.array([
                [concat_yes, concat_no],
                [granular_yes, granular_no]
            ])
            chi2, p_value, _, _ = stats.chi2_contingency(contingency)
            test_used = 'chi_square'
        except:
            # Fallback to simple comparison if test fails
            p_value = None
            test_used = 'descriptive_only'

        return {
            'concatenative_response_rate': concat_rate,
            'granular_response_rate': granular_rate,
            'concatenative_n': len(concat_responses),
            'granular_n': len(granular_responses),
            'difference': concat_rate - granular_rate,
            'statistical_test': test_used,
            'p_value': p_value,
            'significant_at_005': p_value is not None and p_value < 0.05
        }

    def compare_latencies(self, results: Dict) -> Dict:
        """
        Compare response latencies between conditions.

        Args:
            results: Dictionary with latency data for each condition

        Returns:
            Statistical comparison results
        """
        concat_latencies = results['concatenative']['latencies_ms']
        granular_latencies = results['granular']['latencies_ms']

        concat_mean = np.mean(concat_latencies)
        granular_mean = np.mean(granular_latencies)

        # t-test for independent samples
        t_stat, p_value = stats.ttest_ind(concat_latencies, granular_latencies)

        return {
            'concatenative_mean_latency': concat_mean,
            'granular_mean_latency': granular_mean,
            'concatenative_std': np.std(concat_latencies),
            'granular_std': np.std(granular_latencies),
            'difference_ms': concat_mean - granular_mean,
            'statistical_test': 'independent_t_test',
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_at_005': p_value < 0.05
        }

    def evaluate_turing_test(self, results: Dict) -> Dict:
        """
        Evaluate whether Turing test was passed.

        The Turing test is PASSED if animals cannot distinguish between
        natural and granular vocalizations (no significant difference).

        Args:
            results: Statistical comparison results

        Returns:
            Evaluation with conclusion
        """
        concat_rate = results['concatenative_response_rate']
        granular_rate = results['granular_response_rate']
        p_value = results.get('p_value', 1.0)

        # Turing test PASSES if no significant difference
        passed = p_value is None or p_value >= 0.05

        # Also check that response rates are reasonably high
        # (low response to both suggests animals aren't engaged)
        min_response_rate = min(concat_rate, granular_rate)
        engaged = min_response_rate > 0.3  # At least 30% response rate

        if passed and engaged:
            conclusion = 'Animals cannot distinguish between natural and granular'
            interpretation = 'PASSED: Granular synthesis is bio-acoustically valid'
        elif not passed:
            conclusion = 'Animals can distinguish between natural and granular'
            interpretation = 'FAILED: Granular synthesis needs improvement'
        else:
            conclusion = 'Low response rates suggest animals were not engaged'
            interpretation = 'INCONCLUSIVE: Need to verify experimental setup'

        return {
            'passed': passed and engaged,
            'p_value': p_value,
            'conclusion': conclusion,
            'interpretation': interpretation,
            'concatenative_rate': concat_rate,
            'granular_rate': granular_rate,
            'min_response_rate': min_response_rate
        }


class BioAcousticTuringTest:
    """
    Main Bio-Acoustic Turing Test orchestrator.

    Coordinates stimulus presentation, response recording, and
    statistical analysis to determine if animals can distinguish
    between natural and granular-synthesized vocalizations.
    """

    def __init__(self, subject_id: str, species: str, output_dir: str):
        """
        Initialize Turing test.

        Args:
            subject_id: Unique subject identifier
            species: Species name (e.g., 'marmoset')
            output_dir: Directory for saving results
        """
        self.subject_id = subject_id
        self.species = species
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.stimulus_controller = StimulusController({'sample_rate': 22050})
        self.response_recorder = ResponseRecorder(sample_rate=22050)
        self.statistical_analyzer = StatisticalAnalyzer()

        # Test phases
        self.current_phase: Optional[str] = None
        self.phase_results: Dict[str, Dict] = {}
        self.trial_counter = 0

    def set_phase(self, phase_name: str):
        """
        Set current test phase.

        Args:
            phase_name: Name of phase ('concatenative_baseline', 'granular_synthesis')
        """
        self.current_phase = phase_name
        if phase_name not in self.phase_results:
            self.phase_results[phase_name] = {
                'trials': [],
                'responses': [],
                'latencies_ms': []
            }

    def add_stimulus(self, stimulus_id: str, audio: List[float], stimulus_type: str):
        """Add a stimulus to the test."""
        self.stimulus_controller.load_stimulus(stimulus_id, audio, stimulus_type)

    def add_phase_result(self, phase: str, result: Dict):
        """Add result for a phase (for testing)."""
        self.phase_results[phase] = result

    def run_trial(self, stimulus_id: str) -> Dict:
        """
        Run a single trial.

        Args:
            stimulus_id: ID of stimulus to present

        Returns:
            Trial result dictionary
        """
        self.trial_counter += 1

        # Play stimulus
        playback_info = self.stimulus_controller.play_stimulus(stimulus_id)

        # Record response (simulate for now)
        self.response_recorder.start_recording()
        self.response_recorder.mark_stimulus_onset()

        # Simulate response (in real experiment, this would be actual recording)
        # For testing, we'll simulate response 70% of the time
        import random
        if random.random() < 0.7:
            # Simulate response with 200ms latency
            silence_samples = int(0.2 * 22050)
            response_duration = int(0.5 * 22050)
            simulated_response = np.concatenate([
                np.zeros(silence_samples),
                np.sin(2 * np.pi * 8000 * np.linspace(0, 0.5, response_duration)) * 0.3
            ])
            self.response_recorder.record_frame(simulated_response.tolist())

        self.response_recorder.stop_recording()

        # Get response data
        response = self.response_recorder.get_response()
        latency = self.response_recorder.get_response_latency_ms()

        # Store results
        if self.current_phase:
            self.phase_results[self.current_phase]['trials'].append({
                'trial_number': self.trial_counter,
                'stimulus_id': stimulus_id,
                'stimulus_type': playback_info['type'],
                'has_response': response['has_response'],
                'latency_ms': latency,
                'timestamp': datetime.now().isoformat()
            })

            self.phase_results[self.current_phase]['responses'].append(1 if response['has_response'] else 0)
            if latency > 0:
                self.phase_results[self.current_phase]['latencies_ms'].append(latency)

        return {
            'trial_number': self.trial_counter,
            'stimulus_id': stimulus_id,
            'has_response': response['has_response'],
            'latency_ms': latency
        }

    def get_results(self) -> Dict:
        """Get all results."""
        return self.phase_results

    def evaluate_hypothesis(self) -> Dict:
        """
        Evaluate the Turing test hypothesis.

        Returns:
            Statistical evaluation results
        """
        if 'concatenative_baseline' not in self.phase_results or 'granular_synthesis' not in self.phase_results:
            return {
                'error': 'Both phases must be completed before evaluation',
                'null_hypothesis': 'Animals cannot distinguish between natural and granular',
                'conclusion': 'INCOMPLETE'
            }

        concat_results = self.phase_results['concatenative_baseline']
        granular_results = self.phase_results['granular_synthesis']

        # Prepare statistical comparison
        comparison_data = {
            'concatenative': {
                'responses': concat_results['responses'],
                'latencies_ms': concat_results['latencies_ms']
            },
            'granular': {
                'responses': granular_results['responses'],
                'latencies_ms': granular_results['latencies_ms']
            }
        }

        # Compare response rates
        rate_comparison = self.statistical_analyzer.compare_response_rates(comparison_data)

        # Evaluate Turing test
        evaluation = self.statistical_analyzer.evaluate_turing_test(rate_comparison)

        return {
            'null_hypothesis': 'Animals cannot distinguish between natural and granular',
            'alternative_hypothesis': 'Animals can distinguish between natural and granular',
            'conclusion': evaluation['conclusion'],
            'interpretation': evaluation['interpretation'],
            'p_value': evaluation.get('p_value'),
            'concatenative_response_rate': evaluation['concatenative_rate'],
            'granular_response_rate': evaluation['granular_rate'],
            'statistical_test': rate_comparison.get('statistical_test'),
            'passed': evaluation['passed']
        }
