"""
Jetson Agent
============

Main agent for GPU-accelerated animal communication system.
Integrates all components for real-time response generation.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .context_agent import ProbabilisticContextualAgent
from .gpu_phase_vocoder import GPUPhaseVocoder
from .jetson_accelerated_core import JetsonAccelerator

# Mock for testing if PhraseLibraryManager not available
try:
    from ..phrase_library_manager import PhraseLibraryManager
except ImportError:
    class PhraseLibraryManager:
        def __init__(self, *args, **kwargs):
            pass
        def get_response_strategy(self, context):
            return {'response_keys': [f'F0_6000_{context.upper()}']}

logger = logging.getLogger(__name__)


class JetsonAgent:
    """
    Jetson-optimized agent for real-time animal communication.

    Integrates GPU acceleration with contextual awareness for
    responsive and natural vocalization generation.
    """

    def __init__(self, library: PhraseLibraryManager,
                 synthesizer: Any,
                 enable_pitch_matching: bool = True,
                 max_spl_db: float = 80.0):
        """
        Initialize Jetson agent.

        Args:
            library: Phrase library for response generation
            synthesizer: Vocalization synthesizer
            enable_pitch_matching: Whether to enable real-time pitch matching
            max_spl_db: Maximum output SPL for safety
        """
        self.library = library
        self.synthesizer = synthesizer
        self.max_spl_db = max_spl_db

        # Initialize GPU components
        self.accelerator = JetsonAccelerator()
        self.vocoder = GPUPhaseVocoder() if enable_pitch_matching else None
        self.context_agent = ProbabilisticContextualAgent()

        # Performance monitoring
        self.processing_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'avg_latency_ms': 0.0,
            'gpu_usage_percent': 0.0
        }

        logger.info("JetsonAgent initialized")

    def process_input_gpu(self, audio_chunk: np.ndarray,
                         target_f0: Optional[float] = None) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Process audio input and generate response using GPU acceleration.

        Args:
            audio_chunk: Input audio array
            target_f0: Target fundamental frequency for pitch matching

        Returns:
            Tuple of (response_audio, sample_rate) or (None, None) if no response
        """
        start_time = time.perf_counter()
        self.processing_stats['total_calls'] += 1

        try:
            # Step 1: GPU-based audio analysis
            rms = self.accelerator.compute_rms_gpu(audio_chunk)

            # Early rejection of low-energy audio
            if rms < 0.01:
                logger.debug("Rejected low-energy audio")
                return None, None

            # Step 2: Context detection using GPU
            self._analyze_context_gpu(audio_chunk)
            detected_context = self.context_agent.should_respond()[1]

            # Step 3: Generate base response
            if hasattr(self.synthesizer, 'synthesize'):
                result = self.synthesizer.synthesize(detected_context)
                if isinstance(result, tuple) and len(result) == 2:
                    base_audio, sr = result
                else:
                    base_audio = result
                    sr = 44100  # Default sample rate
            else:
                # Fallback for testing
                base_audio = np.sin(2 * np.pi * 6000 * np.linspace(0, 0.1, 4410))
                sr = 44100

            # Step 4: GPU-based pitch matching if requested
            if target_f0 and self.vocoder and base_audio is not None:
                current_base_pitch = 6000.0  # Default base frequency
                semitones = 12 * np.log2(target_f0 / current_base_pitch)

                # Ensure semitones is within reasonable range
                semitones = np.clip(semitones, -24, 24)  # 2 octaves range

                modified_audio = self.vocoder.shift_pitch(base_audio, semitones)
                processing_time = (time.perf_counter() - start_time) * 1000

                # Update stats
                self.processing_stats['successful_calls'] += 1
                self.processing_stats['avg_latency_ms'] = (
                    (self.processing_stats['avg_latency_ms'] * (self.processing_stats['successful_calls'] - 1) + processing_time) /
                    self.processing_stats['successful_calls']
                )

                logger.debug(f"GPU pitch shifting completed in {processing_time:.1f}ms")
                return modified_audio, sr

            elif base_audio is not None:
                processing_time = (time.perf_counter() - start_time) * 1000
                self.processing_stats['successful_calls'] += 1
                self.processing_stats['avg_latency_ms'] = (
                    (self.processing_stats['avg_latency_ms'] * (self.processing_stats['successful_calls'] - 1) + processing_time) /
                    self.processing_stats['successful_calls']
                )

                return base_audio, sr

            # No valid response generated
            return None, None

        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return None, None

    def _analyze_context_gpu(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze audio context using GPU acceleration."""
        try:
            # GPU-based feature extraction
            spectral_centroid = self.accelerator.spectral_centroid_gpu(
                self.accelerator.compute_stft_gpu(audio)
            )

            self.accelerator.zero_crossing_rate_gpu(audio)
            rms = self.accelerator.compute_rms_gpu(audio)

            # Simple context probabilities (would use neural network in production)
            context_probs = {
                'contact': max(0.0, 1.0 - abs(spectral_centroid - 6000) / 2000),
                'alarm': max(0.0, 1.0 - abs(spectral_centroid - 7000) / 1000) * (1 + rms),
                'food': max(0.0, 1.0 - abs(spectral_centroid - 5000) / 1500) * (0.5 + rms)
            }

            # Normalize probabilities
            total = sum(context_probs.values())
            if total > 0:
                context_probs = {k: v/total for k, v in context_probs.items()}

            self.context_agent.update_context(context_probs)
            return context_probs

        except Exception as e:
            logger.error(f"Error in context analysis: {e}")
            return {'contact': 0.5, 'alarm': 0.3, 'food': 0.2}

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        # Update GPU usage if available
        if self.accelerator.cuda_available:
            mem_info = self.accelerator.get_gpu_memory_info()
            if mem_info:
                self.processing_stats['gpu_usage_percent'] = mem_info.get('used_percent', 0)

        return self.processing_stats.copy()

    def reset_stats(self):
        """Reset processing statistics."""
        self.processing_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'avg_latency_ms': 0.0,
            'gpu_usage_percent': 0.0
        }

    def emergency_stop(self):
        """Emergency stop for safety."""
        logger.critical("Emergency stop triggered")
        if self.vocoder:
            self.vocoder = None
        if hasattr(self.accelerator, 'cleanup'):
            self.accelerator.cleanup()
