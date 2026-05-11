#!/usr/bin/env python3
"""
Chaos Corpus Generator

Generates dense overlapping vocalization mixes for stress testing the
syntactic coherence of the AI under chaotic multi-speaker conditions.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChaosMixConfig:
    """Configuration for chaos mix generation."""
    duration_seconds: int = 600
    overlap_count: int = 5
    sample_rate: int = 22050
    min_gain: float = 0.3
    max_gain: float = 1.0
    normalize: bool = True
    target_level: float = 0.9


@dataclass
class ChaosMixResult:
    """Result of chaos mix generation."""
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    files_used: int
    overlap_count: int


class ChaosCorpusGenerator:
    """
    Generates dense overlapping vocalization mixes for stress testing.

    Uses real bat vocalizations from the 91K corpus to simulate
    colony dispute scenarios with overlapping calls.
    """

    def __init__(
        self,
        corpus_dir: str = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio/",
    ):
        """
        Initialize the chaos corpus generator.

        Args:
            corpus_dir: Path to directory containing vocalization audio files
        """
        self.corpus_dir = Path(corpus_dir)

        if not self.corpus_dir.exists():
            logger.warning(f"Corpus directory not found: {self.corpus_dir}")
            self.audio_files = []
        else:
            # Find all audio files
            self.audio_files = list(self.corpus_dir.glob("*.wav"))
            self.audio_files.extend(self.corpus_dir.glob("*.flac"))
            self.audio_files.extend(self.corpus_dir.glob("*.ogg"))

            logger.info(f"Found {len(self.audio_files)} audio files in corpus")

    def _load_audio_file(self, filepath: Path) -> Optional[Tuple[np.ndarray, int]]:
        """
        Load an audio file.

        Args:
            filepath: Path to audio file

        Returns:
            Tuple of (audio_samples, sample_rate) or None if failed
        """
        try:
            import soundfile as sf

            audio, sr = sf.read(str(filepath))

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            return audio, sr
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
            return None

    def generate_chaos_mix(
        self,
        config: Optional[ChaosMixConfig] = None,
    ) -> ChaosMixResult:
        """
        Generate overlapping vocalization mix simulating dense colony dispute.

        Args:
            config: Configuration for mix generation

        Returns:
            ChaosMixResult with generated audio

        Raises:
            ValueError: If no audio files available
        """
        if config is None:
            config = ChaosMixConfig()

        if not self.audio_files:
            raise ValueError("No audio files available in corpus")

        sample_rate = config.sample_rate
        total_samples = config.duration_seconds * sample_rate
        output = np.zeros(total_samples)
        files_used = 0

        # Calculate number of vocalizations to add
        num_vocalizations = config.duration_seconds * config.overlap_count

        logger.info(
            f"Generating {config.duration_seconds}s chaos mix "
            f"with ~{num_vocalizations} overlapping vocalizations"
        )

        for _ in range(num_vocalizations):
            # Randomly select audio file
            audio_file = random.choice(self.audio_files)
            result = self._load_audio_file(audio_file)

            if result is None:
                continue

            audio, sr = result

            # Resample if needed
            if sr != sample_rate:
                # Simple linear interpolation (for production, use scipy.signal.resample)
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * sample_rate / sr))

            # Random start time within output
            start_sample = random.randint(0, max(1, total_samples - len(audio)))
            end_sample = min(start_sample + len(audio), total_samples)
            audio_segment = audio[:end_sample - start_sample]

            # Mix with random gain (simulate different distances)
            gain = random.uniform(config.min_gain, config.max_gain)
            output[start_sample:end_sample] += audio_segment * gain
            files_used += 1

        # Normalize to prevent clipping
        if config.normalize and np.max(np.abs(output)) > 0:
            output = output / np.max(np.abs(output)) * config.target_level

        logger.info(
            f"Generated chaos mix: {len(output)/sample_rate:.2f}s, "
            f"{files_used} files mixed"
        )

        return ChaosMixResult(
            audio=output,
            sample_rate=sample_rate,
            duration_seconds=len(output) / sample_rate,
            files_used=files_used,
            overlap_count=config.overlap_count,
        )

    def save_chaos_mix(
        self,
        result: ChaosMixResult,
        output_path: str,
    ) -> None:
        """
        Save generated chaos mix to file.

        Args:
            result: Chaos mix result
            output_path: Path to save audio file
        """
        try:
            import soundfile as sf

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            sf.write(
                str(output_path),
                result.audio,
                result.sample_rate,
            )

            logger.info(f"Saved chaos mix to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save chaos mix: {e}")

    def generate_test_batches(
        self,
        batch_duration_seconds: int = 60,
        num_batches: int = 10,
    ) -> List[ChaosMixResult]:
        """
        Generate multiple test batches of chaos audio.

        Args:
            batch_duration_seconds: Duration of each batch
            num_batches: Number of batches to generate

        Returns:
            List of chaos mix results
        """
        results = []

        for i in range(num_batches):
            config = ChaosMixConfig(
                duration_seconds=batch_duration_seconds,
                overlap_count=5,
            )

            result = self.generate_chaos_mix(config)
            results.append(result)

            logger.info(f"Generated batch {i+1}/{num_batches}")

        return results
