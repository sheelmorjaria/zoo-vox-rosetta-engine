#!/usr/bin/env python3
"""
Sub-50ms Phoneme Discovery Pipeline

The old 50ms debounce merged rapid ultrasonic trills and chirps
into single "syllables," completely missing micro-syntax.

This module uses the CPC/Mamba Predictive NBD to detect sub-50ms
boundaries, feeding micro-units into VQ-VAE for tokenization.
Enables discovery of combinatorial phonology in bat vocalizations.

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.stats import entropy

logger = logging.getLogger(__name__)


@dataclass
class MicroUnit:
    """
    A sub-50ms phonetic unit in bat vocalization.

    Attributes:
        start_ms: Start time
        end_ms: End time
        duration_ms: Duration (always <50ms)
        audio: Audio segment
        token_id: VQ-VAE token ID
        spectral_centroid: Mean spectral centroid
        f0_mean: Mean fundamental frequency
        f0_std: Standard deviation of F0
        bandwidth: Spectral bandwidth
    """
    start_ms: float
    end_ms: float
    duration_ms: float
    audio: np.ndarray
    token_id: int
    spectral_centroid: float
    f0_mean: float
    f0_std: float
    bandwidth: float


@dataclass
class PhonemeSequence:
    """
    A sequence of micro-units forming a combinatorial phoneme.

    Unlike old pipeline's single "syllable," this captures
    the true micro-syntax: e.g., [20ms "A"] + [30ms "B"] = unique meaning.
    """
    units: List[MicroUnit]
    sequence_id: str
    bat_id: int
    start_ms: float
    end_ms: float

    @property
    def token_sequence(self) -> Tuple[int, ...]:
        """Get token IDs as tuple."""
        return tuple(u.token_id for u in self.units)

    @property
    def duration_ms(self) -> float:
        """Total duration."""
        return self.end_ms - self.start_ms

    @property
    def unit_count(self) -> int:
        """Number of micro-units."""
        return len(self.units)


@dataclass
class PhonotacticRule:
    """
    Discovered rule governing phoneme combination.

    Attributes:
        prefix: Token sequence prefix
        possible_next: Allowed next tokens
        probabilities: Probability of each next token
        count: How often this rule was observed
    """
    prefix: Tuple[int, ...]
    possible_next: List[int]
    probabilities: Dict[int, float]
    count: int


class MicroPhonologyAnalyzer:
    """
    Discovers combinatorial phonology in sub-50ms micro-units.

    Pipeline:
    1. CPC/Mamba NBD detects sub-50ms boundaries
    2. Extract features for each micro-unit
    3. VQ-VAE tokenizes into discrete tokens
    4. Analyze phonotactic rules (bigrams, trigrams, etc.)
    5. Test for combinatorial meaning (A+B ≠ B+A)
    """

    def __init__(
        self,
        min_duration_ms: float = 10,
        max_duration_ms: float = 50,
        sample_rate: int = 48000,
    ):
        """
        Initialize micro-phonology analyzer.

        Args:
            min_duration_ms: Minimum micro-unit duration
            max_duration_ms: Maximum micro-unit duration
            sample_rate: Audio sample rate
        """
        self.min_duration = min_duration_ms
        self.max_duration = max_duration_ms
        self.sample_rate = sample_rate
        self.min_samples = int(min_duration_ms * sample_rate / 1000)
        self.max_samples = int(max_duration_ms * sample_rate / 1000)

        # Discovered phonotactic rules
        self.unigrams: Dict[int, int] = {}
        self.bigrams: Dict[Tuple[int, int], int] = {}
        self.trigrams: Dict[Tuple[int, int, int], int] = {}

        logger.info("MicroPhonologyAnalyzer initialized")

    def detect_micro_boundaries(
        self,
        audio: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Detect sub-50ms phonetic boundaries using energy-based detection.

        In production, this would use CPC/Mamba Predictive NBD.

        Args:
            audio: Audio samples

        Returns:
            List of (start_sample, end_sample) tuples
        """
        # Compute RMS energy
        frame_size = 512
        hop_size = 128

        # Frame the audio
        n_frames = 1 + (len(audio) - frame_size) // hop_size
        energy = []

        for i in range(n_frames):
            start = i * hop_size
            end = min(start + frame_size, len(audio))
            frame = audio[start:end]
            rms = np.sqrt(np.mean(frame ** 2))
            energy.append(rms)

        energy = np.array(energy)

        # Find onset/offset using threshold
        threshold = np.mean(energy) + 2 * np.std(energy)
        above_threshold = energy > threshold

        # Find transitions
        boundaries = []
        in_segment = False
        start_sample = 0

        for i, is_above in enumerate(above_threshold):
            sample = i * hop_size

            if is_above and not in_segment:
                # Onset
                start_sample = sample
                in_segment = True
            elif not is_above and in_segment:
                # Offset
                end_sample = min(sample + frame_size, len(audio))

                # Check duration constraints
                duration_samples = end_sample - start_sample
                duration_ms = duration_samples * 1000 / self.sample_rate

                if self.min_duration <= duration_ms <= self.max_duration:
                    boundaries.append((start_sample, end_sample))

                in_segment = False

        # Handle case where audio ends while above threshold
        if in_segment:
            end_sample = len(audio)
            duration_ms = (end_sample - start_sample) * 1000 / self.sample_rate
            if self.min_duration <= duration_ms <= self.max_duration:
                boundaries.append((start_sample, end_sample))

        return boundaries

    def extract_micro_unit(
        self,
        audio: np.ndarray,
        start_sample: int,
        end_sample: int,
        token_id: int,
    ) -> MicroUnit:
        """
        Extract features for a micro-unit.

        Args:
            audio: Full audio
            start_sample: Start index
            end_sample: End index
            token_id: VQ-VAE token ID

        Returns:
            MicroUnit with extracted features
        """
        unit_audio = audio[start_sample:end_sample]
        duration_ms = (end_sample - start_sample) * 1000 / self.sample_rate

        # Compute spectral features
        freqs, times, Sxx = signal.spectrogram(
            unit_audio,
            fs=self.sample_rate,
            nperseg=min(256, len(unit_audio) // 2)
        )

        # Spectral centroid
        centroid = np.sum(freqs[:, None] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-10)
        spectral_centroid = np.mean(centroid)

        # Bandwidth
        bandwidth = np.std(centroid)

        # F0 estimation (autocorrelation)
        f0_values = []
        for i in range(Sxx.shape[1]):
            col = Sxx[:, i]
            if np.sum(col) > 1e-10:
                # Peak frequency
                peak_idx = np.argmax(col)
                f0 = freqs[peak_idx]
                if 1000 < f0 < 100000:  # Bat frequency range
                    f0_values.append(f0)

        if f0_values:
            f0_mean = np.mean(f0_values)
            f0_std = np.std(f0_values)
        else:
            f0_mean = 0
            f0_std = 0

        return MicroUnit(
            start_ms=start_sample * 1000 / self.sample_rate,
            end_ms=end_sample * 1000 / self.sample_rate,
            duration_ms=duration_ms,
            audio=unit_audio,
            token_id=token_id,
            spectral_centroid=spectral_centroid,
            f0_mean=f0_mean,
            f0_std=f0_std,
            bandwidth=bandwidth,
        )

    def create_phoneme_sequence(
        self,
        audio: np.ndarray,
        boundaries: List[Tuple[int, int]],
        token_ids: List[int],
        bat_id: int,
        sequence_id: str,
    ) -> PhonemeSequence:
        """
        Create a phoneme sequence from boundaries and tokens.

        Args:
            audio: Full audio
            boundaries: List of (start, end) sample indices
            token_ids: Token ID for each boundary
            bat_id: Bat identifier
            sequence_id: Unique sequence ID

        Returns:
            PhonemeSequence with micro-units
        """
        units = []

        for (start, end), token_id in zip(boundaries, token_ids):
            unit = self.extract_micro_unit(audio, start, end, token_id)
            units.append(unit)

        return PhonemeSequence(
            units=units,
            sequence_id=sequence_id,
            bat_id=bat_id,
            start_ms=units[0].start_ms if units else 0,
            end_ms=units[-1].end_ms if units else 0,
        )

    def analyze_phonotactics(
        self,
        sequences: List[PhonemeSequence],
    ) -> Dict[str, List[PhonotacticRule]]:
        """
        Discover phonotactic rules from sequences.

        Args:
            sequences: List of phoneme sequences

        Returns:
            Dictionary with 'unigrams', 'bigrams', 'trigrams' rules
        """
        # Reset counts
        self.unigrams = {}
        self.bigrams = {}
        self.trigrams = {}

        for seq in sequences:
            tokens = seq.token_sequence

            # Unigrams
            for token in tokens:
                self.unigrams[token] = self.unigrams.get(token, 0) + 1

            # Bigrams
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                self.bigrams[bigram] = self.bigrams.get(bigram, 0) + 1

            # Trigrams
            for i in range(len(tokens) - 2):
                trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
                self.trigrams[trigram] = self.trigrams.get(trigram, 0) + 1

        # Convert to rules
        unigram_rules = self._create_unigram_rules()
        bigram_rules = self._create_bigram_rules()
        trigram_rules = self._create_trigram_rules()

        return {
            "unigrams": unigram_rules,
            "bigrams": bigram_rules,
            "trigrams": trigram_rules,
        }

    def _create_unigram_rules(self) -> List[PhonotacticRule]:
        """Create unigram rules (single token probabilities)."""
        rules = []
        total = sum(self.unigrams.values())

        for token, count in self.unigrams.items():
            rules.append(PhonotacticRule(
                prefix=(),
                possible_next=[token],
                probabilities={token: count / total},
                count=count,
            ))

        return rules

    def _create_bigram_rules(self) -> List[PhonotacticRule]:
        """Create bigram rules (token transition probabilities)."""
        rules = {}

        for (prev_token, curr_token), count in self.bigrams.items():
            if prev_token not in rules:
                rules[prev_token] = {
                    "prefix": (prev_token,),
                    "possible_next": [],
                    "probabilities": {},
                    "count": 0,
                }

            rules[prev_token]["possible_next"].append(curr_token)
            rules[prev_token]["probabilities"][curr_token] = count
            rules[prev_token]["count"] += count

        # Normalize probabilities
        result = []
        for r in rules.values():
            total = r["count"]
            normalized_probs = {
                k: v / total for k, v in r["probabilities"].items()
            }
            result.append(PhonotacticRule(
                prefix=r["prefix"],
                possible_next=r["possible_next"],
                probabilities=normalized_probs,
                count=r["count"],
            ))

        return result

    def _create_trigram_rules(self) -> List[PhonotacticRule]:
        """Create trigram rules (two-token context)."""
        rules = {}

        for (t1, t2, t3), count in self.trigrams.items():
            prefix = (t1, t2)
            if prefix not in rules:
                rules[prefix] = {
                    "possible_next": [],
                    "probabilities": {},
                    "count": 0,
                }

            rules[prefix]["possible_next"].append(t3)
            rules[prefix]["probabilities"][t3] = count
            rules[prefix]["count"] += count

        # Normalize
        result = []
        for prefix, r in rules.items():
            total = r["count"]
            normalized_probs = {
                k: v / total for k, v in r["probabilities"].items()
            }
            result.append(PhonotacticRule(
                prefix=prefix,
                possible_next=r["possible_next"],
                probabilities=normalized_probs,
                count=total,
            ))

        return result

    def test_combinatorial_meaning(
        self,
        sequences: List[PhonemeSequence],
    ) -> Dict[Tuple[int, int], Dict[Tuple[int, int], float]]:
        """
        Test if phoneme order carries combinatorial meaning.

        Checks if sequence A+B has different behavioral response
        than B+A.

        Args:
            sequences: Phoneme sequences with response data

        Returns:
            Dictionary mapping (token_a, token_b) to response differences
        """
        # Group by two-token sequences
        pairs = {}

        for seq in sequences:
            tokens = seq.token_sequence
            if len(tokens) == 2:
                pairs[tokens] = pairs.get(tokens, 0) + 1

        # Check for both orders
        combinatorial = {}

        for (a, b), count in pairs.items():
            if (b, a) in pairs:
                # Both orders exist
                # Check if response differs (would need behavioral data)
                # For now, just flag the pair
                if (a, b) not in combinatorial:
                    combinatorial[(a, b)] = {}
                combinatorial[(a, b)][(b, a)] = count

        return combinatorial


def visualize_phonotactic_rules(
    rules: Dict[str, List[PhonotacticRule]],
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize discovered phonotactic rules.

    Args:
        rules: Dictionary of rules from analyze_phonotactics
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Unigrams (token frequencies)
    unigrams = rules["unigrams"]
    if unigrams:
        tokens = [r.possible_next[0] for r in unigrams]
        probs = [r.probabilities[t] for r, t in zip(unigrams, tokens)]

        axes[0].bar(tokens, probs)
        axes[0].set_title("Unigram Frequencies")
        axes[0].set_xlabel("Token ID")
        axes[0].set_ylabel("Probability")

    # Bigrams (transition network)
    bigrams = rules["bigrams"]
    if bigrams:
        G = nx.DiGraph()

        for rule in bigrams:
            source = rule.prefix[0]
            for target, prob in rule.probabilities.items():
                G.add_edge(source, target, weight=prob)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=axes[1], with_labels=True,
                node_size=500, arrowsize=20)
        axes[1].set_title("Bigram Transition Network")

    # Trigrams (context-dependent)
    trigrams = rules["trigrams"]
    if trigrams:
        # Show top trigrams by count
        sorted_trigrams = sorted(trigrams, key=lambda r: r.count, reverse=True)[:10]

        trigram_labels = [str(r.prefix) + "→" + str(r.possible_next)
                          for r in sorted_trigrams]
        counts = [r.count for r in sorted_trigrams]

        axes[2].barh(range(len(trigram_labels)), counts)
        axes[2].set_yticks(range(len(trigram_labels)))
        axes[2].set_yticklabels(trigram_labels, fontsize=8)
        axes[2].set_title("Top Trigram Rules")
        axes[2].set_xlabel("Count")

    plt.suptitle("Phonotactic Rule Discovery")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved phonotactics plot to {save_path}")
    else:
        plt.show()


# Preset configurations

# Default micro-phonology analyzer for bats
BAT_MICRO_PHONOLOGY = MicroPhonologyAnalyzer(
    min_duration_ms=10,
    max_duration_ms=50,
    sample_rate=48000,
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Micro-Phonology Analysis Demo")
    print("=" * 50)

    analyzer = BAT_MICRO_PHONOLOGY

    # Generate synthetic bat audio with micro-units
    sample_rate = 48000
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create three distinct micro-units (different frequencies)
    audio = np.zeros_like(t)

    # Unit 1: 10ms chirp at 10kHz
    unit1_start = int(0.0 * sample_rate)
    unit1_end = int(0.01 * sample_rate)
    audio[unit1_start:unit1_end] = 0.5 * np.sin(
        2 * np.pi * (10000 + 5000 * np.linspace(0, 1, unit1_end - unit1_start)) *
        np.linspace(0, 1, unit1_end - unit1_start)
    )

    # Unit 2: 30ms trill at 15kHz
    unit2_start = int(0.02 * sample_rate)
    unit2_end = int(0.05 * sample_rate)
    audio[unit2_start:unit2_end] = 0.3 * np.sin(
        2 * np.pi * 15000 * np.linspace(0, 1, unit2_end - unit2_start) +
        0.5 * np.sin(2 * np.pi * 16000 * np.linspace(0, 1, unit2_end - unit2_start))
    )

    # Unit 3: 20ms sweep at 20kHz
    unit3_start = int(0.06 * sample_rate)
    unit3_end = int(0.08 * sample_rate)
    audio[unit3_start:unit3_end] = 0.4 * np.sin(
        2 * np.pi * (20000 - 5000 * np.linspace(0, 1, unit3_end - unit3_start)) *
        np.linspace(0, 1, unit3_end - unit3_start)
    )

    # Detect boundaries
    boundaries = analyzer.detect_micro_boundaries(audio)

    print(f"\nDetected {len(boundaries)} micro-units:")
    for i, (start, end) in enumerate(boundaries):
        duration_ms = (end - start) * 1000 / sample_rate
        print(f"  Unit {i+1}: {duration_ms:.1f}ms")

    # Create sequences
    token_ids = [1, 5, 3]  # Simulated VQ-VAE tokens
    sequence = analyzer.create_phoneme_sequence(
        audio=audio,
        boundaries=boundaries,
        token_ids=token_ids,
        bat_id=1,
        sequence_id="test_seq_001",
    )

    print(f"\nPhoneme Sequence: {sequence.sequence_id}")
    print(f"  Token sequence: {sequence.token_sequence}")
    print(f"  Duration: {sequence.duration_ms:.1f}ms")
    print(f"  Units: {sequence.unit_count}")

    # Analyze multiple sequences
    sequences = [sequence]
    rules = analyzer.analyze_phonotactics(sequences)

    print(f"\nDiscovered Rules:")
    print(f"  Unigrams: {len(rules['unigrams'])}")
    print(f"  Bigrams: {len(rules['bigrams'])}")
    print(f"  Trigrams: {len(rules['trigrams'])}")
