# phrase_audio_library.py
"""
Phrase Audio Library for Animal Vocalization Synthesis
======================================================

This module provides a comprehensive system for:
1. Building phrase audio libraries during analysis
2. Audio segmentation synchronized with phrase detection
3. Multi-mode synthesis:
   - Concatenative synthesis (horizontal/sequential encoding)
   - Superpositional synthesis (vertical/simultaneous encoding)
   - Combined synthesis (mixed horizontal and vertical encoding)

The library enables real-time synthesis of animal vocalizations based on
discovered phrase structures, supporting both sequential phrase combinations
and simultaneous superposition of phrases.

Usage:
    from phrase_audio_library import (
        PhraseAudioLibrary,
        PhraseAudioSegment,
        VocalizationSynthesizer
    )

    # Create library during analysis
    library = PhraseAudioLibrary(species='marmoset', sr=44100)

    # Extract and store phrase audio
    segment = library.extract_phrase_segment(audio, sr, start_ms, end_ms, phrase_key)

    # Synthesize new vocalizations
    synthesizer = VocalizationSynthesizer(library)

    # Horizontal (concatenative) synthesis
    result = synthesizer.synthesize_horizontal(['F0_6400_DUR_5_RANGE_0', 'F0_6600_DUR_10_RANGE_0'])

    # Vertical (superpositional) synthesis
    result = synthesizer.synthesize_vertical(['F0_6400_DUR_5_RANGE_0', 'F0_6600_DUR_5_RANGE_0'])

    # Combined synthesis
    result = synthesizer.synthesize_combined([
        ('horizontal', ['F0_6400_DUR_5_RANGE_0', 'F0_6600_DUR_10_RANGE_0']),
        ('vertical', ['F0_6400_DUR_5_RANGE_0', 'F0_6800_DUR_5_RANGE_0'])
    ])

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ============================================================================
# Phrase Audio Segment Data Structure
# ============================================================================


@dataclass
class PhraseAudioSegment:
    """
    A single phrase audio segment with metadata.

    Stores the actual audio waveform along with all relevant metadata
    for synthesis and analysis, including microharmonic signature.
    """

    # Audio data
    audio: np.ndarray  # Audio waveform
    sr: int  # Sample rate

    # Phrase metadata
    phrase_key: str  # Phrase signature (e.g., "F0_6400_DUR_5_RANGE_0")
    source_file: str  # Original audio file
    start_time_ms: float  # Start time in source
    end_time_ms: float  # End time in source

    # Acoustic features
    mean_f0_hz: float
    std_f0_hz: float
    mean_duration_ms: float
    mean_range_hz: float

    # Encoding information
    encoding: str = "horizontal"  # "horizontal" or "vertical"
    superposed_with: List[str] = field(default_factory=list)

    # Additional metadata
    occurrence_id: str = ""  # Unique ID for this occurrence
    context: Optional[str] = None  # Behavioral context if available
    individual_id: Optional[str] = None  # Individual ID if available

    # Quality metrics
    snr_db: float = 0.0  # Signal-to-noise ratio
    quality_score: float = 1.0  # Overall quality (0-1)

    # Microharmonic signature
    microharmonic_signature: Optional[Dict[str, Any]] = None  # Microharmonic features

    def __post_init__(self):
        """Generate occurrence ID if not provided."""
        if not self.occurrence_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.occurrence_id = f"{self.phrase_key}_{timestamp}"

        # Initialize microharmonic signature dict if None
        if self.microharmonic_signature is None:
            self.microharmonic_signature = {}

    @property
    def duration_samples(self) -> int:
        """Get duration in samples."""
        return len(self.audio)

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return len(self.audio) / self.sr

    @property
    def has_microharmonics(self) -> bool:
        """Check if segment has microharmonic data."""
        return bool(self.microharmonic_signature)

    @property
    def dominant_harmonic(self) -> Optional[int]:
        """Get dominant harmonic index (1-based) if available."""
        if self.microharmonic_signature and "dominant_harmonic" in self.microharmonic_signature:
            return self.microharmonic_signature["dominant_harmonic"]
        return None

    @property
    def harmonic_entropy(self) -> Optional[float]:
        """Get harmonic entropy (spectral complexity) if available."""
        if self.microharmonic_signature and "harmonic_entropy" in self.microharmonic_signature:
            return self.microharmonic_signature["harmonic_entropy"]
        return None

    @property
    def spectral_centroid_hz(self) -> Optional[float]:
        """Get spectral centroid in Hz if available."""
        if self.microharmonic_signature and "spectral_centroid_hz" in self.microharmonic_signature:
            return self.microharmonic_signature["spectral_centroid_hz"]
        return None

    def get_harmonic_ratios(self) -> Optional[np.ndarray]:
        """Get harmonic amplitude ratios if available."""
        if self.microharmonic_signature and "harmonic_ratios" in self.microharmonic_signature:
            return self.microharmonic_signature["harmonic_ratios"]
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "phrase_key": self.phrase_key,
            "source_file": self.source_file,
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "mean_f0_hz": self.mean_f0_hz,
            "std_f0_hz": self.std_f0_hz,
            "mean_duration_ms": self.mean_duration_ms,
            "mean_range_hz": self.mean_range_hz,
            "encoding": self.encoding,
            "superposed_with": self.superposed_with,
            "occurrence_id": self.occurrence_id,
            "context": self.context,
            "individual_id": self.individual_id,
            "duration_samples": self.duration_samples,
            "duration_seconds": self.duration_seconds,
            "snr_db": self.snr_db,
            "quality_score": self.quality_score,
            "has_microharmonics": self.has_microharmonics,
            # Note: audio data and microharmonic_signature are serialized separately
        }

        # Add microharmonic summary fields
        if self.has_microharmonics:
            result.update(
                {
                    "dominant_harmonic": self.dominant_harmonic,
                    "harmonic_entropy": self.harmonic_entropy,
                    "spectral_centroid_hz": self.spectral_centroid_hz,
                }
            )

        return result


# ============================================================================
# Phrase Audio Library
# ============================================================================


class PhraseAudioLibrary:
    """
    Library of phrase audio segments for synthesis and analysis.

    This class manages the storage and retrieval of phrase audio segments,
    organized by phrase signature for efficient lookup during synthesis.

    Features:
    - Storage of audio segments by phrase key
    - Quality filtering and ranking
    - Context-aware selection
    - Serialization to disk
    - Statistical analysis
    """

    def __init__(
        self,
        species: str,
        sr: int,
        library_dir: Optional[Union[str, Path]] = None,
        max_segments_per_phrase: int = 100,
        min_quality_score: float = 0.3,
    ):
        """
        Initialize the phrase audio library.

        Args:
            species: Species name
            sr: Sample rate
            library_dir: Directory for saving/loading library
            max_segments_per_phrase: Maximum segments to store per phrase
            min_quality_score: Minimum quality score for storage
        """
        self.species = species
        self.sr = sr
        self.library_dir = Path(library_dir) if library_dir else None
        self.max_segments_per_phrase = max_segments_per_phrase
        self.min_quality_score = min_quality_score

        # Storage: phrase_key -> list of segments
        self.phrase_segments: Dict[str, List[PhraseAudioSegment]] = defaultdict(list)

        # Metadata
        self.total_segments = 0
        self.total_phrases = 0
        self.creation_time = datetime.now()

        logger.info(f"PhraseAudioLibrary initialized for {species} at {sr} Hz")

    def add_segment(self, segment: PhraseAudioSegment, allow_duplicate: bool = False) -> bool:
        """
        Add a phrase audio segment to the library.

        Args:
            segment: PhraseAudioSegment to add
            allow_duplicate: Whether to allow duplicate segments

        Returns:
            True if segment was added, False otherwise
        """
        # Quality check
        if segment.quality_score < self.min_quality_score:
            return False

        phrase_key = segment.phrase_key

        # Check for duplicates
        if not allow_duplicate:
            for existing in self.phrase_segments[phrase_key]:
                if (
                    existing.source_file == segment.source_file
                    and abs(existing.start_time_ms - segment.start_time_ms) < 10
                ):
                    return False

        # Enforce maximum segments per phrase
        if len(self.phrase_segments[phrase_key]) >= self.max_segments_per_phrase:
            # Remove lowest quality segment
            self.phrase_segments[phrase_key].sort(key=lambda s: s.quality_score, reverse=True)
            if segment.quality_score > self.phrase_segments[phrase_key][-1].quality_score:
                self.phrase_segments[phrase_key].pop()
            else:
                return False

        self.phrase_segments[phrase_key].append(segment)
        self.total_segments += 1

        return True

    def extract_phrase_segment(
        self,
        audio: np.ndarray,
        sr: int,
        start_ms: float,
        end_ms: float,
        phrase_key: str,
        source_file: str = "",
        mean_f0_hz: float = 0.0,
        std_f0_hz: float = 0.0,
        mean_range_hz: float = 0.0,
        encoding: str = "horizontal",
        superposed_with: List[str] = None,
        context: Optional[str] = None,
        individual_id: Optional[str] = None,
        microharmonic_signature: Optional[Dict[str, Any]] = None,
    ) -> Optional[PhraseAudioSegment]:
        """
        Extract a phrase audio segment from audio and add to library.

        This is the primary method called during phrase detection to
        automatically build the phrase audio library.

        Args:
            audio: Full audio waveform
            sr: Sample rate
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            phrase_key: Phrase signature
            source_file: Original file name
            mean_f0_hz: Mean F0 in Hz
            std_f0_hz: Standard deviation of F0
            mean_range_hz: F0 range
            encoding: Encoding type (horizontal/vertical)
            superposed_with: List of superposed phrase keys
            context: Behavioral context
            individual_id: Individual identifier
            microharmonic_signature: Optional microharmonic features dict

        Returns:
            PhraseAudioSegment if extracted successfully, None otherwise
        """
        try:
            # Convert time to samples
            start_sample = int(start_ms / 1000 * sr)
            end_sample = int(end_ms / 1000 * sr)

            # Ensure bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)

            if end_sample <= start_sample:
                return None

            # Extract segment
            segment_audio = audio[start_sample:end_sample].copy()

            # Apply fade in/out to prevent clicks
            fade_samples = min(int(0.002 * sr), len(segment_audio) // 10)  # 2ms fade
            if fade_samples > 0:
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                segment_audio[:fade_samples] *= fade_in
                segment_audio[-fade_samples:] *= fade_out

            # Resample if necessary
            if sr != self.sr:
                segment_audio = librosa.resample(segment_audio, orig_sr=sr, target_sr=self.sr)

            # Calculate quality metrics
            snr_db = self._calculate_snr(segment_audio)
            quality_score = self._calculate_quality_score(segment_audio, snr_db)

            # Create segment
            duration_ms = (end_sample - start_sample) / sr * 1000

            segment = PhraseAudioSegment(
                audio=segment_audio,
                sr=self.sr,
                phrase_key=phrase_key,
                source_file=source_file,
                start_time_ms=start_ms,
                end_time_ms=end_ms,
                mean_f0_hz=mean_f0_hz,
                std_f0_hz=std_f0_hz,
                mean_duration_ms=duration_ms,
                mean_range_hz=mean_range_hz,
                encoding=encoding,
                superposed_with=superposed_with or [],
                context=context,
                individual_id=individual_id,
                snr_db=snr_db,
                quality_score=quality_score,
                microharmonic_signature=microharmonic_signature,
            )

            # Add to library
            if self.add_segment(segment):
                return segment

            return None

        except Exception as e:
            logger.error(f"Error extracting phrase segment: {e}")
            return None

    def create_phrase_segment(
        self,
        audio: np.ndarray,
        phrase_key: str,
        context: Optional[str] = None,
        individual_id: Optional[str] = None,
        **kwargs,
    ) -> Optional[PhraseAudioSegment]:
        """
        Create a phrase audio segment manually and add to library.

        This is a convenience method for testing when automatic extraction
        might fail due to quality constraints.

        Args:
            audio: Audio waveform
            phrase_key: Phrase signature
            context: Behavioral context
            individual_id: Individual identifier
            **kwargs: Additional segment attributes

        Returns:
            PhraseAudioSegment if created successfully, None otherwise
        """
        try:
            # Create a basic segment with default parameters
            duration_ms = len(audio) / self.sr * 1000

            segment = PhraseAudioSegment(
                audio=audio,
                sr=self.sr,
                phrase_key=phrase_key,
                source_file="manual_test",
                start_time_ms=0,
                end_time_ms=duration_ms,
                mean_f0_hz=kwargs.get("mean_f0_hz", 1000.0),
                std_f0_hz=kwargs.get("std_f0_hz", 100.0),
                mean_duration_ms=duration_ms,
                mean_range_hz=kwargs.get("mean_range_hz", 500.0),
                encoding=kwargs.get("encoding", "horizontal"),
                superposed_with=kwargs.get("superposed_with", []),
                context=context,
                individual_id=individual_id,
                snr_db=kwargs.get("snr_db", 20.0),
                quality_score=kwargs.get("quality_score", 0.8),
                microharmonic_signature=kwargs.get("microharmonic_signature"),
            )

            # Add to library
            if self.add_segment(segment):
                return segment
            return None

        except Exception as e:
            logger.error(f"Error creating phrase segment: {e}")
            return None

    def get_segment(
        self, phrase_key: str, strategy: str = "random", min_quality: float = 0.5
    ) -> Optional[PhraseAudioSegment]:
        """
        Get a phrase audio segment from the library.

        Args:
            phrase_key: Phrase signature to retrieve
            strategy: Selection strategy ("random", "best", "highest_snr", "most_recent")
            min_quality: Minimum quality score

        Returns:
            PhraseAudioSegment if found, None otherwise
        """
        if phrase_key not in self.phrase_segments:
            return None

        candidates = [s for s in self.phrase_segments[phrase_key] if s.quality_score >= min_quality]

        if not candidates:
            return None

        if strategy == "best":
            # Best overall quality
            return max(candidates, key=lambda s: s.quality_score)
        elif strategy == "highest_snr":
            # Highest SNR
            return max(candidates, key=lambda s: s.snr_db)
        elif strategy == "most_recent":
            # Most recently added
            return candidates[-1]
        else:  # random
            return np.random.choice(candidates)

    def get_segments(
        self,
        phrase_key: str,
        count: int = 1,
        strategy: str = "random",
        min_quality: float = 0.5,
        allow_repeats: bool = False,
    ) -> List[PhraseAudioSegment]:
        """
        Get multiple phrase audio segments.

        Args:
            phrase_key: Phrase signature
            count: Number of segments to retrieve
            strategy: Selection strategy
            min_quality: Minimum quality score
            allow_repeats: Whether to allow returning the same segment multiple times

        Returns:
            List of PhraseAudioSegments
        """
        segments = []

        for _ in range(count):
            segment = self.get_segment(phrase_key, strategy, min_quality)
            if segment is None:
                break

            if not allow_repeats and segments:
                # Avoid duplicates
                if all(s.occurrence_id != segment.occurrence_id for s in segments):
                    segments.append(segment)
            else:
                segments.append(segment)

        return segments

    def get_available_phrases(self) -> List[str]:
        """Get list of all available phrase keys."""
        return list(self.phrase_segments.keys())

    def get_phrase_count(self, phrase_key: str) -> int:
        """Get number of segments for a phrase."""
        return len(self.phrase_segments.get(phrase_key, []))

    def get_library_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        phrase_counts = {k: len(v) for k, v in self.phrase_segments.items()}

        total_audio_duration = sum(
            sum(s.duration_seconds for s in segments) for segments in self.phrase_segments.values()
        )

        encoding_counts = {"horizontal": 0, "vertical": 0}
        for segments in self.phrase_segments.values():
            for segment in segments:
                encoding_counts[segment.encoding] += 1

        # Context statistics
        context_counts = defaultdict(int)
        individual_counts = defaultdict(int)
        for segments in self.phrase_segments.values():
            for segment in segments:
                if segment.context:
                    context_counts[segment.context] += 1
                if segment.individual_id:
                    individual_counts[segment.individual_id] += 1

        return {
            "species": self.species,
            "sr": self.sr,
            "total_phrases": len(self.phrase_segments),
            "total_segments": self.total_segments,
            "total_audio_duration_seconds": total_audio_duration,
            "phrase_counts": phrase_counts,
            "encoding_distribution": encoding_counts,
            "context_distribution": dict(context_counts),
            "individual_distribution": dict(individual_counts),
            "creation_time": self.creation_time.isoformat(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Alias for get_library_stats() for compatibility."""
        return self.get_library_stats()

    # ========================================================================
    # Context-Aware Methods
    # ========================================================================

    def get_contexts_for_phrase(self, phrase_key: str) -> List[str]:
        """
        Get all unique contexts associated with a phrase.

        Args:
            phrase_key: Phrase signature

        Returns:
            List of unique context labels
        """
        if phrase_key not in self.phrase_segments:
            return []

        contexts = set()
        for segment in self.phrase_segments[phrase_key]:
            if segment.context:
                contexts.add(segment.context)

        return sorted(list(contexts))

    def get_individuals_for_phrase(self, phrase_key: str) -> List[str]:
        """
        Get all unique individuals associated with a phrase.

        Args:
            phrase_key: Phrase signature

        Returns:
            List of unique individual IDs
        """
        if phrase_key not in self.phrase_segments:
            return []

        individuals = set()
        for segment in self.phrase_segments[phrase_key]:
            if segment.individual_id:
                individuals.add(segment.individual_id)

        return sorted(list(individuals))

    def get_context_statistics(self) -> Dict[str, Any]:
        """
        Get statistical associations between contexts and phrases.

        Returns:
            Dictionary with context-phrase association statistics
        """
        context_phrase_counts = defaultdict(lambda: defaultdict(int))
        context_totals = defaultdict(int)

        for phrase_key, segments in self.phrase_segments.items():
            for segment in segments:
                if segment.context:
                    context_phrase_counts[segment.context][phrase_key] += 1
                    context_totals[segment.context] += 1

        # Calculate probabilities and enrichments
        context_phrase_stats = {}
        phrase_totals = {k: len(v) for k, v in self.phrase_segments.items()}
        total_segments = self.total_segments

        for context, phrase_counts in context_phrase_counts.items():
            context_phrases = []
            for phrase_key, count in phrase_counts.items():
                # Probability of phrase given context
                p_phrase_given_context = count / context_totals[context]
                # Probability of phrase in general
                p_phrase = phrase_totals[phrase_key] / total_segments
                # Enrichment (how much more likely in this context)
                enrichment = p_phrase_given_context / p_phrase if p_phrase > 0 else 0

                context_phrases.append(
                    {
                        "phrase_key": phrase_key,
                        "count": count,
                        "probability": p_phrase_given_context,
                        "enrichment": enrichment,
                    }
                )

            # Sort by enrichment
            context_phrases.sort(key=lambda x: x["enrichment"], reverse=True)

            context_phrase_stats[context] = {
                "total_occurrences": context_totals[context],
                "phrases": context_phrases[:20],  # Top 20 enriched phrases
                "num_unique_phrases": len(phrase_counts),
            }

        return {
            "context_statistics": context_phrase_stats,
            "total_contexts": len(context_phrase_counts),
            "segments_with_context": sum(context_totals.values()),
            "segments_without_context": self.total_segments - sum(context_totals.values()),
        }

    def get_all_phrase_keys(self) -> List[str]:
        """
        Get all unique phrase keys in the library.

        Returns:
            List of all phrase keys
        """
        return list(self.phrase_segments.keys())

    def get_phrase_occurrence_statistics(self) -> Dict[str, int]:
        """
        Get statistics on phrase occurrences.

        Returns:
            Dictionary mapping phrase keys to occurrence counts
        """
        occurrence_stats = {}
        for phrase_key, segments in self.phrase_segments.items():
            occurrence_stats[phrase_key] = len(segments)
        return occurrence_stats

    def get_quality_distribution_statistics(self) -> Dict[str, float]:
        """
        Get quality distribution statistics across all segments.

        Returns:
            Dictionary with quality statistics
        """
        if not self.total_segments:
            return {
                "mean_quality": 0.0,
                "min_quality": 0.0,
                "max_quality": 0.0,
                "std_quality": 0.0,
                "total_segments": 0,
            }

        all_qualities = []
        for segments in self.phrase_segments.values():
            for segment in segments:
                if segment.quality_score is not None:
                    all_qualities.append(segment.quality_score)

        if not all_qualities:
            return {
                "mean_quality": 0.0,
                "min_quality": 0.0,
                "max_quality": 0.0,
                "std_quality": 0.0,
                "total_segments": self.total_segments,
            }

        qualities_array = np.array(all_qualities)
        return {
            "mean_quality": float(np.mean(qualities_array)),
            "min_quality": float(np.min(qualities_array)),
            "max_quality": float(np.max(qualities_array)),
            "std_quality": float(np.std(qualities_array)),
            "total_segments": len(all_qualities),
        }

    def select_phrases_by_context(
        self, context: str, min_quality: float = 0.5, max_results: Optional[int] = None
    ) -> List[PhraseAudioSegment]:
        """
        Select phrases by context with filtering options.

        Args:
            context: Context to filter by
            min_quality: Minimum quality score
            max_results: Maximum number of results to return

        Returns:
            List of PhraseAudioSegment objects matching the criteria
        """
        selected_segments = []

        for phrase_key, segments in self.phrase_segments.items():
            for segment in segments:
                if segment.context == context and segment.quality_score >= min_quality:
                    selected_segments.append(segment)

        # Sort by quality score (highest first)
        selected_segments.sort(key=lambda s: s.quality_score, reverse=True)

        # Limit results if specified
        if max_results is not None and len(selected_segments) > max_results:
            selected_segments = selected_segments[:max_results]

        return selected_segments

    def get_segment_by_context(
        self,
        phrase_key: str,
        context: Optional[str] = None,
        individual_id: Optional[str] = None,
        strategy: str = "random",
        min_quality: float = 0.5,
    ) -> Optional[PhraseAudioSegment]:
        """
        Get a phrase audio segment filtered by context and/or individual.

        Args:
            phrase_key: Phrase signature to retrieve
            context: Optional context filter (only return segments with this context)
            individual_id: Optional individual filter (only return segments from this individual)
            strategy: Selection strategy ("random", "best", "highest_snr")
            min_quality: Minimum quality score

        Returns:
            PhraseAudioSegment if found, None otherwise
        """
        if phrase_key not in self.phrase_segments:
            return None

        # Filter by context and individual
        candidates = [
            s
            for s in self.phrase_segments[phrase_key]
            if s.quality_score >= min_quality
            and (context is None or s.context == context)
            and (individual_id is None or s.individual_id == individual_id)
        ]

        if not candidates:
            return None

        if strategy == "best":
            return max(candidates, key=lambda s: s.quality_score)
        elif strategy == "highest_snr":
            return max(candidates, key=lambda s: s.snr_db)
        elif strategy == "most_recent":
            return candidates[-1]
        else:  # random
            return np.random.choice(candidates)

    def get_phrases_by_context(self, context: str, min_occurrences: int = 1) -> List[str]:
        """
        Get all phrase keys associated with a specific context.

        Args:
            context: Context label
            min_occurrences: Minimum number of occurrences required

        Returns:
            List of phrase keys
        """
        phrases = []
        for phrase_key, segments in self.phrase_segments.items():
            count = sum(1 for s in segments if s.context == context)
            if count >= min_occurrences:
                phrases.append(phrase_key)

        return sorted(phrases)

    def get_segments_for_synthesis_by_context(
        self,
        phrase_keys: List[str],
        context: Optional[str] = None,
        individual_id: Optional[str] = None,
        strategy: str = "random",
        min_quality: float = 0.5,
    ) -> List[PhraseAudioSegment]:
        """
        Get multiple segments for synthesis, filtered by context.

        Args:
            phrase_keys: List of phrase keys to retrieve
            context: Optional context filter
            individual_id: Optional individual filter
            strategy: Selection strategy
            min_quality: Minimum quality score

        Returns:
            List of PhraseAudioSegments (one per requested phrase key)
        """
        segments = []
        for phrase_key in phrase_keys:
            segment = self.get_segment_by_context(
                phrase_key=phrase_key,
                context=context,
                individual_id=individual_id,
                strategy=strategy,
                min_quality=min_quality,
            )
            if segment is not None:
                segments.append(segment)
            else:
                logger.warning(
                    f"Could not find segment for phrase {phrase_key} with context={context}"
                )

        return segments

    # ========================================================================
    # Microharmonic-Aware Methods
    # ========================================================================

    def get_segment_by_microharmonic(
        self,
        phrase_key: str,
        dominant_harmonic: Optional[int] = None,
        harmonic_entropy_range: Optional[Tuple[float, float]] = None,
        spectral_centroid_range: Optional[Tuple[float, float]] = None,
        harmonic_stability_min: float = 0.0,
        strategy: str = "random",
        min_quality: float = 0.5,
    ) -> Optional[PhraseAudioSegment]:
        """
        Get a phrase audio segment filtered by microharmonic signature.

        Args:
            phrase_key: Phrase signature to retrieve
            dominant_harmonic: Filter by dominant harmonic (1-based index)
            harmonic_entropy_range: Filter by (min, max) harmonic entropy
            spectral_centroid_range: Filter by (min, max) spectral centroid Hz
            harmonic_stability_min: Minimum harmonic stability (0-1)
            strategy: Selection strategy
            min_quality: Minimum quality score

        Returns:
            PhraseAudioSegment if found, None otherwise
        """
        if phrase_key not in self.phrase_segments:
            return None

        # Filter by microharmonic criteria
        candidates = []
        for segment in self.phrase_segments[phrase_key]:
            if segment.quality_score < min_quality:
                continue
            if not segment.has_microharmonics:
                continue

            # Filter by dominant harmonic
            if dominant_harmonic is not None:
                if segment.dominant_harmonic != dominant_harmonic:
                    continue

            # Filter by harmonic entropy
            if harmonic_entropy_range is not None:
                entropy = segment.harmonic_entropy
                if entropy is None or not (
                    harmonic_entropy_range[0] <= entropy <= harmonic_entropy_range[1]
                ):
                    continue

            # Filter by spectral centroid
            if spectral_centroid_range is not None:
                centroid = segment.spectral_centroid_hz
                if centroid is None or not (
                    spectral_centroid_range[0] <= centroid <= spectral_centroid_range[1]
                ):
                    continue

            # Filter by harmonic stability
            if harmonic_stability_min > 0:
                stability = segment.microharmonic_signature.get("harmonic_stability", 0)
                if stability < harmonic_stability_min:
                    continue

            candidates.append(segment)

        if not candidates:
            return None

        if strategy == "best":
            return max(candidates, key=lambda s: s.quality_score)
        elif strategy == "highest_snr":
            return max(candidates, key=lambda s: s.snr_db)
        elif strategy == "most_stable":
            return max(
                candidates, key=lambda s: s.microharmonic_signature.get("harmonic_stability", 0)
            )
        else:  # random
            return np.random.choice(candidates)

    def get_segments_for_synthesis_by_microharmonic(
        self,
        phrase_keys: List[str],
        dominant_harmonic: Optional[int] = None,
        harmonic_entropy_range: Optional[Tuple[float, float]] = None,
        spectral_centroid_range: Optional[Tuple[float, float]] = None,
        harmonic_stability_min: float = 0.0,
        strategy: str = "random",
        min_quality: float = 0.5,
    ) -> List[PhraseAudioSegment]:
        """
        Get multiple segments for synthesis, filtered by microharmonic signature.

        Args:
            phrase_keys: List of phrase keys to retrieve
            dominant_harmonic: Filter by dominant harmonic
            harmonic_entropy_range: Filter by harmonic entropy range
            spectral_centroid_range: Filter by spectral centroid range
            harmonic_stability_min: Minimum harmonic stability
            strategy: Selection strategy
            min_quality: Minimum quality score

        Returns:
            List of PhraseAudioSegments
        """
        segments = []
        for phrase_key in phrase_keys:
            segment = self.get_segment_by_microharmonic(
                phrase_key=phrase_key,
                dominant_harmonic=dominant_harmonic,
                harmonic_entropy_range=harmonic_entropy_range,
                spectral_centroid_range=spectral_centroid_range,
                harmonic_stability_min=harmonic_stability_min,
                strategy=strategy,
                min_quality=min_quality,
            )
            if segment is not None:
                segments.append(segment)
            else:
                logger.warning(
                    f"Could not find segment for phrase {phrase_key} with microharmonic filters"
                )

        return segments

    def get_microharmonic_statistics(self) -> Dict[str, Any]:
        """
        Get statistical distribution of microharmonic signatures.

        Returns:
            Dictionary with microharmonic statistics
        """
        dominant_harmonic_counts = defaultdict(int)
        harmonic_entropy_values = []
        spectral_centroid_values = []
        harmonic_stability_values = []

        segments_with_microharmonics = 0
        total_segments = 0

        for segments in self.phrase_segments.values():
            for segment in segments:
                total_segments += 1
                if segment.has_microharmonics:
                    segments_with_microharmonics += 1

                    # Dominant harmonic
                    if segment.dominant_harmonic is not None:
                        dominant_harmonic_counts[segment.dominant_harmonic] += 1

                    # Harmonic entropy
                    if segment.harmonic_entropy is not None:
                        harmonic_entropy_values.append(segment.harmonic_entropy)

                    # Spectral centroid
                    if segment.spectral_centroid_hz is not None:
                        spectral_centroid_values.append(segment.spectral_centroid_hz)

                    # Harmonic stability
                    stability = segment.microharmonic_signature.get("harmonic_stability")
                    if stability is not None:
                        harmonic_stability_values.append(stability)

        stats = {
            "total_segments": total_segments,
            "segments_with_microharmonics": segments_with_microharmonics,
            "coverage_percent": (segments_with_microharmonics / total_segments * 100)
            if total_segments > 0
            else 0,
            "dominant_harmonic_distribution": dict(dominant_harmonic_counts),
        }

        if harmonic_entropy_values:
            stats["harmonic_entropy"] = {
                "mean": float(np.mean(harmonic_entropy_values)),
                "std": float(np.std(harmonic_entropy_values)),
                "min": float(np.min(harmonic_entropy_values)),
                "max": float(np.max(harmonic_entropy_values)),
                "median": float(np.median(harmonic_entropy_values)),
            }

        if spectral_centroid_values:
            stats["spectral_centroid_hz"] = {
                "mean": float(np.mean(spectral_centroid_values)),
                "std": float(np.std(spectral_centroid_values)),
                "min": float(np.min(spectral_centroid_values)),
                "max": float(np.max(spectral_centroid_values)),
                "median": float(np.median(spectral_centroid_values)),
            }

        if harmonic_stability_values:
            stats["harmonic_stability"] = {
                "mean": float(np.mean(harmonic_stability_values)),
                "std": float(np.std(harmonic_stability_values)),
                "min": float(np.min(harmonic_stability_values)),
                "max": float(np.max(harmonic_stability_values)),
                "median": float(np.median(harmonic_stability_values)),
            }

        return stats

    def get_phrases_by_microharmonic_profile(
        self,
        dominant_harmonic: Optional[int] = None,
        min_harmonic_entropy: float = 0.0,
        max_harmonic_entropy: float = 1.0,
        min_harmonic_stability: float = 0.0,
    ) -> List[str]:
        """
        Get all phrase keys matching a microharmonic profile.

        Args:
            dominant_harmonic: Required dominant harmonic (None = any)
            min_harmonic_entropy: Minimum harmonic entropy
            max_harmonic_entropy: Maximum harmonic entropy
            min_harmonic_stability: Minimum harmonic stability

        Returns:
            List of phrase keys
        """
        matching_phrases = []

        for phrase_key, segments in self.phrase_segments.items():
            # Check if any segment matches the profile
            for segment in segments:
                if not segment.has_microharmonics:
                    continue

                # Check dominant harmonic
                if dominant_harmonic is not None:
                    if segment.dominant_harmonic != dominant_harmonic:
                        continue

                # Check harmonic entropy
                entropy = segment.harmonic_entropy
                if entropy is None or not (min_harmonic_entropy <= entropy <= max_harmonic_entropy):
                    continue

                # Check harmonic stability
                stability = segment.microharmonic_signature.get("harmonic_stability", 0)
                if stability < min_harmonic_stability:
                    continue

                # If we get here, this phrase has at least one matching segment
                matching_phrases.append(phrase_key)
                break

        return sorted(matching_phrases)

    def calculate_microharmonic_similarity(
        self, phrase_key1: str, phrase_key2: str
    ) -> Optional[float]:
        """
        Calculate microharmonic similarity between two phrases.

        Uses cosine similarity of harmonic ratio vectors.

        Args:
            phrase_key1: First phrase key
            phrase_key2: Second phrase key

        Returns:
            Similarity score (0-1) or None if either phrase lacks microharmonics
        """
        if phrase_key1 not in self.phrase_segments or phrase_key2 not in self.phrase_segments:
            return None

        # Get representative segments (best quality)
        segment1 = self.get_segment(phrase_key1, strategy="best")
        segment2 = self.get_segment(phrase_key2, strategy="best")

        if segment1 is None or segment2 is None:
            return None

        if not segment1.has_microharmonics or not segment2.has_microharmonics:
            return None

        ratios1 = segment1.get_harmonic_ratios()
        ratios2 = segment2.get_harmonic_ratios()

        if ratios1 is None or ratios2 is None:
            return None

        # Ensure same length
        min_len = min(len(ratios1), len(ratios2))
        if min_len == 0:
            return None

        ratios1 = ratios1[:min_len]
        ratios2 = ratios2[:min_len]

        # Cosine similarity
        dot_product = np.dot(ratios1, ratios2)
        norm1 = np.linalg.norm(ratios1)
        norm2 = np.linalg.norm(ratios2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def get_similar_phrases_by_microharmonics(
        self, phrase_key: str, top_k: int = 10, min_similarity: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Find phrases with similar microharmonic signatures.

        Args:
            phrase_key: Reference phrase
            top_k: Maximum number of similar phrases to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (phrase_key, similarity) tuples, sorted by similarity
        """
        similar_phrases = []

        for other_phrase in self.get_available_phrases():
            if other_phrase == phrase_key:
                continue

            similarity = self.calculate_microharmonic_similarity(phrase_key, other_phrase)
            if similarity is not None and similarity >= min_similarity:
                similar_phrases.append((other_phrase, similarity))

        # Sort by similarity (descending)
        similar_phrases.sort(key=lambda x: x[1], reverse=True)

        return similar_phrases[:top_k]

    def save(self, filepath: Optional[Union[str, Path]] = None):
        """
        Save library to disk.

        Args:
            filepath: Path to save to (uses library_dir if None)
        """
        if filepath is None:
            if self.library_dir is None:
                raise ValueError("No filepath or library_dir specified")
            filepath = self.library_dir / f"{self.species}_phrase_library.pkl"

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save library state
        library_data = {
            "species": self.species,
            "sr": self.sr,
            "phrase_segments": {
                k: [s.to_dict() for s in v] for k, v in self.phrase_segments.items()
            },
            "total_segments": self.total_segments,
            "creation_time": self.creation_time.isoformat(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(library_data, f)

        logger.info(f"Library metadata saved to {filepath}")

    def load(self, filepath: Optional[Union[str, Path]] = None):
        """
        Load library metadata from disk.

        Note: Audio segments are not loaded to save memory.
        Use get_segment() to load specific segments on demand.

        Args:
            filepath: Path to load from (uses library_dir if None)
        """
        if filepath is None:
            if self.library_dir is None:
                raise ValueError("No filepath or library_dir specified")
            filepath = self.library_dir / f"{self.species}_phrase_library.pkl"

        with open(filepath, "rb") as f:
            library_data = pickle.load(f)

        # Load library state
        self.species = library_data["species"]
        self.sr = library_data["sr"]
        self.total_segments = library_data["total_segments"]
        self.creation_time = datetime.fromisoformat(library_data["creation_time"])

        # Reconstruct phrase_segments from loaded metadata
        self.phrase_segments = defaultdict(list)
        for phrase_key, segment_dicts in library_data["phrase_segments"].items():
            for segment_dict in segment_dicts:
                # Create segment without audio data for memory efficiency
                segment = PhraseAudioSegment(
                    audio=np.array([]),  # Empty audio - load on demand
                    sr=self.sr,
                    phrase_key=segment_dict["phrase_key"],
                    source_file=segment_dict["source_file"],
                    start_time_ms=segment_dict["start_time_ms"],
                    end_time_ms=segment_dict["end_time_ms"],
                    mean_f0_hz=segment_dict["mean_f0_hz"],
                    std_f0_hz=segment_dict["std_f0_hz"],
                    mean_duration_ms=segment_dict["mean_duration_ms"],
                    mean_range_hz=segment_dict["mean_range_hz"],
                    encoding=segment_dict["encoding"],
                    superposed_with=segment_dict["superposed_with"],
                    context=segment_dict["context"],
                    individual_id=segment_dict["individual_id"],
                    snr_db=segment_dict["snr_db"],
                    quality_score=segment_dict["quality_score"],
                    microharmonic_signature=segment_dict.get("microharmonic_signature"),
                )
                self.phrase_segments[phrase_key].append(segment)

        logger.info(f"Library metadata loaded from {filepath}")
        logger.info(f"  Species: {self.species}")
        logger.info(f"  Total segments: {self.total_segments}")

    def _calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        # Simple SNR estimate based on energy distribution
        energy = np.sum(audio**2)

        # Use low-energy frames as noise estimate
        frame_size = len(audio) // 10
        frame_energies = [
            np.sum(audio[i : i + frame_size] ** 2)
            for i in range(0, len(audio) - frame_size, frame_size)
        ]

        if frame_energies:
            noise_energy = np.min(frame_energies)
            if noise_energy > 0:
                snr_linear = energy / (len(audio) * noise_energy)
                return 10 * np.log10(max(snr_linear, 1e-10))

        return 0.0

    def _calculate_quality_score(self, audio: np.ndarray, snr_db: float) -> float:
        """Calculate overall quality score (0-1)."""
        # SNR component
        snr_score = min(1.0, max(0.0, (snr_db + 10) / 40))  # -10dB to 30dB range

        # Dynamic range component
        peak = np.max(np.abs(audio))
        if peak > 0:
            rms = np.sqrt(np.mean(audio**2))
            crest_factor = peak / rms if rms > 0 else 1
            dynamic_score = min(1.0, crest_factor / 10)  # Good crest factor around 3-10
        else:
            dynamic_score = 0.0

        # Combined score
        quality = 0.7 * snr_score + 0.3 * dynamic_score

        return float(quality)


# ============================================================================
# Vocalization Synthesizer
# ============================================================================


class VocalizationSynthesizer:
    """
    Multi-mode vocalization synthesizer using phrase audio library.

    Supports three synthesis modes:
    1. Horizontal (Concatenative): Sequential phrase combination
    2. Vertical (Superpositional): Simultaneous phrase overlay
    3. Combined: Mixed horizontal and vertical synthesis

    This enables generation of novel vocalizations that respect the
    syntactic and semantic structure discovered in the species'
    communication system.
    """

    def __init__(
        self,
        library: PhraseAudioLibrary,
        crossfade_ms: float = 5.0,
        vertical_mix_mode: str = "average",
    ):
        """
        Initialize the synthesizer.

        Args:
            library: PhraseAudioLibrary to use for synthesis
            crossfade_ms: Crossfade duration for concatenative synthesis (ms)
            vertical_mix_mode: How to mix superposed phrases:
                - "average": Average the signals
                - "add": Add the signals (may clip)
                - "normalized_add": Add and normalize to prevent clipping
        """
        self.library = library
        self.crossfade_ms = crossfade_ms
        self.vertical_mix_mode = vertical_mix_mode

        self.synthesis_count = 0

    def synthesize_horizontal(
        self,
        phrase_sequence: List[str],
        gap_ms: float = 0.0,
        variation_strategy: str = "random",
        min_quality: float = 0.5,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Synthesize vocalization using horizontal (concatenative) encoding.

        This creates a sequential vocalization by concatenating phrase
        audio segments in order, with optional gaps and crossfades.

        Args:
            phrase_sequence: List of phrase keys to synthesize in order
            gap_ms: Gap between phrases in milliseconds
            variation_strategy: Strategy for selecting phrase variants
            min_quality: Minimum quality score for segments

        Returns:
            Tuple of (audio, sr) if successful, None otherwise
        """
        if not phrase_sequence:
            return None

        segments = []
        total_duration = 0.0

        for phrase_key in phrase_sequence:
            segment = self.library.get_segment(
                phrase_key, strategy=variation_strategy, min_quality=min_quality
            )

            if segment is None:
                logger.warning(f"Could not find segment for phrase: {phrase_key}")
                continue

            segments.append(segment)
            total_duration += segment.duration_seconds

        if not segments:
            return None

        # Calculate gap in samples
        gap_samples = int(gap_ms / 1000 * self.library.sr)

        # Calculate total output duration
        total_samples = sum(s.duration_samples for s in segments)
        total_samples += gap_samples * (len(segments) - 1)

        # Initialize output
        output = np.zeros(total_samples)

        # Concatenate segments
        current_sample = 0
        crossfade_samples = int(self.crossfade_ms / 1000 * self.library.sr)

        for i, segment in enumerate(segments):
            segment_audio = segment.audio.copy()
            segment_length = len(segment_audio)

            # Add gap (except for first segment)
            if i > 0:
                current_sample += gap_samples

            # Handle crossfade with previous segment
            if crossfade_samples > 0 and i > 0 and current_sample >= crossfade_samples:
                # Crossfade region
                crossfade_end = current_sample
                crossfade_start = crossfade_end - crossfade_samples

                # Create crossfade windows
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)

                # Apply crossfade
                output[crossfade_start:crossfade_end] *= fade_out
                segment_start_idx = 0
                segment_end_idx = min(crossfade_samples, segment_length)

                output[crossfade_start:crossfade_end] += (
                    segment_audio[segment_start_idx:segment_end_idx] * fade_in[:segment_end_idx]
                )

                # Place remaining segment
                remaining_start = crossfade_end
                remaining_end = remaining_start + (segment_length - segment_end_idx)
                if remaining_end <= len(output):
                    output[remaining_start:remaining_end] = segment_audio[segment_end_idx:]
            else:
                # Simple placement
                end_sample = current_sample + segment_length
                if end_sample <= len(output):
                    output[current_sample:end_sample] = segment_audio

            current_sample += segment_length

        # Normalize to prevent clipping
        if np.max(np.abs(output)) > 0:
            output = output / np.max(np.abs(output)) * 0.95

        self.synthesis_count += 1

        return output, self.library.sr

    def synthesize_vertical(
        self,
        phrase_set: List[str],
        alignment: str = "start",
        variation_strategy: str = "random",
        min_quality: float = 0.5,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Synthesize vocalization using vertical (superpositional) encoding.

        This creates simultaneous vocalization by overlaying multiple
        phrase audio segments. Useful for simulating multi-individual
        vocalizations or testing encoding detection.

        Args:
            phrase_set: List of phrase keys to superpose
            alignment: How to align phrases ("start", "center", "end", "random")
            variation_strategy: Strategy for selecting phrase variants
            min_quality: Minimum quality score for segments

        Returns:
            Tuple of (audio, sr) if successful, None otherwise
        """
        if not phrase_set:
            return None

        segments = []
        max_duration = 0

        for phrase_key in phrase_set:
            segment = self.library.get_segment(
                phrase_key, strategy=variation_strategy, min_quality=min_quality
            )

            if segment is None:
                logger.warning(f"Could not find segment for phrase: {phrase_key}")
                continue

            segments.append(segment)
            max_duration = max(max_duration, segment.duration_seconds)

        if not segments:
            return None

        # Calculate output size
        max_samples = int(max_duration * self.library.sr) + 1000  # Add buffer

        # Initialize output
        output = np.zeros(max_samples)

        # Mix segments
        for segment in segments:
            segment_audio = segment.audio
            segment_length = len(segment_audio)

            # Determine start position
            if alignment == "start":
                start_pos = 0
            elif alignment == "center":
                start_pos = (max_samples - segment_length) // 2
            elif alignment == "end":
                start_pos = max_samples - segment_length
            elif alignment == "random":
                start_pos = np.random.randint(0, max(1, max_samples - segment_length))
            else:
                start_pos = 0

            end_pos = start_pos + segment_length

            # Ensure bounds
            if end_pos > max_samples:
                end_pos = max_samples
                segment_audio = segment_audio[: end_pos - start_pos]

            if start_pos >= 0 and end_pos > start_pos:
                output[start_pos:end_pos] += segment_audio

        # Apply mixing mode
        if self.vertical_mix_mode == "normalized_add":
            if np.max(np.abs(output)) > 0:
                output = output / np.max(np.abs(output)) * 0.95

        # Trim trailing silence
        trim_threshold = 0.01 * np.max(np.abs(output))
        above_threshold = np.abs(output) > trim_threshold
        if np.any(above_threshold):
            last_valid = np.where(above_threshold)[0][-1]
            output = output[: last_valid + 1]

        self.synthesis_count += 1

        return output, self.library.sr

    def synthesize_combined(
        self,
        synthesis_plan: List[Tuple[str, List[str]]],
        gap_ms: float = 0.0,
        variation_strategy: str = "random",
        min_quality: float = 0.5,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Synthesize vocalization using combined horizontal and vertical encoding.

        This creates complex vocalizations that mix sequential phrases
        with simultaneous superposition at each step.

        Args:
            synthesis_plan: List of (mode, phrases) tuples where:
                - mode is "horizontal" or "vertical"
                - phrases is a list of phrase keys
            gap_ms: Gap between horizontal steps in milliseconds
            variation_strategy: Strategy for selecting phrase variants
            min_quality: Minimum quality score for segments

        Returns:
            Tuple of (audio, sr) if successful, None otherwise

        Example:
            synthesizer.synthesize_combined([
                ('horizontal', ['F0_6400_DUR_5_RANGE_0', 'F0_6600_DUR_10_RANGE_0']),
                ('vertical', ['F0_6400_DUR_5_RANGE_0', 'F0_6800_DUR_5_RANGE_0']),
                ('horizontal', ['F0_7000_DUR_15_RANGE_0'])
            ])
        """
        if not synthesis_plan:
            return None

        output_segments = []

        for mode, phrases in synthesis_plan:
            if mode == "horizontal":
                # Sequential synthesis
                result = self.synthesize_horizontal(
                    phrases,
                    gap_ms=gap_ms,
                    variation_strategy=variation_strategy,
                    min_quality=min_quality,
                )
            elif mode == "vertical":
                # Simultaneous synthesis
                result = self.synthesize_vertical(
                    phrases,
                    alignment="start",
                    variation_strategy=variation_strategy,
                    min_quality=min_quality,
                )
            else:
                logger.warning(f"Unknown synthesis mode: {mode}")
                continue

            if result is not None:
                output_segments.append(result)

        if not output_segments:
            return None

        # Concatenate all synthesis results
        final_sr = output_segments[0][1]
        combined = np.concatenate([s[0] for s in output_segments])

        # Normalize
        if np.max(np.abs(combined)) > 0:
            combined = combined / np.max(np.abs(combined)) * 0.95

        self.synthesis_count += 1

        return combined, final_sr

    def generate_random_vocalization(
        self,
        num_phrases: int = 5,
        encoding_ratio: float = 0.7,  # 70% horizontal, 30% vertical
        min_quality: float = 0.5,
    ) -> Optional[Tuple[np.ndarray, int, List]]:
        """
        Generate a random vocalization using available phrases.

        Args:
            num_phrases: Approximate number of phrases to use
            encoding_ratio: Ratio of horizontal to vertical encoding (0-1)
            min_quality: Minimum quality score

        Returns:
            Tuple of (audio, sr, synthesis_plan) if successful, None otherwise
        """
        available_phrases = self.library.get_available_phrases()

        if not available_phrases:
            return None

        synthesis_plan = []
        phrases_used = 0

        while phrases_used < num_phrases:
            # Decide encoding type
            use_horizontal = np.random.random() < encoding_ratio

            if use_horizontal:
                # Horizontal: 1-3 phrases in sequence
                n_phrases = min(np.random.randint(1, 4), num_phrases - phrases_used)
                selected = np.random.choice(available_phrases, n_phrases).tolist()
                synthesis_plan.append(("horizontal", selected))
                phrases_used += n_phrases
            else:
                # Vertical: 2-4 phrases superposed
                n_phrases = min(np.random.randint(2, 5), num_phrases - phrases_used)
                selected = np.random.choice(available_phrases, n_phrases).tolist()
                synthesis_plan.append(("vertical", selected))
                phrases_used += n_phrases

            if phrases_used >= num_phrases:
                break

            # Small chance to stop early
            if np.random.random() < 0.1:
                break

        result = self.synthesize_combined(
            synthesis_plan, variation_strategy="random", min_quality=min_quality
        )

        if result is not None:
            return result[0], result[1], synthesis_plan

        return None

    # ========================================================================
    # Context-Aware Synthesis Methods
    # ========================================================================

    def synthesize_horizontal_with_context(
        self,
        phrase_sequence: List[str],
        context: Optional[str] = None,
        individual_id: Optional[str] = None,
        gap_ms: float = 0.0,
        strategy: str = "random",
        min_quality: float = 0.5,
    ) -> Optional[Tuple[np.ndarray, int, Dict]]:
        """
        Synthesize vocalization using horizontal encoding with context filtering.

        This creates a sequential vocalization using phrase segments that match
        the specified behavioral context and/or individual.

        Args:
            phrase_sequence: List of phrase keys to synthesize in order
            context: Optional behavioral context filter
            individual_id: Optional individual ID filter
            gap_ms: Gap between phrases in milliseconds
            strategy: Selection strategy ("random", "best", "highest_snr")
            min_quality: Minimum quality score for segments

        Returns:
            Tuple of (audio, sr, metadata) if successful, None otherwise
        """
        if not phrase_sequence:
            return None

        segments = self.library.get_segments_for_synthesis_by_context(
            phrase_keys=phrase_sequence,
            context=context,
            individual_id=individual_id,
            strategy=strategy,
            min_quality=min_quality,
        )

        if not segments:
            logger.warning(f"No segments found for context={context}, individual={individual_id}")
            return None

        # Calculate gap in samples
        gap_samples = int(gap_ms / 1000 * self.library.sr)

        # Calculate total output duration
        total_samples = sum(s.duration_samples for s in segments)
        total_samples += gap_samples * (len(segments) - 1)

        # Initialize output
        output = np.zeros(total_samples)

        # Concatenate segments
        current_sample = 0
        crossfade_samples = int(self.crossfade_ms / 1000 * self.library.sr)

        for i, segment in enumerate(segments):
            segment_audio = segment.audio.copy()
            segment_length = len(segment_audio)

            # Add gap (except for first segment)
            if i > 0:
                current_sample += gap_samples

            # Handle crossfade with previous segment
            if crossfade_samples > 0 and i > 0 and current_sample >= crossfade_samples:
                crossfade_end = current_sample
                crossfade_start = crossfade_end - crossfade_samples

                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)

                output[crossfade_start:crossfade_end] *= fade_out
                segment_start_idx = 0
                segment_end_idx = min(crossfade_samples, segment_length)

                output[crossfade_start:crossfade_end] += (
                    segment_audio[segment_start_idx:segment_end_idx] * fade_in[:segment_end_idx]
                )

                remaining_start = crossfade_end
                remaining_end = remaining_start + (segment_length - segment_end_idx)
                if remaining_end <= len(output):
                    output[remaining_start:remaining_end] = segment_audio[segment_end_idx:]
            else:
                output[current_sample : current_sample + segment_length] = segment_audio

            current_sample += segment_length

        self.synthesis_count += 1

        metadata = {
            "synthesis_type": "horizontal_with_context",
            "context": context,
            "individual_id": individual_id,
            "phrase_sequence": phrase_sequence,
            "num_segments": len(segments),
        }

        return output, self.library.sr, metadata

    def synthesize_vertical_with_context(
        self,
        phrase_set: List[str],
        context: Optional[str] = None,
        individual_id: Optional[str] = None,
        alignment: str = "start",
        strategy: str = "random",
        min_quality: float = 0.5,
    ) -> Optional[Tuple[np.ndarray, int, Dict]]:
        """
        Synthesize vocalization using vertical encoding with context filtering.

        This creates a superposed vocalization using phrase segments that match
        the specified behavioral context and/or individual.

        Args:
            phrase_set: List of phrase keys to overlay
            context: Optional behavioral context filter
            individual_id: Optional individual ID filter
            alignment: How to align phrases ('start', 'center', 'end', 'random')
            strategy: Selection strategy ("random", "best", "highest_snr")
            min_quality: Minimum quality score for segments

        Returns:
            Tuple of (audio, sr, metadata) if successful, None otherwise
        """
        if not phrase_set:
            return None

        segments = self.library.get_segments_for_synthesis_by_context(
            phrase_keys=phrase_set,
            context=context,
            individual_id=individual_id,
            strategy=strategy,
            min_quality=min_quality,
        )

        if not segments:
            logger.warning(f"No segments found for context={context}, individual={individual_id}")
            return None

        # Calculate output size
        max_duration = max(s.duration_seconds for s in segments)
        max_samples = int(max_duration * self.library.sr) + 1000

        # Initialize output
        output = np.zeros(max_samples)

        # Mix segments
        for segment in segments:
            segment_audio = segment.audio
            segment_length = len(segment_audio)

            # Determine start position
            if alignment == "start":
                start_pos = 0
            elif alignment == "center":
                start_pos = (max_samples - segment_length) // 2
            elif alignment == "end":
                start_pos = max_samples - segment_length
            elif alignment == "random":
                start_pos = np.random.randint(0, max(1, max_samples - segment_length))
            else:
                start_pos = 0

            end_pos = start_pos + segment_length

            # Ensure bounds
            if end_pos > max_samples:
                end_pos = max_samples
                segment_audio = segment_audio[: end_pos - start_pos]

            if start_pos >= 0 and end_pos > start_pos:
                output[start_pos:end_pos] += segment_audio

        # Apply mixing mode
        if self.vertical_mix_mode == "normalized_add":
            if np.max(np.abs(output)) > 0:
                output = output / np.max(np.abs(output)) * 0.95

        # Trim trailing silence
        trim_threshold = 0.01 * np.max(np.abs(output))
        above_threshold = np.abs(output) > trim_threshold
        if np.any(above_threshold):
            last_valid = np.where(above_threshold)[0][-1]
            output = output[: last_valid + 1]

        self.synthesis_count += 1

        metadata = {
            "synthesis_type": "vertical_with_context",
            "context": context,
            "individual_id": individual_id,
            "phrase_set": phrase_set,
            "num_segments": len(segments),
            "alignment": alignment,
        }

        return output, self.library.sr, metadata

    def synthesize_combined_with_context(
        self,
        synthesis_plan: List[Tuple[str, List[str]]],
        context: Optional[str] = None,
        individual_id: Optional[str] = None,
        gap_ms: float = 0.0,
        strategy: str = "random",
        min_quality: float = 0.5,
    ) -> Optional[Tuple[np.ndarray, int, Dict]]:
        """
        Synthesize vocalization using combined encoding with context filtering.

        This creates complex vocalizations using context-filtered segments
        in mixed sequential and simultaneous patterns.

        Args:
            synthesis_plan: List of (mode, phrases) tuples
            context: Optional behavioral context filter
            individual_id: Optional individual ID filter
            gap_ms: Gap between horizontal steps in milliseconds
            strategy: Selection strategy
            min_quality: Minimum quality score for segments

        Returns:
            Tuple of (audio, sr, metadata) if successful, None otherwise
        """
        if not synthesis_plan:
            return None

        output_segments = []
        total_duration = 0.0

        for mode, phrases in synthesis_plan:
            if mode == "horizontal":
                result = self.synthesize_horizontal_with_context(
                    phrases,
                    context=context,
                    individual_id=individual_id,
                    gap_ms=gap_ms,
                    strategy=strategy,
                    min_quality=min_quality,
                )
            elif mode == "vertical":
                result = self.synthesize_vertical_with_context(
                    phrases,
                    context=context,
                    individual_id=individual_id,
                    strategy=strategy,
                    min_quality=min_quality,
                )
            else:
                logger.warning(f"Unknown synthesis mode: {mode}")
                continue

            if result is not None:
                audio, sr, metadata = result
                output_segments.append((audio, metadata))
                total_duration += len(audio) / sr

        if not output_segments:
            return None

        # Concatenate all segments
        total_samples = int(total_duration * self.library.sr) + 1000
        output = np.zeros(total_samples)
        current_sample = 0

        for audio, metadata in output_segments:
            segment_length = len(audio)
            output[current_sample : current_sample + segment_length] = audio
            current_sample += segment_length

        # Trim trailing silence
        trim_threshold = 0.01 * np.max(np.abs(output))
        above_threshold = np.abs(output) > trim_threshold
        if np.any(above_threshold):
            last_valid = np.where(above_threshold)[0][-1]
            output = output[: last_valid + 1]

        self.synthesis_count += 1

        metadata = {
            "synthesis_type": "combined_with_context",
            "context": context,
            "individual_id": individual_id,
            "synthesis_plan": synthesis_plan,
            "num_steps": len(output_segments),
        }

        return output, self.library.sr, metadata

    def synthesize_random_with_context(
        self,
        num_phrases: int = 5,
        context: Optional[str] = None,
        individual_id: Optional[str] = None,
        encoding_ratio: float = 0.6,
        min_quality: float = 0.5,
    ) -> Optional[Tuple[np.ndarray, int, Dict]]:
        """
        Generate a random context-constrained vocalization.

        Creates a random vocalization using only phrases associated with
        the specified behavioral context.

        Args:
            num_phrases: Approximate number of phrases to use
            context: Behavioral context to constrain selection
            individual_id: Optional individual ID to constrain selection
            encoding_ratio: Ratio of horizontal to vertical encoding (0-1)
            min_quality: Minimum quality score

        Returns:
            Tuple of (audio, sr, synthesis_plan) if successful, None otherwise
        """
        # Get phrases available for this context
        if context:
            available_phrases = self.library.get_phrases_by_context(context, min_occurrences=1)
        else:
            available_phrases = self.library.get_available_phrases()

        if not available_phrases:
            logger.warning(f"No phrases found for context={context}")
            return None

        synthesis_plan = []
        phrases_used = 0

        while phrases_used < num_phrases:
            # Decide encoding type
            use_horizontal = np.random.random() < encoding_ratio

            if use_horizontal:
                # Horizontal: 1-3 phrases in sequence
                n_phrases = min(np.random.randint(1, 4), num_phrases - phrases_used)
                selected = np.random.choice(
                    available_phrases, min(n_phrases, len(available_phrases))
                ).tolist()
                if selected:
                    synthesis_plan.append(("horizontal", selected))
                    phrases_used += n_phrases
            else:
                # Vertical: 2-4 phrases superposed
                n_phrases = min(np.random.randint(2, 5), num_phrases - phrases_used)
                selected = np.random.choice(
                    available_phrases, min(n_phrases, len(available_phrases))
                ).tolist()
                if selected:
                    synthesis_plan.append(("vertical", selected))
                    phrases_used += n_phrases

            if phrases_used >= num_phrases:
                break

            # Small chance to stop early
            if np.random.random() < 0.1:
                break

        result = self.synthesize_combined_with_context(
            synthesis_plan, context=context, individual_id=individual_id, min_quality=min_quality
        )

        if result is not None:
            audio, sr, metadata = result
            metadata["synthesis_plan"] = synthesis_plan
            return audio, sr, metadata

        return None

    # ========================================================================
    # Microharmonic-Aware Synthesis Methods
    # ========================================================================

    def synthesize_horizontal_with_microharmonics(
        self,
        phrase_sequence: List[str],
        dominant_harmonic: Optional[int] = None,
        harmonic_entropy_range: Optional[Tuple[float, float]] = None,
        spectral_centroid_range: Optional[Tuple[float, float]] = None,
        harmonic_stability_min: float = 0.0,
        gap_ms: float = 0.0,
        strategy: str = "random",
        min_quality: float = 0.5,
    ) -> Optional[Tuple[np.ndarray, int, Dict]]:
        """
        Synthesize vocalization using horizontal encoding with microharmonic filtering.

        This creates a sequential vocalization using phrase segments that match
        the specified microharmonic signature.

        Args:
            phrase_sequence: List of phrase keys to synthesize in order
            dominant_harmonic: Filter by dominant harmonic (1-based index)
            harmonic_entropy_range: Filter by (min, max) harmonic entropy
            spectral_centroid_range: Filter by spectral centroid range
            harmonic_stability_min: Minimum harmonic stability
            gap_ms: Gap between phrases in milliseconds
            strategy: Selection strategy ("random", "best", "highest_snr", "most_stable")
            min_quality: Minimum quality score for segments

        Returns:
            Tuple of (audio, sr, metadata) if successful, None otherwise
        """
        if not phrase_sequence:
            return None

        segments = self.library.get_segments_for_synthesis_by_microharmonic(
            phrase_keys=phrase_sequence,
            dominant_harmonic=dominant_harmonic,
            harmonic_entropy_range=harmonic_entropy_range,
            spectral_centroid_range=spectral_centroid_range,
            harmonic_stability_min=harmonic_stability_min,
            strategy=strategy,
            min_quality=min_quality,
        )

        if not segments:
            logger.warning("No segments found with microharmonic filters")
            return None

        # Calculate gap in samples
        gap_samples = int(gap_ms / 1000 * self.library.sr)

        # Calculate total output duration
        total_samples = sum(s.duration_samples for s in segments)
        total_samples += gap_samples * (len(segments) - 1)

        # Initialize output
        output = np.zeros(total_samples)

        # Concatenate segments
        current_sample = 0
        crossfade_samples = int(self.crossfade_ms / 1000 * self.library.sr)

        for i, segment in enumerate(segments):
            segment_audio = segment.audio.copy()
            segment_length = len(segment_audio)

            # Add gap (except for first segment)
            if i > 0:
                current_sample += gap_samples

            # Handle crossfade with previous segment
            if crossfade_samples > 0 and i > 0 and current_sample >= crossfade_samples:
                crossfade_end = current_sample
                crossfade_start = crossfade_end - crossfade_samples

                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)

                output[crossfade_start:crossfade_end] *= fade_out
                segment_start_idx = 0
                segment_end_idx = min(crossfade_samples, segment_length)

                output[crossfade_start:crossfade_end] += (
                    segment_audio[segment_start_idx:segment_end_idx] * fade_in[:segment_end_idx]
                )

                remaining_start = crossfade_end
                remaining_end = remaining_start + (segment_length - segment_end_idx)
                if remaining_end <= len(output):
                    output[remaining_start:remaining_end] = segment_audio[segment_end_idx:]
            else:
                output[current_sample : current_sample + segment_length] = segment_audio

            current_sample += segment_length

        self.synthesis_count += 1

        metadata = {
            "synthesis_type": "horizontal_with_microharmonics",
            "phrase_sequence": phrase_sequence,
            "num_segments": len(segments),
            "microharmonic_filters": {
                "dominant_harmonic": dominant_harmonic,
                "harmonic_entropy_range": harmonic_entropy_range,
                "spectral_centroid_range": spectral_centroid_range,
                "harmonic_stability_min": harmonic_stability_min,
            },
        }

        return output, self.library.sr, metadata

    def synthesize_vertical_with_microharmonics(
        self,
        phrase_set: List[str],
        dominant_harmonic: Optional[int] = None,
        harmonic_entropy_range: Optional[Tuple[float, float]] = None,
        spectral_centroid_range: Optional[Tuple[float, float]] = None,
        harmonic_stability_min: float = 0.0,
        alignment: str = "start",
        strategy: str = "random",
        min_quality: float = 0.5,
    ) -> Optional[Tuple[np.ndarray, int, Dict]]:
        """
        Synthesize vocalization using vertical encoding with microharmonic filtering.

        This creates a superposed vocalization using phrase segments that match
        the specified microharmonic signature, useful for harmonic compatibility.

        Args:
            phrase_set: List of phrase keys to overlay
            dominant_harmonic: Filter by dominant harmonic
            harmonic_entropy_range: Filter by harmonic entropy range
            spectral_centroid_range: Filter by spectral centroid range
            harmonic_stability_min: Minimum harmonic stability
            alignment: How to align phrases ('start', 'center', 'end', 'random')
            strategy: Selection strategy
            min_quality: Minimum quality score for segments

        Returns:
            Tuple of (audio, sr, metadata) if successful, None otherwise
        """
        if not phrase_set:
            return None

        segments = self.library.get_segments_for_synthesis_by_microharmonic(
            phrase_keys=phrase_set,
            dominant_harmonic=dominant_harmonic,
            harmonic_entropy_range=harmonic_entropy_range,
            spectral_centroid_range=spectral_centroid_range,
            harmonic_stability_min=harmonic_stability_min,
            strategy=strategy,
            min_quality=min_quality,
        )

        if not segments:
            logger.warning("No segments found with microharmonic filters")
            return None

        # Calculate output size
        max_duration = max(s.duration_seconds for s in segments)
        max_samples = int(max_duration * self.library.sr) + 1000

        # Initialize output
        output = np.zeros(max_samples)

        # Mix segments
        for segment in segments:
            segment_audio = segment.audio
            segment_length = len(segment_audio)

            # Determine start position
            if alignment == "start":
                start_pos = 0
            elif alignment == "center":
                start_pos = (max_samples - segment_length) // 2
            elif alignment == "end":
                start_pos = max_samples - segment_length
            elif alignment == "random":
                start_pos = np.random.randint(0, max(1, max_samples - segment_length))
            else:
                start_pos = 0

            end_pos = start_pos + segment_length

            # Ensure bounds
            if end_pos > max_samples:
                end_pos = max_samples
                segment_audio = segment_audio[: end_pos - start_pos]

            if start_pos >= 0 and end_pos > start_pos:
                output[start_pos:end_pos] += segment_audio

        # Apply mixing mode
        if self.vertical_mix_mode == "normalized_add":
            if np.max(np.abs(output)) > 0:
                output = output / np.max(np.abs(output)) * 0.95

        # Trim trailing silence
        trim_threshold = 0.01 * np.max(np.abs(output))
        above_threshold = np.abs(output) > trim_threshold
        if np.any(above_threshold):
            last_valid = np.where(above_threshold)[0][-1]
            output = output[: last_valid + 1]

        self.synthesis_count += 1

        metadata = {
            "synthesis_type": "vertical_with_microharmonics",
            "phrase_set": phrase_set,
            "num_segments": len(segments),
            "alignment": alignment,
            "microharmonic_filters": {
                "dominant_harmonic": dominant_harmonic,
                "harmonic_entropy_range": harmonic_entropy_range,
                "spectral_centroid_range": spectral_centroid_range,
                "harmonic_stability_min": harmonic_stability_min,
            },
        }

        return output, self.library.sr, metadata

    def synthesize_combined_with_microharmonics(
        self,
        synthesis_plan: List[Tuple[str, List[str]]],
        dominant_harmonic: Optional[int] = None,
        harmonic_entropy_range: Optional[Tuple[float, float]] = None,
        spectral_centroid_range: Optional[Tuple[float, float]] = None,
        harmonic_stability_min: float = 0.0,
        gap_ms: float = 0.0,
        strategy: str = "random",
        min_quality: float = 0.5,
    ) -> Optional[Tuple[np.ndarray, int, Dict]]:
        """
        Synthesize vocalization using combined encoding with microharmonic filtering.

        This creates complex vocalizations using microharmonic-filtered segments
        in mixed sequential and simultaneous patterns.

        Args:
            synthesis_plan: List of (mode, phrases) tuples
            dominant_harmonic: Filter by dominant harmonic
            harmonic_entropy_range: Filter by harmonic entropy range
            spectral_centroid_range: Filter by spectral centroid range
            harmonic_stability_min: Minimum harmonic stability
            gap_ms: Gap between horizontal steps in milliseconds
            strategy: Selection strategy
            min_quality: Minimum quality score for segments

        Returns:
            Tuple of (audio, sr, metadata) if successful, None otherwise
        """
        if not synthesis_plan:
            return None

        output_segments = []
        total_duration = 0.0

        for mode, phrases in synthesis_plan:
            if mode == "horizontal":
                result = self.synthesize_horizontal_with_microharmonics(
                    phrases,
                    dominant_harmonic=dominant_harmonic,
                    harmonic_entropy_range=harmonic_entropy_range,
                    spectral_centroid_range=spectral_centroid_range,
                    harmonic_stability_min=harmonic_stability_min,
                    gap_ms=gap_ms,
                    strategy=strategy,
                    min_quality=min_quality,
                )
            elif mode == "vertical":
                result = self.synthesize_vertical_with_microharmonics(
                    phrases,
                    dominant_harmonic=dominant_harmonic,
                    harmonic_entropy_range=harmonic_entropy_range,
                    spectral_centroid_range=spectral_centroid_range,
                    harmonic_stability_min=harmonic_stability_min,
                    alignment="start",
                    strategy=strategy,
                    min_quality=min_quality,
                )
            else:
                logger.warning(f"Unknown synthesis mode: {mode}")
                continue

            if result is not None:
                audio, sr, metadata = result
                output_segments.append((audio, metadata))
                total_duration += len(audio) / sr

        if not output_segments:
            return None

        # Concatenate all segments
        total_samples = int(total_duration * self.library.sr) + 1000
        output = np.zeros(total_samples)
        current_sample = 0

        for audio, metadata in output_segments:
            segment_length = len(audio)
            output[current_sample : current_sample + segment_length] = audio
            current_sample += segment_length

        # Trim trailing silence
        trim_threshold = 0.01 * np.max(np.abs(output))
        above_threshold = np.abs(output) > trim_threshold
        if np.any(above_threshold):
            last_valid = np.where(above_threshold)[0][-1]
            output = output[: last_valid + 1]

        self.synthesis_count += 1

        metadata = {
            "synthesis_type": "combined_with_microharmonics",
            "synthesis_plan": synthesis_plan,
            "num_steps": len(output_segments),
            "microharmonic_filters": {
                "dominant_harmonic": dominant_harmonic,
                "harmonic_entropy_range": harmonic_entropy_range,
                "spectral_centroid_range": spectral_centroid_range,
                "harmonic_stability_min": harmonic_stability_min,
            },
        }

        return output, self.library.sr, metadata

    def synthesize_harmonically_compatible(
        self,
        phrase_sequence: List[str],
        gap_ms: float = 0.0,
        min_similarity: float = 0.7,
        output_path: Optional[Path] = None,
    ) -> Optional[Tuple[np.ndarray, int, Dict]]:
        """
        Synthesize a harmonically compatible vocalization sequence.

        This method selects phrases that are microharmonically similar to each other,
        creating a sequence with smooth harmonic transitions.

        Args:
            phrase_sequence: List of phrase keys to synthesize
            gap_ms: Gap between phrases in milliseconds
            min_similarity: Minimum microharmonic similarity between consecutive phrases
            output_path: Optional path to save audio

        Returns:
            Tuple of (audio, sr, metadata) if successful, None otherwise
        """
        if len(phrase_sequence) < 2:
            # Single phrase, use regular horizontal synthesis
            return self.synthesize_horizontal(phrase_sequence, gap_ms=gap_ms)

        # Check microharmonic compatibility between consecutive phrases
        compatible_sequence = [phrase_sequence[0]]
        similarities = []

        for i in range(len(phrase_sequence) - 1):
            similarity = self.library.calculate_microharmonic_similarity(
                phrase_sequence[i], phrase_sequence[i + 1]
            )

            if similarity is None:
                logger.warning(
                    f"No microharmonic data for phrases {phrase_sequence[i]} and {
                        phrase_sequence[i + 1]
                    }"
                )
                return None

            similarities.append(similarity)

            if similarity < min_similarity:
                logger.warning(
                    f"Low microharmonic similarity ({similarity:.2f}) between "
                    f"{phrase_sequence[i]} and {phrase_sequence[i + 1]}"
                )

            compatible_sequence.append(phrase_sequence[i + 1])

        # Synthesize with the compatible sequence
        result = self.synthesize_horizontal(
            phrase_sequence=compatible_sequence, gap_ms=gap_ms, variation_strategy="best"
        )

        if result is not None:
            audio, sr = result
            metadata = {
                "synthesis_type": "harmonically_compatible",
                "phrase_sequence": compatible_sequence,
                "similarities": similarities,
                "mean_similarity": float(np.mean(similarities)) if similarities else 0.0,
                "min_similarity": float(np.min(similarities)) if similarities else 0.0,
            }

            if output_path:
                self.save_synthesis(audio, output_path, metadata)

            return audio, sr, metadata

        return None

    def save_synthesis(
        self,
        audio: np.ndarray,
        output_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save synthesized audio to file.

        Args:
            audio: Audio waveform
            output_path: Output file path
            metadata: Optional metadata to save as JSON sidecar
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save audio
        sf.write(str(output_path), audio, self.library.sr)

        # Save metadata if provided
        if metadata:
            metadata_path = output_path.with_suffix(".json")
            import json

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Synthesis saved to {output_path}")


# ============================================================================
# Convenience Functions for Integration with Analyzers
# ============================================================================


def create_phrase_library_during_analysis(
    analyzer, output_dir: Optional[Union[str, Path]] = None, enable_segmentation: bool = True
) -> PhraseAudioLibrary:
    """
    Create and configure a phrase audio library for use during analysis.

    This function is designed to be called by V3 analyzers to enable
    automatic audio segmentation during phrase detection.

    Args:
        analyzer: RosettaStoneAnalyzerV2 instance
        output_dir: Output directory for library
        enable_segmentation: Whether to enable audio segmentation

    Returns:
        Configured PhraseAudioLibrary

    Example integration in analyzer:
        self.phrase_audio_library = create_phrase_library_during_analysis(self)
    """
    if output_dir is None:
        output_dir = Path(analyzer.config.output_dir) / "phrase_audio_library"
    else:
        output_dir = Path(output_dir)

    species = getattr(analyzer, "species", "unknown")
    sr = analyzer.config.sr

    library = PhraseAudioLibrary(
        species=species,
        sr=sr,
        library_dir=output_dir,
        max_segments_per_phrase=100,
        min_quality_score=0.3,
    )

    if enable_segmentation:
        logger.info(f"Phrase audio library enabled for {species}")
        logger.info(f"  Library directory: {output_dir}")
        logger.info(f"  Sample rate: {sr} Hz")
    else:
        logger.info("Phrase audio library created but segmentation disabled")

    return library


def extract_microharmonic_signature_from_audio(
    audio: np.ndarray, sr: int, start_ms: float, end_ms: float, species: str
) -> Optional[Dict[str, Any]]:
    """
    Extract microharmonic signature from audio segment.

    This function attempts to extract microharmonic features using the
    existing microharmonic encoder infrastructure.

    Args:
        audio: Full audio waveform
        sr: Sample rate
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        species: Species name

    Returns:
        Microharmonic signature dictionary or None if extraction fails
    """
    try:
        # Extract segment
        start_sample = int(start_ms / 1000 * sr)
        end_sample = int(end_ms / 1000 * sr)
        segment_audio = audio[start_sample:end_sample].copy()

        # Try to import microharmonic encoder
        try:
            from intracall_linguistic_analysis import IntraCallLinguisticAnalyzer  # noqa: F401
            from microharmonic_encoder_phase1 import MicroharmonicDatasetBuilder
        except ImportError:
            logger.debug("Microharmonic encoder not available, skipping signature extraction")
            return None

        # Initialize analyzer
        builder = MicroharmonicDatasetBuilder()

        # Extract comprehensive features (includes microharmonic structure)
        features = builder.extract_comprehensive_features(segment_audio, sr, species)

        if not features:
            return None

        # Build microharmonic signature
        signature = {
            "harmonic_ratios": features.get("harmonic_ratios"),
            "dominant_harmonic": features.get("dominant_harmonic"),
            "harmonic_entropy": features.get("harmonic_entropy"),
            "spectral_centroid_hz": features.get("spectral_centroid_hz"),
            "harmonic_stability": features.get("harmonic_stability"),
            "mean_harmonics_per_frame": features.get("mean_harmonics_per_frame"),
            "frequency_modulation_rate": features.get("frequency_modulation_rate"),
            "amplitude_modulation_rate": features.get("amplitude_modulation_rate"),
        }

        return signature

    except Exception as e:
        logger.debug(f"Microharmonic extraction failed: {e}")
        return None


def extract_and_store_phrases_from_analysis(
    analyzer,
    audio_path: Path,
    phrases: List[Dict[str, Any]],
    audio: Optional[np.ndarray] = None,
    enable_microharmonic_extraction: bool = False,
) -> int:
    """
    Extract and store phrase audio segments during analysis.

    This function is called within the analyzer's phrase extraction
    pipeline to automatically build the phrase audio library.

    Args:
        analyzer: RosettaStoneAnalyzerV2 instance
        audio_path: Path to audio file
        phrases: List of detected phrases
        audio: Pre-loaded audio (will load if None)
        enable_microharmonic_extraction: Whether to extract microharmonic signatures

    Returns:
        Number of segments successfully extracted and stored

    Example integration in analyzer:
        n_extracted = extract_and_store_phrases_from_analysis(
            self, audio_path, result['phrases'], enable_microharmonic_extraction=True
        )
    """
    if not hasattr(analyzer, "phrase_audio_library"):
        return 0

    library = analyzer.phrase_audio_library
    sr = analyzer.config.sr
    species = getattr(analyzer, "species", "unknown")

    # Load audio if not provided
    if audio is None:
        try:
            import librosa

            audio, sr_actual = librosa.load(str(audio_path), sr=sr)
        except Exception as e:
            logger.error(f"Error loading audio for segmentation: {e}")
            return 0
    else:
        sr_actual = sr

    n_extracted = 0

    for phrase in phrases:
        # Extract microharmonic signature if enabled
        microharmonic_signature = None
        if enable_microharmonic_extraction:
            microharmonic_signature = extract_microharmonic_signature_from_audio(
                audio=audio,
                sr=sr_actual,
                start_ms=phrase.get("start_time_ms", 0),
                end_ms=phrase.get(
                    "end_time_ms",
                    phrase.get("start_time_ms", 0) + phrase.get("mean_duration_ms", 50),
                ),
                species=species,
            )

        # Extract segment
        segment = library.extract_phrase_segment(
            audio=audio,
            sr=sr_actual,
            start_ms=phrase.get("start_time_ms", 0),
            end_ms=phrase.get(
                "end_time_ms", phrase.get("start_time_ms", 0) + phrase.get("mean_duration_ms", 50)
            ),
            phrase_key=phrase["phrase_key"],
            source_file=phrase.get("source_file", audio_path.name),
            mean_f0_hz=phrase.get("mean_f0_hz", 0),
            std_f0_hz=phrase.get("std_f0_hz", 0),
            mean_range_hz=phrase.get("mean_range_hz", 0),
            encoding=phrase.get("encoding", "horizontal"),
            superposed_with=phrase.get("superposed_with", []),
            context=phrase.get("context"),
            individual_id=phrase.get("individual_id"),
            microharmonic_signature=microharmonic_signature,
        )

        if segment is not None:
            n_extracted += 1

    if n_extracted > 0:
        logger.debug(f"Extracted {n_extracted} phrase segments from {audio_path.name}")

    return n_extracted
