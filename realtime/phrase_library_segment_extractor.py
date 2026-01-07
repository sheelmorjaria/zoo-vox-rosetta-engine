"""
Phrase Library Segment Extractor
==================================

Utility to extract audio segments from source files using .pkl phrase library metadata.

The .pkl files contain all the information needed to re-segment audio from source files:
- source_file: Original audio file path
- start_time_ms, end_time_ms: Exact timestamps
- phrase_key: Phrase identifier

This enables:
1. Loading phrase library metadata
2. Extracting segments from original source audio
3. Building audio library for synthesis
4. Cross-session phrase recovery

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class SegmentExtractionConfig:
    """Configuration for segment extraction."""

    output_dir: str = "audio_library"
    audio_format: str = "wav"
    normalize: bool = True
    target_level_db: float = -3.0


class PhraseLibrarySegmentExtractor:
    """
    Extract audio segments from source files using .pkl library metadata.

    Uses the segmentation information stored in phrase libraries to extract
    the actual audio segments from source recordings.
    """

    def __init__(
        self,
        library_path: str,
        source_audio_dir: Optional[str] = None,
        config: Optional[SegmentExtractionConfig] = None,
    ):
        """
        Initialize extractor.

        Args:
            library_path: Path to .pkl phrase library
            source_audio_dir: Directory containing source audio files
            config: Extraction configuration
        """
        self.library_path = Path(library_path)
        self.source_audio_dir = Path(source_audio_dir) if source_audio_dir else None
        self.config = config or SegmentExtractionConfig()

        # Load library
        with open(library_path, "rb") as f:
            self.library_data = pickle.load(f)

        self.species = self.library_data["species"]
        self.sample_rate = self.library_data["sr"]
        self.phrase_segments = self.library_data["phrase_segments"]

        logger.info(f"Loaded phrase library for {self.species}")
        logger.info(f"  Sample rate: {self.sample_rate} Hz")
        logger.info(f"  Phrase types: {len(self.phrase_segments)}")

    def extract_segments(
        self, output_dir: Optional[str] = None, progress: bool = True
    ) -> Dict[str, List[str]]:
        """
        Extract all audio segments from source files.

        Args:
            output_dir: Output directory for segments
            progress: Show progress bar

        Returns:
            Dictionary mapping phrase_key to list of output file paths
        """
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create species subdirectory
        species_dir = output_dir / self.species
        species_dir.mkdir(exist_ok=True)

        output_files = {}
        total_segments = sum(len(segments) for segments in self.phrase_segments.values())

        logger.info(f"Extracting {total_segments} segments...")

        segments_iter = self._iter_segments()
        if progress:
            segments_iter = tqdm(segments_iter, total=total_segments, desc="Extracting")

        for phrase_key, segment, segment_idx in segments_iter:
            try:
                output_path = self._extract_single_segment(
                    phrase_key, segment, segment_idx, species_dir
                )

                if output_path:
                    if phrase_key not in output_files:
                        output_files[phrase_key] = []
                    output_files[phrase_key].append(output_path)

            except Exception as e:
                logger.warning(f"Failed to extract {phrase_key} segment {segment_idx}: {e}")

        # Save index
        self._save_index(output_files, species_dir)

        logger.info(f"✓ Extracted {sum(len(v) for v in output_files.values())} segments")
        logger.info(f"  Output: {species_dir}")

        return output_files

    def _iter_segments(self):
        """Iterate through all segments in the library."""
        for phrase_key, segments in self.phrase_segments.items():
            for segment_idx, segment in enumerate(segments):
                yield phrase_key, segment, segment_idx

    def _extract_single_segment(
        self, phrase_key: str, segment: dict, segment_idx: int, output_dir: Path
    ) -> Optional[str]:
        """
        Extract a single audio segment.

        Args:
            phrase_key: Phrase identifier
            segment: Segment metadata
            segment_idx: Segment index
            output_dir: Output directory

        Returns:
            Path to extracted file or None
        """
        source_file = segment["source_file"]
        start_ms = segment["start_time_ms"]
        end_ms = segment["end_time_ms"]

        # Resolve source file path
        source_path = self._resolve_source_path(source_file)
        if not source_path or not source_path.exists():
            logger.warning(f"Source file not found: {source_file}")
            return None

        # Load source audio
        try:
            audio, sr = sf.read(source_path)

            # Convert to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Resample if needed
            if sr != self.sample_rate:
                from scipy import signal

                num_samples = int(len(audio) * self.sample_rate / sr)
                audio = signal.resample(audio, num_samples)

            # Convert ms to samples
            start_sample = int(start_ms / 1000 * self.sample_rate)
            end_sample = int(end_ms / 1000 * self.sample_rate)

            # Validate bounds
            start_sample = max(0, min(start_sample, len(audio)))
            end_sample = max(0, min(end_sample, len(audio)))

            if end_sample <= start_sample:
                logger.warning(f"Invalid time range for {phrase_key}")
                return None

            # Extract segment
            segment_audio = audio[start_sample:end_sample]

            # Normalize if requested
            if self.config.normalize and len(segment_audio) > 0:
                rms = np.sqrt(np.mean(segment_audio**2))
                if rms > 0:
                    target_rms = 10 ** (self.config.target_level_db / 20)
                    segment_audio = segment_audio * (target_rms / rms)
                    # Clip to prevent distortion
                    segment_audio = np.clip(segment_audio, -1.0, 1.0)

            # Create output filename
            _ = segment.get("occurrence_id", f"{segment_idx}")
            safe_phrase_key = phrase_key.replace("/", "_").replace("\\", "_")
            output_filename = f"{safe_phrase_key}_{segment_idx:03d}.wav"
            output_path = output_dir / output_filename

            # Save segment
            sf.write(output_path, segment_audio, self.sample_rate)

            return str(output_path)

        except Exception as e:
            logger.error(f"Error extracting {phrase_key}: {e}")
            return None

    def _resolve_source_path(self, source_file: str) -> Optional[Path]:
        """
        Resolve the full path to source audio file.

        Args:
            source_file: Source file name or path from metadata

        Returns:
            Full path to source file or None
        """
        # If it's already a full path
        source_path = Path(source_file)
        if source_path.is_absolute() and source_path.exists():
            return source_path

        # Try source_audio_dir
        if self.source_audio_dir:
            path = self.source_audio_dir / source_file
            if path.exists():
                return path

            # Try with .wav extension
            path = self.source_audio_dir / f"{source_file}.wav"
            if path.exists():
                return path

        # Try relative to library path
        library_dir = self.library_path.parent
        path = library_dir / source_file
        if path.exists():
            return path

        path = library_dir / f"{source_file}.wav"
        if path.exists():
            return path

        # Try parent directories
        for parent in [self.library_path, self.library_path.parent, Path.cwd()]:
            path = parent / source_file
            if path.exists():
                return path

            path = parent / f"{source_file}.wav"
            if path.exists():
                return path

        return None

    def _save_index(self, output_files: Dict[str, List[str]], output_dir: Path):
        """Save segment index to JSON."""
        import json

        index = {
            "species": self.species,
            "sample_rate": self.sample_rate,
            "total_phrases": len(output_files),
            "total_segments": sum(len(v) for v in output_files.values()),
            "phrases": {},
        }

        for phrase_key, files in output_files.items():
            index["phrases"][phrase_key] = {"count": len(files), "files": files}

        index_path = output_dir / "segment_index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        logger.info(f"Saved index to {index_path}")

    def get_segment_info(self) -> pd.DataFrame:
        """
        Get information about all segments as a DataFrame.

        Returns:
            DataFrame with segment information
        """
        import pandas as pd

        rows = []
        for phrase_key, segments in self.phrase_segments.items():
            for segment in segments:
                rows.append(
                    {
                        "phrase_key": phrase_key,
                        "source_file": segment["source_file"],
                        "start_ms": segment["start_time_ms"],
                        "end_ms": segment["end_time_ms"],
                        "duration_ms": segment["end_time_ms"] - segment["start_time_ms"],
                        "mean_f0_hz": segment["mean_f0_hz"],
                        "std_f0_hz": segment["std_f0_hz"],
                        "quality_score": segment["quality_score"],
                    }
                )

        return pd.DataFrame(rows)


# ============================================================================
# Convenience Functions
# ============================================================================


def extract_from_library(
    library_path: str, source_audio_dir: Optional[str] = None, output_dir: str = "audio_library"
) -> Dict[str, List[str]]:
    """
    Extract all audio segments from a phrase library.

    Args:
        library_path: Path to .pkl phrase library
        source_audio_dir: Directory containing source audio
        output_dir: Output directory for segments

    Returns:
        Dictionary mapping phrase_key to list of extracted file paths
    """
    extractor = PhraseLibrarySegmentExtractor(
        library_path=library_path, source_audio_dir=source_audio_dir
    )

    return extractor.extract_segments(output_dir=output_dir)


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python phrase_library_segment_extractor.py")
        print("       <library.pkl> [source_audio_dir] [output_dir]")
        sys.exit(1)

    library_path = sys.argv[1]
    source_dir = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "audio_library"

    print(f"\nExtracting segments from: {library_path}")
    if source_dir:
        print(f"Source audio directory: {source_dir}")
    print(f"Output directory: {output_dir}\n")

    output_files = extract_from_library(
        library_path=library_path, source_audio_dir=source_dir, output_dir=output_dir
    )

    print(f"\n✓ Extracted {sum(len(v) for v in output_files.values())} segments")
    print(f"  Phrase types: {len(output_files)}")
