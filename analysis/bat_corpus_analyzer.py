#!/usr/bin/env python3
"""
Egyptian Fruit Bat Corpus Analyzer - Acoustic-First Pipeline

Processes the 91K Egyptian bat vocalizations through the 3-stage Acoustic-First
Pipeline to extract:
- 112D Rosetta Features (BioMAE)
- 16D Affective Embeddings (pUMAP + β-VAE)
- Syntactic Token Sequences (VQ-VAE)

This is a NEW extraction using:
- Linear (non-mel) spectrograms for ultrasonic preservation
- BioMAE ViT encoder
- Dual-stream affective/syntactic encoding

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Iterator, Any
import pickle

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from tqdm import tqdm

# Import pipeline components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_extraction.bio_spectrogram import (
    UltrasonicSpectrogram,
    SpectrogramConfig,
    BAT_CONFIG,
)
from pipeline.acoustic_first_pipeline import (
    AcousticFirstPipeline,
    PipelineConfig,
    BAT_PIPELINE,
    PipelineOutput,
)

logger = logging.getLogger(__name__)


@dataclass
class VocalizationMetadata:
    """Metadata for a single vocalization."""
    file_name: str
    emitter: int
    addressee: int
    context: int
    emitter_prev_action: int
    addressee_prev_action: int
    emitter_post_action: int
    addressee_post_action: int


@dataclass
class CorpusExtraction:
    """Extracted features for a single vocalization."""
    metadata: VocalizationMetadata

    # 112D Rosetta features (BioMAE)
    rosetta_features_112d: np.ndarray  # Shape: (n_segments, 112)

    # 16D Affective embeddings (pUMAP + β-VAE)
    affective_features_16d: np.ndarray  # Shape: (n_segments, 16)

    # Syntactic tokens (VQ-VAE)
    syntactic_tokens: np.ndarray  # Shape: (n_segments,)

    # Segmentation info
    segment_boundaries: List[Tuple[int, int]]  # List of (start_ms, end_ms)

    # Processing info
    sample_rate: int
    duration_ms: float
    n_segments: int


class BatCorpusLoader:
    """Loads Egyptian bat vocalization data with annotations."""

    def __init__(
        self,
        corpus_dir: str = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/",
    ):
        self.corpus_dir = Path(corpus_dir)
        self.audio_dir = self.corpus_dir / "audio"
        self.annotations_file = self.corpus_dir / "annotations.csv"

        self.annotations: Dict[str, VocalizationMetadata] = {}
        self._load_annotations()

        logger.info(f"Loaded {len(self.annotations)} annotations")

    def _load_annotations(self):
        """Load annotations from CSV file."""
        import csv

        with open(self.annotations_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row['File Name']
                self.annotations[file_name] = VocalizationMetadata(
                    file_name=file_name,
                    emitter=int(row['Emitter']),
                    addressee=int(row['Addressee']),
                    context=int(row['Context']),
                    emitter_prev_action=int(row['Emitter pre-vocalization action']),
                    addressee_prev_action=int(row['Addressee pre-vocalization action']),
                    emitter_post_action=int(row['Emitter post-vocalization action']),
                    addressee_post_action=int(row['Addressee post-vocalization action']),
                )

    def get_audio_files(self) -> List[Path]:
        """Get all audio file paths."""
        return sorted(self.audio_dir.glob("*.wav"))

    def load_audio(self, file_name: str) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        audio_path = self.audio_dir / file_name
        audio, sr = sf.read(str(audio_path))

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        return audio.astype(np.float32), sr

    def get_metadata(self, file_name: str) -> Optional[VocalizationMetadata]:
        """Get metadata for a file."""
        return self.annotations.get(file_name)

    def iter_vocalizations(self) -> Iterator[Tuple[str, np.ndarray, int, VocalizationMetadata]]:
        """Iterate over all vocalizations with audio and metadata."""
        for file_name in self.annotations.keys():
            try:
                audio, sr = self.load_audio(file_name)
                metadata = self.annotations[file_name]
                yield file_name, audio, sr, metadata
            except Exception as e:
                logger.warning(f"Failed to load {file_name}: {e}")
                continue


class CorpusAnalyzer:
    """
    Analyzes bat corpus through Acoustic-First Pipeline.

    Extracts 112D Rosetta features, 16D affective embeddings, and syntactic tokens.
    """

    def __init__(
        self,
        pipeline: Optional[AcousticFirstPipeline] = None,
        device: str = "cpu",
    ):
        self.device = device

        if pipeline is None:
            # Create default pipeline with specified device
            config = BAT_PIPELINE
            # Override device in config
            config.device = device
            pipeline = AcousticFirstPipeline(config)

        self.pipeline = pipeline

        logger.info(f"CorpusAnalyzer initialized on device: {device}")

    def process_vocalization(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: VocalizationMetadata,
    ) -> Optional[CorpusExtraction]:
        """
        Process a single vocalization through the pipeline.

        Args:
            audio: Audio waveform
            sample_rate: Sample rate in Hz
            metadata: Vocalization metadata

        Returns:
            CorpusExtraction with extracted features, or None if processing failed
        """
        try:
            # Run pipeline
            output: PipelineOutput = self.pipeline.process_audio(
                audio, sample_rate
            )

            # Check if we got valid output
            if output.features_112d is None:
                logger.warning(f"No features extracted for {metadata.file_name}")
                return None

            # Convert to numpy (already numpy from pipeline)
            rosetta_features = output.features_112d
            affective_features = output.affective_latent_16d
            syntactic_tokens = np.array(output.syntactic_tokens) if output.syntactic_tokens else np.array([])

            # Extract segment boundaries from output
            segment_boundaries = []
            if output.boundaries:
                for start_ms, end_ms in output.boundaries:
                    segment_boundaries.append((start_ms, end_ms))

            # Create extraction
            extraction = CorpusExtraction(
                metadata=metadata,
                rosetta_features_112d=rosetta_features,
                affective_features_16d=affective_features,
                syntactic_tokens=syntactic_tokens,
                segment_boundaries=segment_boundaries,
                sample_rate=sample_rate,
                duration_ms=len(audio) / sample_rate * 1000,
                n_segments=len(rosetta_features),
            )

            return extraction

        except Exception as e:
            logger.error(f"Error processing {metadata.file_name}: {e}")
            return None

    def process_corpus(
        self,
        loader: BatCorpusLoader,
        output_dir: Optional[Path] = None,
        max_files: Optional[int] = None,
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Process entire corpus and extract features.

        Args:
            loader: Corpus loader
            output_dir: Directory to save results
            max_files: Maximum number of files to process (for testing)
            batch_size: Number of files before intermediate save

        Returns:
            Summary statistics
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        stats = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "total_segments": 0,
            "total_duration_ms": 0.0,
        }

        # Storage for batch processing
        extractions_batch = []

        # Get files to process
        audio_files = loader.get_audio_files()
        if max_files:
            audio_files = audio_files[:max_files]

        stats["total_files"] = len(audio_files)
        logger.info(f"Processing {len(audio_files)} vocalizations...")

        # Process each vocalization
        for file_name, audio, sr, metadata in tqdm(
            loader.iter_vocalizations(),
            total=len(audio_files),
            desc="Extracting features"
        ):
            if file_name not in [f.name for f in audio_files]:
                continue

            # Process
            extraction = self.process_vocalization(audio, sr, metadata)

            if extraction:
                extractions_batch.append(extraction)
                stats["processed_files"] += 1
                stats["total_segments"] += extraction.n_segments
                stats["total_duration_ms"] += extraction.duration_ms
            else:
                stats["failed_files"] += 1

            # Intermediate save
            if len(extractions_batch) >= batch_size and output_dir:
                self._save_batch(extractions_batch, output_dir)
                extractions_batch = []

        # Final save
        if extractions_batch and output_dir:
            self._save_batch(extractions_batch, output_dir)

        logger.info(f"Processing complete:")
        logger.info(f"  Processed: {stats['processed_files']}/{stats['total_files']}")
        logger.info(f"  Failed: {stats['failed_files']}")
        logger.info(f"  Total segments: {stats['total_segments']}")
        logger.info(f"  Total duration: {stats['total_duration_ms']/1000/60:.1f} minutes")

        return stats

    def _save_batch(self, extractions: List[CorpusExtraction], output_dir: Path):
        """Save a batch of extractions."""
        # Save as JSONL for streaming
        output_file = output_dir / "corpus_extractions.jsonl"

        with open(output_file, 'a') as f:
            for extraction in extractions:
                # Convert to dict for JSON serialization
                record = {
                    "file_name": extraction.metadata.file_name,
                    "emitter": extraction.metadata.emitter,
                    "addressee": extraction.metadata.addressee,
                    "context": extraction.metadata.context,
                    "rosetta_features_112d": extraction.rosetta_features_112d.tolist(),
                    "affective_features_16d": extraction.affective_features_16d.tolist(),
                    "syntactic_tokens": extraction.syntactic_tokens.tolist(),
                    "segment_boundaries": extraction.segment_boundaries,
                    "sample_rate": extraction.sample_rate,
                    "duration_ms": extraction.duration_ms,
                    "n_segments": extraction.n_segments,
                }
                f.write(json.dumps(record) + '\n')


class AggregatedFeaturesBuilder:
    """
    Builds aggregated feature matrices from corpus extractions.

    Creates:
    - Rosetta feature matrix: (N_segments, 112)
    - Affective feature matrix: (N_segments, 16)
    - Syntactic token sequences: list of token lists
    - Metadata for each segment
    """

    def __init__(self, extraction_dir: Path):
        self.extraction_dir = Path(extraction_dir)
        self.extraction_file = self.extraction_dir / "corpus_extractions.jsonl"

    def build_aggregates(self) -> Dict[str, Any]:
        """
        Build aggregated feature matrices from JSONL extractions.

        Returns:
            Dictionary with aggregated features and metadata
        """
        rosetta_features_list = []
        affective_features_list = []
        syntactic_tokens_list = []
        metadata_list = []

        segment_counter = 0
        file_counter = 0

        logger.info("Building aggregated features...")

        with open(self.extraction_file, 'r') as f:
            for line in tqdm(f, desc="Aggregating"):
                record = json.loads(line)

                n_segs = record['n_segments']
                if n_segs == 0:
                    continue

                rosetta = np.array(record['rosetta_features_112d'])
                affective = np.array(record['affective_features_16d'])
                tokens = np.array(record['syntactic_tokens'])

                rosetta_features_list.append(rosetta)
                affective_features_list.append(affective)
                syntactic_tokens_list.append(tokens)

                # Metadata for each segment
                for i in range(n_segs):
                    metadata_list.append({
                        "file_name": record['file_name'],
                        "segment_index": i,
                        "global_segment_id": segment_counter + i,
                        "emitter": record['emitter'],
                        "addressee": record['addressee'],
                        "context": record['context'],
                        "boundary": record['segment_boundaries'][i] if i < len(record['segment_boundaries']) else None,
                    })

                segment_counter += n_segs
                file_counter += 1

        # Concatenate into matrices
        rosetta_matrix = np.vstack(rosetta_features_list)
        affective_matrix = np.vstack(affective_features_list)

        # Flatten syntactic tokens
        all_tokens = np.concatenate(syntactic_tokens_list)

        logger.info(f"Aggregated {segment_counter} segments from {file_counter} files")
        logger.info(f"  Rosetta matrix: {rosetta_matrix.shape}")
        logger.info(f"  Affective matrix: {affective_matrix.shape}")
        logger.info(f"  Total tokens: {len(all_tokens)}")

        return {
            "rosetta_features_112d": rosetta_matrix,
            "affective_features_16d": affective_matrix,
            "syntactic_tokens": all_tokens,
            "metadata": metadata_list,
            "n_segments": segment_counter,
            "n_files": file_counter,
        }

    def save_aggregates(self, output_path: Path):
        """Save aggregated features to disk."""
        aggregates = self.build_aggregates()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as NPZ for efficient loading
        np.savez(
            output_path,
            rosetta_features_112d=aggregates["rosetta_features_112d"],
            affective_features_16d=aggregates["affective_features_16d"],
            syntactic_tokens=aggregates["syntactic_tokens"],
        )

        # Save metadata separately
        metadata_file = output_path.with_suffix('.metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                "metadata": aggregates["metadata"],
                "n_segments": aggregates["n_segments"],
                "n_files": aggregates["n_files"],
            }, f)

        logger.info(f"Saved aggregates to {output_path}")
        logger.info(f"Saved metadata to {metadata_file}")


def main():
    """Main entry point for corpus analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze bat corpus with Acoustic-First Pipeline")
    parser.add_argument("--corpus-dir", default="/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/",
                        help="Path to corpus directory")
    parser.add_argument("--output-dir", default="/mnt/c/Users/sheel/Desktop/src/analysis/results/bat_corpus_acoustic_first/",
                        help="Output directory for extracted features")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum number of files to process (for testing)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for intermediate saves")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device to run pipeline on")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only aggregate existing extractions")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.aggregate_only:
        # Only aggregate existing extractions
        builder = AggregatedFeaturesBuilder(args.output_dir)
        builder.save_aggregates(Path(args.output_dir) / "aggregated_features.npz")
        return

    # Load corpus
    logger.info("Loading corpus...")
    loader = BatCorpusLoader(args.corpus_dir)

    # Create analyzer
    analyzer = CorpusAnalyzer(device=args.device)

    # Process corpus
    stats = analyzer.process_corpus(
        loader=loader,
        output_dir=Path(args.output_dir),
        max_files=args.max_files,
        batch_size=args.batch_size,
    )

    # Build aggregates
    logger.info("\nBuilding aggregated features...")
    builder = AggregatedFeaturesBuilder(args.output_dir)
    builder.save_aggregates(Path(args.output_dir) / "aggregated_features.npz")

    # Save stats
    stats_file = Path(args.output_dir) / "extraction_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\nExtraction complete! Results saved to {args.output_dir}")
    logger.info(f"Stats: {stats_file}")


if __name__ == "__main__":
    main()
