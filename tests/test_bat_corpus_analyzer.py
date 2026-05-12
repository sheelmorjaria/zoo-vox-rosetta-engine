#!/usr/bin/env python3
"""
Tests for Bat Corpus Analyzer

Tests the corpus loading, feature extraction, and aggregation
for the 91K Egyptian bat vocalizations.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

import numpy as np
import torch

# Import classes to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.bat_corpus_analyzer import (
    VocalizationMetadata,
    CorpusExtraction,
    BatCorpusLoader,
    CorpusAnalyzer,
    AggregatedFeaturesBuilder,
)


class TestVocalizationMetadata(unittest.TestCase):
    """Test metadata dataclass."""

    def test_metadata_creation(self):
        """Should create metadata from annotation data."""
        metadata = VocalizationMetadata(
            file_name="0.wav",
            emitter=118,
            addressee=0,
            context=9,
            emitter_prev_action=2,
            addressee_prev_action=2,
            emitter_post_action=3,
            addressee_post_action=3,
        )

        self.assertEqual(metadata.file_name, "0.wav")
        self.assertEqual(metadata.emitter, 118)
        self.assertEqual(metadata.addressee, 0)


class TestBatCorpusLoader(unittest.TestCase):
    """Test corpus loading functionality."""

    def setUp(self):
        """Set up test corpus directory."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.audio_dir = self.test_dir / "audio"
        self.audio_dir.mkdir(parents=True)

        # Create test audio files
        self.test_files = ["0.wav", "1.wav", "2.wav"]
        for fname in self.test_files:
            (self.audio_dir / fname).touch()

        # Create test annotations
        self.annotations_file = self.test_dir / "annotations.csv"
        with open(self.annotations_file, 'w') as f:
            f.write("Emitter,Addressee,Context,Emitter pre-vocalization action,Addressee pre-vocalization action,Emitter post-vocalization action,Addressee post-vocalization action,File Name\n")
            f.write("118,0,9,2,2,3,3,0.wav\n")
            f.write("0,0,11,0,0,0,0,1.wav\n")
            f.write("118,0,12,2,2,3,3,2.wav\n")

    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir)

    def test_loader_initializes(self):
        """Should load annotations from CSV."""
        loader = BatCorpusLoader(str(self.test_dir))

        self.assertEqual(len(loader.annotations), 3)
        self.assertIn("0.wav", loader.annotations)
        self.assertIn("1.wav", loader.annotations)

    def test_get_metadata(self):
        """Should retrieve metadata for a file."""
        loader = BatCorpusLoader(str(self.test_dir))
        metadata = loader.get_metadata("0.wav")

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.emitter, 118)
        self.assertEqual(metadata.addressee, 0)
        self.assertEqual(metadata.context, 9)

    def test_get_audio_files(self):
        """Should list all audio files."""
        loader = BatCorpusLoader(str(self.test_dir))
        files = loader.get_audio_files()

        self.assertEqual(len(files), 3)
        self.assertTrue(all(f.suffix == '.wav' for f in files))

    @patch('soundfile.read')
    def test_load_audio(self, mock_sf_read):
        """Should load audio file."""
        # Mock audio data
        mock_audio = np.random.randn(4800).astype(np.float32)
        mock_sr = 22050
        mock_sf_read.return_value = (mock_audio, mock_sr)

        loader = BatCorpusLoader(str(self.test_dir))
        audio, sr = loader.load_audio("0.wav")

        self.assertEqual(sr, 22050)
        self.assertEqual(len(audio), 4800)

    @patch('soundfile.read')
    def test_iter_vocalizations(self, mock_sf_read):
        """Should iterate over all vocalizations."""
        # Mock audio data
        mock_audio = np.random.randn(4800).astype(np.float32)
        mock_sf_read.return_value = (mock_audio, 22050)

        loader = BatCorpusLoader(str(self.test_dir))
        vocalizations = list(loader.iter_vocalizations())

        self.assertEqual(len(vocalizations), 3)


class TestCorpusAnalyzer(unittest.TestCase):
    """Test corpus feature extraction."""

    def setUp(self):
        """Set up test analyzer."""
        self.device = "cpu"

    @patch('analysis.bat_corpus_analyzer.AcousticFirstPipeline')
    def test_analyzer_initializes(self, mock_pipeline_class):
        """Should initialize with default pipeline."""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline

        analyzer = CorpusAnalyzer(device=self.device)

        self.assertIsNotNone(analyzer.pipeline)
        self.assertEqual(analyzer.device, "cpu")

    @patch('analysis.bat_corpus_analyzer.AcousticFirstPipeline')
    def test_process_vocalization_success(self, mock_pipeline_class):
        """Should successfully process a vocalization."""
        # Mock pipeline output - use correct field names from PipelineOutput
        mock_output = Mock()
        mock_output.features_112d = np.random.randn(5, 112)  # numpy array, not tensor
        mock_output.affective_latent_16d = np.random.randn(5, 16)  # numpy array
        mock_output.syntactic_tokens = [5, 10, 15, 20, 25]  # list of ints
        mock_output.boundaries = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250)]

        mock_pipeline = Mock()
        mock_pipeline.process_audio.return_value = mock_output
        mock_pipeline_class.return_value = mock_pipeline

        analyzer = CorpusAnalyzer(device=self.device)

        # Create test metadata
        metadata = VocalizationMetadata(
            file_name="test.wav",
            emitter=1,
            addressee=0,
            context=1,
            emitter_prev_action=0,
            addressee_prev_action=0,
            emitter_post_action=0,
            addressee_post_action=0,
        )

        # Process
        audio = np.random.randn(22050).astype(np.float32)
        extraction = analyzer.process_vocalization(audio, 22050, metadata)

        self.assertIsNotNone(extraction)
        self.assertEqual(extraction.n_segments, 5)
        self.assertEqual(extraction.rosetta_features_112d.shape, (5, 112))
        self.assertEqual(extraction.affective_features_16d.shape, (5, 16))

    @patch('analysis.bat_corpus_analyzer.AcousticFirstPipeline')
    def test_process_vocalization_failure(self, mock_pipeline_class):
        """Should return None if processing fails."""
        # Mock pipeline to fail
        mock_pipeline = Mock()
        mock_pipeline.process_audio.side_effect = Exception("Processing failed")
        mock_pipeline_class.return_value = mock_pipeline

        analyzer = CorpusAnalyzer(device=self.device)

        metadata = VocalizationMetadata(
            file_name="test.wav",
            emitter=1,
            addressee=0,
            context=1,
            emitter_prev_action=0,
            addressee_prev_action=0,
            emitter_post_action=0,
            addressee_post_action=0,
        )

        audio = np.random.randn(22050).astype(np.float32)
        extraction = analyzer.process_vocalization(audio, 22050, metadata)

        self.assertIsNone(extraction)

    @patch('analysis.bat_corpus_analyzer.AcousticFirstPipeline')
    def test_process_vocalization_no_features(self, mock_pipeline_class):
        """Should return None if no features extracted."""
        # Mock pipeline with no features
        mock_output = Mock()
        mock_output.rosetta_features_112d = None

        mock_pipeline = Mock()
        mock_pipeline.process_audio.return_value = mock_output
        mock_pipeline_class.return_value = mock_pipeline

        analyzer = CorpusAnalyzer(device=self.device)

        metadata = VocalizationMetadata(
            file_name="test.wav",
            emitter=1,
            addressee=0,
            context=1,
            emitter_prev_action=0,
            addressee_prev_action=0,
            emitter_post_action=0,
            addressee_post_action=0,
        )

        audio = np.random.randn(22050).astype(np.float32)
        extraction = analyzer.process_vocalization(audio, 22050, metadata)

        self.assertIsNone(extraction)


class TestAggregatedFeaturesBuilder(unittest.TestCase):
    """Test feature aggregation."""

    def setUp(self):
        """Set up test extraction file."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.extraction_file = self.test_dir / "corpus_extractions.jsonl"

        # Create test extractions
        test_data = [
            {
                "file_name": "0.wav",
                "emitter": 118,
                "addressee": 0,
                "context": 9,
                "rosetta_features_112d": [[1.0] * 112, [2.0] * 112],
                "affective_features_16d": [[0.5] * 16, [0.6] * 16],
                "syntactic_tokens": [5, 10],
                "segment_boundaries": [(0, 50), (50, 100)],
                "sample_rate": 22050,
                "duration_ms": 100.0,
                "n_segments": 2,
            },
            {
                "file_name": "1.wav",
                "emitter": 0,
                "addressee": 0,
                "context": 11,
                "rosetta_features_112d": [[3.0] * 112],
                "affective_features_16d": [[0.7] * 16],
                "syntactic_tokens": [15],
                "segment_boundaries": [(0, 50)],
                "sample_rate": 22050,
                "duration_ms": 50.0,
                "n_segments": 1,
            },
        ]

        with open(self.extraction_file, 'w') as f:
            for record in test_data:
                f.write(json.dumps(record) + '\n')

    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir)

    def test_build_aggregates(self):
        """Should build aggregated feature matrices."""
        builder = AggregatedFeaturesBuilder(self.test_dir)
        aggregates = builder.build_aggregates()

        # Check shapes
        self.assertEqual(aggregates["n_segments"], 3)
        self.assertEqual(aggregates["n_files"], 2)
        self.assertEqual(aggregates["rosetta_features_112d"].shape, (3, 112))
        self.assertEqual(aggregates["affective_features_16d"].shape, (3, 16))
        self.assertEqual(len(aggregates["syntactic_tokens"]), 3)
        self.assertEqual(len(aggregates["metadata"]), 3)

        # Check metadata
        self.assertEqual(aggregates["metadata"][0]["file_name"], "0.wav")
        self.assertEqual(aggregates["metadata"][0]["segment_index"], 0)
        self.assertEqual(aggregates["metadata"][1]["file_name"], "0.wav")
        self.assertEqual(aggregates["metadata"][1]["segment_index"], 1)
        self.assertEqual(aggregates["metadata"][2]["file_name"], "1.wav")

    def test_save_aggregates(self):
        """Should save aggregates to disk."""
        output_file = self.test_dir / "aggregated_features.npz"

        builder = AggregatedFeaturesBuilder(self.test_dir)
        builder.save_aggregates(output_file)

        # Check files exist
        self.assertTrue(output_file.exists())
        self.assertTrue(output_file.with_suffix('.metadata.json').exists())

        # Load and verify
        data = np.load(output_file)
        self.assertEqual(data['rosetta_features_112d'].shape, (3, 112))
        self.assertEqual(data['affective_features_16d'].shape, (3, 16))


class TestCorpusIntegration(unittest.TestCase):
    """Integration tests for corpus analysis."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.audio_dir = self.test_dir / "audio"
        self.audio_dir.mkdir(parents=True)

        # Create annotations
        self.annotations_file = self.test_dir / "annotations.csv"
        with open(self.annotations_file, 'w') as f:
            f.write("Emitter,Addressee,Context,Emitter pre-vocalization action,Addressee pre-vocalization action,Emitter post-vocalization action,Addressee post-vocalization action,File Name\n")
            f.write("118,0,9,2,2,3,3,0.wav\n")

    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir)

    @patch('soundfile.read')
    @patch('analysis.bat_corpus_analyzer.AcousticFirstPipeline')
    def test_end_to_end_extraction(self, mock_pipeline_class, mock_sf_read):
        """Test full extraction pipeline."""
        # Mock audio
        mock_audio = np.random.randn(22050).astype(np.float32)
        mock_sf_read.return_value = (mock_audio, 22050)

        # Mock pipeline with realistic output that matches real pipeline structure
        mock_output = Mock()
        mock_output.features_112d = np.random.randn(2, 112)  # 2 segments
        mock_output.affective_latent_16d = np.random.randn(2, 16)
        mock_output.syntactic_tokens = [5, 10]  # 2 tokens total
        mock_output.boundaries = [(0, 50), (50, 100)]
        mock_output.segment_count = 2
        mock_output.perplexity = 5.5
        mock_output.processing_time_ms = 150.0

        mock_pipeline = Mock()
        mock_pipeline.process_audio.return_value = mock_output
        mock_pipeline.to_device = Mock()
        mock_pipeline_class.return_value = mock_pipeline

        # Create dummy audio file
        (self.audio_dir / "0.wav").touch()

        # Run extraction
        loader = BatCorpusLoader(str(self.test_dir))
        analyzer = CorpusAnalyzer(device="cpu")

        output_dir = self.test_dir / "output"
        stats = analyzer.process_corpus(
            loader=loader,
            output_dir=output_dir,
            max_files=1,
            batch_size=1,
        )

        # Check results
        self.assertEqual(stats["processed_files"], 1)
        self.assertEqual(stats["total_segments"], 2)
        self.assertTrue((output_dir / "corpus_extractions.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
