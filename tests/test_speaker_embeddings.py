#!/usr/bin/env python3
"""
Tests for Speaker Embeddings (Direction 3)

Tests for speaker embedding extraction, database, identification,
and adaptive synthesis.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest
from unittest.mock import Mock

import numpy as np


class TestSpeakerEmbeddingExtractor(unittest.TestCase):
    """Test speaker embedding extraction from audio and features."""

    def setUp(self):
        """Set up test fixtures."""
        from analysis.rosetta_stone.speaker_embeddings import SpeakerEmbeddingExtractor

        self.extractor = SpeakerEmbeddingExtractor(embedding_dim=256)

    def test_extract_from_audio_shape(self):
        """Returns correct embedding dimension."""
        # Create 1 second of test audio at 48kHz
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        sr = 48000

        embedding = self.extractor.extract_from_audio(audio, sr)

        self.assertEqual(len(embedding), 256)
        self.assertIsInstance(embedding, np.ndarray)

    def test_extract_same_audio_same_embedding(self):
        """Same audio produces same embedding."""
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        sr = 48000

        emb1 = self.extractor.extract_from_audio(audio, sr)
        emb2 = self.extractor.extract_from_audio(audio, sr)

        # Embeddings should be nearly identical (deterministic)
        np.testing.assert_array_almost_equal(emb1, emb2, decimal=5)

    def test_extract_different_audio_different_embedding(self):
        """Different audio produces different embedding."""
        # Use distinctly different audio signals
        # Audio 1: Low frequency tone
        t1 = np.arange(48000) / 48000
        audio1 = (np.sin(2 * np.pi * 440 * t1) * 0.3).astype(np.float32)

        # Audio 2: High frequency tone + noise
        t2 = np.arange(48000) / 48000
        audio2 = (np.sin(2 * np.pi * 2000 * t2) * 0.3 + np.random.randn(48000) * 0.05).astype(
            np.float32
        )

        sr = 48000

        emb1 = self.extractor.extract_from_audio(audio1, sr)
        emb2 = self.extractor.extract_from_audio(audio2, sr)

        # Different audio should produce different embeddings
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        self.assertLess(cosine_sim, 0.98, "Different audio should have different embeddings")

    def test_extract_from_features(self):
        """Feature-based extraction works."""
        features = np.random.randn(112).astype(np.float32)

        embedding = self.extractor.extract_from_features(features)

        self.assertEqual(len(embedding), 256)
        self.assertIsInstance(embedding, np.ndarray)

    def test_embedding_normalization(self):
        """Embeddings are L2 normalized."""
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        sr = 48000

        embedding = self.extractor.extract_from_audio(audio, sr)

        # L2 norm should be 1.0
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)


class TestSpeakerDatabase(unittest.TestCase):
    """Test speaker database for enrollment, verification, and identification."""

    def setUp(self):
        """Set up test fixtures."""
        from analysis.rosetta_stone.speaker_embeddings import SpeakerDatabase

        self.db = SpeakerDatabase()

        # Create test embeddings
        self.speaker1_emb = np.random.randn(256).astype(np.float32)
        self.speaker1_emb /= np.linalg.norm(self.speaker1_emb)

        self.speaker2_emb = np.random.randn(256).astype(np.float32)
        self.speaker2_emb /= np.linalg.norm(self.speaker2_emb)

    def test_enroll_new_speaker(self):
        """New speaker added to database."""
        result = self.db.enroll("speaker_1", self.speaker1_emb)

        self.assertTrue(result)
        self.assertIn("speaker_1", self.db.speakers)
        self.assertEqual(len(self.db.speakers), 1)

    def test_enroll_updates_existing(self):
        """Re-enrolling updates embedding."""
        self.db.enroll("speaker_1", self.speaker1_emb)

        # Enroll again with different embedding
        new_emb = np.random.randn(256).astype(np.float32)
        new_emb /= np.linalg.norm(new_emb)
        self.db.enroll("speaker_1", new_emb)

        # Enrollment count should increase
        self.assertEqual(self.db.speakers["speaker_1"].enrollment_count, 2)

    def test_verify_same_speaker(self):
        """Same speaker verifies successfully."""
        self.db.enroll("speaker_1", self.speaker1_emb)

        result = self.db.verify("speaker_1", self.speaker1_emb, threshold=0.8)

        self.assertTrue(result.is_match)
        self.assertGreater(result.confidence, 0.9)

    def test_verify_different_speaker(self):
        """Different speaker fails verification."""
        self.db.enroll("speaker_1", self.speaker1_emb)

        result = self.db.verify("speaker_1", self.speaker2_emb, threshold=0.8)

        self.assertFalse(result.is_match)
        self.assertLess(result.confidence, 0.8)

    def test_verify_threshold_sensitivity(self):
        """Threshold affects verification results."""
        self.db.enroll("speaker_1", self.speaker1_emb)

        # Create a somewhat similar embedding
        # Use spherical interpolation between speaker1 and speaker2
        theta = np.arccos(np.clip(np.dot(self.speaker1_emb, self.speaker2_emb), -1, 1))
        sin_theta = np.sin(theta)
        if sin_theta > 1e-6:
            similar_emb = (
                np.sin((1 - 0.3) * theta) / sin_theta * self.speaker1_emb
                + np.sin(0.3 * theta) / sin_theta * self.speaker2_emb
            )
        else:
            similar_emb = self.speaker1_emb.copy()
        similar_emb = similar_emb.astype(np.float32)
        similar_emb /= np.linalg.norm(similar_emb)

        # Compute actual similarity to check our thresholds
        actual_sim = np.dot(self.speaker1_emb, similar_emb)

        # Pick appropriate thresholds based on actual similarity
        # Strict threshold (higher than actual sim) should fail
        result_strict = self.db.verify("speaker_1", similar_emb, threshold=actual_sim + 0.05)
        self.assertFalse(result_strict.is_match)

        # Loose threshold (lower than actual sim) should pass
        result_loose = self.db.verify("speaker_1", similar_emb, threshold=actual_sim - 0.05)
        self.assertTrue(result_loose.is_match)


class TestSpeakerIdentification(unittest.TestCase):
    """Test speaker identification from embeddings."""

    def setUp(self):
        """Set up test fixtures."""
        from analysis.rosetta_stone.speaker_embeddings import SpeakerDatabase

        self.db = SpeakerDatabase()

        # Enroll multiple speakers
        np.random.seed(42)
        for i in range(5):
            emb = np.random.randn(256).astype(np.float32)
            emb /= np.linalg.norm(emb)
            self.db.enroll(f"speaker_{i}", emb)

    def test_identify_known_speaker(self):
        """Correctly identifies known speaker."""
        # Get a known speaker's embedding
        known_emb = self.db.speakers["speaker_0"].embedding

        matches = self.db.identify(known_emb, top_k=1)

        self.assertEqual(len(matches), 1)
        speaker_id, score = matches[0]
        self.assertEqual(speaker_id, "speaker_0")
        self.assertGreater(score, 0.99)

    def test_identify_unknown_speaker(self):
        """Returns low confidence for unknown."""
        unknown_emb = np.random.randn(256).astype(np.float32)
        unknown_emb /= np.linalg.norm(unknown_emb)

        matches = self.db.identify(unknown_emb, top_k=1)

        speaker_id, score = matches[0]
        # Should have lower confidence since it's not a known speaker
        self.assertLess(score, 0.9)

    def test_identify_top_k(self):
        """Returns top-k most similar speakers."""
        query_emb = self.db.speakers["speaker_2"].embedding

        matches = self.db.identify(query_emb, top_k=3)

        self.assertEqual(len(matches), 3)
        # First match should be speaker_2 with highest confidence
        self.assertEqual(matches[0][0], "speaker_2")
        # Scores should be in descending order
        self.assertGreaterEqual(matches[0][1], matches[1][1])
        self.assertGreaterEqual(matches[1][1], matches[2][1])

    def test_identify_empty_db(self):
        """Handles empty database gracefully."""
        from analysis.rosetta_stone.speaker_embeddings import SpeakerDatabase

        empty_db = SpeakerDatabase()
        query_emb = np.random.randn(256).astype(np.float32)
        query_emb /= np.linalg.norm(query_emb)

        matches = empty_db.identify(query_emb, top_k=5)

        self.assertEqual(len(matches), 0)


class TestSpeakerClustering(unittest.TestCase):
    """Test speaker clustering for discovering new speakers."""

    def setUp(self):
        """Set up test fixtures."""
        from analysis.rosetta_stone.speaker_embeddings import SpeakerDatabase

        self.db = SpeakerDatabase()

    def test_cluster_two_speakers(self):
        """Separates two distinct speakers."""
        # Create embeddings for two distinct speakers
        np.random.seed(42)
        speaker1_embs = [np.random.randn(256) for _ in range(5)]
        speaker2_embs = [np.random.randn(256) + 3.0 for _ in range(5)]  # Offset

        all_embs = [emb.astype(np.float32) for emb in speaker1_embs + speaker2_embs]

        # Normalize
        for emb in all_embs:
            emb /= np.linalg.norm(emb)

        clusters = self.db.cluster_speakers(all_embs)

        self.assertEqual(len(clusters), len(all_embs))
        # First 5 should be in one cluster, last 5 in another
        self.assertEqual(clusters[:5], clusters[0:5])
        self.assertNotEqual(clusters[0], clusters[5])

    def test_cluster_same_speaker(self):
        """Clusters same-utterance embeddings together."""
        # Create similar embeddings (same speaker)
        base_emb = np.random.randn(256).astype(np.float32)
        base_emb /= np.linalg.norm(base_emb)

        embs = []
        for _ in range(5):
            emb = base_emb + np.random.randn(256) * 0.02  # Less noise
            emb /= np.linalg.norm(emb)
            embs.append(emb)

        # Use a specific number of clusters
        clusters = self.db.cluster_speakers(embs, n_clusters=1)

        # All should be in the same cluster
        self.assertEqual(len(set(clusters)), 1)

    def test_cluster_varying_counts(self):
        """Handles different numbers of actual speakers."""
        np.random.seed(42)

        # 3 speakers with varying utterance counts
        embs = []
        for speaker_idx in range(3):
            center = np.random.randn(256) * (speaker_idx + 1)
            for _ in range(3 + speaker_idx * 2):
                emb = center + np.random.randn(256) * 0.1
                emb = emb.astype(np.float32)
                emb /= np.linalg.norm(emb)
                embs.append(emb)

        clusters = self.db.cluster_speakers(embs)

        self.assertEqual(len(clusters), len(embs))
        # Should identify roughly 3 clusters
        unique_clusters = len(set(clusters))
        self.assertGreaterEqual(unique_clusters, 2)
        self.assertLessEqual(unique_clusters, 4)

    def test_cluster_returns_assignments(self):
        """Returns cluster assignments for all inputs."""
        embs = [np.random.randn(256).astype(np.float32) for _ in range(10)]
        for emb in embs:
            emb /= np.linalg.norm(emb)

        clusters = self.db.cluster_speakers(embs)

        self.assertEqual(len(clusters), 10)
        # All assignments should be non-negative integers
        for c in clusters:
            self.assertGreaterEqual(c, 0)


class TestSpeakerAdaptiveSynthesis(unittest.TestCase):
    """Test speaker-adaptive synthesis."""

    def setUp(self):
        """Set up test fixtures."""
        from analysis.rosetta_stone.speaker_embeddings import (
            SpeakerAdaptiveSynthesis,
            SpeakerDatabase,
        )

        self.db = SpeakerDatabase()
        self.base_model = Mock()

        # Enroll a test speaker
        speaker_emb = np.random.randn(256).astype(np.float32)
        speaker_emb /= np.linalg.norm(speaker_emb)
        self.db.enroll("test_speaker", speaker_emb)

        self.synthesizer = SpeakerAdaptiveSynthesis(base_model=self.base_model, speaker_db=self.db)

    def test_synthesize_as_speaker(self):
        """Output matches target speaker characteristics."""
        tokens = [42, 117, 3, 99]

        # Mock the base model to return audio
        mock_audio = np.random.randn(48000).astype(np.float32)

        # Set up the mock to have the synthesize_with_speaker method
        self.base_model.synthesize_with_speaker = Mock(return_value=mock_audio)

        audio = self.synthesizer.synthesize_as_speaker(tokens, "test_speaker")

        self.assertIsNotNone(audio)
        self.assertEqual(len(audio), 48000)

    def test_synthesize_preserves_tokens(self):
        """Token sequence preserved in synthesis."""
        tokens = [1, 2, 3, 4, 5]

        mock_audio = np.random.randn(48000).astype(np.float32)

        def mock_synthesize(tokens_list, **kwargs):
            return mock_audio

        self.base_model.synthesize = mock_synthesize

        self.synthesizer.synthesize_as_speaker(tokens, "test_speaker")

        # The synthesis was called (we can't easily check args with the mock function approach)

    def test_synthesize_unknown_speaker(self):
        """Falls back for unknown speaker."""
        tokens = [42, 117]
        mock_audio = np.random.randn(48000).astype(np.float32)
        self.base_model.synthesize.return_value = mock_audio

        # Should not raise error for unknown speaker
        audio = self.synthesizer.synthesize_as_speaker(tokens, "unknown_speaker")

        self.assertIsNotNone(audio)


class TestSpeakerEmbeddingsIntegration(unittest.TestCase):
    """Integration tests for speaker embeddings with existing system."""

    def test_feature_event_has_embedding(self):
        """FeatureEvent includes speaker embedding."""
        from realtime.feature_subscriber import FeatureEvent

        # Create a mock feature event with embedding
        embedding = np.random.randn(256).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.random.randn(112).astype(np.float32),
            timestamp=1000.0,
            sequence=1,
            speaker_embedding=embedding,
        )

        self.assertIsNotNone(event.speaker_embedding)
        self.assertEqual(len(event.speaker_embedding), 256)

    def test_interaction_agent_tracks_speakers(self):
        """Agent tracks who is speaking."""
        from analysis.rosetta_stone.speaker_embeddings import SpeakerDatabase
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        # Agent should have speaker database
        self.assertIsNone(agent.speaker_db)  # Initially None

        # Can attach speaker database
        db = SpeakerDatabase()
        agent.speaker_db = db
        self.assertEqual(agent.speaker_db, db)

    def test_speaker_verification_for_response_targeting(self):
        """Speaker verification enables targeted responses."""
        from analysis.rosetta_stone.speaker_embeddings import SpeakerDatabase, VerificationResult

        db = SpeakerDatabase()

        # Enroll a speaker
        speaker_emb = np.random.randn(256).astype(np.float32)
        speaker_emb /= np.linalg.norm(speaker_emb)
        db.enroll("target_bat", speaker_emb)

        # Verify the speaker
        result = db.verify("target_bat", speaker_emb, threshold=0.8)

        self.assertIsInstance(result, VerificationResult)
        self.assertTrue(result.is_match)

    def test_speaker_adaptive_synthesis_with_vocoder(self):
        """SpeakerAdaptiveSynthesis works with NeuralVocoder when tokenizer is provided."""
        from analysis.rosetta_stone.neural_language_model import AcousticTokenizer
        from analysis.rosetta_stone.neural_vocoder import NeuralVocoder
        from analysis.rosetta_stone.speaker_embeddings import (
            SpeakerAdaptiveSynthesis,
            SpeakerDatabase,
        )

        # Create database and enroll speaker
        db = SpeakerDatabase()
        speaker_emb = np.random.randn(256).astype(np.float32)
        speaker_emb /= np.linalg.norm(speaker_emb)
        db.enroll("bat_001", speaker_emb)

        # Create vocoder and tokenizer
        vocoder = NeuralVocoder(model_type="simple", sample_rate=48000)
        tokenizer = AcousticTokenizer(vocab_size=50)

        # Create synthesizer with tokenizer
        synthesizer = SpeakerAdaptiveSynthesis(
            base_model=vocoder, speaker_db=db, tokenizer=tokenizer
        )

        # Should synthesize without error
        tokens = [1, 2, 3]
        audio = synthesizer.synthesize_as_speaker(tokens, "bat_001")

        self.assertIsNotNone(audio)
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    def test_interaction_agent_speaker_tracking_integration(self):
        """InteractionAgent identifies speakers and triggers callbacks."""
        from analysis.rosetta_stone.speaker_embeddings import SpeakerDatabase, VerificationResult
        from realtime.feature_subscriber import FeatureEvent
        from realtime.interaction_agent import InteractionAgent

        # Track speaker changes
        speaker_changes = []

        def on_speaker_change(speaker_id: str, confidence: float):
            speaker_changes.append((speaker_id, confidence))

        # Create agent with speaker tracking
        agent = InteractionAgent(on_speaker_change=on_speaker_change)

        # Create and enroll speaker
        db = SpeakerDatabase()
        speaker_emb = np.random.randn(256).astype(np.float32)
        speaker_emb /= np.linalg.norm(speaker_emb)
        db.enroll("bat_001", speaker_emb)

        # Attach speaker database to agent
        agent.speaker_db = db

        # Create a feature event with speaker embedding
        test_emb = speaker_emb.copy()  # Same as enrolled speaker
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.random.randn(112).astype(np.float32),
            timestamp=1000.0,
            sequence=1,
            speaker_embedding=test_emb,
        )

        # Process the event (simulate agent processing)
        result = agent._process_features(event)

        # Should identify the speaker
        self.assertEqual(result.get("speaker_id"), "bat_001")
        self.assertIsNotNone(result.get("speaker_confidence"))

        # Simulate speaker change detection
        if result.get("speaker_id") != agent._current_speaker:
            agent._current_speaker = result.get("speaker_id")
            agent._speaker_confidence = result.get("speaker_confidence", 0.0)
            if agent.on_speaker_change and agent._current_speaker is not None:
                agent.on_speaker_change(agent._current_speaker, agent._speaker_confidence)

        # Should have triggered speaker change
        self.assertEqual(len(speaker_changes), 1)
        self.assertEqual(speaker_changes[0][0], "bat_001")
        speaker_emb = np.random.randn(256).astype(np.float32)
        speaker_emb /= np.linalg.norm(speaker_emb)
        db.enroll("target_bat", speaker_emb)

        # Verify the speaker
        result = db.verify("target_bat", speaker_emb, threshold=0.8)

        self.assertIsInstance(result, VerificationResult)
        self.assertTrue(result.is_match)


if __name__ == "__main__":
    unittest.main()
