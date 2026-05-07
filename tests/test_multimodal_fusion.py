#!/usr/bin/env python3
"""
Tests for Multimodal Fusion - Vision + Audio

These tests verify the multimodal fusion mechanism for combining
audio features with visual information for enhanced understanding.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest

import numpy as np


class TestVisualFeatureExtractor(unittest.TestCase):
    """Test visual feature extraction from video frames"""

    def test_extract_frame_features(self):
        """Should extract features from video frame"""
        from cognitive_intelligence.multimodal_fusion import VisualFeatureExtractor

        extractor = VisualFeatureExtractor(frame_height=224, frame_width=224)

        # Simulate video frame (RGB)
        frame = np.random.randn(224, 224, 3).astype(np.float32)
        features = extractor.extract_frame_features(frame)

        self.assertEqual(features.shape[0], 512)  # Feature dimension

    def test_temporal_aggregation(self):
        """Should aggregate features across temporal window"""
        from cognitive_intelligence.multimodal_fusion import VisualFeatureExtractor

        extractor = VisualFeatureExtractor(frame_height=224, frame_width=224)

        # Simulate video frames
        frames = [np.random.randn(224, 224, 3).astype(np.float32) for _ in range(10)]
        features = extractor.aggregate_temporal(frames)

        self.assertEqual(features.shape[0], 512)

    def test_motion_detection(self):
        """Should detect motion between frames"""
        from cognitive_intelligence.multimodal_fusion import VisualFeatureExtractor

        extractor = VisualFeatureExtractor(frame_height=224, frame_width=224)

        frame1 = np.random.randn(224, 224, 3).astype(np.float32)
        frame2 = np.random.randn(224, 224, 3).astype(np.float32)

        motion_score = extractor.detect_motion(frame1, frame2)

        self.assertGreaterEqual(motion_score, 0.0)
        self.assertLessEqual(motion_score, 1.0)


class TestAudioVisualFusion(unittest.TestCase):
    """Test fusion of audio and visual features"""

    def test_cross_modal_attention(self):
        """Should apply cross-modal attention between audio and visual"""
        from cognitive_intelligence.multimodal_fusion import AudioVisualFusion

        fusion = AudioVisualFusion(audio_dim=112, visual_dim=512, fusion_dim=256)

        audio_features = np.random.randn(10, 112).astype(np.float32)
        visual_features = np.random.randn(10, 512).astype(np.float32)

        fused = fusion.cross_modal_attention(audio_features, visual_features)

        self.assertEqual(fused.shape[0], 10)
        self.assertEqual(fused.shape[1], 256)

    def test_late_fusion(self):
        """Should perform late fusion (concatenation + projection)"""
        from cognitive_intelligence.multimodal_fusion import AudioVisualFusion

        fusion = AudioVisualFusion(audio_dim=112, visual_dim=512, fusion_dim=256)

        audio_features = np.random.randn(100, 112).astype(np.float32)
        visual_features = np.random.randn(100, 512).astype(np.float32)

        fused = fusion.late_fusion(audio_features, visual_features)

        self.assertEqual(fused.shape[0], 100)
        self.assertEqual(fused.shape[1], 256)

    def test_early_fusion(self):
        """Should perform early fusion (feature-level combination)"""
        from cognitive_intelligence.multimodal_fusion import AudioVisualFusion

        fusion = AudioVisualFusion(audio_dim=112, visual_dim=512, fusion_dim=256)

        audio_features = np.random.randn(50, 112).astype(np.float32)
        visual_features = np.random.randn(50, 512).astype(np.float32)

        fused = fusion.early_fusion(audio_features, visual_features)

        self.assertEqual(fused.shape[0], 50)
        self.assertEqual(fused.shape[1], 256)


class TestMultimodalContextClassifier(unittest.TestCase):
    """Test context classification with multimodal input"""

    def test_multimodal_classification(self):
        """Should classify using both audio and visual input"""
        from cognitive_intelligence.multimodal_fusion import MultimodalContextClassifier

        classifier = MultimodalContextClassifier(audio_dim=112, visual_dim=512, num_classes=4)

        audio_features = np.random.randn(20, 112).astype(np.float32)
        visual_features = np.random.randn(20, 512).astype(np.float32)

        logits = classifier.classify(audio_features, visual_features)

        self.assertEqual(logits.shape[0], 4)  # num_classes

    def test_audio_only_fallback(self):
        """Should handle audio-only input"""
        from cognitive_intelligence.multimodal_fusion import MultimodalContextClassifier

        classifier = MultimodalContextClassifier(audio_dim=112, visual_dim=512, num_classes=4)

        audio_features = np.random.randn(20, 112).astype(np.float32)

        logits = classifier.classify(audio_features, visual_features=None)

        self.assertEqual(logits.shape[0], 4)

    def test_visual_only_fallback(self):
        """Should handle visual-only input"""
        from cognitive_intelligence.multimodal_fusion import MultimodalContextClassifier

        classifier = MultimodalContextClassifier(audio_dim=112, visual_dim=512, num_classes=4)

        visual_features = np.random.randn(20, 512).astype(np.float32)

        logits = classifier.classify(audio_features=None, visual_features=visual_features)

        self.assertEqual(logits.shape[0], 4)


class TestVisualVocalizationCorrelation(unittest.TestCase):
    """Test correlation between visual and vocalization features"""

    def test_learn_correlation(self):
        """Should learn audio-visual correlation"""
        from cognitive_intelligence.multimodal_fusion import VisualVocalizationCorrelation

        correlation = VisualVocalizationCorrelation(audio_dim=112, visual_dim=512)

        # Paired training data
        audio_features = [np.random.randn(112).astype(np.float32) for _ in range(10)]
        visual_features = [np.random.randn(512).astype(np.float32) for _ in range(10)]

        correlation.learn_correlation(audio_features, visual_features)

        # Should have learned some correlation
        self.assertGreater(correlation.correlation_strength, 0.0)

    def test_predict_visual_from_audio(self):
        """Should predict visual features from audio"""
        from cognitive_intelligence.multimodal_fusion import VisualVocalizationCorrelation

        correlation = VisualVocalizationCorrelation(audio_dim=112, visual_dim=512)

        # Train first
        audio_features = [np.random.randn(112).astype(np.float32) for _ in range(10)]
        visual_features = [np.random.randn(512).astype(np.float32) for _ in range(10)]
        correlation.learn_correlation(audio_features, visual_features)

        # Predict
        test_audio = np.random.randn(112).astype(np.float32)
        predicted_visual = correlation.predict_visual(test_audio)

        self.assertEqual(predicted_visual.shape[0], 512)

    def test_retrieve_similar_visuals(self):
        """Should retrieve similar visual contexts"""
        from cognitive_intelligence.multimodal_fusion import VisualVocalizationCorrelation

        correlation = VisualVocalizationCorrelation(audio_dim=112, visual_dim=512)

        # Build visual index
        visual_features = [np.random.randn(512).astype(np.float32) for _ in range(20)]
        contexts = ["feeding", "alarm", "contact", "feeding", "alarm"] * 4

        correlation.build_visual_index(visual_features, contexts)

        # Query
        query_audio = np.random.randn(112).astype(np.float32)
        results = correlation.retrieve_similar(query_audio, top_k=3)

        self.assertLessEqual(len(results), 3)


class TestMultimodalAttentionWeights(unittest.TestCase):
    """Test attention weights for interpretability"""

    def test_audio_attention_weights(self):
        """Should output audio attention weights"""
        from cognitive_intelligence.multimodal_fusion import AudioVisualFusion

        fusion = AudioVisualFusion(audio_dim=112, visual_dim=512, fusion_dim=256)

        audio_features = np.random.randn(10, 112).astype(np.float32)
        visual_features = np.random.randn(10, 512).astype(np.float32)

        fused, audio_attn, visual_attn = fusion.fuse_with_attention_weights(
            audio_features, visual_features
        )

        # Attention weights should sum to ~1
        self.assertAlmostEqual(np.sum(audio_attn), 1.0, places=4)
        self.assertAlmostEqual(np.sum(visual_attn), 1.0, places=4)

    def test_modality_importance(self):
        """Should compute relative importance of each modality"""
        from cognitive_intelligence.multimodal_fusion import AudioVisualFusion

        fusion = AudioVisualFusion(audio_dim=112, visual_dim=512, fusion_dim=256)

        audio_features = np.random.randn(10, 112).astype(np.float32)
        visual_features = np.random.randn(10, 512).astype(np.float32)

        importance = fusion.compute_modality_importance(audio_features, visual_features)

        self.assertIn("audio", importance)
        self.assertIn("visual", importance)
        self.assertAlmostEqual(importance["audio"] + importance["visual"], 1.0)


class TestTemporalAlignment(unittest.TestCase):
    """Test temporal alignment between audio and video"""

    def test_align_audio_to_video(self):
        """Should align audio timestamps to video frames"""
        from cognitive_intelligence.multimodal_fusion import TemporalAlignment

        aligner = TemporalAlignment(fps=30, audio_rate=48000)

        # Audio timestamps
        audio_timestamps = np.linspace(0, 2, 96000)  # 2 seconds at 48kHz

        # Get corresponding frame indices
        frame_indices = aligner.audio_to_frame_indices(audio_timestamps)

        self.assertEqual(len(frame_indices), 96000)
        self.assertGreaterEqual(frame_indices[0], 0)

    def test_sync_audio_visual_windows(self):
        """Should create synchronized audio-visual windows"""
        from cognitive_intelligence.multimodal_fusion import TemporalAlignment

        aligner = TemporalAlignment(fps=30, audio_rate=48000)

        audio_features = np.random.randn(4800, 112).astype(np.float32)  # 100ms
        visual_features = np.random.randn(3, 512).astype(np.float32)  # 3 frames

        synced = aligner.sync_windows(audio_features, visual_features)

        self.assertIn("audio", synced)
        self.assertIn("visual", synced)


if __name__ == "__main__":
    unittest.main()
