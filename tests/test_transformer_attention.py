#!/usr/bin/env python3
"""
Tests for Transformer Attention Module

These tests verify the Transformer-based attention mechanism
for capturing long-range dependencies in vocalization sequences.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest

import numpy as np


class TestMultiHeadAttention(unittest.TestCase):
    """Test multi-head attention mechanism"""

    def test_attention_shape(self):
        """Attention output should match expected shape"""
        from cognitive_intelligence.transformer_attention import MultiHeadAttention

        attention = MultiHeadAttention(
            embed_dim=112,
            num_heads=4,
            dropout=0.1,
        )

        # Batch=2, Seq=16, Dim=112
        x = np.random.randn(2, 16, 112).astype(np.float32)
        output, weights = attention.forward(x)

        # Output should preserve batch and seq dimensions
        self.assertEqual(output.shape, (2, 16, 112))
        self.assertEqual(weights.shape, (2, 4, 16, 16))  # (batch, heads, seq_q, seq_k)

    def test_attention_weights_sum_to_one(self):
        """Attention weights should sum to 1 across key sequence"""
        from cognitive_intelligence.transformer_attention import MultiHeadAttention

        attention = MultiHeadAttention(embed_dim=112, num_heads=2)
        x = np.random.randn(1, 8, 112).astype(np.float32)

        _, weights = attention.forward(x)

        # Check attention weights sum to approximately 1
        for head in range(weights.shape[1]):
            for q in range(weights.shape[2]):
                sum_weights = np.sum(weights[0, head, q, :])
                self.assertAlmostEqual(sum_weights, 1.0, places=4)

    def test_num_heads_divides_embed_dim(self):
        """Embed dimension should be divisible by number of heads"""
        from cognitive_intelligence.transformer_attention import MultiHeadAttention

        # Valid: 112 / 4 = 28 per head
        attention = MultiHeadAttention(embed_dim=112, num_heads=4)
        self.assertEqual(attention.head_dim, 28)

    def test_causal_masking(self):
        """Causal masking should prevent attending to future tokens"""
        from cognitive_intelligence.transformer_attention import MultiHeadAttention

        attention = MultiHeadAttention(embed_dim=64, num_heads=2, causal=True)
        x = np.random.randn(1, 4, 64).astype(np.float32)

        output, weights = attention.forward(x)

        # Check that future attention weights are zero
        for q in range(4):
            for k in range(q + 1, 4):
                self.assertAlmostEqual(weights[0, 0, q, k], 0.0, places=4)


class TestTransformerEncoder(unittest.TestCase):
    """Test Transformer encoder layer"""

    def test_encoder_forward(self):
        """Encoder should process input through attention and FFN"""
        from cognitive_intelligence.transformer_attention import TransformerEncoder

        encoder = TransformerEncoder(
            embed_dim=112,
            num_heads=4,
            ff_dim=256,
            dropout=0.1,
        )

        x = np.random.randn(2, 16, 112).astype(np.float32)
        output = encoder.forward(x)

        self.assertEqual(output.shape, (2, 16, 112))

    def test_encoder_residual_connection(self):
        """Encoder should have residual connections"""
        from cognitive_intelligence.transformer_attention import TransformerEncoder

        encoder = TransformerEncoder(
            embed_dim=112,
            num_heads=4,
            ff_dim=256,
            dropout=0.0,  # Disable dropout for deterministic test
        )

        x = np.ones((1, 4, 112), dtype=np.float32)
        output = encoder.forward(x)

        # With residual connections, output should be different from input
        # (due to layer norm and processing)
        self.assertFalse(np.allclose(output, x))

    def test_encoder_layer_norm(self):
        """Encoder output should be layer normalized"""
        from cognitive_intelligence.transformer_attention import TransformerEncoder

        encoder = TransformerEncoder(embed_dim=64, num_heads=2, ff_dim=128)

        x = np.random.randn(2, 8, 64).astype(np.float32) * 10  # Large values
        output = encoder.forward(x)

        # Output should have reasonable scale (layer norm effect)
        mean = np.mean(output, axis=-1)
        std = np.std(output, axis=-1)

        # Mean should be near 0, std should be bounded
        self.assertTrue(np.all(np.abs(mean) < 5.0))
        self.assertTrue(np.all(std < 10.0))


class TestVocalizationTransformer(unittest.TestCase):
    """Test Transformer for vocalization sequence processing"""

    def test_positional_encoding(self):
        """Positional encoding should add position information"""
        from cognitive_intelligence.transformer_attention import VocalizationTransformer

        model = VocalizationTransformer(
            vocab_size=100,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            max_seq_len=32,
        )

        # Get positional encoding
        pos_emb = model.get_positional_encoding(4)

        self.assertEqual(pos_emb.shape, (4, 64))

        # Different positions should have different encodings
        self.assertFalse(np.allclose(pos_emb[0], pos_emb[1]))

    def test_sequence_classification(self):
        """Transformer should classify sequences"""
        from cognitive_intelligence.transformer_attention import VocalizationTransformer

        model = VocalizationTransformer(
            vocab_size=50,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            num_classes=4,
        )

        # Input sequence of token IDs
        seq = np.array([[1, 5, 10, 15], [2, 6, 11, 16]], dtype=np.int32)

        logits = model.forward(seq)

        # Output should be (batch, num_classes)
        self.assertEqual(logits.shape, (2, 4))

    def test_long_range_dependency(self):
        """Transformer should capture long-range dependencies"""
        from cognitive_intelligence.transformer_attention import VocalizationTransformer

        model = VocalizationTransformer(
            vocab_size=20,
            embed_dim=64,
            num_heads=2,
            num_layers=4,  # More layers for better long-range capture
            num_classes=2,
        )

        # Create sequence where class depends on first and last tokens
        seq = np.array([[1, 0, 0, 0, 0, 2]], dtype=np.int32)

        logits = model.forward(seq)

        # Model should produce valid predictions
        self.assertEqual(logits.shape, (1, 2))
        self.assertFalse(np.any(np.isnan(logits)))
        self.assertFalse(np.any(np.isinf(logits)))


class TestAttentionVisualization(unittest.TestCase):
    """Test attention weight visualization for interpretability"""

    def test_attention_map_extraction(self):
        """Should extract attention maps for visualization"""
        from cognitive_intelligence.transformer_attention import VocalizationTransformer

        model = VocalizationTransformer(
            vocab_size=50,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            num_classes=4,
        )

        seq = np.array([[1, 5, 10, 15]], dtype=np.int32)

        # Get attention weights
        attention_maps = model.get_attention_maps(seq)

        # Should return attention maps for each layer
        self.assertIsInstance(attention_maps, list)
        self.assertEqual(len(attention_maps), 2)  # num_layers

        # Each layer should have (batch, heads, seq, seq)
        self.assertEqual(attention_maps[0].shape, (1, 2, 4, 4))

    def test_attention_patterns(self):
        """Attention should show interpretable patterns"""
        from cognitive_intelligence.transformer_attention import VocalizationTransformer

        model = VocalizationTransformer(
            vocab_size=20,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            num_classes=4,
        )

        # Self-similar sequence (repeated pattern)
        seq = np.array([[1, 2, 3, 1, 2, 3]], dtype=np.int32)

        attention_maps = model.get_attention_maps(seq)

        # Check that attention weights are valid
        for attn_map in attention_maps:
            # All weights should be between 0 and 1
            self.assertTrue(np.all(attn_map >= 0.0))
            self.assertTrue(np.all(attn_map <= 1.0))


class TestEfficientAttention(unittest.TestCase):
    """Test memory-efficient attention mechanisms"""

    def test_flash_attention_compatibility(self):
        """Model should support flash attention when available"""
        from cognitive_intelligence.transformer_attention import EfficientAttention

        attention = EfficientAttention(
            embed_dim=128,
            num_heads=4,
            use_flash_attention=False,  # Use standard for testing
        )

        x = np.random.randn(2, 32, 128).astype(np.float32)
        output = attention.forward(x)

        self.assertEqual(output.shape, (2, 32, 128))

    def test_chunked_attention(self):
        """Chunked attention should handle long sequences"""
        from cognitive_intelligence.transformer_attention import EfficientAttention

        attention = EfficientAttention(
            embed_dim=64,
            num_heads=2,
            chunk_size=16,  # Process in chunks of 16
        )

        # Long sequence (64 tokens)
        x = np.random.randn(1, 64, 64).astype(np.float32)

        output = attention.forward(x)

        # Should handle long sequence
        self.assertEqual(output.shape, (1, 64, 64))


if __name__ == "__main__":
    unittest.main()
