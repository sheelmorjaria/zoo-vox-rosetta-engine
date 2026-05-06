#!/usr/bin/env python3
"""
Tests for State Space Model (Mamba-style)

These tests verify the SSM implementation for efficient
long-range modeling in neural boundary detection.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest

import numpy as np


class TestSelectiveSSM(unittest.TestCase):
    """Test selective state space model core"""

    def test_ssm_forward(self):
        """SSM forward pass should produce correct output shape"""
        from cognitive_intelligence.state_space_model import SelectiveSSM

        ssm = SelectiveSSM(
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        x = np.random.randn(2, 100, 64).astype(np.float32)
        output = ssm.forward(x)

        # Output should preserve input shape
        self.assertEqual(output.shape, (2, 100, 64))

    def test_ssm_state_propagation(self):
        """Hidden state should propagate across sequence"""
        from cognitive_intelligence.state_space_model import SelectiveSSM

        ssm = SelectiveSSM(d_model=32, d_state=8, d_conv=4, expand=2)

        # Input with dependency across time
        x = np.zeros((1, 10, 32), dtype=np.float32)
        x[0, 0, 0] = 1.0  # Set first time step

        output = ssm.forward(x)

        # Later time steps should be affected by earlier input
        self.assertFalse(np.allclose(output[0, -1], output[0, 0]))

    def test_selective_mechanism(self):
        """Selective mechanism should modulate based on input"""
        from cognitive_intelligence.state_space_model import SelectiveSSM

        ssm = SelectiveSSM(d_model=32, d_state=8, d_conv=4, expand=2)

        # Two different inputs
        x1 = np.ones((1, 10, 32), dtype=np.float32)
        x2 = np.ones((1, 10, 32), dtype=np.float32) * 2.0

        out1 = ssm.forward(x1)
        out2 = ssm.forward(x2)

        # Different inputs should produce different outputs
        self.assertFalse(np.allclose(out1, out2))


class TestMambaBlock(unittest.TestCase):
    """Test Mamba block with conv and SSM"""

    def test_mamba_block_forward(self):
        """Mamba block should process input correctly"""
        from cognitive_intelligence.state_space_model import MambaBlock

        block = MambaBlock(
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        x = np.random.randn(2, 50, 64).astype(np.float32)
        output = block.forward(x)

        self.assertEqual(output.shape, (2, 50, 64))

    def test_conv1d_processing(self):
        """Conv1d should capture local patterns"""
        from cognitive_intelligence.state_space_model import MambaBlock

        block = MambaBlock(d_model=32, d_state=8, d_conv=4, expand=2)

        # Input with local pattern
        x = np.zeros((1, 20, 32), dtype=np.float32)
        x[0, 5:10, :] = 1.0  # Localized pattern

        output = block.forward(x)

        # Output should reflect the pattern
        self.assertGreater(np.sum(np.abs(output)), 0.0)

    def test_residual_connection(self):
        """Mamba block should have residual connection"""
        from cognitive_intelligence.state_space_model import MambaBlock

        block = MambaBlock(d_model=32, d_state=8, d_conv=4, expand=2)

        x = np.random.randn(1, 10, 32).astype(np.float32)
        output = block.forward(x)

        # Output should be different from input (processing applied)
        # but similar scale (residual connection)
        input_norm = np.linalg.norm(x)
        output_norm = np.linalg.norm(output)

        # Norms should be in similar range
        ratio = output_norm / (input_norm + 1e-6)
        self.assertGreater(ratio, 0.1)
        self.assertLess(ratio, 10.0)


class TestMambaBoundaryDetector(unittest.TestCase):
    """Test Mamba-based boundary detector"""

    def test_boundary_detection(self):
        """Mamba detector should find boundaries in sequences"""
        from cognitive_intelligence.state_space_model import MambaBoundaryDetector

        detector = MambaBoundaryDetector(
            input_dim=112,
            d_model=64,
            d_state=16,
            n_layers=2,
        )

        # Input sequence with potential boundary
        x = np.random.randn(1, 100, 112).astype(np.float32)
        boundaries = detector.detect_boundaries(x)

        # Should output boundary probabilities
        self.assertEqual(boundaries.shape[0], 1)  # Batch
        self.assertLessEqual(boundaries.shape[1], 100)  # Seq length (may differ)

    def test_detector_confidence_scores(self):
        """Detector should output confidence scores"""
        from cognitive_intelligence.state_space_model import MambaBoundaryDetector

        detector = MambaBoundaryDetector(
            input_dim=112,
            d_model=64,
            d_state=16,
            n_layers=2,
        )

        x = np.random.randn(1, 50, 112).astype(np.float32)

        boundaries, confidence = detector.detect_with_confidence(x)

        # Confidence should be between 0 and 1
        self.assertTrue(np.all(confidence >= 0.0))
        self.assertTrue(np.all(confidence <= 1.0))

    def test_long_sequence_efficiency(self):
        """Mamba should handle long sequences efficiently"""
        from cognitive_intelligence.state_space_model import MambaBoundaryDetector

        detector = MambaBoundaryDetector(
            input_dim=112,
            d_model=64,
            d_state=16,
            n_layers=2,
        )

        # Long sequence (1000 steps)
        x = np.random.randn(1, 1000, 112).astype(np.float32)
        boundaries = detector.detect_boundaries(x)

        # Should process without error
        self.assertEqual(boundaries.shape[0], 1)


class TestSSMConfig(unittest.TestCase):
    """Test SSM configuration"""

    def test_default_config(self):
        """Default config should have reasonable values"""
        from cognitive_intelligence.state_space_model import SSMConfig

        config = SSMConfig()

        self.assertEqual(config.d_model, 64)
        self.assertEqual(config.d_state, 16)
        self.assertEqual(config.d_conv, 4)
        self.assertEqual(config.expand, 2)

    def test_custom_config(self):
        """Should accept custom configuration"""
        from cognitive_intelligence.state_space_model import SSMConfig

        config = SSMConfig(
            d_model=128,
            d_state=32,
            d_conv=8,
            expand=4,
        )

        self.assertEqual(config.d_model, 128)
        self.assertEqual(config.d_state, 32)
        self.assertEqual(config.d_conv, 8)
        self.assertEqual(config.expand, 4)


class TestEfficiencyMetrics(unittest.TestCase):
    """Test computational efficiency metrics"""

    def test_linear_complexity(self):
        """SSM should have linear complexity in sequence length"""
        from cognitive_intelligence.state_space_model import MambaBoundaryDetector

        detector = MambaBoundaryDetector(
            input_dim=112,
            d_model=64,
            d_state=16,
            n_layers=2,
        )

        import time

        # Short sequence
        x_short = np.random.randn(1, 100, 112).astype(np.float32)
        start = time.time()
        detector.detect_boundaries(x_short)
        time_short = time.time() - start

        # Long sequence (10x longer)
        x_long = np.random.randn(1, 1000, 112).astype(np.float32)
        start = time.time()
        detector.detect_boundaries(x_long)
        time_long = time.time() - start

        # Long sequence should not take 10x time (linear vs quadratic)
        # Allow some overhead but should be significantly less than 10x
        ratio = time_long / (time_short + 1e-6)
        self.assertLess(ratio, 15.0, "SSM should have near-linear complexity")

    def test_memory_efficiency(self):
        """SSM should be memory efficient for long sequences"""
        from cognitive_intelligence.state_space_model import MambaBoundaryDetector

        detector = MambaBoundaryDetector(
            input_dim=112,
            d_model=64,
            d_state=16,
            n_layers=2,
        )

        # This should not cause memory issues
        x = np.random.randn(4, 5000, 112).astype(np.float32)
        boundaries = detector.detect_boundaries(x)

        self.assertEqual(boundaries.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
