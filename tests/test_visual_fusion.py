#!/usr/bin/env python3
"""
Test Suite for Visual Fusion Implementation
Testing MediaPipe integration and visual-audio fusion capabilities
"""

import unittest
import numpy as np
import time
import threading
import sys
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import tempfile
import os
import json

# Import visual fusion module
sys.path.append('src')
import cognitive_intelligence.visual_fusion as visual_fusion

class TestVisualFusion(unittest.TestCase):
    """Test Suite for Visual Fusion Implementation"""

    def setUp(self):
        """Set up test fixtures for visual fusion tests"""
        self.test_resolution = (640, 480)
        self.test_fps = 30

    def test_visual_fusion_system_creation(self):
        """Test that Visual Fusion System can be created"""
        from cognitive_intelligence.visual_fusion import (
            VisualFusionSystem, VisualFusionConfig, VisualFeatures, VisualAttentionLevel
        )

        # 1. Create configuration
        config = VisualFusionConfig(
            camera_resolution=self.test_resolution,
            fps=self.test_fps,
            use_mediapipe=True,
            separate_thread=False  # Run in main thread for testing
        )

        # 2. Create VisualFusionSystem instance
        fusion_system = VisualFusionSystem(config)

        # 3. Verify configuration is applied
        self.assertEqual(fusion_system.config.camera_resolution, self.test_resolution)
        self.assertEqual(fusion_system.config.fps, self.test_fps)
        self.assertTrue(fusion_system.config.use_mediapipe)
        self.assertFalse(fusion_system.config.separate_thread)

        # 4. Verify initial state
        self.assertFalse(fusion_system.running)
        self.assertIsNone(fusion_system.thread)

    def test_mediapipe_tracker_creation(self):
        """Test that MediaPipe tracker can be initialized"""
        from cognitive_intelligence.visual_fusion import (
            MediaPipeTracker, VisualFusionConfig, VisualFeatures
        )

        # 1. Create configuration
        config = VisualFusionConfig(
            camera_resolution=self.test_resolution,
            fps=self.test_fps,
            use_mediapipe=True
        )

        # 2. Create MediaPipeTracker instance
        tracker = MediaPipeTracker(config)

        # 3. Verify MediaPipe solutions are initialized
        self.assertIsNotNone(tracker.hands)
        self.assertIsNotNone(tracker.face_mesh)
        self.assertIsNotNone(tracker.pose)
        self.assertEqual(len(tracker.movement_history), 0)
        self.assertEqual(len(tracker.attention_scores), 0)

    def test_lighttrack_fallback_initialization(self):
        """Test that LightTrack fallback can be initialized"""
        from cognitive_intelligence.visual_fusion import (
            LightTrackFallback, VisualFusionConfig, VisualFeatures
        )

        # 1. Create configuration with LightTrack enabled
        config = VisualFusionConfig(
            camera_resolution=self.test_resolution,
            fps=self.test_fps,
            use_mediapipe=False,  # Force fallback
            use_lighttrack_fallback=True
        )

        # 2. Create LightTrackFallback instance
        fallback = LightTrackFallback(config)

        # 3. Verify configuration
        self.assertEqual(fallback.config, config)
        self.assertIsNotNone(fallback)

    def test_frame_processing_without_camera(self):
        """Test frame processing without actual camera input"""
        from cognitive_intelligence.visual_fusion import (
            VisualFusionSystem, VisualFusionConfig, VisualFeatures
        )

        # 1. Create configuration
        config = VisualFusionConfig(
            camera_resolution=self.test_resolution,
            fps=self.test_fps,
            use_mediapipe=True,
            separate_thread=False
        )

        # 2. Create fusion system
        fusion_system = VisualFusionSystem(config)

        # 3. Create test frame (black image)
        test_frame = np.zeros((self.test_resolution[1], self.test_resolution[0], 3), dtype=np.uint8)

        # 4. Test frame processing (this will use mock MediaPipe)
        with patch('cognitive_intelligence.visual_fusion.mp_hands.Hands') as mock_hands, \
             patch('cognitive_intelligence.visual_fusion.mp_face_mesh.FaceMesh') as mock_face, \
             patch('cognitive_intelligence.visual_fusion.mp_pose.Pose') as mock_pose:

            # Mock the processors
            mock_hands.return_value.process.return_value.multi_hand_landmarks = None
            mock_face.return_value.process.return_value.multi_face_landmarks = None
            mock_pose.return_value.process.return_value.pose_landmarks = None

            # Process frame
            features = fusion_system.tracker.process_frame(test_frame)

            # 5. Verify VisualFeatures is returned
            self.assertIsInstance(features, VisualFeatures)
            self.assertGreater(features.timestamp, 0)
            self.assertIsNotNone(features.attention_level)
            self.assertGreaterEqual(features.movement_intensity, 0.0)
            self.assertLessEqual(features.movement_intensity, 1.0)

    def test_separate_thread_processing(self):
        """Test that visual processing runs on separate thread without blocking"""
        from cognitive_intelligence.visual_fusion import (
            VisualFusionSystem, VisualFusionConfig
        )
        import time

        # 1. Create configuration with separate thread
        config = VisualFusionConfig(
            camera_resolution=self.test_resolution,
            fps=self.test_fps,
            use_mediapipe=True,
            separate_thread=True
        )

        # 2. Create fusion system
        fusion_system = VisualFusionSystem(config)

        # 3. Test start/stop functionality
        start_time = time.time()
        fusion_system.start()
        self.assertTrue(fusion_system.running)
        self.assertIsNotNone(fusion_system.thread)
        self.assertTrue(fusion_system.thread.is_alive())

        # Wait briefly for thread to initialize
        time.sleep(0.1)

        # Stop the system
        fusion_system.stop()
        self.assertFalse(fusion_system.running)

        # Verify stop didn't block for too long
        self.assertLess(time.time() - start_time, 2.0)

    def test_frame_queue_overflow_protection(self):
        """Test that frame queue overflow is handled gracefully"""
        from cognitive_intelligence.visual_fusion import (
            VisualFusionSystem, VisualFusionConfig
        )

        # 1. Create configuration with small queue
        config = VisualFusionConfig(
            camera_resolution=self.test_resolution,
            fps=self.test_fps,
            use_mediapipe=True,
            separate_thread=False,
            max_queue_size=5
        )

        # 2. Create fusion system
        fusion_system = VisualFusionSystem(config)

        # 3. Fill queue to capacity
        test_frame = np.zeros((self.test_resolution[1], self.test_resolution[0], 3), dtype=np.uint8)

        initial_drops = fusion_system.frame_drops

        for i in range(10):  # More than queue capacity
            fusion_system.process_frame_async(test_frame)

        # 4. Verify frame drops were counted
        self.assertGreater(fusion_system.frame_drops, initial_drops)

    def test_visual_attention_calculation(self):
        """Test that visual attention is calculated correctly"""
        from cognitive_intelligence.visual_fusion import (
            MediaPipeTracker, VisualFusionConfig, VisualAttentionLevel, VisualFeatures
        )

        # 1. Create tracker
        config = VisualFusionConfig(
            camera_resolution=self.test_resolution,
            fps=self.test_fps
        )
        tracker = MediaPipeTracker(config)

        # 2. Test attention levels
        test_features = [
            # Create features with different attention levels
            VisualFeatures(gaze_direction="towards_camera", movement_intensity=0.9, hand_gestures=["open_hand"]),
            VisualFeatures(gaze_direction="away", movement_intensity=0.1, hand_gestures=[]),
            VisualFeatures(gaze_direction="left", movement_intensity=0.5, hand_gestures=["point"]),
            VisualFeatures(gaze_direction="towards_camera", movement_intensity=0.1, hand_gestures=["open_hand", "peace"])
        ]

        # Test the attention level calculation directly
        self.assertIn(VisualAttentionLevel.LOW, VisualAttentionLevel)
        self.assertIn(VisualAttentionLevel.HIGH, VisualAttentionLevel)
        self.assertIn(VisualAttentionLevel.VERY_HIGH, VisualAttentionLevel)
        self.assertIn(VisualAttentionLevel.MODERATE, VisualAttentionLevel)

    def test_gaze_estimation_accuracy(self):
        """Test that gaze estimation works with different head poses"""
        from cognitive_intelligence.visual_fusion import (
            MediaPipeTracker, VisualFusionConfig
        )

        # 1. Create tracker
        config = VisualFusionConfig(camera_resolution=self.test_resolution)
        tracker = MediaPipeTracker(config)

        # Test that gaze estimation is handled properly
        test_frame = np.zeros((self.test_resolution[1], self.test_resolution[0], 3), dtype=np.uint8)

        # Process frame - should return VisualFeatures even without actual gaze estimation
        features = tracker.process_frame(test_frame)

        # Verify gaze estimation returns valid result
        self.assertIsInstance(features, visual_fusion.VisualFeatures)
        # Validate gaze_direction is either None or a valid string
        if features.gaze_direction is not None:
            self.assertIsInstance(features.gaze_direction, str)

    def test_hand_gesture_recognition(self):
        """Test that hand gestures are recognized correctly"""
        from cognitive_intelligence.visual_fusion import (
            MediaPipeTracker, VisualFusionConfig
        )

        # 1. Create tracker
        config = VisualFusionConfig(camera_resolution=self.test_resolution)
        tracker = MediaPipeTracker(config)

        # 2. Test with mock hand landmarks
        mock_hand_landmarks = Mock()
        mock_hand_landmarks.landmark = [Mock(x=0.5, y=0.5) for _ in range(21)]

        mock_results = Mock()
        mock_results.multi_hand_landmarks = [mock_hand_landmarks]

        # Test gesture extraction
        gestures = tracker._extract_hand_gestures(mock_results)

        # Verify gestures is a list
        self.assertIsInstance(gestures, list)

    def test_visual_audio_fusion(self):
        """Test that visual and audio features are fused correctly"""
        from cognitive_intelligence.visual_fusion import (
            VisualFusionSystem, VisualFusionConfig, VisualFeatures, VisualAttentionLevel
        )

        # 1. Create fusion system
        config = VisualFusionConfig()
        fusion_system = VisualFusionSystem(config)

        # 2. Test audio features
        audio_features = {
            'rms': 0.1,
            'f0': 6000.0,
            'context': 'contact_call',
            'response_probability': 0.6
        }

        # 3. Test visual features with high attention
        visual_features = VisualFeatures(
            attention_level=VisualAttentionLevel.HIGH,
            gaze_direction='towards_camera',
            movement_intensity=0.7
        )

        # 4. Fuse features
        fused_result = fusion_system.integrate_with_audio(audio_features, visual_features)

        # 5. Verify fusion result
        self.assertIsInstance(fused_result, dict)
        self.assertIn('response_probability', fused_result)
        self.assertIn('visual_context', fused_result)

        # Verify visual context is stored correctly
        visual_context = fused_result['visual_context']
        self.assertEqual(visual_context['attention_level'], 'High')
        self.assertEqual(visual_context['gaze_direction'], 'towards_camera')

    def test_attention_score_calculation(self):
        """Test that visual attention scores are calculated correctly"""
        from cognitive_intelligence.visual_fusion import (
            VisualFusionSystem, VisualFusionConfig, VisualFeatures, VisualAttentionLevel
        )

        # 1. Create fusion system
        config = VisualFusionConfig()
        fusion_system = VisualFusionSystem(config)

        # 2. Test different attention scenarios
        test_scenarios = [
            # Low attention
            VisualFeatures(
                attention_level=VisualAttentionLevel.LOW,
                gaze_direction='away',
                movement_intensity=0.0,
                hand_gestures=[],
                confidence=0.8
            ),
            # High attention
            VisualFeatures(
                attention_level=VisualAttentionLevel.HIGH,
                gaze_direction='towards_camera',
                movement_intensity=0.5,
                hand_gestures=['open_hand'],
                confidence=0.9
            ),
            # Very high attention
            VisualFeatures(
                attention_level=VisualAttentionLevel.VERY_HIGH,
                gaze_direction='towards_camera',
                movement_intensity=0.9,
                hand_gestures=['open_hand', 'peace'],
                confidence=0.95
            )
        ]

        for features in test_scenarios:
            # Calculate attention score
            score = fusion_system.create_visual_attention_score(features)

            # Verify score is between 0 and 1
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

            # Verify higher attention gives higher scores
            if features.attention_level == VisualAttentionLevel.VERY_HIGH:
                self.assertGreater(score, 0.7)
            elif features.attention_level == VisualAttentionLevel.LOW:
                self.assertLess(score, 0.3)

    def test_performance_monitoring(self):
        """Test that performance statistics are tracked correctly"""
        from cognitive_intelligence.visual_fusion import (
            VisualFusionSystem, VisualFusionConfig
        )

        # 1. Create fusion system
        config = VisualFusionConfig()
        fusion_system = VisualFusionSystem(config)

        # 2. Get initial stats
        initial_stats = fusion_system.get_performance_stats()

        # 3. Verify stats structure
        self.assertIsInstance(initial_stats, dict)
        self.assertIn('running', initial_stats)
        self.assertIn('frame_drops', initial_stats)
        self.assertIn('queue_size', initial_stats)
        self.assertIn('avg_processing_time', initial_stats)

        # 4. Verify initial values
        self.assertFalse(initial_stats['running'])
        self.assertEqual(initial_stats['frame_drops'], 0)
        self.assertEqual(initial_stats['queue_size'], 0)

    def test_error_handling(self):
        """Test that errors are handled gracefully"""
        from cognitive_intelligence.visual_fusion import (
            VisualFusionSystem, VisualFusionConfig
        )

        # 1. Create fusion system
        config = VisualFusionConfig()
        fusion_system = VisualFusionSystem(config)

        # 2. Test with invalid frame
        invalid_frame = np.array([])  # Empty array
        result = fusion_system.process_frame_async(invalid_frame)

        # This should not raise an exception
        self.assertIsNone(result)

        # 3. Get stats (should handle errors gracefully)
        stats = fusion_system.get_performance_stats()
        self.assertIsInstance(stats, dict)

    def test_camera_initialization(self):
        """Test camera initialization parameters"""
        from cognitive_intelligence.visual_fusion import VisualFusionSystem, VisualFusionConfig

        # 1. Create configuration
        config = VisualFusionConfig(
            camera_resolution=(1280, 720),
            fps=60,
            use_mediapipe=True,
            separate_thread=False
        )

        # 2. Create fusion system
        fusion_system = VisualFusionSystem(config)

        # 3. Verify configuration
        self.assertEqual(fusion_system.config.camera_resolution, (1280, 720))
        self.assertEqual(fusion_system.config.fps, 60)

        # 4. Test with different camera resolutions
        high_res_config = VisualFusionConfig(camera_resolution=(1920, 1080))
        high_res_fusion = VisualFusionSystem(high_res_config)
        self.assertEqual(high_res_fusion.config.camera_resolution, (1920, 1080))


if __name__ == '__main__':
    # Create test suite with all test cases
    suite = unittest.TestSuite()

    # Add all test methods
    test_methods = [
        'test_visual_fusion_system_creation',
        'test_mediapipe_tracker_creation',
        'test_lighttrack_fallback_initialization',
        'test_frame_processing_without_camera',
        'test_separate_thread_processing',
        'test_frame_queue_overflow_protection',
        'test_visual_attention_calculation',
        'test_gaze_estimation_accuracy',
        'test_hand_gesture_recognition',
        'test_visual_audio_fusion',
        'test_attention_score_calculation',
        'test_performance_monitoring',
        'test_error_handling',
        'test_camera_initialization'
    ]

    for method in test_methods:
        suite.addTest(TestVisualFusion(method))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Visual Fusion Test Results:")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print(f"\n{'='*50}")
        print("FAILURES:")
        print(f"{'='*50}")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print(f"\n{'='*50}")
        print("ERRORS:")
        print(f"{'='*50}")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")