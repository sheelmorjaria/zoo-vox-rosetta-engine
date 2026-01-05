#!/usr/bin/env python3
"""
Test Suite for A/B Testing Controller with Blind Mode
Using Test-Driven Development methodology to implement:

1. A/B Testing Controller with Blind Mode
2. Statistical analysis and significance testing
3. Blind mode for unbiased evaluation
4. Multi-variant testing support
"""

import unittest
import numpy as np
import time
import threading
import json
import tempfile
import os
import random
import sys
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import hashlib

# Import all enhancement modules
sys.path.append('src')

class TestABTestingController(unittest.TestCase):
    """Test Suite for A/B Testing Controller Implementation"""

    def setUp(self):
        """Set up test fixtures for A/B testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'ab_testing_config.json')

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ab_testing_controller_creation(self):
        """Test that A/B Testing Controller can be created"""
        # Import and create controller
        from scientific_validation.ab_testing_controller import ABTestingController

        controller = ABTestingController(
            experiment_name="test_experiment",
            blind_mode=True,
            significance_threshold=0.05
        )

        self.assertIsNotNone(controller)
        self.assertEqual(controller.experiment_name, "test_experiment")
        self.assertTrue(controller.blind_mode)
        self.assertEqual(controller.significance_threshold, 0.05)

    def test_variant_creation(self):
        """Test that variants can be created and managed"""
        from scientific_validation.ab_testing_controller import ABTestingController

        controller = ABTestingController(
            experiment_name="test_experiment",
            blind_mode=True
        )

        # Add variants
        variant_a = controller.create_variant(
            variant_id="A",
            name="Control Group",
            parameters={"method": "traditional"}
        )

        variant_b = controller.create_variant(
            variant_id="B",
            name="Treatment Group",
            parameters={"method": "enhanced"}
        )

        self.assertEqual(len(controller.variants), 2)
        self.assertIn("A", controller.variants)
        self.assertIn("B", controller.variants)
        self.assertEqual(variant_a.name, "Control Group")
        self.assertEqual(variant_b.name, "Treatment Group")

    def test_participant_assignment(self):
        """Test that participants can be assigned to variants"""
        from scientific_validation.ab_testing_controller import ABTestingController

        controller = ABTestingController(
            experiment_name="test_experiment",
            blind_mode=True
        )

        # Create variants
        controller.create_variant("A", "Control", {})
        controller.create_variant("B", "Treatment", {})

        # Assign participants
        participant1_id = "user_001"
        participant2_id = "user_002"

        variant1 = controller.assign_participant(participant1_id)
        variant2 = controller.assign_participant(participant2_id)

        self.assertIn(participant1_id, controller.participant_assignments)
        self.assertIn(participant2_id, controller.participant_assignments)
        self.assertIn(variant1, ["A", "B"])
        self.assertIn(variant2, ["A", "B"])

    def test_blind_mode_functionality(self):
        """Test that blind mode hides variant information"""
        from scientific_validation.ab_testing_controller import ABTestingController

        controller = ABTestingController(
            experiment_name="test_experiment",
            blind_mode=True
        )

        # Create variants with different parameters
        controller.create_variant("A", "Control", {"method": "traditional"})
        controller.create_variant("B", "Treatment", {"method": "enhanced"})

        # Assign participant
        variant_id = controller.assign_participant("user_001")

        # In blind mode, participant should not see variant details
        participant_info = controller.get_participant_info("user_001")

        self.assertEqual(participant_info["assigned_variant"], variant_id)
        # In blind mode, name should be generic (either "Variant A" or "Variant B")
        self.assertIn(participant_info["variant_name"], ["Variant A", "Variant B"])
        # Parameters should be hidden in blind mode
        self.assertNotIn("parameters", participant_info)

    def test_statistical_significance_testing(self):
        """Test statistical significance calculation"""
        from scientific_validation.ab_testing_controller import ABTestingController

        controller = ABTestingController(
            experiment_name="test_experiment",
            significance_threshold=0.05
        )

        # Add variants
        controller.create_variant("A", "Control", {})
        controller.create_variant("B", "Treatment", {})

        # Assign participants
        controller.assign_participant("user_001")
        controller.assign_participant("user_002")
        controller.assign_participant("user_003")
        controller.assign_participant("user_004")

        # Record some results
        for i in range(100):
            controller.record_result("user_001", success=True)
            controller.record_result("user_002", success=False)

        for i in range(120):
            controller.record_result("user_003", success=True)
            controller.record_result("user_004", success=False)

        # Calculate significance
        significance = controller.calculate_significance("A", "B")

        self.assertIsInstance(significance, float)
        self.assertGreaterEqual(significance, 0.0)
        self.assertLessEqual(significance, 1.0)

    def test_experiment_results_export(self):
        """Test that experiment results can be exported"""
        from scientific_validation.ab_testing_controller import ABTestingController

        controller = ABTestingController(
            experiment_name="test_experiment"
        )

        # Add variants and record results
        controller.create_variant("A", "Control", {})
        controller.create_variant("B", "Treatment", {})

        # Assign participants first
        controller.assign_participant("user_001")
        controller.assign_participant("user_002")
        controller.assign_participant("user_003")
        controller.assign_participant("user_004")

        controller.record_result("user_001", success=True)
        controller.record_result("user_002", success=False)
        controller.record_result("user_003", success=True)
        controller.record_result("user_004", success=True)

        # Export results
        results = controller.export_results()

        self.assertIn("experiment_name", results)
        self.assertIn("variants", results)
        self.assertIn("significance_tests", results)
        self.assertIn("summary_stats", results)

    def test_multiple_variant_support(self):
        """Test support for more than two variants"""
        from scientific_validation.ab_testing_controller import ABTestingController

        controller = ABTestingController(
            experiment_name="multi_variant_test",
            blind_mode=True
        )

        # Create multiple variants
        variants = ["A", "B", "C", "D"]
        for variant_id in variants:
            controller.create_variant(variant_id, f"Variant {variant_id}", {})

        self.assertEqual(len(controller.variants), 4)

        # Assign participants to all variants
        participants = [f"user_{i:03d}" for i in range(10)]
        for participant in participants:
            variant_id = controller.assign_participant(participant)
            self.assertIn(variant_id, variants)

    def test_real_time_monitoring(self):
        """Test real-time monitoring of experiment progress"""
        from scientific_validation.ab_testing_controller import ABTestingController

        controller = ABTestingController(
            experiment_name="real_time_test"
        )

        # Add variants
        controller.create_variant("A", "Control", {})
        controller.create_variant("B", "Treatment", {})

        # Assign and record results in real-time
        for i in range(50):
            user_a = f"user_A_{i}"
            user_b = f"user_B_{i}"
            controller.assign_participant(user_a)
            controller.assign_participant(user_b)
            controller.record_result(user_a, success=i % 2 == 0)
            controller.record_result(user_b, success=i % 3 == 0)
            time.sleep(0.001)  # Small delay to simulate real-time

        # Get stats
        stats = controller.get_experiment_stats()

        self.assertIn("total_participants", stats)
        self.assertIn("success_rates", stats)
        self.assertIn("completion_rates", stats)
        self.assertTrue(stats["total_participants"] > 0)

if __name__ == '__main__':
    import sys
    sys.path.append('src/scientific_validation')
    unittest.main()