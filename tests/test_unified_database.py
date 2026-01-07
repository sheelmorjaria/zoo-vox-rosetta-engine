#!/usr/bin/env python3
"""
Test Suite for Unified Database System
======================================

Tests all database functionality including:
1. SQLite operations
2. File-based caching
3. Cloud synchronization
4. Backup and recovery
5. Integration with existing systems

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import asyncio
import json
import os
import pickle
import shutil

# Add parent directory to path
import sys
import tempfile
import unittest
from datetime import datetime

import numpy as np

sys.path.append("/mnt/c/Users/sheel/Desktop/src")

# Import database modules
try:
    from realtime.unified_database import (
        CloudSync,
        DatabaseBackup,
        DatabaseConfig,
        FileBasedCache,
        SQLiteDatabase,
        UnifiedDatabaseManager,
    )

    DATABASE_AVAILABLE = True
except ImportError as e:
    DATABASE_AVAILABLE = False
    print(f"Database modules not available: {e}")


@unittest.skipIf(not DATABASE_AVAILABLE, "Database modules not available")
class TestUnifiedDatabase(unittest.TestCase):
    """Test suite for unified database system"""

    def setUp(self):
        """Set up test environment"""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.cache_path = os.path.join(self.temp_dir, "cache")
        self.backup_path = os.path.join(self.temp_dir, "backups")

        # Create test config
        self.config = DatabaseConfig(
            sqlite_path=self.db_path,
            cache_path=self.cache_path,
            backup_path=self.backup_path,
            max_cache_size=100,
            cache_ttl=3600,
            enable_cloud_sync=False,  # Disable cloud sync for testing
            auto_backup=False,
        )

        # Create database manager
        self.db_manager = UnifiedDatabaseManager(self.config)

    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, "db_manager"):
            asyncio.run(self.db_manager.stop())

        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sqlite_connection(self):
        """Test SQLite database connection"""
        self.assertTrue(self.db_manager.sqlite_db.connect())

        # Test basic query
        result = self.db_manager.sqlite_db.execute_query("SELECT 1")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 1)

    def test_cache_operations(self):
        """Test cache operations"""
        test_data = {"test": "value", "number": 42}

        # Test set and get
        self.db_manager.cache.set("test_key", test_data)
        retrieved = self.db_manager.cache.get("test_key")
        self.assertEqual(retrieved, test_data)

        # Test delete
        self.db_manager.cache.delete("test_key")
        self.assertIsNone(self.db_manager.cache.get("test_key"))

        # Test clear
        self.db_manager.cache.set("key1", "value1")
        self.db_manager.cache.set("key2", "value2")
        self.assertEqual(self.db_manager.cache.size(), 2)

        self.db_manager.cache.clear()
        self.assertEqual(self.db_manager.cache.size(), 0)

    def test_decision_logging(self):
        """Test decision logging to database"""
        decision_record = {
            "session_id": "test_session_001",
            "timestamp": datetime.now().isoformat(),
            "input_features": {"f0": 7400, "duration": 0.1},
            "context_probabilities": {"contact": 0.8, "alarm": 0.2},
            "phrase_selection": {"phrase_key": "F0_7400"},
            "synthesis_method": "microharmonic",
            "output_audio": [0.1, 0.2, 0.3] * 100,
            "processing_time_ms": 45.6,
            "adaptation_parameters": {"learning_rate": 0.01},
            "safety_applied": True,
            "cognitive_context": None,
            "visual_context": None,
            "experimental_conditions": {},
        }

        # Log decision
        self.db_manager.log_decision(decision_record)

        # Verify it was logged
        result = self.db_manager.sqlite_db.execute_query(
            "SELECT * FROM decisions WHERE session_id = ?", (decision_record["session_id"],)
        )
        self.assertEqual(len(result), 1)

        # Check stored data
        row = result[0]
        self.assertEqual(row["session_id"], decision_record["session_id"])
        self.assertEqual(row["synthesis_method"], decision_record["synthesis_method"])
        self.assertEqual(json.loads(row["input_features"]), decision_record["input_features"])

    def test_phrase_database_operations(self):
        """Test phrase database operations"""
        phrase_key = "test_phrase_001"
        species = "marmoset"
        audio_data = np.array([0.1, 0.2, 0.3])
        acoustic_features = {"f0": 5000, "duration": 0.1}
        context_info = {"context": "contact", "probability": 0.9}

        # Save phrase to database
        self.db_manager.save_phrase_to_cache(
            phrase_key=phrase_key,
            species=species,
            audio_data=pickle.dumps(audio_data),
            acoustic_features=acoustic_features,
            context_info=context_info,
        )

        # Load phrase from database
        loaded_phrase = self.db_manager.load_phrase_database(phrase_key, species)

        self.assertIsNotNone(loaded_phrase)
        self.assertEqual(loaded_phrase["acoustic_features"], acoustic_features)
        self.assertEqual(loaded_phrase["context_info"], context_info)
        np.testing.assert_array_equal(pickle.loads(loaded_phrase["audio_data"]), audio_data)

    def test_experiments_management(self):
        """Test experiment data management"""
        experiment_data = {
            "experiment_id": "exp_test_001",
            "name": "Test Experiment",
            "description": "A test experiment",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "conditions": {"param1": "value1"},
            "metrics": {"accuracy": 0.95},
            "status": "active",
        }

        # Save experiment
        self.db_manager.save_experiment_data(experiment_data)

        # Load experiment
        loaded = self.db_manager.get_experiment_data("exp_test_001")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["experiment_id"], experiment_data["experiment_id"])
        self.assertEqual(loaded["name"], experiment_data["name"])

    def test_backup_and_recovery(self):
        """Test backup and recovery functionality"""
        # Create some data
        decision_record = {
            "session_id": "backup_test_001",
            "timestamp": datetime.now().isoformat(),
            "input_features": {"backup": True},
            "context_probabilities": {},
            "phrase_selection": {},
            "synthesis_method": "backup_test",
            "output_audio": [],
            "processing_time_ms": 0.0,
            "adaptation_parameters": {},
            "safety_applied": False,
            "cognitive_context": None,
            "visual_context": None,
            "experimental_conditions": {},
        }
        self.db_manager.log_decision(decision_record)

        # Create backup
        backup_file = self.db_manager.create_backup()
        self.assertIsNotNone(backup_file)
        self.assertTrue(os.path.exists(backup_file))

        # Get backup list
        backups = self.db_manager.backup.get_backup_list()
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0]["name"], os.path.basename(backup_file))

    def test_database_stats(self):
        """Test database statistics"""
        # Get initial stats
        stats = self.db_manager.get_database_stats()
        self.assertIn("database_size_bytes", stats)
        self.assertIn("cache_size", stats)
        self.assertIn("cache_max_size", stats)
        self.assertIn("decisions_count", stats)
        self.assertEqual(stats["cache_size"], 0)

    def test_concurrent_operations(self):
        """Test concurrent database operations"""
        import threading
        import time

        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    decision_record = {
                        "session_id": f"concurrent_{worker_id}_{i}",
                        "timestamp": datetime.now().isoformat(),
                        "input_features": {"worker": worker_id, "iteration": i},
                        "context_probabilities": {},
                        "phrase_selection": {},
                        "synthesis_method": "concurrent_test",
                        "output_audio": [],
                        "processing_time_ms": 0.0,
                        "adaptation_parameters": {},
                        "safety_applied": False,
                        "cognitive_context": None,
                        "visual_context": None,
                        "experimental_conditions": {},
                    }
                    self.db_manager.log_decision(decision_record)
                    time.sleep(0.01)
                results.append(True)
            except Exception as e:
                errors.append(str(e))

        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for threads to complete
        for t in threads:
            t.join()

        # Check results
        self.assertEqual(len(results), 5)
        self.assertEqual(len(errors), 0)

        # Verify data was logged
        stats = self.db_manager.get_database_stats()
        self.assertGreater(stats["decisions_count"], 0)


class TestDatabaseIntegration(unittest.TestCase):
    """Test database integration with other systems"""

    def test_edge_computing_framework_integration(self):
        """Test EdgeComputingFramework integration with database"""
        if not DATABASE_AVAILABLE:
            self.skipTest("Database modules not available")

        # Create temporary database
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_edge.db")

            # Create database config
            config = {
                "sqlite_path": db_path,
                "cache_path": os.path.join(temp_dir, "cache"),
                "enable_cloud_sync": False,
                "auto_backup": False,
            }

            # Test EdgeComputingFramework initialization with database
            try:
                from realtime.advanced_technical_enhancements import EdgeComputingFramework

                framework = EdgeComputingFramework(
                    model_path=None,  # Use default models
                    device="cpu",
                    database_config=config,
                )

                # Verify database manager was initialized
                self.assertIsNotNone(framework.db_manager)

                # Test sync functionality
                framework.sync_queue.append(
                    {
                        "id": "test_sync_001",
                        "timestamp": datetime.now(),
                        "results": {"test": True},
                        "metadata": {"type": "test"},
                    }
                )

                # Sync to database
                framework.sync_when_connected()

                # Verify sync was processed
                stats = framework.db_manager.get_database_stats()
                self.assertGreater(stats["decisions_count"], 0)

            except ImportError:
                self.skipTest("EdgeComputingFramework not available")

    def test_dual_path_analyzer_integration(self):
        """Test DualPathAnalyzer integration with database"""
        if not DATABASE_AVAILABLE:
            self.skipTest("Database modules not available")

        # Create temporary database
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_dual.db")

            # Create database config
            config = {
                "sqlite_path": db_path,
                "cache_path": os.path.join(temp_dir, "cache"),
                "enable_cloud_sync": False,
                "auto_backup": False,
            }

            # Create database manager
            from realtime.unified_database import DatabaseConfig, UnifiedDatabaseManager

            db_config = DatabaseConfig(**config)
            db_manager = UnifiedDatabaseManager(db_config)

            # Test DualPathAnalyzer initialization with database
            try:
                from realtime.dual_path_analyzer import DualPathAnalyzer

                analyzer = DualPathAnalyzer(sr=44100, db_manager=db_manager)

                # Verify database manager was passed
                self.assertIsNotNone(analyzer.fast_path.db_manager)
                self.assertIsNotNone(analyzer.slow_path.db_manager)

                # Test pre-canned response loading
                responses = analyzer.fast_path._load_pre_canned_responses()
                self.assertIsInstance(responses, dict)

            except ImportError:
                self.skipTest("DualPathAnalyzer not available")


def run_tests():
    """Run all database tests"""
    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(unittest.makeSuite(TestUnifiedDatabase))
    suite.addTest(unittest.makeSuite(TestDatabaseIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests
    success = run_tests()

    if success:
        print("\n✅ All database tests passed!")
    else:
        print("\n❌ Some database tests failed!")
        sys.exit(1)
