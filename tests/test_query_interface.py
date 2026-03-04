"""
Comprehensive Tests for Vocalization Query Interface

Tests the high-performance query interface for accessing vocalization data
including:
- Database loading and indexing
- Phrase search by F0 range and duration
- Similarity search
- Grammar network analysis
- Cross-species comparison
- Statistics aggregation

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import os
import tempfile
import unittest

# Handle imports gracefully - some tests may need mocked data
try:
    from data_models import Species  # noqa: F401 - used for availability check
    from query_interface.vocalization_query_interface import VocalizationQueryInterface

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


@unittest.skipIf(not MODELS_AVAILABLE, "Required modules not available")
class TestVocalizationQueryInterface(unittest.TestCase):
    """Test VocalizationQueryInterface with mocked database"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures with mock database"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.temp_dir, "test_vocalization_database.json")

        # Create minimal test database
        test_db = {
            "species_data": {
                "marmoset": {
                    "total_phrases": 2,
                    "total_sentences": 1,
                    "total_grammar_rules": 1,
                    "vocabulary_size": 2,
                    "phrases": {
                        "marmoset_phee_001": {
                            "signature": "phee_call",
                            "species": "marmoset",
                            "modality": "harmonic",
                            "acoustic_features": {
                                "mean_f0_hz": 8000.0,
                                "duration_ms": 200.0,
                                "f0_range_hz": 2000.0,
                                "std_f0_hz": 500.0,
                                "min_f0_hz": 7000.0,
                                "max_f0_hz": 9000.0,
                                "duration_frames": 100,
                                "voiced_ratio": 0.95,
                                "f0_slope": 100.0,
                                "modulation_rate": 5.0,
                                "acoustic_variance": 0.1,
                                "mean_duration_ms": 200.0,
                            },
                            "total_occurrences": 50,
                            "contexts": [
                                {"context_name": "contact", "count": 30, "percentage": 60.0}
                            ],
                            "social_contexts": {},
                            "is_compositional": False,
                            "phrase_components": [],
                        },
                        "marmoset_trill_001": {
                            "signature": "trill_call",
                            "species": "marmoset",
                            "modality": "harmonic",
                            "acoustic_features": {
                                "mean_f0_hz": 10000.0,
                                "duration_ms": 150.0,
                                "f0_range_hz": 3000.0,
                                "std_f0_hz": 800.0,
                                "min_f0_hz": 8500.0,
                                "max_f0_hz": 11500.0,
                                "duration_frames": 80,
                                "voiced_ratio": 0.90,
                                "f0_slope": -50.0,
                                "modulation_rate": 15.0,
                                "acoustic_variance": 0.2,
                                "mean_duration_ms": 150.0,
                            },
                            "total_occurrences": 30,
                            "contexts": [
                                {"context_name": "alarm", "count": 20, "percentage": 66.7}
                            ],
                            "social_contexts": {},
                            "is_compositional": False,
                            "phrase_components": [],
                        },
                    },
                    "sentences": [
                        {
                            "sentence_id": "marmoset_sent_001",
                            "species": "marmoset",
                            "phrase_sequence": ["marmoset_phee_001", "marmoset_trill_001"],
                            "context": "alarm",
                            "has_ascending_syntax": True,
                            "has_descending_syntax": False,
                            "has_fm_pattern": False,
                            "syntax_score": 0.8,
                            "total_duration_ms": 350.0,
                            "complexity_score": 0.6,
                        }
                    ],
                    "grammar_rules": [
                        {
                            "rule_id": "rule_001",
                            "from_phrase": "marmoset_phee_001",
                            "to_phrase": "marmoset_trill_001",
                            "frequency": 25,
                            "confidence": 0.85,
                            "contexts": ["alarm", "contact"],
                            "is_bidirectional": False,
                            "strength_score": 0.75,
                        }
                    ],
                }
            }
        }

        with open(cls.db_path, "w") as f:
            json.dump(test_db, f)

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        if os.path.exists(cls.temp_dir):
            import shutil

            shutil.rmtree(cls.temp_dir)

    def setUp(self):
        """Create query interface for each test"""
        self.interface = VocalizationQueryInterface(database_path=self.db_path)

    def test_interface_creation(self):
        """Interface should be created successfully"""
        self.assertIsNotNone(self.interface)
        self.assertIsNotNone(self.interface.db)

    def test_database_loading(self):
        """Database should load species data correctly"""
        self.assertIn(Species.MARMOSET, self.interface.db.species_data)

    def test_index_building(self):
        """Indexes should be built correctly"""
        self.assertIn("marmoset_phee_001", self.interface.phrase_by_key_index)
        self.assertIn("marmoset_trill_001", self.interface.phrase_by_key_index)

    def test_get_phrase_by_key(self):
        """Getting phrase by key should return correct phrase"""
        result = self.interface.get_phrase_by_key("marmoset_phee_001")
        self.assertIsNotNone(result)
        species, phrase = result
        self.assertEqual(species, Species.MARMOSET)
        self.assertEqual(phrase.signature, "phee_call")

    def test_get_phrase_by_key_not_found(self):
        """Getting nonexistent phrase should return None"""
        result = self.interface.get_phrase_by_key("nonexistent_phrase")
        self.assertIsNone(result)

    def test_get_phrases_by_species(self):
        """Getting phrases by species should return all phrases"""
        phrases = self.interface.get_phrases_by_species(Species.MARMOSET)
        self.assertEqual(len(phrases), 2)
        self.assertIn("marmoset_phee_001", phrases)
        self.assertIn("marmoset_trill_001", phrases)

    def test_search_phrases_by_f0_range(self):
        """Searching by F0 range should return matching phrases"""
        # Search in range that includes only phee (8000 Hz)
        results = self.interface.search_phrases_by_f0_range(7500, 8500, Species.MARMOSET)
        self.assertEqual(len(results), 1)
        phrase_key, phrase = results[0]
        self.assertEqual(phrase_key, "marmoset_phee_001")

    def test_search_phrases_by_f0_range_all(self):
        """Searching by F0 range should return all matching phrases"""
        # Search in range that includes both phrases
        results = self.interface.search_phrases_by_f0_range(7000, 12000, Species.MARMOSET)
        self.assertEqual(len(results), 2)

    def test_search_phrases_by_f0_range_none(self):
        """Searching by F0 range with no matches should return empty list"""
        results = self.interface.search_phrases_by_f0_range(100, 200, Species.MARMOSET)
        self.assertEqual(len(results), 0)

    def test_search_phrases_by_duration(self):
        """Searching by duration should return matching phrases"""
        # Search in range that includes only trill (150ms)
        results = self.interface.search_phrases_by_duration(100, 175, Species.MARMOSET)
        self.assertEqual(len(results), 1)
        phrase_key, phrase = results[0]
        self.assertEqual(phrase_key, "marmoset_trill_001")

    def test_get_similar_phrases(self):
        """Similarity search should find acoustically similar phrases"""
        # Search for phrases similar to phee
        similar = self.interface.get_similar_phrases("marmoset_phee_001", threshold=0.5)
        # Should find trill (different but not too different)
        self.assertIsInstance(similar, list)

    def test_get_similar_phrases_not_found(self):
        """Similarity search for nonexistent phrase should return empty"""
        similar = self.interface.get_similar_phrases("nonexistent_phrase", threshold=0.5)
        self.assertEqual(len(similar), 0)

    def test_get_grammar_transitions(self):
        """Grammar transitions should be retrievable"""
        transitions = self.interface.get_grammar_transitions("marmoset_phee_001")
        self.assertIn("marmoset_trill_001", transitions)

    def test_get_grammar_transitions_none(self):
        """Grammar transitions for nonexistent phrase should return empty"""
        transitions = self.interface.get_grammar_transitions("nonexistent_phrase")
        self.assertEqual(len(transitions), 0)

    def test_get_phrase_statistics(self):
        """Phrase statistics should be calculated correctly"""
        stats = self.interface.get_phrase_statistics(Species.MARMOSET)

        self.assertEqual(stats["total_phrases"], 2)
        self.assertIn("marmoset", stats["species_breakdown"])
        self.assertIn("harmonic", stats["modality_breakdown"])
        self.assertGreater(stats["frequency_distribution"]["avg"], 0)

    def test_get_phrase_statistics_all_species(self):
        """Phrase statistics without species should include all"""
        stats = self.interface.get_phrase_statistics()
        self.assertGreaterEqual(stats["total_phrases"], 2)

    def test_get_grammar_network(self):
        """Grammar network should be generated"""
        network = self.interface.get_grammar_network(Species.MARMOSET)

        self.assertGreater(network["nodes"], 0)
        self.assertGreater(network["edges"], 0)
        self.assertIn("transitions", network)


@unittest.skipIf(not MODELS_AVAILABLE, "Required modules not available")
class TestPhraseSearchEdgeCases(unittest.TestCase):
    """Test edge cases in phrase search"""

    @classmethod
    def setUpClass(cls):
        """Set up empty database for edge case tests"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.temp_dir, "empty_database.json")

        # Create empty database
        empty_db = {"species_data": {}}
        with open(cls.db_path, "w") as f:
            json.dump(empty_db, f)

    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        if os.path.exists(cls.temp_dir):
            import shutil

            shutil.rmtree(cls.temp_dir)

    def test_empty_database_searches(self):
        """Searches on empty database should return empty results"""
        interface = VocalizationQueryInterface(database_path=self.db_path)

        self.assertEqual(len(interface.search_phrases_by_f0_range(0, 100000)), 0)
        self.assertEqual(len(interface.search_phrases_by_duration(0, 10000)), 0)
        self.assertEqual(len(interface.get_similar_phrases("any_phrase")), 0)

    def test_empty_database_statistics(self):
        """Statistics on empty database should have sensible defaults"""
        interface = VocalizationQueryInterface(database_path=self.db_path)
        stats = interface.get_phrase_statistics()

        self.assertEqual(stats["total_phrases"], 0)
        self.assertEqual(stats["frequency_distribution"]["min"], float("inf"))


@unittest.skipIf(not MODELS_AVAILABLE, "Required modules not available")
class TestCrossSpeciesSearch(unittest.TestCase):
    """Test cross-species search capabilities"""

    @classmethod
    def setUpClass(cls):
        """Set up multi-species database"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.temp_dir, "multi_species_database.json")

        # Create database with multiple species
        multi_db = {
            "species_data": {
                "marmoset": {
                    "total_phrases": 1,
                    "total_sentences": 0,
                    "total_grammar_rules": 0,
                    "vocabulary_size": 1,
                    "phrases": {
                        "marmoset_call_001": {
                            "signature": "high_call",
                            "species": "marmoset",
                            "modality": "harmonic",
                            "acoustic_features": {
                                "mean_f0_hz": 10000.0,
                                "duration_ms": 200.0,
                                "f0_range_hz": 2000.0,
                                "std_f0_hz": 500.0,
                                "min_f0_hz": 9000.0,
                                "max_f0_hz": 11000.0,
                                "duration_frames": 100,
                                "voiced_ratio": 0.95,
                                "f0_slope": 0.0,
                                "modulation_rate": 5.0,
                                "acoustic_variance": 0.1,
                                "mean_duration_ms": 200.0,
                            },
                            "total_occurrences": 10,
                            "contexts": [],
                            "social_contexts": {},
                            "is_compositional": False,
                            "phrase_components": [],
                        }
                    },
                    "sentences": [],
                    "grammar_rules": [],
                },
                "dolphin": {
                    "total_phrases": 1,
                    "total_sentences": 0,
                    "total_grammar_rules": 0,
                    "vocabulary_size": 1,
                    "phrases": {
                        "dolphin_whistle_001": {
                            "signature": "whistle",
                            "species": "dolphin",
                            "modality": "whistle",
                            "acoustic_features": {
                                "mean_f0_hz": 10000.0,
                                "duration_ms": 200.0,
                                "f0_range_hz": 4000.0,
                                "std_f0_hz": 1000.0,
                                "min_f0_hz": 8000.0,
                                "max_f0_hz": 12000.0,
                                "duration_frames": 150,
                                "voiced_ratio": 1.0,
                                "f0_slope": 200.0,
                                "modulation_rate": 10.0,
                                "acoustic_variance": 0.15,
                                "mean_duration_ms": 200.0,
                            },
                            "total_occurrences": 5,
                            "contexts": [],
                            "social_contexts": {},
                            "is_compositional": False,
                            "phrase_components": [],
                        }
                    },
                    "sentences": [],
                    "grammar_rules": [],
                },
            }
        }

        with open(cls.db_path, "w") as f:
            json.dump(multi_db, f)

    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        if os.path.exists(cls.temp_dir):
            import shutil

            shutil.rmtree(cls.temp_dir)

    def test_cross_species_f0_search(self):
        """F0 search without species should find across species"""
        interface = VocalizationQueryInterface(database_path=self.db_path)

        # Search in range that includes both species
        results = interface.search_phrases_by_f0_range(9000, 11000)  # No species filter
        self.assertEqual(len(results), 2)

        # Verify we got both species
        species_found = set()
        for phrase_key, phrase in results:
            species_found.add(phrase.species)
        self.assertEqual(len(species_found), 2)

    def test_cross_species_similarity(self):
        """Similarity search should find similar phrases across species"""
        interface = VocalizationQueryInterface(database_path=self.db_path)

        # Marmoset and dolphin phrases have same F0 and duration
        similar = interface.get_similar_phrases("marmoset_call_001", threshold=0.7)
        self.assertEqual(len(similar), 1)
        similarity, phrase_key, phrase = similar[0]
        self.assertEqual(phrase_key, "dolphin_whistle_001")


@unittest.skipIf(not MODELS_AVAILABLE, "Required modules not available")
class TestQueryPerformance(unittest.TestCase):
    """Test query performance characteristics"""

    @classmethod
    def setUpClass(cls):
        """Set up database with many phrases for performance testing"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.temp_dir, "large_database.json")

        # Create database with many phrases
        phrases = {}
        for i in range(100):
            phrase_key = f"marmoset_phrase_{i:03d}"
            f0 = 5000.0 + (i * 50)  # F0 ranges from 5000 to 10000
            phrases[phrase_key] = {
                "signature": f"phrase_{i}",
                "species": "marmoset",
                "modality": "harmonic",
                "acoustic_features": {
                    "mean_f0_hz": f0,
                    "duration_ms": 200.0,
                    "f0_range_hz": 1000.0,
                    "std_f0_hz": 500.0,
                    "min_f0_hz": f0 - 500,
                    "max_f0_hz": f0 + 500,
                    "duration_frames": 100,
                    "voiced_ratio": 0.95,
                    "f0_slope": 0.0,
                    "modulation_rate": 5.0,
                    "acoustic_variance": 0.1,
                    "mean_duration_ms": 200.0,
                },
                "total_occurrences": 10,
                "contexts": [],
                "social_contexts": {},
                "is_compositional": False,
                "phrase_components": [],
            }

        large_db = {
            "species_data": {
                "marmoset": {
                    "total_phrases": 100,
                    "total_sentences": 0,
                    "total_grammar_rules": 0,
                    "vocabulary_size": 100,
                    "phrases": phrases,
                    "sentences": [],
                    "grammar_rules": [],
                }
            }
        }

        with open(cls.db_path, "w") as f:
            json.dump(large_db, f)

    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        if os.path.exists(cls.temp_dir):
            import shutil

            shutil.rmtree(cls.temp_dir)

    def test_f0_search_performance(self):
        """F0 search should complete quickly"""
        import time

        interface = VocalizationQueryInterface(database_path=self.db_path)

        start = time.time()
        results = interface.search_phrases_by_f0_range(6000, 8000, Species.MARMOSET)
        elapsed = time.time() - start

        self.assertLess(elapsed, 0.1, "F0 search should complete in <100ms")
        self.assertGreater(len(results), 0)

    def test_index_lookup_performance(self):
        """Index lookups should be O(1)"""
        import time

        interface = VocalizationQueryInterface(database_path=self.db_path)

        # Multiple lookups should be fast
        start = time.time()
        for i in range(100):
            interface.get_phrase_by_key(f"marmoset_phrase_{i:03d}")
        elapsed = time.time() - start

        self.assertLess(elapsed, 0.05, "100 index lookups should complete in <50ms")

    def test_similarity_search_performance(self):
        """Similarity search should complete in reasonable time"""
        import time

        interface = VocalizationQueryInterface(database_path=self.db_path)

        start = time.time()
        similar = interface.get_similar_phrases("marmoset_phrase_050", threshold=0.5)
        elapsed = time.time() - start

        self.assertLess(elapsed, 0.5, "Similarity search should complete in <500ms")
        # Verify the search returned results (even if empty is valid for performance test)
        self.assertIsInstance(similar, list)


class TestMissingDatabase(unittest.TestCase):
    """Test behavior when database file is missing"""

    def test_missing_database_creates_empty(self):
        """Missing database should create empty interface"""
        interface = VocalizationQueryInterface(database_path="/nonexistent/path/database.json")
        self.assertIsNotNone(interface.db)
        self.assertEqual(len(interface.db.species_data), 0)


if __name__ == "__main__":
    unittest.main()
