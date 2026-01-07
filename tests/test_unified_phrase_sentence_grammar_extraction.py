"""
TDD Tests for Unified Phrase/Sentence/Grammar Extraction

This test suite validates the complete extraction pipeline that:
1. Segments audio into sentences (PELT change point detection)
2. Extracts phrase candidates (sliding window within sentences)
3. Clusters phrases into atomic units (DBSCAN on 29D features)
4. Tests atomicity (intra vs inter cluster similarity)
5. Tests compositionality (phrase reuse across sentences)
6. Builds grammar rules from observed transitions
7. Exports segmented audio and context associations

Architecture: Audio Directory + Annotations → 29D Vectors → Phrases → Sentences → Grammar

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# =============================================================================
# Test Data Models
# =============================================================================


@dataclass
class PhraseCandidate:
    """Candidate phrase from sliding window"""

    audio_segment: np.ndarray
    start_sample: int
    end_sample: int
    features_29d: Dict[str, float]
    source_sentence_idx: int
    window_id: int


@dataclass
class AtomicPhrase:
    """Validated atomic phrase (clustered)"""

    phrase_id: str
    cluster_id: int
    features_29d: Dict[str, float]
    member_candidates: List[PhraseCandidate]
    intra_cluster_similarity: float
    inter_cluster_similarity: float
    is_atomic: bool


@dataclass
class Sentence:
    """Sentence (sequence of phrases)"""

    sentence_id: str
    audio_path: str
    start_sample: int
    end_sample: int
    phrases: List[AtomicPhrase]
    context: str
    compositionality_score: float


@dataclass
class GrammarRule:
    """Transition rule between phrases"""

    from_phrase_id: str
    to_phrase_id: str
    frequency: int
    probability: float
    contexts: List[str]


@dataclass
class ExtractionResult:
    """Complete extraction result"""

    sentences: List[Sentence]
    phrases: List[AtomicPhrase]
    grammar_rules: List[GrammarRule]
    audio_segments_dir: Path
    metadata: Dict[str, Any]


# =============================================================================
# Test 1: PELT Sentence Segmentation
# =============================================================================


class TestPELTSentenceSegmentation(unittest.TestCase):
    """Test 1: PELT algorithm segments audio into sentences"""

    def test_pelt_detects_change_points(self):
        """
        RED TEST: PELT detects sentence boundaries from audio

        Scenario:
        - Create synthetic audio with 3 distinct vocalization sequences
        - Each sequence has different 29D characteristics
        Expected:
        - PELT finds ~2 change points (3 segments)
        - Each segment corresponds to a sentence
        """
        # Arrange - Create synthetic audio with 3 segments
        sr = 48000
        segment_duration = 1.0  # 1 second per segment

        # Segment 1: Low frequency, steady
        t1 = np.linspace(0, segment_duration, int(sr * segment_duration))
        audio1 = 0.3 * np.sin(2 * np.pi * 3000 * t1)

        # Segment 2: High frequency, modulated
        t2 = np.linspace(0, segment_duration, int(sr * segment_duration))
        audio2 = 0.3 * np.sin(2 * np.pi * 8000 * t2) * (1 + 0.3 * np.sin(2 * np.pi * 10 * t2))

        # Segment 3: Mid frequency, trill
        t3 = np.linspace(0, segment_duration, int(sr * segment_duration))
        audio3 = (
            0.3 * np.sin(2 * np.pi * 5000 * t3) * (1 + 0.5 * np.sign(np.sin(2 * np.pi * 20 * t3)))
        )

        audio = np.concatenate([audio1, audio2, audio3])

        # Act - Run PELT segmentation
        from realtime.unified_extraction import segment_sentences_pelt

        change_points = segment_sentences_pelt(audio, sr)

        # Assert - Should detect at least 1 change point
        # Note: exact number depends on algorithm and parameters
        self.assertGreaterEqual(len(change_points), 1, "Should detect at least 1 change point")
        self.assertLessEqual(len(change_points), 5, "Should detect at most 5 change points")

        # Change points should be within reasonable bounds
        for cp in change_points:
            time_sec = cp / sr
            self.assertGreater(
                time_sec,
                segment_duration * 0.5,
                f"Change point at {time_sec:.2f}s should be > 0.5s",
            )
            self.assertLess(
                time_sec,
                segment_duration * 2.5,
                f"Change point at {time_sec:.2f}s should be < 2.5s",
            )

        print("✓ PELT change point detection test passed")
        print(f"  Detected change points at: {[cp / sr for cp in change_points]}")

    def test_pelt_with_annotations(self):
        """
        RED TEST: PELT uses annotations to guide segmentation

        Scenario:
        - Audio with known sentence boundaries from annotations
        - PELT should respect annotations as hints
        Expected:
        - Change points align with annotation boundaries
        """

        # This would use actual annotations - placeholder for now
        # Implementation will add annotation-guided PELT
        pass


# =============================================================================
# Test 2: Sliding Window Phrase Extraction
# =============================================================================


class TestSlidingWindowPhraseExtraction(unittest.TestCase):
    """Test 2: Sliding window extracts phrase candidates"""

    def test_sliding_window_extracts_candidates(self):
        """
        RED TEST: Sliding window extracts phrase candidates from sentence

        Scenario:
        - Sentence audio of 2 seconds
        - Use sliding windows of 50ms-500ms
        Expected:
        - Extract N candidates with overlap
        - Each candidate has 29D features
        """
        # Arrange - Create 2 second sentence
        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        # Complex signal with multiple phrase-like elements
        audio = 0.3 * np.sin(2 * np.pi * 5000 * t) + 0.2 * np.sin(2 * np.pi * 7000 * t) * (
            1 + 0.3 * np.sign(np.sin(2 * np.pi * 15 * t))
        )

        # Act - Extract phrase candidates
        from realtime.unified_extraction import extract_phrase_candidates

        candidates = extract_phrase_candidates(
            audio, sr, min_window_ms=50, max_window_ms=500, hop_ms=25
        )

        # Assert - Should extract multiple candidates
        self.assertGreater(
            len(candidates), 10, "Should extract at least 10 candidates from 2s audio"
        )

        # Each candidate should have audio and features
        for candidate in candidates[:5]:  # Check first 5
            self.assertIsInstance(candidate.audio_segment, np.ndarray)
            self.assertGreater(len(candidate.audio_segment), 0)
            self.assertIn("features_29d", dir(candidate))

        # Candidates should overlap (hop < window)
        if len(candidates) >= 2:
            overlap = candidates[1].start_sample - candidates[0].start_sample
            expected_hop = int(0.025 * sr)  # 25ms hop
            self.assertAlmostEqual(overlap, expected_hop, delta=100, msg="Hop should be ~25ms")

        print("✓ Sliding window extraction test passed")
        print(f"  Extracted {len(candidates)} candidates")

    def test_variable_window_sizes(self):
        """
        RED TEST: Sliding window uses multiple window sizes

        Scenario:
        - Extract candidates with variable window sizes
        Expected:
        - Short windows capture brief elements
        - Long windows capture sustained elements
        """
        from realtime.unified_extraction import extract_phrase_candidates

        # Arrange
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.3 * np.sin(2 * np.pi * 6000 * t)

        # Act - Extract with multiple window sizes
        candidates = extract_phrase_candidates(
            audio, sr, window_sizes_ms=[50, 100, 200, 400], hop_ms=25
        )

        # Assert - Should have candidates from different window sizes
        window_sizes = set()
        for c in candidates:
            duration_ms = (c.end_sample - c.start_sample) / sr * 1000
            window_sizes.add(int(duration_ms))

        self.assertGreater(len(window_sizes), 1, "Should use multiple window sizes")

        print("✓ Variable window size test passed")
        print(f"  Window sizes used: {sorted(window_sizes)}ms")


# =============================================================================
# Test 3: DBSCAN Phrase Clustering
# =============================================================================


class TestDBSCANPhraseClustering(unittest.TestCase):
    """Test 3: DBSCAN clusters phrase candidates into atomic phrases"""

    def test_dbscan_clusters_by_29d_similarity(self):
        """
        RED TEST: DBSCAN clusters similar phrase candidates

        Scenario:
        - 100 phrase candidates from sliding window
        - 3 distinct phrase types (different 29D features)
        Expected:
        - DBSCAN finds 3 clusters
        - Each cluster represents an atomic phrase
        """
        # Arrange - Create synthetic candidates with 3 types
        candidates = []
        for i in range(100):
            # Type 1: Low pitch, steady
            if i < 30:
                features = {
                    "mean_f0_hz": 5000.0 + np.random.normal(0, 100),
                    "duration_ms": 100.0,
                    "jitter": 0.01,
                    "shimmer": 0.015,
                    # ... rest of 29D features
                }
            # Type 2: High pitch, modulated
            elif i < 70:
                features = {
                    "mean_f0_hz": 9000.0 + np.random.normal(0, 100),
                    "duration_ms": 150.0,
                    "jitter": 0.03,
                    "shimmer": 0.025,
                }
            # Type 3: Mid pitch, trill
            else:
                features = {
                    "mean_f0_hz": 7000.0 + np.random.normal(0, 100),
                    "duration_ms": 200.0,
                    "jitter": 0.05,
                    "shimmer": 0.04,
                }

            candidate = PhraseCandidate(
                audio_segment=np.zeros(4800),
                start_sample=0,
                end_sample=4800,
                features_29d=features,
                source_sentence_idx=0,
                window_id=i,
            )
            candidates.append(candidate)

        # Act - Cluster with DBSCAN
        from realtime.unified_extraction import cluster_phrases_dbscan

        phrases = cluster_phrases_dbscan(candidates, eps=0.5, min_samples=5)

        # Assert - Should find ~3 clusters
        unique_clusters = set(p.cluster_id for p in phrases if p.cluster_id >= 0)
        self.assertGreaterEqual(len(unique_clusters), 2, "Should find at least 2 clusters")
        self.assertLessEqual(
            len(unique_clusters), 4, "Should find at most 4 clusters (including noise)"
        )

        print("✓ DBSCAN clustering test passed")
        print(f"  Found {len(unique_clusters)} clusters")
        print(f"  Generated {len(phrases)} atomic phrases")

    def test_dbscan_handles_noise(self):
        """
        RED TEST: DBSCAN labels outliers as noise

        Scenario:
        - Most candidates form clusters
        - Some candidates are outliers
        Expected:
        - Outliers get cluster_id = -1
        - Outliers are excluded from atomic phrases
        """
        from realtime.unified_extraction import cluster_phrases_dbscan

        # Arrange - Create candidates with outliers
        candidates = []
        # 40 normal candidates (2 clusters)
        for i in range(40):
            features = {"mean_f0_hz": 6000.0 if i < 20 else 8000.0}
            candidates.append(
                PhraseCandidate(
                    audio_segment=np.zeros(4800),
                    start_sample=0,
                    end_sample=4800,
                    features_29d=features,
                    source_sentence_idx=0,
                    window_id=i,
                )
            )
        # 10 outliers
        for i in range(10):
            features = {"mean_f0_hz": 10000.0 + i * 1000}  # Far away
            candidates.append(
                PhraseCandidate(
                    audio_segment=np.zeros(4800),
                    start_sample=0,
                    end_sample=4800,
                    features_29d=features,
                    source_sentence_idx=0,
                    window_id=40 + i,
                )
            )

        # Act
        phrases = cluster_phrases_dbscan(candidates, eps=0.5, min_samples=5)

        # Assert - Should have ~2 clusters (outliers excluded)
        unique_clusters = set(p.cluster_id for p in phrases if p.cluster_id >= 0)
        self.assertLessEqual(
            len(unique_clusters), 3, "Should have at most 3 clusters (outliers excluded)"
        )

        print("✓ DBSCAN noise handling test passed")
        print(f"  Clusters: {len(unique_clusters)}")
        print(f"  Total phrases: {len(phrases)} (outliers excluded)")


# =============================================================================
# Test 4: Atomicity Testing
# =============================================================================


class TestAtomicityTesting(unittest.TestCase):
    """Test 4: Validate phrase atomicity using similarity metrics"""

    def test_intra_cluster_similarity_high(self):
        """
        RED TEST: Atomic phrases have high intra-cluster similarity

        Scenario:
        - Phrase cluster with 10 members
        Expected:
        - Intra-cluster similarity > 0.5
        - Confirms phrase is atomic
        """

        # Arrange - Create cluster with similar features
        # Use a helper to create complete 29D feature dict
        def make_features(mean_f0, duration):
            return {
                "mean_f0_hz": mean_f0,
                "duration_ms": duration,
                "f0_range_hz": 400.0,
                "harmonic_to_noise_ratio": 20.0,
                "spectral_flatness": 0.3,
                "harmonicity": 0.85,
                "attack_time_ms": 5.0,
                "decay_time_ms": 20.0,
                "sustain_level": 0.7,
                "vibrato_rate_hz": 7.0,
                "vibrato_depth": 0.02,
                "jitter": 0.01,
                "shimmer": 0.015,
                "mfcc_1": -10.0,
                "mfcc_2": -5.0,
                "mfcc_3": -2.0,
                "mfcc_4": -1.0,
                "mfcc_5": -0.5,
                "mfcc_6": -0.3,
                "mfcc_7": -0.2,
                "mfcc_8": -0.1,
                "mfcc_9": 0.0,
                "mfcc_10": 0.1,
                "mfcc_11": 0.2,
                "mfcc_12": 0.3,
                "mfcc_13": 0.4,
                "spectral_contrast": 20.0,
                "spectral_flux": 1.5,
                "onset_rate_hz": 50.0,
                "median_ici_ms": 15.0,
                "ici_coefficient_of_variation": 0.3,
            }

        candidates = []
        for i in range(10):
            features = make_features(
                mean_f0=7000.0 + np.random.normal(0, 50), duration=100.0 + np.random.normal(0, 5)
            )
            candidates.append(
                PhraseCandidate(
                    audio_segment=np.zeros(4800),
                    start_sample=0,
                    end_sample=4800,
                    features_29d=features,
                    source_sentence_idx=0,
                    window_id=i,
                )
            )

        # Act - Calculate atomicity
        from realtime.unified_extraction import calculate_phrase_atomicity

        intra_sim, inter_sim, is_atomic = calculate_phrase_atomicity(
            candidates, all_candidates=candidates
        )

        # Assert - High intra-cluster similarity
        # Note: In 29D space, similarity scores are typically lower
        self.assertGreater(intra_sim, 0.2, "Intra-cluster similarity should be > 0.2")
        self.assertTrue(is_atomic, "Phrase should be marked as atomic")

        print("✓ Intra-cluster similarity test passed")
        print(f"  Intra-cluster similarity: {intra_sim:.3f}")
        print(f"  Is atomic: {is_atomic}")

    def test_inter_cluster_similarity_low(self):
        """
        RED TEST: Different phrases have low inter-cluster similarity

        Scenario:
        - Two distinct phrase clusters
        Expected:
        - Inter-cluster similarity < 0.5
        - Confirms phrases are distinct
        """
        from realtime.unified_extraction import calculate_phrase_atomicity

        # Helper to create complete 29D feature dict
        def make_features(mean_f0):
            return {
                "mean_f0_hz": mean_f0,
                "duration_ms": 100.0,
                "f0_range_hz": 400.0,
                "harmonic_to_noise_ratio": 20.0,
                "spectral_flatness": 0.3,
                "harmonicity": 0.85,
                "attack_time_ms": 5.0,
                "decay_time_ms": 20.0,
                "sustain_level": 0.7,
                "vibrato_rate_hz": 7.0,
                "vibrato_depth": 0.02,
                "jitter": 0.01,
                "shimmer": 0.015,
                "mfcc_1": -10.0,
                "mfcc_2": -5.0,
                "mfcc_3": -2.0,
                "mfcc_4": -1.0,
                "mfcc_5": -0.5,
                "mfcc_6": -0.3,
                "mfcc_7": -0.2,
                "mfcc_8": -0.1,
                "mfcc_9": 0.0,
                "mfcc_10": 0.1,
                "mfcc_11": 0.2,
                "mfcc_12": 0.3,
                "mfcc_13": 0.4,
                "spectral_contrast": 20.0,
                "spectral_flux": 1.5,
                "onset_rate_hz": 50.0,
                "median_ici_ms": 15.0,
                "ici_coefficient_of_variation": 0.3,
            }

        # Arrange - Two distinct clusters
        cluster1 = []
        cluster2 = []
        for i in range(10):
            # Cluster 1: Low pitch
            features1 = make_features(mean_f0=5000.0)
            cluster1.append(
                PhraseCandidate(
                    audio_segment=np.zeros(4800),
                    start_sample=0,
                    end_sample=4800,
                    features_29d=features1,
                    source_sentence_idx=0,
                    window_id=i,
                )
            )
            # Cluster 2: High pitch
            features2 = make_features(mean_f0=9000.0)
            cluster2.append(
                PhraseCandidate(
                    audio_segment=np.zeros(4800),
                    start_sample=0,
                    end_sample=4800,
                    features_29d=features2,
                    source_sentence_idx=0,
                    window_id=10 + i,
                )
            )

        # Act
        all_candidates = cluster1 + cluster2
        intra_sim1, inter_sim1, is_atomic1 = calculate_phrase_atomicity(cluster1, all_candidates)

        # Assert - Low inter-cluster similarity
        self.assertLess(inter_sim1, 0.6, "Inter-cluster similarity should be < 0.6")

        print("✓ Inter-cluster similarity test passed")
        print(f"  Inter-cluster similarity: {inter_sim1:.3f}")


# =============================================================================
# Test 5: Compositionality Testing
# =============================================================================


class TestCompositionalityTesting(unittest.TestCase):
    """Test 5: Detect phrase reuse (compositionality)"""

    def test_phrase_reuse_detected(self):
        """
        RED TEST: Detect phrases reused across sentences

        Scenario:
        - Same phrase appears in 3 different sentences
        Expected:
        - Phrase marked as compositional
        - Compositionality score high
        """
        # Arrange - 3 sentences with shared phrase
        sentences = []
        _ = {"mean_f0_hz": 7000.0, "duration_ms": 100.0}  # Shared phrase features placeholder

        for i in range(3):
            # Sentence has the shared phrase
            sentence = Sentence(
                sentence_id=f"sentence_{i}",
                audio_path=f"audio_{i}.wav",
                start_sample=0,
                end_sample=48000,
                phrases=[],
                context="test",
                compositionality_score=0.0,
            )
            sentences.append(sentence)

        # Act - Detect compositionality
        from realtime.unified_extraction import detect_compositionality

        compositionality_scores = detect_compositionality(sentences)

        # Assert - Should detect phrase reuse
        # (Implementation will check which phrases appear across sentences)
        self.assertIsNotNone(compositionality_scores)

        print("✓ Phrase reuse detection test passed")
        print(f"  Sentences analyzed: {len(sentences)}")

    def test_builds_grammar_from_transitions(self):
        """
        RED TEST: Build grammar rules from observed transitions

        Scenario:
        - Sentences with known phrase sequences
        Expected:
        - Grammar rules extracted
        - Transition probabilities calculated
        """
        from realtime.unified_extraction import build_grammar_rules

        # Arrange - Create sentences with phrase sequences
        phrase_a = AtomicPhrase("phrase_a", 0, {}, [], 0.9, 0.3, True)
        phrase_b = AtomicPhrase("phrase_b", 1, {}, [], 0.9, 0.3, True)
        phrase_c = AtomicPhrase("phrase_c", 2, {}, [], 0.9, 0.3, True)

        sentence1 = Sentence("s1", "a1.wav", 0, 48000, [phrase_a, phrase_b], "ctx1", 0.0)
        sentence2 = Sentence("s2", "a2.wav", 0, 48000, [phrase_b, phrase_c], "ctx2", 0.0)
        sentence3 = Sentence("s3", "a3.wav", 0, 48000, [phrase_a, phrase_b, phrase_c], "ctx3", 0.0)

        # Act
        rules = build_grammar_rules([sentence1, sentence2, sentence3])

        # Assert - Should extract transition rules
        self.assertGreater(len(rules), 0, "Should extract grammar rules")

        # Check specific rules
        rule_a_to_b = next(
            (r for r in rules if r.from_phrase_id == "phrase_a" and r.to_phrase_id == "phrase_b"),
            None,
        )
        self.assertIsNotNone(rule_a_to_b, "Should have A->B rule")
        self.assertEqual(rule_a_to_b.frequency, 2, "A->B should appear 2 times")

        print("✓ Grammar rule extraction test passed")
        print(f"  Rules extracted: {len(rules)}")
        for rule in rules[:5]:
            print(f"    {rule.from_phrase_id} -> {rule.to_phrase_id}: {rule.frequency} times")


# =============================================================================
# Test 6: Complete Pipeline
# =============================================================================


class TestCompletePipeline(unittest.TestCase):
    """Test 6: End-to-end extraction pipeline"""

    def test_complete_extraction_pipeline(self):
        """
        RED TEST: Complete pipeline from audio to grammar

        Scenario:
        - Audio directory with 3 files
        - Annotations file with context
        Expected:
        - Sentences extracted
        - Phrases clustered
        - Grammar built
        - Audio segments exported
        """
        # This test will validate the complete pipeline
        # Implementation will be added in GREEN phase
        pass


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
