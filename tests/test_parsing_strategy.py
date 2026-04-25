#!/usr/bin/env python3
"""
TDD Tests for Parsing Strategy Module
=====================================

Implements test cases from the TDD plan:
- Sprint 1.1: Strategy Selection
- Sprint 1.2: Bat-Specific Idiom Detection
- Sprint 1.3: Context Integration

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime.parsing_strategy import (
    CompositionalStrategy,
    HolophrasticStrategy,
    ParsedToken,
    ParseResult,
    ParsingStrategyFactory,
    TokenType,
)


class TestTokenType(unittest.TestCase):
    """Test TokenType enum"""

    def test_token_types_exist(self):
        """Verify all token types are defined"""
        self.assertEqual(TokenType.COMPOSITIONAL.value, "compositional")
        self.assertEqual(TokenType.IDIOM.value, "idiom")
        self.assertEqual(TokenType.CONTENT.value, "content")
        self.assertEqual(TokenType.NOISE.value, "noise")
        self.assertEqual(TokenType.OPENER.value, "opener")
        self.assertEqual(TokenType.CLOSER.value, "closer")


class TestParsedToken(unittest.TestCase):
    """Test ParsedToken dataclass"""

    def test_create_compositional_token(self):
        """Test creating a compositional token"""
        token = ParsedToken(
            token_type=TokenType.COMPOSITIONAL,
            segments=[114],
            meaning="test_meaning",
            confidence=0.9,
            position=0,
        )

        self.assertEqual(token.token_type, TokenType.COMPOSITIONAL)
        self.assertEqual(token.segments, [114])
        self.assertEqual(token.meaning, "test_meaning")
        self.assertEqual(token.confidence, 0.9)
        self.assertEqual(token.position, 0)

    def test_create_idiom_token(self):
        """Test creating an idiom token with multiple segments"""
        idiom = [114, 464, 604, 324, 94, 714]
        token = ParsedToken(
            token_type=TokenType.IDIOM,
            segments=idiom,
            meaning="LRN-6_IDIOM",
            confidence=0.98,
            position=0,
        )

        self.assertEqual(len(token.segments), 6)
        self.assertEqual(token.meaning, "LRN-6_IDIOM")


class TestParseResult(unittest.TestCase):
    """Test ParseResult dataclass"""

    def test_create_parse_result(self):
        """Test creating a parse result"""
        tokens = [
            ParsedToken(token_type=TokenType.COMPOSITIONAL, segments=[1], position=0),
            ParsedToken(token_type=TokenType.COMPOSITIONAL, segments=[2], position=1),
        ]

        result = ParseResult(
            tokens=tokens,
            original_sequence=[1, 2],
            strategy_used="compositional",
            compositional_count=2,
        )

        self.assertEqual(len(result.tokens), 2)
        self.assertEqual(result.original_sequence, [1, 2])
        self.assertEqual(result.strategy_used, "compositional")


class TestCompositionalStrategy(unittest.TestCase):
    """Sprint 1.1: Strategy Selection - General Mode"""

    def setUp(self):
        """Set up test fixtures"""
        self.strategy = CompositionalStrategy()

    def test_strategy_name(self):
        """Test Case 1.1.0: Strategy name is correct"""
        self.assertEqual(self.strategy.name, "compositional")

    def test_is_not_holophrastic(self):
        """Test Case 1.1.0: Compositional is not holophrastic"""
        self.assertFalse(self.strategy.is_holophrastic)

    def test_general_mode_fallback(self):
        """Test Case 1.1.1: General Mode Fallback

        Verify that compositional strategy treats each segment as independent.
        """
        result = self.strategy.parse([114, 464])

        self.assertEqual(len(result.tokens), 2)
        self.assertEqual(result.tokens[0].token_type, TokenType.COMPOSITIONAL)
        self.assertEqual(result.tokens[1].token_type, TokenType.COMPOSITIONAL)
        self.assertEqual(result.compositional_count, 2)

    def test_empty_sequence(self):
        """Test parsing empty sequence"""
        result = self.strategy.parse([])

        self.assertEqual(len(result.tokens), 0)
        self.assertEqual(result.original_sequence, [])

    def test_single_segment(self):
        """Test parsing single segment"""
        result = self.strategy.parse([42])

        self.assertEqual(len(result.tokens), 1)
        self.assertEqual(result.tokens[0].segments, [42])
        self.assertEqual(result.tokens[0].position, 0)

    def test_with_segment_meanings(self):
        """Test compositional strategy with meaning lookup"""
        strategy = CompositionalStrategy(segment_meanings={114: "alert", 464: "query"})

        result = strategy.parse([114, 464])

        self.assertEqual(result.tokens[0].meaning, "alert")
        self.assertEqual(result.tokens[1].meaning, "query")
        self.assertEqual(result.tokens[0].confidence, 1.0)

    def test_unknown_segment_confidence(self):
        """Test that unknown segments have lower confidence"""
        result = self.strategy.parse([999])

        self.assertEqual(result.tokens[0].confidence, 0.5)

    def test_context_isolation(self):
        """Test Case 1.3.1: Isolation of Context Logic

        In general mode, position should not affect interpretation.
        """
        result = self.strategy.parse([384, 304])  # Opener + Closer segments

        # Both should be treated as COMPOSITIONAL, not OPENER/CLOSER
        self.assertEqual(result.tokens[0].token_type, TokenType.COMPOSITIONAL)
        self.assertEqual(result.tokens[1].token_type, TokenType.COMPOSITIONAL)


class TestHolophrasticStrategy(unittest.TestCase):
    """Sprint 1.2: Bat-Specific Idiom Detection"""

    def setUp(self):
        """Set up test fixtures"""
        self.strategy = HolophrasticStrategy()

    def test_strategy_name(self):
        """Test Case 1.2.0: Strategy name is correct"""
        self.assertEqual(self.strategy.name, "holophrastic")

    def test_is_holophrastic(self):
        """Test Case 1.2.0: Holophrastic is holophrastic"""
        self.assertTrue(self.strategy.is_holophrastic)

    def test_bat_mode_activation_lrn6(self):
        """Test Case 1.2.1: Bat Mode Activation - LRN-6 Idiom

        Verify that LRN-6 pattern is collapsed to single IDIOM token.
        """
        lrn6 = [114, 464, 604, 324, 94, 714]
        result = self.strategy.parse(lrn6)

        # Should collapse to single token
        self.assertEqual(len(result.tokens), 1)
        self.assertEqual(result.tokens[0].token_type, TokenType.IDIOM)
        self.assertEqual(result.tokens[0].segments, lrn6)
        self.assertEqual(result.tokens[0].meaning, "LRN-6_IDIOM")
        self.assertGreater(result.tokens[0].confidence, 0.9)
        self.assertEqual(result.idiom_count, 1)

    def test_lrn6_with_following_segments(self):
        """Test LRN-6 idiom followed by other segments"""
        lrn6_plus = [114, 464, 604, 324, 94, 714, 304, 394]
        result = self.strategy.parse(lrn6_plus)

        # First token should be the idiom
        self.assertEqual(result.tokens[0].token_type, TokenType.IDIOM)
        self.assertEqual(len(result.tokens[0].segments), 6)

        # Should have additional tokens for the rest
        self.assertGreater(len(result.tokens), 1)

    def test_no_partial_lrn6_match(self):
        """Test that partial LRN-6 doesn't trigger idiom detection"""
        partial = [114, 464, 604]  # First 3 of LRN-6
        result = self.strategy.parse(partial)

        # Should NOT be collapsed
        for token in result.tokens:
            self.assertNotEqual(token.token_type, TokenType.IDIOM)

    def test_opener_detection(self):
        """Test opener segment detection at position 0"""
        result = self.strategy.parse([384])  # 384 is an opener

        self.assertEqual(result.tokens[0].token_type, TokenType.OPENER)
        self.assertEqual(result.tokens[0].meaning, "OPENER_STACCATO_ALERT")
        self.assertGreater(result.tokens[0].confidence, 0.8)

    def test_closer_detection(self):
        """Test closer segment detection at position 1"""
        # Use a valid sequence: opener + closer
        result = self.strategy.parse([384, 444])  # opener + closer

        self.assertEqual(result.tokens[0].token_type, TokenType.OPENER)
        self.assertEqual(result.tokens[1].token_type, TokenType.CLOSER)

    def test_noise_classification(self):
        """Test noise classification for unknown segments"""
        result = self.strategy.parse([999])  # Unknown segment

        self.assertEqual(result.tokens[0].token_type, TokenType.NOISE)
        self.assertLess(result.tokens[0].confidence, 0.5)
        self.assertEqual(result.noise_count, 1)

    def test_valid_bigram_transition(self):
        """Test that valid bigrams are recognized"""
        # (764, 304) is a valid bigram
        transitions = self.strategy.get_valid_transitions(764)

        self.assertIn(304, transitions)

    def test_empty_sequence(self):
        """Test parsing empty sequence in holophrastic mode"""
        result = self.strategy.parse([])

        self.assertEqual(len(result.tokens), 0)
        self.assertEqual(result.idiom_count, 0)

    def test_add_custom_idiom(self):
        """Test adding custom idiom at runtime"""
        strategy = HolophrasticStrategy()

        # Add custom idiom
        strategy.add_idiom([100, 200, 300], "CUSTOM_IDIOM", confidence=0.95)

        result = strategy.parse([100, 200, 300])

        self.assertEqual(len(result.tokens), 1)
        self.assertEqual(result.tokens[0].token_type, TokenType.IDIOM)
        self.assertEqual(result.tokens[0].meaning, "CUSTOM_IDIOM")


class TestParsingStrategyFactory(unittest.TestCase):
    """Test the factory for creating strategies"""

    def test_create_general_strategy(self):
        """Test factory creates compositional strategy for general mode"""
        strategy = ParsingStrategyFactory.create(domain_mode="general")

        self.assertIsInstance(strategy, CompositionalStrategy)
        self.assertEqual(strategy.name, "compositional")

    def test_create_bat_strategy(self):
        """Test factory creates holophrastic strategy for bat mode"""
        strategy = ParsingStrategyFactory.create(domain_mode="bat")

        self.assertIsInstance(strategy, HolophrasticStrategy)
        self.assertEqual(strategy.name, "holophrastic")

    def test_create_holophrastic_alias(self):
        """Test that 'holophrastic' also creates bat strategy"""
        strategy = ParsingStrategyFactory.create(domain_mode="holophrastic")

        self.assertIsInstance(strategy, HolophrasticStrategy)

    def test_case_insensitive(self):
        """Test domain mode is case insensitive"""
        strategy1 = ParsingStrategyFactory.create(domain_mode="BAT")
        strategy2 = ParsingStrategyFactory.create(domain_mode="Bat")

        self.assertIsInstance(strategy1, HolophrasticStrategy)
        self.assertIsInstance(strategy2, HolophrasticStrategy)

    def test_custom_meanings_passed(self):
        """Test that custom meanings are passed to strategy"""
        meanings = {1: "alpha", 2: "beta"}
        strategy = ParsingStrategyFactory.create(domain_mode="general", segment_meanings=meanings)

        result = strategy.parse([1, 2])

        self.assertEqual(result.tokens[0].meaning, "alpha")
        self.assertEqual(result.tokens[1].meaning, "beta")


class TestBackwardsCompatibility(unittest.TestCase):
    """Test backwards compatibility guarantees"""

    def test_default_is_compositional(self):
        """Test that default factory creates compositional strategy"""
        strategy = ParsingStrategyFactory.create()

        self.assertIsInstance(strategy, CompositionalStrategy)

    def test_unknown_domain_defaults_to_compositional(self):
        """Test that unknown domain mode defaults to compositional"""
        strategy = ParsingStrategyFactory.create(domain_mode="unknown_species")

        self.assertIsInstance(strategy, CompositionalStrategy)

    def test_original_behavior_preserved(self):
        """Test Case 1.1.1: Original behavior preserved

        Verify that compositional strategy produces same results
        as before the Strategy Pattern was introduced.
        """
        strategy = CompositionalStrategy()

        # Test with typical sequence
        sequence = [114, 464, 604, 324, 94, 714]  # LRN-6
        result = strategy.parse(sequence)

        # Should NOT collapse to single token (original behavior)
        self.assertEqual(len(result.tokens), 6)
        for token in result.tokens:
            self.assertEqual(token.token_type, TokenType.COMPOSITIONAL)


class TestValidBigrams(unittest.TestCase):
    """Test the valid bigram constraints from Phase 2 research"""

    def test_valid_bigrams_count(self):
        """Test that we have some valid bigrams defined"""
        strategy = HolophrasticStrategy()

        # Phase 2 found 50 valid bigrams (0.02% of 260,100 possible)
        self.assertGreater(len(strategy.VALID_BIGRAMS), 0)
        self.assertLess(len(strategy.VALID_BIGRAMS), 100)

    def test_valid_bigram_examples(self):
        """Test specific valid bigrams from Phase 2 research"""
        strategy = HolophrasticStrategy()

        # Top bigrams from Phase 2 analysis
        self.assertIn((764, 304), strategy.VALID_BIGRAMS)
        self.assertIn((534, 434), strategy.VALID_BIGRAMS)
        self.assertIn((304, 394), strategy.VALID_BIGRAMS)

    def test_invalid_bigram_not_in_set(self):
        """Test that random bigrams are not valid"""
        strategy = HolophrasticStrategy()

        # Random segment pair should not be valid
        self.assertNotIn((999, 888), strategy.VALID_BIGRAMS)


class TestPositionSpecialists(unittest.TestCase):
    """Test opener/closer position specialists from Phase 3 research"""

    def test_openers_defined(self):
        """Test that openers are defined"""
        strategy = HolophrasticStrategy()

        self.assertIn(384, strategy.OPENERS)
        self.assertIn(264, strategy.OPENERS)

    def test_closers_defined(self):
        """Test that closers are defined"""
        strategy = HolophrasticStrategy()

        self.assertIn(444, strategy.CLOSERS)
        self.assertIn(304, strategy.CLOSERS)

    def test_opener_only_at_position_0(self):
        """Test that opener segment at position 0 is classified as OPENER"""
        strategy = HolophrasticStrategy()

        # 384 should be OPENER at position 0
        result = strategy.parse([384])
        self.assertEqual(result.tokens[0].token_type, TokenType.OPENER)

    def test_opener_not_opener_at_other_positions(self):
        """Test that opener at non-zero position is not OPENER"""
        strategy = HolophrasticStrategy()

        # 384 at position 1 (after another segment) should not be OPENER
        result = strategy.parse([100, 384])

        # First token might be content/noise
        # Second token (384) should not be OPENER since not at position 0
        self.assertNotEqual(result.tokens[1].token_type, TokenType.OPENER)


if __name__ == "__main__":
    unittest.main(verbosity=2)
