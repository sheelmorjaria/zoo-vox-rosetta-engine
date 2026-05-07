#!/usr/bin/env python3
"""
Tests for PCFG Induction - Formal Language Theory

These tests verify Probabilistic Context-Free Grammar induction
for discovering grammatical structure in animal vocalizations.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest

import numpy as np


class TestGrammarRule(unittest.TestCase):
    """Test grammar rule representation"""

    def test_rule_creation(self):
        """Should create a grammar rule with LHS and RHS"""
        from semiotics.pcfg_induction import GrammarRule

        rule = GrammarRule(lhs="S", rhs=["NP", "VP"], prob=0.8)

        self.assertEqual(rule.lhs, "S")
        self.assertEqual(rule.rhs, ["NP", "VP"])
        self.assertEqual(rule.prob, 0.8)

    def test_rule_terminal(self):
        """Should handle terminal rules"""
        from semiotics.pcfg_induction import GrammarRule

        rule = GrammarRule(lhs="NP", rhs=["word"], prob=1.0, is_terminal=True)

        self.assertTrue(rule.is_terminal)
        self.assertEqual(rule.rhs, ["word"])

    def test_rule_str_representation(self):
        """Should have string representation"""
        from semiotics.pcfg_induction import GrammarRule

        rule = GrammarRule(lhs="S", rhs=["A", "B"], prob=0.5)
        rule_str = str(rule)

        self.assertIn("S", rule_str)
        self.assertIn("A", rule_str)
        self.assertIn("B", rule_str)


class TestPCFG(unittest.TestCase):
    """Test Probabilistic Context-Free Grammar"""

    def test_grammar_creation(self):
        """Should create a PCFG with rules"""
        from semiotics.pcfg_induction import PCFG

        grammar = PCFG(start_symbol="S")

        self.assertEqual(grammar.start_symbol, "S")
        self.assertEqual(len(grammar.rules), 0)

    def test_add_rule(self):
        """Should add rules to grammar"""
        from semiotics.pcfg_induction import PCFG, GrammarRule

        grammar = PCFG(start_symbol="S")
        rule = GrammarRule(lhs="S", rhs=["A", "B"], prob=0.5)

        grammar.add_rule(rule)

        self.assertEqual(len(grammar.rules), 1)
        self.assertEqual(grammar.rules[0], rule)

    def test_get_rules_for_lhs(self):
        """Should retrieve rules by LHS"""
        from semiotics.pcfg_induction import PCFG, GrammarRule

        grammar = PCFG(start_symbol="S")
        grammar.add_rule(GrammarRule(lhs="S", rhs=["A"], prob=0.3))
        grammar.add_rule(GrammarRule(lhs="S", rhs=["B"], prob=0.7))

        rules = grammar.get_rules_for_lhs("S")

        self.assertEqual(len(rules), 2)

    def test_normalize_probabilities(self):
        """Should normalize probabilities for same LHS"""
        from semiotics.pcfg_induction import PCFG, GrammarRule

        grammar = PCFG(start_symbol="S")
        grammar.add_rule(GrammarRule(lhs="S", rhs=["A"], prob=0.3))
        grammar.add_rule(GrammarRule(lhs="S", rhs=["B"], prob=0.7))
        grammar.add_rule(GrammarRule(lhs="S", rhs=["C"], prob=1.0))  # Sum > 1

        grammar.normalize()

        rules = grammar.get_rules_for_lhs("S")
        total_prob = sum(r.prob for r in rules)

        self.assertAlmostEqual(total_prob, 1.0, places=5)


class TestPCFGInduction(unittest.TestCase):
    """Test PCFG induction from sequences"""

    def test_induce_from_sequences(self):
        """Should induce grammar from sequences"""
        from semiotics.pcfg_induction import PCFGInducer

        sequences = [
            ["a", "b", "c"],
            ["a", "b", "c"],
            ["x", "y", "z"],
        ]

        inducer = PCFGInducer(max_rule_length=3)
        grammar = inducer.induce(sequences)

        self.assertGreater(len(grammar.rules), 0)

    def test_learn_frequent_patterns(self):
        """Should learn frequently occurring patterns"""
        from semiotics.pcfg_induction import PCFGInducer

        # "ab" pattern appears frequently
        sequences = [
            ["a", "b", "c"],
            ["a", "b", "d"],
            ["a", "b", "e"],
            ["x", "y", "z"],
        ]

        inducer = PCFGInducer()
        grammar = inducer.induce(sequences)

        # Check that "ab" pattern is captured
        has_ab_rule = any("a" in rule.rhs and "b" in rule.rhs for rule in grammar.rules)

        self.assertTrue(has_ab_rule)

    def test_vocabulary_discovery(self):
        """Should discover vocabulary from sequences"""
        from semiotics.pcfg_induction import PCFGInducer

        sequences = [
            ["call", "type", "A"],
            ["call", "type", "B"],
            ["food", "request"],
        ]

        inducer = PCFGInducer()
        vocabulary = inducer.discover_vocabulary(sequences)

        self.assertIn("call", vocabulary)
        self.assertIn("type", vocabulary)
        self.assertIn("A", vocabulary)
        self.assertIn("B", vocabulary)
        self.assertIn("food", vocabulary)
        self.assertIn("request", vocabulary)


class TestGrammarParser(unittest.TestCase):
    """Test probabilistic parsing with induced grammar"""

    def test_parse_sequence(self):
        """Should parse sequence using grammar"""
        from semiotics.pcfg_induction import PCFG, GrammarParser, GrammarRule

        grammar = PCFG(start_symbol="S")
        grammar.add_rule(GrammarRule(lhs="S", rhs=["A", "B"], prob=0.6))
        grammar.add_rule(GrammarRule(lhs="S", rhs=["C"], prob=0.4))
        grammar.add_rule(GrammarRule(lhs="A", rhs=["a"], prob=1.0, is_terminal=True))
        grammar.add_rule(GrammarRule(lhs="B", rhs=["b"], prob=1.0, is_terminal=True))
        grammar.add_rule(GrammarRule(lhs="C", rhs=["c"], prob=1.0, is_terminal=True))

        parser = GrammarParser(grammar)
        parse_trees = parser.parse(["a", "b"])

        self.assertGreater(len(parse_trees), 0)

    def test_parse_probability(self):
        """Should compute parse probability"""
        from semiotics.pcfg_induction import PCFG, GrammarParser, GrammarRule

        grammar = PCFG(start_symbol="S")
        grammar.add_rule(GrammarRule(lhs="S", rhs=["A", "B"], prob=0.6))
        grammar.add_rule(GrammarRule(lhs="A", rhs=["a"], prob=1.0, is_terminal=True))
        grammar.add_rule(GrammarRule(lhs="B", rhs=["b"], prob=1.0, is_terminal=True))

        parser = GrammarParser(grammar)
        prob = parser.parse_probability(["a", "b"])

        # Probability should be 0.6 (S->AB) * 1.0 (A->a) * 1.0 (B->b)
        self.assertAlmostEqual(prob, 0.6, places=5)

    def test_most_likely_parse(self):
        """Should find most likely parse"""
        from semiotics.pcfg_induction import PCFG, GrammarParser, GrammarRule

        grammar = PCFG(start_symbol="S")
        grammar.add_rule(GrammarRule(lhs="S", rhs=["A"], prob=0.3))
        grammar.add_rule(GrammarRule(lhs="S", rhs=["B"], prob=0.7))
        grammar.add_rule(GrammarRule(lhs="A", rhs=["a"], prob=1.0, is_terminal=True))
        grammar.add_rule(GrammarRule(lhs="B", rhs=["a"], prob=1.0, is_terminal=True))

        parser = GrammarParser(grammar)
        best_parse = parser.most_likely_parse(["a"])

        # Should prefer S->B (higher probability)
        self.assertEqual(best_parse[0].lhs, "B")


class TestVocalizationGrammar(unittest.TestCase):
    """Test grammar induction for vocalizations"""

    def test_segment_to_phrase_mapping(self):
        """Should map segments to phrases"""
        from semiotics.pcfg_induction import VocalizationGrammar

        grammar = VocalizationGrammar(species="marmoset")

        # Add phrase types
        grammar.add_phrase_type("contact_call")
        grammar.add_phrase_type("alarm_call")

        self.assertIn("contact_call", grammar.phrase_types)
        self.assertIn("alarm_call", grammar.phrase_types)

    def test_learn_sequence_patterns(self):
        """Should learn patterns from segment sequences"""
        from semiotics.pcfg_induction import VocalizationGrammar

        grammar = VocalizationGrammar(species="marmoset")

        # Training sequences (segment IDs)
        sequences = [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 4],
            [5, 6, 7],
        ]

        grammar.learn_from_sequences(sequences)

        # Should have learned rules
        self.assertGreater(len(grammar.grammar.rules), 0)

    def test_predict_next_segment(self):
        """Should predict next segment in sequence"""
        from semiotics.pcfg_induction import VocalizationGrammar

        grammar = VocalizationGrammar(species="marmoset")

        # Train with patterns
        sequences = [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 4],
        ]

        grammar.learn_from_sequences(sequences)

        # Predict next segment after [1, 2]
        predictions = grammar.predict_next([1, 2], top_k=2)

        self.assertGreater(len(predictions), 0)
        # 3 should be most likely (appears twice)
        self.assertEqual(predictions[0][0], 3)

    def test_detect_phrase_boundaries(self):
        """Should detect phrase boundaries in sequence"""
        from semiotics.pcfg_induction import VocalizationGrammar

        grammar = VocalizationGrammar(species="marmoset")

        sequences = [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 6, 7],
        ]

        grammar.learn_from_sequences(sequences)

        boundaries = grammar.detect_boundaries([1, 2, 3, 4, 5])

        # Should detect boundary after 3 (pattern change)
        self.assertIn(3, boundaries)


class TestGrammarComplexity(unittest.TestCase):
    """Test grammar complexity metrics"""

    def test_count_terminals(self):
        """Should count terminal symbols"""
        from semiotics.pcfg_induction import PCFG, GrammarRule

        grammar = PCFG(start_symbol="S")
        grammar.add_rule(GrammarRule(lhs="A", rhs=["a"], prob=0.5, is_terminal=True))
        grammar.add_rule(GrammarRule(lhs="B", rhs=["b"], prob=0.5, is_terminal=True))

        terminals = grammar.count_terminals()

        self.assertEqual(terminals, 2)

    def test_count_non_terminals(self):
        """Should count non-terminal symbols"""
        from semiotics.pcfg_induction import PCFG, GrammarRule

        grammar = PCFG(start_symbol="S")
        grammar.add_rule(GrammarRule(lhs="S", rhs=["A", "B"], prob=1.0))
        grammar.add_rule(GrammarRule(lhs="A", rhs=["a"], prob=1.0, is_terminal=True))

        non_terminals = grammar.count_non_terminals()

        self.assertEqual(non_terminals, 3)  # S, A, B

    def test_compute_entropy(self):
        """Should compute grammar entropy"""
        from semiotics.pcfg_induction import PCFG, GrammarRule

        grammar = PCFG(start_symbol="S")
        grammar.add_rule(GrammarRule(lhs="S", rhs=["A"], prob=0.5))
        grammar.add_rule(GrammarRule(lhs="S", rhs=["B"], prob=0.5))

        entropy = grammar.compute_entropy()

        # Two equally likely rules: entropy = -2 * 0.5 * log(0.5) = log(2)
        self.assertAlmostEqual(entropy, np.log(2), places=5)


if __name__ == "__main__":
    unittest.main()
