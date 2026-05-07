#!/usr/bin/env python3
"""
PCFG Induction - Formal Language Theory
========================================

Probabilistic Context-Free Grammar induction for discovering
grammatical structure in animal vocalizations.

This module implements:
- Probabilistic Context-Free Grammar (PCFG) representation
- Grammar rule induction from observation sequences
- Probabilistic parsing with CYK algorithm
- Vocalization-specific grammar learning
- Grammar complexity metrics (entropy, rule count)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GrammarRule:
    """
    A probabilistic grammar rule.

    Represents a production rule in a PCFG:
    - lhs: Left-hand side (non-terminal symbol)
    - rhs: Right-hand side (list of symbols)
    - prob: Probability of this rule given the LHS
    - is_terminal: Whether this is a terminal rule (produces actual tokens)
    """

    lhs: str
    rhs: List[str]
    prob: float
    is_terminal: bool = False

    def __str__(self) -> str:
        """String representation of the rule."""
        rhs_str = " ".join(self.rhs)
        return f"{self.lhs} -> {rhs_str} [{self.prob:.3f}]"

    def __eq__(self, other) -> bool:
        """Rules are equal if LHS and RHS match."""
        if not isinstance(other, GrammarRule):
            return False
        return self.lhs == other.lhs and self.rhs == other.rhs

    def __hash__(self) -> int:
        """Hash rule for use in sets/dicts."""
        return hash((self.lhs, tuple(self.rhs)))


class PCFG:
    """
    Probabilistic Context-Free Grammar.

    A PCFG consists of:
    - A set of non-terminal symbols
    - A set of terminal symbols
    - A start symbol
    - Production rules with probabilities
    """

    def __init__(self, start_symbol: str = "S"):
        """
        Initialize a PCFG.

        Args:
            start_symbol: The start symbol for the grammar
        """
        self.start_symbol = start_symbol
        self.rules: List[GrammarRule] = []

    def add_rule(self, rule: GrammarRule) -> None:
        """
        Add a rule to the grammar.

        Args:
            rule: The grammar rule to add
        """
        self.rules.append(rule)

    def get_rules_for_lhs(self, lhs: str) -> List[GrammarRule]:
        """
        Get all rules with the given left-hand side.

        Args:
            lhs: The left-hand side symbol

        Returns:
            List of rules with the given LHS
        """
        return [r for r in self.rules if r.lhs == lhs]

    def normalize(self) -> None:
        """
        Normalize probabilities for each LHS.

        Ensures that probabilities for each LHS sum to 1.0.
        """
        # Group rules by LHS
        rules_by_lhs: Dict[str, List[GrammarRule]] = defaultdict(list)
        for rule in self.rules:
            rules_by_lhs[rule.lhs].append(rule)

        # Normalize each group
        new_rules = []
        for lhs, lhs_rules in rules_by_lhs.items():
            total_prob = sum(r.prob for r in lhs_rules)
            if total_prob > 0:
                for rule in lhs_rules:
                    normalized_rule = GrammarRule(
                        lhs=rule.lhs,
                        rhs=rule.rhs.copy(),
                        prob=rule.prob / total_prob,
                        is_terminal=rule.is_terminal,
                    )
                    new_rules.append(normalized_rule)
            else:
                # Equal probabilities if all are zero
                uniform_prob = 1.0 / len(lhs_rules)
                for rule in lhs_rules:
                    normalized_rule = GrammarRule(
                        lhs=rule.lhs,
                        rhs=rule.rhs.copy(),
                        prob=uniform_prob,
                        is_terminal=rule.is_terminal,
                    )
                    new_rules.append(normalized_rule)

        self.rules = new_rules

    def count_terminals(self) -> int:
        """
        Count unique terminal symbols.

        Returns:
            Number of unique terminal symbols
        """
        terminals: Set[str] = set()
        for rule in self.rules:
            if rule.is_terminal:
                for symbol in rule.rhs:
                    terminals.add(symbol)
        return len(terminals)

    def count_non_terminals(self) -> int:
        """
        Count unique non-terminal symbols.

        Returns:
            Number of unique non-terminal symbols
        """
        non_terminals: Set[str] = set()
        terminals: Set[str] = set()

        non_terminals.add(self.start_symbol)

        # First pass: identify terminals (symbols only in RHS of terminal rules)
        for rule in self.rules:
            non_terminals.add(rule.lhs)
            if rule.is_terminal:
                for symbol in rule.rhs:
                    terminals.add(symbol)

        # Second pass: add non-terminals from RHS of non-terminal rules
        for rule in self.rules:
            if not rule.is_terminal:
                for symbol in rule.rhs:
                    # If not a known terminal, it's a non-terminal
                    if symbol not in terminals:
                        non_terminals.add(symbol)

        return len(non_terminals)

    def compute_entropy(self) -> float:
        """
        Compute the entropy of the grammar.

        Returns:
            Entropy in nats
        """
        # Group rules by LHS
        rules_by_lhs: Dict[str, List[GrammarRule]] = defaultdict(list)
        for rule in self.rules:
            rules_by_lhs[rule.lhs].append(rule)

        # Compute entropy for each LHS and sum
        total_entropy = 0.0
        for lhs, lhs_rules in rules_by_lhs.items():
            probs = [r.prob for r in lhs_rules]
            # Normalize to ensure valid probabilities
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
                for p in probs:
                    if p > 0:
                        total_entropy -= p * math.log(p)

        return total_entropy


class PCFGInducer:
    """
    Induce a PCFG from observed sequences.

    Uses frequency-based learning to discover grammar rules
    from sequences of observed symbols.
    """

    def __init__(self, max_rule_length: int = 3, min_frequency: int = 1):
        """
        Initialize the PCFG inducer.

        Args:
            max_rule_length: Maximum length of RHS to consider
            min_frequency: Minimum frequency for a rule to be included
        """
        self.max_rule_length = max_rule_length
        self.min_frequency = min_frequency

    def induce(self, sequences: List[List]) -> PCFG:
        """
        Induce a grammar from sequences.

        Args:
            sequences: List of symbol sequences

        Returns:
            Induced PCFG
        """
        # Build vocabulary
        vocabulary = self.discover_vocabulary(sequences)

        # Count n-grams and patterns
        rule_counter: Dict[Tuple, int] = Counter()

        # Learn patterns from sequences
        for seq in sequences:
            for length in range(1, min(len(seq) + 1, self.max_rule_length + 1)):
                for i in range(len(seq) - length + 1):
                    pattern = tuple(seq[i : i + length])
                    rule_counter[pattern] += 1

        # Create grammar
        grammar = PCFG(start_symbol="S")

        # Create non-terminal symbols for frequent patterns
        non_terminal_id = 0
        pattern_to_nt: Dict[Tuple, str] = {}

        # Sort patterns by frequency
        sorted_patterns = sorted(rule_counter.items(), key=lambda x: x[1], reverse=True)

        for pattern, count in sorted_patterns:
            if count < self.min_frequency:
                continue

            # Skip single symbols that are in vocabulary
            if len(pattern) == 1 and pattern[0] in vocabulary:
                continue

            # Create non-terminal for this pattern
            nt = f"N{non_terminal_id}"
            non_terminal_id += 1
            pattern_to_nt[pattern] = nt

            # Add rule expanding the non-terminal
            total_pattern_count = sum(c for p, c in rule_counter.items() if p[0] == pattern[0])

            if total_pattern_count > 0:
                prob = count / total_pattern_count
                grammar.add_rule(GrammarRule(lhs="S", rhs=[nt], prob=prob, is_terminal=False))

                # Add terminal rule for the pattern
                grammar.add_rule(GrammarRule(lhs=nt, rhs=list(pattern), prob=1.0, is_terminal=True))

        # Add direct vocabulary rules
        for symbol in vocabulary:
            count = sum(1 for seq in sequences if symbol in seq)
            if count >= self.min_frequency:
                grammar.add_rule(GrammarRule(lhs="S", rhs=[symbol], prob=0.1, is_terminal=True))

        grammar.normalize()
        return grammar

    def discover_vocabulary(self, sequences: List[List]) -> Set:
        """
        Discover the vocabulary (set of unique symbols).

        Args:
            sequences: List of symbol sequences

        Returns:
            Set of unique symbols
        """
        vocabulary: Set = set()
        for seq in sequences:
            for symbol in seq:
                vocabulary.add(symbol)
        return vocabulary


class GrammarParser:
    """
    Probabilistic parser using recursive descent.

    Parses sequences using a PCFG and computes probabilities.
    """

    def __init__(self, grammar: PCFG):
        """
        Initialize the parser.

        Args:
            grammar: The PCFG to use for parsing
        """
        self.grammar = grammar

    def parse(self, sequence: List) -> List[Dict]:
        """
        Parse a sequence using the grammar.

        Args:
            sequence: Input sequence of symbols

        Returns:
            List of possible parse trees (simplified representation)
        """
        trees = self._parse_recursive(sequence, self.grammar.start_symbol, 0)
        return trees

    def _parse_recursive(self, sequence: List, symbol: str, pos: int) -> List[Dict]:
        """
        Recursively parse sequence starting from position.

        Args:
            sequence: Input sequence
            symbol: Current non-terminal to expand
            pos: Current position in sequence

        Returns:
            List of parse trees
        """
        trees = []
        remaining = len(sequence) - pos

        for rule in self.grammar.get_rules_for_lhs(symbol):
            if rule.is_terminal:
                # Check if RHS matches sequence starting at pos
                if (
                    len(rule.rhs) <= remaining
                    and list(rule.rhs) == sequence[pos : pos + len(rule.rhs)]
                ):
                    # Create terminal node
                    trees.append(
                        {
                            "symbol": symbol,
                            "children": [{"symbol": s, "children": []} for s in rule.rhs],
                            "rule": rule,
                        }
                    )
            else:
                # Try to expand non-terminals
                expanded_trees = self._expand_non_terminal(sequence, rule, pos, {symbol})
                trees.extend(expanded_trees)

        return trees

    def _expand_non_terminal(
        self, sequence: List, rule: GrammarRule, pos: int, visited: Set[str]
    ) -> List[Dict]:
        """Expand a non-terminal rule."""
        if len(rule.rhs) == 0:
            return []

        # Expand first symbol
        first = rule.rhs[0]
        first_trees = self._parse_recursive(sequence, first, pos)

        if len(rule.rhs) == 1:
            # Single RHS symbol
            return [{"symbol": rule.lhs, "children": [t], "rule": rule} for t in first_trees]

        # Multiple RHS symbols - need to expand all
        result = []
        for first_tree in first_trees:
            first_len = self._tree_length(first_tree)
            next_pos = pos + first_len

            if next_pos >= len(sequence):
                continue

            # Recursively expand remaining symbols
            remaining = self._expand_remaining(sequence, rule.rhs[1:], next_pos, visited)

            for rem_tree in remaining:
                result.append(
                    {"symbol": rule.lhs, "children": [first_tree, rem_tree], "rule": rule}
                )

        return result

    def _expand_remaining(
        self, sequence: List, symbols: List[str], pos: int, visited: Set[str]
    ) -> List[Dict]:
        """Expand remaining symbols in a multi-symbol RHS."""
        if not symbols:
            return [{"symbol": "epsilon", "children": []}]

        if len(symbols) == 1:
            return self._parse_recursive(sequence, symbols[0], pos)

        result = []
        first_trees = self._parse_recursive(sequence, symbols[0], pos)

        for first_tree in first_trees:
            first_len = self._tree_length(first_tree)
            next_pos = pos + first_len

            if next_pos >= len(sequence):
                continue

            remaining = self._expand_remaining(sequence, symbols[1:], next_pos, visited)

            for rem_tree in remaining:
                result.append({"symbol": symbols[0], "children": [first_tree, rem_tree]})

        return result

    def _tree_length(self, tree: Dict) -> int:
        """Count leaf nodes in a parse tree."""
        if not tree.get("children"):
            return 1
        return sum(self._tree_length(child) for child in tree["children"])

    def parse_probability(self, sequence: List) -> float:
        """
        Compute the probability of a sequence.

        Args:
            sequence: Input sequence

        Returns:
            Probability of the sequence under the grammar
        """
        trees = self.parse(sequence)
        if not trees:
            return 0.0

        # Compute probability for each tree and take max
        max_prob = 0.0
        for tree in trees:
            prob = self._tree_probability(tree)
            max_prob = max(max_prob, prob)

        return max_prob

    def _tree_probability(self, tree: Dict) -> float:
        """Compute probability of a parse tree."""
        rule = tree.get("rule")
        if not rule:
            return 0.0

        prob = rule.prob

        # Multiply by probabilities of children
        for child in tree.get("children", []):
            child_rule = child.get("rule")
            if child_rule:
                prob *= child_rule.prob

        return prob

    def most_likely_parse(self, sequence: List) -> List:
        """
        Find the most likely parse rules.

        Args:
            sequence: Input sequence

        Returns:
            List of rules that could derive the sequence, sorted by probability
        """
        # Find all terminal rules that directly match the sequence
        direct_matches = []
        for rule in self.grammar.rules:
            if rule.is_terminal and tuple(rule.rhs) == tuple(sequence):
                # Compute full derivation probability from start symbol
                full_prob = self._derivation_probability(rule.lhs)
                direct_matches.append((rule, full_prob))

        # Sort by full derivation probability
        direct_matches.sort(key=lambda x: x[1], reverse=True)

        # Return just the rules
        return [rule for rule, _ in direct_matches]

    def _derivation_probability(self, symbol: str) -> float:
        """
        Compute probability of deriving a symbol from start symbol.

        Args:
            symbol: The symbol to derive

        Returns:
            Maximum probability of reaching this symbol from start
        """
        if symbol == self.grammar.start_symbol:
            return 1.0

        # Find rules from start that can reach this symbol
        # For simplicity, use BFS to find path probability
        visited = set()
        max_prob = 0.0

        def bfs(current: str, prob: float):
            nonlocal max_prob
            if current in visited:
                return
            visited.add(current)

            if current == symbol:
                max_prob = max(max_prob, prob)
                return

            for rule in self.grammar.get_rules_for_lhs(current):
                for rhs_symbol in rule.rhs:
                    if not rule.is_terminal or rhs_symbol == symbol:
                        bfs(rhs_symbol, prob * rule.prob)

        bfs(self.grammar.start_symbol, 1.0)
        return max_prob


class VocalizationGrammar:
    """
    Specialized PCFG for vocalization analysis.

    Handles species-specific grammatical patterns in
    animal vocalization sequences.
    """

    def __init__(self, species: str):
        """
        Initialize vocalization grammar.

        Args:
            species: The species this grammar models
        """
        self.species = species
        self.grammar = PCFG(start_symbol="S")
        self.phrase_types: Dict[str, List[List[int]]] = {}

    def add_phrase_type(self, phrase_type: str) -> None:
        """
        Add a phrase type to the grammar.

        Args:
            phrase_type: Name of the phrase type
        """
        if phrase_type not in self.phrase_types:
            self.phrase_types[phrase_type] = []

    def learn_from_sequences(self, sequences: List[List[int]]) -> None:
        """
        Learn grammar patterns from segment sequences.

        Args:
            sequences: List of segment ID sequences
        """
        # Flatten to get all segment IDs
        all_segments = []
        for seq in sequences:
            all_segments.extend(seq)

        # Create inducer and learn
        inducer = PCFGInducer(max_rule_length=3, min_frequency=1)
        self.grammar = inducer.induce(sequences)

    def predict_next(self, prefix: List[int], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Predict the next segment in a sequence.

        Args:
            prefix: The prefix sequence
            top_k: Number of top predictions to return

        Returns:
            List of (segment_id, probability) tuples
        """
        # Count what typically follows this prefix
        follower_counts: Dict[int, int] = Counter()
        prefix_len = len(prefix)

        # Extract patterns from learned grammar
        for rule in self.grammar.rules:
            if rule.is_terminal and len(rule.rhs) > prefix_len:
                # Check if prefix matches
                matches = True
                for i in range(prefix_len):
                    if rule.rhs[i] != prefix[i]:
                        matches = False
                        break
                if matches:
                    next_symbol = rule.rhs[prefix_len]
                    follower_counts[next_symbol] += int(rule.prob * 100)

        # Convert to probabilities and sort
        total = sum(follower_counts.values())
        if total > 0:
            predictions = [(sym, count / total) for sym, count in follower_counts.items()]
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:top_k]

        return []

    def detect_boundaries(self, sequence: List[int]) -> List[int]:
        """
        Detect phrase boundaries in a sequence.

        Args:
            sequence: Input sequence of segment IDs

        Returns:
            List of boundary indices
        """
        boundaries = []
        n = len(sequence)

        if n < 2:
            return boundaries

        # Look for pattern changes based on learned grammar rules
        # Check each position for whether continuation is "predictable"
        for i in range(1, n):
            prefix = sequence[:i]
            sequence[i:]  # suffix unused

            # Get predictions for what should follow the prefix
            predictions = self.predict_next(prefix, top_k=10)

            if not predictions:
                # No predictions found -> likely a boundary
                boundaries.append(i - 1)
                continue

            # Check if the next symbol is in top predictions
            next_symbol = sequence[i] if i < n else None
            if next_symbol is not None:
                predicted_symbols = [p[0] for p in predictions]
                if next_symbol not in predicted_symbols:
                    # Next symbol not predicted -> likely a boundary
                    boundaries.append(i - 1)
                else:
                    # Check probability drop
                    next_prob = next((p[1] for p in predictions if p[0] == next_symbol), 0.0)
                    # If probability is very low, it might be a boundary
                    if next_prob < 0.3:  # Threshold for low probability
                        boundaries.append(i - 1)

        return boundaries

    def _sequence_probability(self, sequence: List[int]) -> float:
        """Compute probability of a sequence."""
        if not sequence:
            return 1.0

        # Check if there's a matching rule
        for rule in self.grammar.rules:
            if rule.is_terminal and tuple(rule.rhs) == tuple(sequence):
                return rule.prob

        # Otherwise estimate from individual symbols
        prob = 1.0
        for symbol in sequence:
            symbol_prob = 0.0
            for rule in self.grammar.rules:
                if rule.is_terminal and len(rule.rhs) == 1 and rule.rhs[0] == symbol:
                    symbol_prob = max(symbol_prob, rule.prob)
            prob *= symbol_prob

        return prob


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("PCFG Induction - Formal Language Theory")
    print("=" * 50)

    # Test grammar creation
    grammar = PCFG(start_symbol="S")
    grammar.add_rule(GrammarRule(lhs="S", rhs=["NP", "VP"], prob=0.8))
    grammar.add_rule(GrammarRule(lhs="S", rhs=["VP"], prob=0.2))
    grammar.add_rule(GrammarRule(lhs="NP", rhs=["Det", "N"], prob=1.0, is_terminal=True))
    grammar.add_rule(GrammarRule(lhs="VP", rhs=["V", "NP"], prob=1.0, is_terminal=True))

    print(f"Grammar: {len(grammar.rules)} rules")
    print(f"Non-terminals: {grammar.count_non_terminals()}")
    print(f"Terminals: {grammar.count_terminals()}")
    print(f"Entropy: {grammar.compute_entropy():.4f}")

    # Test PCFG induction
    sequences = [
        ["a", "b", "c"],
        ["a", "b", "c"],
        ["x", "y", "z"],
    ]

    inducer = PCFGInducer(max_rule_length=3)
    learned_grammar = inducer.induce(sequences)

    print(f"\nLearned grammar: {len(learned_grammar.rules)} rules")
    for rule in learned_grammar.rules:
        print(f"  {rule}")

    # Test vocalization grammar
    vocal_grammar = VocalizationGrammar(species="marmoset")
    vocal_grammar.add_phrase_type("contact_call")
    vocal_grammar.add_phrase_type("alarm_call")

    train_seqs = [
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 4],
    ]

    vocal_grammar.learn_from_sequences(train_seqs)

    predictions = vocal_grammar.predict_next([1, 2], top_k=2)
    print(f"\nPredictions after [1, 2]: {predictions}")
