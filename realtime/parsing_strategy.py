#!/usr/bin/env python3
"""
Parsing Strategy Module - Strategy Pattern for Vocalization Parsing
====================================================================

This module implements the Strategy Pattern for parsing animal vocalizations,
allowing different species to use different parsing approaches while maintaining
a unified interface.

Key Strategies:
- CompositionalStrategy: Original behavior where each segment = semantic unit
- HolophrasticStrategy: Bat-specific where rigid idioms are atomic units

Background Research (Egyptian Fruit Bat Phase 2/3):
- Only 0.02% of possible bigrams are used (extremely restrictive grammar)
- LRN-6 [114, 464, 604, 324, 94, 714] is an unbreakable rigid idiom
- No function words detected (all segments have <5 unique transitions)
- Context is determined by external factors, not segment identity
- Position-based roles: Openers (position 0), Closers (position 1), Content (2+)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class TokenType(Enum):
    """Classification of parsed token types"""

    COMPOSITIONAL = "compositional"  # Original: segment = word (general mode)
    IDIOM = "idiom"  # Bat: rigid pattern = single meaning (holophrastic)
    CONTENT = "content"  # Bat: individual segment in known position
    NOISE = "noise"  # Bat: segment not in known patterns
    OPENER = "opener"  # Bat: position 0 specialist (staccato alert)
    CLOSER = "closer"  # Bat: position 1 specialist (clean termination)


@dataclass
class ParsedToken:
    """Result of parsing a segment sequence into a semantic token"""

    token_type: TokenType
    segments: List[int]  # Segment IDs (multiple for idioms)
    meaning: Optional[str] = None  # Semantic label
    confidence: float = 1.0
    position: Optional[int] = None  # Position in original sequence
    acoustic_hints: Optional[Dict[str, Any]] = None  # Optional acoustic metadata


@dataclass
class ParseResult:
    """Complete result of parsing a segment sequence"""

    tokens: List[ParsedToken]
    original_sequence: List[int]
    strategy_used: str
    idiom_count: int = 0
    compositional_count: int = 0
    noise_count: int = 0
    confidence_avg: float = 1.0


class ParsingStrategy(ABC):
    """
    Abstract base class for vocalization parsing strategies.

    This defines the interface that all parsing strategies must implement,
    enabling the Strategy Pattern for species-specific parsing behavior.
    """

    @abstractmethod
    def parse(self, segment_sequence: List[int]) -> ParseResult:
        """
        Parse segment sequence into semantic tokens.

        Args:
            segment_sequence: List of segment IDs from audio processing

        Returns:
            ParseResult with parsed tokens and metadata
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name"""
        pass

    @property
    @abstractmethod
    def is_holophrastic(self) -> bool:
        """Return True if this strategy uses holophrastic (idiom-based) parsing"""
        pass


class CompositionalStrategy(ParsingStrategy):
    """
    Original behavior: each segment is a semantic unit.

    This is the default strategy used for most species where vocalizations
    follow a compositional grammar (segments combine to form meaning).

    Use Cases:
    - Marmosets (phrase-type encoding)
    - Songbirds (combinatorial syntax)
    - General unknown species
    """

    def __init__(self, segment_meanings: Optional[Dict[int, str]] = None):
        """
        Initialize compositional strategy.

        Args:
            segment_meanings: Optional mapping from segment ID to semantic label
        """
        self._segment_meanings = segment_meanings or {}

    @property
    def name(self) -> str:
        return "compositional"

    @property
    def is_holophrastic(self) -> bool:
        return False

    def parse(self, segment_sequence: List[int]) -> ParseResult:
        """Parse each segment as an independent semantic unit."""
        tokens = []

        for i, segment in enumerate(segment_sequence):
            meaning = self._segment_meanings.get(segment)
            confidence = 1.0 if meaning else 0.5

            tokens.append(
                ParsedToken(
                    token_type=TokenType.COMPOSITIONAL,
                    segments=[segment],
                    meaning=meaning,
                    confidence=confidence,
                    position=i,
                )
            )

        return ParseResult(
            tokens=tokens,
            original_sequence=segment_sequence.copy(),
            strategy_used=self.name,
            compositional_count=len(tokens),
            confidence_avg=sum(t.confidence for t in tokens) / max(len(tokens), 1),
        )

    def _lookup_segment(self, segment: int) -> Optional[str]:
        """Look up semantic meaning for a segment."""
        return self._segment_meanings.get(segment)


class HolophrasticStrategy(ParsingStrategy):
    """
    Bat-specific: rigid idioms are atomic units.

    This strategy implements the holophrastic communication model discovered
    in Egyptian fruit bat research (Phase 2/3). Key characteristics:

    1. Rigid Idioms: Some patterns are unbreakable (must be matched as a whole)
    2. Position-Based Roles: Segments have different meanings based on position
    3. No Composition: Meanings are NOT built from segment parts

    From Phase 2 Research:
    - LRN-6 [114, 464, 604, 324, 94, 714] is completely unbreakable
    - 0 of its sub-patterns appear independently
    - Diagnosis: RIGID IDIOM (like "kick the bucket" - not about kicking)

    From Phase 3 Research:
    - Openers: Short duration (~31.6ms), high energy, low HNR (staccato alerts)
    - Closers: Long duration (~58.0ms), low energy, high HNR (clean termination)
    - Position determines role, not acoustics
    """

    # Default rigid idioms from Phase 2 research (fallback when Rust unavailable)
    _DEFAULT_RIGID_IDIOMS: List[Tuple[List[int], str, float]] = [
        ([114, 464, 604, 324, 94, 714], "LRN-6_IDIOM", 0.98),
    ]

    # Default bigrams from Phase 2 research (fallback)
    _DEFAULT_VALID_BIGRAMS: set = {
        (764, 304),
        (534, 434),
        (304, 394),
        (514, 504),
        (384, 464),
        (574, 324),
        (444, 544),
        (1014, 684),
        (384, 44),
        (154, 204),
        (264, 44),
        (764, 464),
        (514, 304),
        (574, 684),
        (434, 504),
        (304, 404),
        (394, 404),
        (544, 504),
        (684, 504),
        (324, 394),
        (464, 604),
        (604, 324),
        (324, 94),
        (94, 714),
        (114, 464),
    }

    # Default position specialists (fallback)
    _DEFAULT_OPENERS: set = {384, 264, 1014, 1004, 534, 434, 514, 764}
    _DEFAULT_CLOSERS: set = {444, 304, 544, 404, 394, 684, 504}

    def __init__(
        self,
        rigid_idioms: Optional[List[Tuple[List[int], str, float]]] = None,
        segment_meanings: Optional[Dict[int, str]] = None,
        valid_bigrams: Optional[set] = None,
        openers: Optional[set] = None,
        closers: Optional[set] = None,
    ):
        """
        Initialize holophrastic strategy.

        Args:
            rigid_idioms: Custom rigid idioms (overrides defaults)
            segment_meanings: Optional mapping for non-idiom segments
            valid_bigrams: Valid bigram pairs (overrides defaults)
            openers: Opener segments (overrides defaults)
            closers: Closer segments (overrides defaults)
        """
        self._rigid_idioms = rigid_idioms or self._DEFAULT_RIGID_IDIOMS
        self._segment_meanings = segment_meanings or {}
        self._valid_bigrams = (
            valid_bigrams if valid_bigrams is not None else self._DEFAULT_VALID_BIGRAMS
        )
        self._openers = openers if openers is not None else self._DEFAULT_OPENERS
        self._closers = closers if closers is not None else self._DEFAULT_CLOSERS

        # Build efficient lookup structures
        self._idiom_first_segments = {idiom[0][0]: i for i, idiom in enumerate(self._rigid_idioms)}

    @property
    def name(self) -> str:
        return "holophrastic"

    @property
    def is_holophrastic(self) -> bool:
        return True

    # Class-level accessors for backward compat (tests reference these)
    @property
    def OPENERS(self) -> set:
        return self._openers

    @property
    def CLOSERS(self) -> set:
        return self._closers

    @property
    def VALID_BIGRAMS(self) -> set:
        return self._valid_bigrams

    def parse(self, segment_sequence: List[int]) -> ParseResult:
        """
        Parse segment sequence with idiom detection.

        Algorithm:
        1. Check for rigid idioms first (highest priority)
        2. Check for position specialists (openers, closers)
        3. Fall back to noise/content classification
        """
        if not segment_sequence:
            return ParseResult(
                tokens=[],
                original_sequence=[],
                strategy_used=self.name,
            )

        tokens = []
        remaining = segment_sequence.copy()
        position = 0
        idiom_count = 0
        noise_count = 0

        while remaining:
            # Step 1: Check for rigid idiom match
            idiom_token = self._try_match_idiom(remaining, position)
            if idiom_token:
                tokens.append(idiom_token)
                remaining = remaining[len(idiom_token.segments) :]
                position += len(idiom_token.segments)
                idiom_count += 1
                continue

            # Step 2: Check for position specialists
            segment = remaining[0]

            if position == 0 and segment in self._openers:
                tokens.append(
                    ParsedToken(
                        token_type=TokenType.OPENER,
                        segments=[segment],
                        meaning="OPENER_STACCATO_ALERT",
                        confidence=0.85,
                        position=position,
                        acoustic_hints={
                            "expected_duration_ms": 31.6,
                            "expected_energy": "high",
                            "expected_hnr": "low",
                        },
                    )
                )
            elif position == 1 and segment in self._closers:
                tokens.append(
                    ParsedToken(
                        token_type=TokenType.CLOSER,
                        segments=[segment],
                        meaning="CLOSER_CLEAN_TERMINATION",
                        confidence=0.85,
                        position=position,
                        acoustic_hints={
                            "expected_duration_ms": 58.0,
                            "expected_energy": "low",
                            "expected_hnr": "high",
                        },
                    )
                )
            else:
                # Step 3: Classify as content or noise
                is_valid = self._is_valid_segment(segment, position, remaining)
                if is_valid:
                    tokens.append(
                        ParsedToken(
                            token_type=TokenType.CONTENT,
                            segments=[segment],
                            meaning=self._segment_meanings.get(segment),
                            confidence=0.7,
                            position=position,
                        )
                    )
                else:
                    tokens.append(
                        ParsedToken(
                            token_type=TokenType.NOISE,
                            segments=[segment],
                            meaning=None,
                            confidence=0.3,
                            position=position,
                        )
                    )
                    noise_count += 1

            remaining = remaining[1:]
            position += 1

        return ParseResult(
            tokens=tokens,
            original_sequence=segment_sequence.copy(),
            strategy_used=self.name,
            idiom_count=idiom_count,
            compositional_count=0,
            noise_count=noise_count,
            confidence_avg=sum(t.confidence for t in tokens) / max(len(tokens), 1),
        )

    def _try_match_idiom(self, sequence: List[int], position: int) -> Optional[ParsedToken]:
        """Try to match a rigid idiom at the start of the sequence."""
        if not sequence:
            return None

        first_segment = sequence[0]

        # Quick lookup check
        if first_segment not in self._idiom_first_segments:
            return None

        idiom_idx = self._idiom_first_segments[first_segment]
        idiom_segments, meaning, confidence = self._rigid_idioms[idiom_idx]

        if self._sequence_starts_with(sequence, idiom_segments):
            return ParsedToken(
                token_type=TokenType.IDIOM,
                segments=idiom_segments,
                meaning=meaning,
                confidence=confidence,
                position=position,
            )

        return None

    def _sequence_starts_with(self, sequence: List[int], prefix: List[int]) -> bool:
        """Check if sequence starts with the given prefix."""
        if len(sequence) < len(prefix):
            return False
        return sequence[: len(prefix)] == prefix

    def _is_valid_segment(self, segment: int, position: int, remaining: List[int]) -> bool:
        """Check if segment is valid at this position."""
        # Check if segment appears in any valid bigram at this position
        if len(remaining) > 1:
            next_segment = remaining[1]
            if (segment, next_segment) in self._valid_bigrams:
                return True

        # Segment is valid if it's in any known set
        return (
            segment in self._openers
            or segment in self._closers
            or segment in {s for bigram in self._valid_bigrams for s in bigram}
        )

    def add_idiom(self, segments: List[int], meaning: str, confidence: float = 0.9) -> None:
        """Add a new rigid idiom to the strategy."""
        self._rigid_idioms.append((segments, meaning, confidence))
        self._idiom_first_segments[segments[0]] = len(self._rigid_idioms) - 1

    def get_valid_transitions(self, segment: int) -> List[int]:
        """Get valid transitions from a segment (for bat grammar)."""
        return [to for (from_s, to) in self._valid_bigrams if from_s == segment]

    @classmethod
    def from_rust_profile(cls, profile_data: Any) -> "HolophrasticStrategy":
        """
        Create a HolophrasticStrategy from an AcousticProfileData loaded from Rust.

        This is the preferred way to construct the strategy when Rust is available,
        as it eliminates data drift by loading from the single source of truth.

        Args:
            profile_data: AcousticProfileData from config_client

        Returns:
            HolophrasticStrategy with data loaded from Rust
        """
        rigid_idioms = [
            (idiom.segments, idiom.meaning, idiom.confidence) for idiom in profile_data.rigid_idioms
        ]
        return cls(
            rigid_idioms=rigid_idioms if rigid_idioms else None,
            openers=set(profile_data.openers),
            closers=set(profile_data.closers),
            valid_bigrams=set(profile_data.valid_bigrams),
        )


class ParsingStrategyFactory:
    """
    Factory for creating parsing strategies based on configuration.

    This enables runtime selection of parsing behavior based on
    domain mode configuration.
    """

    @staticmethod
    def create(
        domain_mode: str = "general",
        segment_meanings: Optional[Dict[int, str]] = None,
        custom_idioms: Optional[List[Tuple[List[int], str, float]]] = None,
        config_endpoint: Optional[str] = None,
    ) -> ParsingStrategy:
        """
        Create a parsing strategy based on configuration.

        For bat/holophrastic mode, tries to load profile data from Rust
        first (via config_client), falling back to hardcoded defaults
        if Rust is unavailable.

        Args:
            domain_mode: "general" for compositional, "bat" for holophrastic
            segment_meanings: Optional mapping from segment ID to meaning
            custom_idioms: Custom idioms for holophrastic mode
            config_endpoint: ZeroMQ endpoint for Rust config server

        Returns:
            Appropriate ParsingStrategy instance
        """
        if domain_mode.lower() in ("bat", "holophrastic"):
            # Try loading from Rust config server first
            if custom_idioms is None:
                try:
                    from realtime.config_client import ConfigClient

                    client = (
                        ConfigClient(endpoint=config_endpoint)
                        if config_endpoint
                        else ConfigClient()
                    )
                    profile = client.request_acoustic_profile("bat")
                    if profile is not None:
                        return HolophrasticStrategy.from_rust_profile(profile)
                except Exception:
                    pass  # Fall through to hardcoded defaults

            return HolophrasticStrategy(
                rigid_idioms=custom_idioms,
                segment_meanings=segment_meanings,
            )
        else:
            return CompositionalStrategy(segment_meanings=segment_meanings)


# Convenience exports
__all__ = [
    "TokenType",
    "ParsedToken",
    "ParseResult",
    "ParsingStrategy",
    "CompositionalStrategy",
    "HolophrasticStrategy",
    "ParsingStrategyFactory",
]
