#!/usr/bin/env python3
"""
Interaction Agent - Closed-Loop Cognitive Agent
================================================

This module implements the main Interaction Agent that orchestrates the
closed-loop communication between the Rust Execution Layer and Python
Logic Layer.

The agent:
1. Subscribes to 112D feature events from Rust
2. Processes them through the CognitiveLayer for context detection
3. Generates synthesis timelines based on behavioral intent
4. Publishes synthesis actions back to Rust for audio output

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "AgentState",
    "SpeakerProfile",
    "BigramProbability",
    "InteractionEvent",
    "SessionMetrics",
    "InteractionAgentConfig",
    "InteractionAgent",
    "build_cluster_context_map",
    "analyze_corpus_bigram_frequencies",
    "build_bigram_probability_map",
    "calculate_ras",
    "create_test_agent",
]

from realtime.action_publisher import (
    ActionPublisher,
    ActionPublisherConfig,
    MicroDynamicsDelta,
    TimelineEvent,
)

# Import components
from realtime.feature_subscriber import (
    FeatureEvent,
    FeatureSubscriber,
    FeatureSubscriberConfig,
)

# Import parsing strategy for Strategy Pattern
from realtime.parsing_strategy import (
    ParseResult,
    ParsingStrategy,
    ParsingStrategyFactory,
)

logger = logging.getLogger(__name__)

# Default endpoints
FEATURES_ENDPOINT = os.environ.get("RUST_FEATURES_ENDPOINT", "ipc:///tmp/cognitive_features.ipc")
ACTIONS_ENDPOINT = os.environ.get("RUST_ACTIONS_ENDPOINT", "ipc:///tmp/cognitive_actions.ipc")


class AgentState(Enum):
    """State of the Interaction Agent"""

    IDLE = "idle"  # Not processing
    LISTENING = "listening"  # Receiving features, analyzing
    RESPONDING = "responding"  # Generating and sending synthesis


@dataclass
class SpeakerProfile:
    """
    Speaker profile for Level 2 Semantic Grounding.

    Represents a known emitter (individual animal) with behavioral
    characteristics that influence response policies.

    This enables the system to differentiate *Who* is speaking from
    *What* is being said—critical for social species where meaning
    depends on both signal content and sender identity.

    Attributes:
        emitter_id: Unique identifier for this speaker (from Rust source separation)
        dominance_rank: Social hierarchy position (0-1, higher = more dominant)
        age_class: Life stage category ("juvenile", "subadult", "adult")
        response_bias: Context-specific response multipliers
                       (e.g., {"alarm": 0.95, "contact": 0.70})
    """
    emitter_id: int
    dominance_rank: Optional[float] = None
    age_class: Optional[str] = None
    response_bias: Optional[Dict[str, float]] = None

    def get_response_bias(self, context: str) -> float:
        """
        Get the response bias multiplier for a given context.

        Args:
            context: The canonical context (alarm, contact, territorial, social)

        Returns:
            Response bias multiplier (default 1.0 if context not in bias dict)
        """
        if self.response_bias is None:
            return 1.0
        return self.response_bias.get(context, 1.0)


@dataclass
class BigramProbability:
    """
    Bigram transition probability for Markov chain-based response weighting.

    Represents a valid bigram (opener → response) with its statistical
    properties derived from corpus analysis. This enables the system to
    distinguish between common transitions (high confidence) and rare
    transitions (requires cognitive attention).

    Attributes:
        opener: The opening cluster ID
        response: The response cluster ID
        count: Number of times this bigram occurs in the corpus
        probability: P(response | opener) - conditional probability
        rarity_score: 0-1 score where higher = more rare (1 - probability)
    """
    opener: int
    response: int
    count: int
    probability: float
    rarity_score: Optional[float] = None

    def calculate_rarity_score(self) -> float:
        """
        Calculate rarity score from probability.

        High probability (common) → low rarity
        Low probability (rare) → high rarity

        Returns:
            Rarity score between 0 and 1
        """
        if self.rarity_score is not None:
            return self.rarity_score
        # Rarity = 1 - probability (simple inverse)
        return 1.0 - self.probability

    def __post_init__(self):
        """Auto-calculate rarity_score if not provided."""
        if self.rarity_score is None:
            self.rarity_score = self.calculate_rarity_score()


@dataclass
class InteractionEvent:
    """
    Single interaction event for ethological validation (v1.5.0).

    Represents either an animal vocalization or a system response
    in the interaction sequence, used for calculating the Response
    Appropriateness Score (RAS).

    Attributes:
        timestamp: Unix timestamp of the event
        source: "animal" or "system"
        cluster_id: The cluster ID of the vocalization
        emitter_id: The animal emitter ID (None for system events)
        response_to: The cluster_id this event responds to (if applicable)
        time_since_previous: Time in seconds since previous event
    """
    timestamp: float
    source: str  # "animal" or "system"
    cluster_id: int
    emitter_id: Optional[int] = None
    response_to: Optional[int] = None
    time_since_previous: float = 0.0


@dataclass
class SessionMetrics:
    """
    Metrics for a single ethological validation session (v1.5.0).

    Contains all metrics needed to evaluate the success of a
    closed-loop bioacoustic interaction session.

    Attributes:
        session_id: Unique identifier for the session
        duration_seconds: Total session duration
        condition: Experimental condition ("baseline", "conspecific", "full_system", etc.)
        total_animal_vocalizations: Count of animal vocalizations
        total_system_responses: Count of system responses
        positive_responses: Count of positive RAS responses
        negative_responses: Count of negative RAS responses
        ras_score: Response Appropriateness Score (0-1)
        start_time: Session start timestamp
        end_time: Session end timestamp (None if ongoing)
    """
    session_id: str
    duration_seconds: float
    condition: str
    total_animal_vocalizations: int = 0
    total_system_responses: int = 0
    positive_responses: int = 0
    negative_responses: int = 0
    ras_score: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "duration_seconds": self.duration_seconds,
            "condition": self.condition,
            "total_animal_vocalizations": self.total_animal_vocalizations,
            "total_system_responses": self.total_system_responses,
            "positive_responses": self.positive_responses,
            "negative_responses": self.negative_responses,
            "ras_score": self.ras_score,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass
class InteractionAgentConfig:
    """Configuration for Interaction Agent"""

    feature_endpoint: str = FEATURES_ENDPOINT
    action_endpoint: str = ACTIONS_ENDPOINT
    response_cooldown_ms: float = 100.0
    max_responses_per_second: float = 5.0
    verbose_logging: bool = False

    # Domain mode for Strategy Pattern (Sprint 1)
    # "general" = CompositionalStrategy (default, segments = words)
    # "bat" = HolophrasticStrategy (rigid idioms = atomic units)
    domain_mode: str = "general"

    # Optional segment meanings for parsing
    segment_meanings: Optional[Dict[int, str]] = None

    # Custom idioms for holophrastic mode (list of (segments, meaning, confidence))
    custom_idioms: Optional[List[Tuple[List[int], str, float]]] = None

    # ZeroMQ endpoint for Rust config REQ/REP channel
    config_endpoint: Optional[str] = None

    # Path to trained ContextClassifier model for semantic alignment
    context_classifier_path: Optional[str] = None

    # Canonical context ontology - the finite set of response contexts
    # Any classifier label must map to one of these canonical contexts
    canonical_contexts: Tuple[str, ...] = ("contact", "alarm", "territorial", "social")

    # Label mapping: maps raw classifier labels to canonical contexts
    # Example: {"context_0": "social", "context_1": "alarm", ...}
    context_label_mapping: Optional[Dict[str, str]] = None

    # Uncertainty threshold for response gating (0-1)
    # Events with uncertainty > threshold will not trigger responses
    uncertainty_threshold: float = 0.6

    # ========================================================================
    # v1.2.0: Cluster-Based Semantic Grounding (Teacher-Student Pipeline)
    # ========================================================================

    # Pre-computed context map for BGMM-discovered clusters
    # Maps cluster_id (0-44) → canonical context (contact/alarm/territorial/social)
    cluster_context_map: Optional[Dict[int, str]] = None

    # Confidence threshold for Rust Student-derived cluster assignments
    # Events with confidence < threshold will not trigger responses
    # This is the distance-derived confidence from the OOD filter
    confidence_threshold: float = 0.5

    # Valid bat bigrams from LRN-6 syntax analysis
    # Set of (opener_cluster, response_cluster) tuples representing
    # the 50 valid transitions in the bat's acoustic grammar
    valid_bigrams: Optional[set] = None

    # ========================================================================
    # v1.3.0: Level 2 Speaker Grounding (Emitter ID Integration)
    # ========================================================================

    # Speaker profiles mapping emitter_id → SpeakerProfile
    # Enables speaker-specific response policies (Alpha vs Juvenile)
    speaker_profiles: Optional[Dict[int, SpeakerProfile]] = None

    # Enable speaker-aware response adaptation
    # When False, speaker_profiles are ignored even if configured
    enable_speaker_adaptation: bool = False

    # Minimum effective bias threshold for response gating
    # If speaker_bias_multiplier × base_confidence < threshold, suppress response
    speaker_bias_threshold: float = 0.3

    # ========================================================================
    # v1.4.0: Probabilistic Bigram Weights (Markov Chain Upgrade)
    # ========================================================================

    # Bigram probability map from corpus analysis
    # Maps (opener_cluster, response_cluster) → BigramProbability
    bigram_probability_map: Optional[Dict[Tuple[int, int], BigramProbability]] = None

    # Enable probability-weighted response modulation
    # When True, common bigrams boost confidence, rare bigrams reduce it
    enable_probabilistic_weighting: bool = False

    # Default probability for bigrams not in the map
    default_bigram_probability: float = 0.5

    # Rarity threshold for cognitive attention flag
    # If rarity_score > threshold, set cognitive_attention=True
    rarity_attention_threshold: float = 0.8

    # ========================================================================
    # v1.5.0: Ethological Validation Protocol (Field Deployment)
    # ========================================================================

    # Enable ethological validation mode for field deployment
    # When True, tracks interaction events and calculates RAS in real-time
    enable_ethological_mode: bool = False

    # Experimental condition label for logging
    # Options: "baseline", "conspecific", "full_system", "invalid_syntax", "synthetic_tones"
    experimental_condition: str = "full_system"

    # Response timeout window for RAS calculation (seconds)
    # Animal must respond within this window for positive RAS scoring
    ras_response_timeout_seconds: float = 2.0

    # Session ID for logging (auto-generated if None)
    session_id: Optional[str] = None

    # Maximum interaction history to keep for RAS calculation
    # Prevents unbounded memory growth during long sessions
    max_interaction_history: int = 10000

    def get_parsing_strategy(self) -> ParsingStrategy:
        """
        Get the appropriate parsing strategy based on domain_mode.

        Returns:
            ParsingStrategy instance (CompositionalStrategy or HolophrasticStrategy)
        """
        return ParsingStrategyFactory.create(
            domain_mode=self.domain_mode,
            segment_meanings=self.segment_meanings,
            custom_idioms=self.custom_idioms,
            config_endpoint=self.config_endpoint,
        )


# =============================================================================
# v1.2.0: Cluster-Based Semantic Grounding Functions
# =============================================================================

def build_cluster_context_map(centroids_112d: List[np.ndarray]) -> Dict[int, str]:
    """
    Build a context map for all BGMM-discovered clusters.

    This function pre-computes the behavioral context for each cluster
    based on the acoustic properties of its centroid (archetype).

    The inference uses the same rules as the original system, but applied
    to the *archetype* rather than noisy instances. This provides stability
    and semantic grounding - the context is derived from the statistical
    structure of the species' own vocalizations.

    Args:
        centroids_112d: List of 112D centroid arrays (one per cluster)

    Returns:
        Dictionary mapping cluster_id → canonical context string
    """
    context_map = {}

    for cluster_id, centroid in enumerate(centroids_112d):
        # Extract key acoustic features from centroid
        f0 = float(centroid[0]) if centroid[0] > 0 else 5000.0
        rms = float(centroid[1]) if len(centroid) > 1 else 0.5

        # Apply rule-based inference to the archetype
        if f0 > 8000 and rms > 0.6:
            context = "alarm"
        elif f0 > 6000:
            context = "territorial"
        elif f0 < 4000:
            context = "social"
        else:
            context = "contact"

        context_map[cluster_id] = context

    logger.info(f"Built cluster context map for {len(context_map)} clusters")
    return context_map


def analyze_corpus_bigram_frequencies(
    corpus_sequence: List[int],
) -> Dict[Tuple[int, int], int]:
    """
    Analyze bigram frequencies from a corpus cluster sequence.

    Counts how many times each bigram (opener → response) appears
    in the corpus. Used to build the Markov chain transition model.

    Args:
        corpus_sequence: List of cluster IDs in temporal order

    Returns:
        Dictionary mapping (opener, response) → count
    """
    bigram_counts = {}

    for i in range(len(corpus_sequence) - 1):
        opener = corpus_sequence[i]
        response = corpus_sequence[i + 1]
        bigram = (opener, response)

        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

    logger.info(f"Analyzed {len(corpus_sequence)} segments, found {len(bigram_counts)} unique bigrams")
    return bigram_counts


def build_bigram_probability_map(
    corpus_sequence: List[int],
    valid_bigrams: set,
) -> Dict[Tuple[int, int], BigramProbability]:
    """
    Build a probability map for all valid bigrams from corpus analysis.

    For each opener cluster, calculates P(response | opener) based on
    corpus frequencies. Only includes bigrams that are in valid_bigrams.

    Args:
        corpus_sequence: List of cluster IDs in temporal order
        valid_bigrams: Set of (opener, response) tuples representing valid transitions

    Returns:
        Dictionary mapping (opener, response) → BigramProbability
    """
    # Step 1: Count all bigrams in corpus
    all_bigram_counts = analyze_corpus_bigram_frequencies(corpus_sequence)

    # Step 2: Filter to only valid bigrams and count per opener
    opener_totals: Dict[int, int] = {}
    valid_bigram_counts: Dict[Tuple[int, int], int] = {}

    for bigram, count in all_bigram_counts.items():
        if bigram in valid_bigrams:
            opener = bigram[0]
            valid_bigram_counts[bigram] = count
            opener_totals[opener] = opener_totals.get(opener, 0) + count

    # Step 3: Build probability map
    prob_map = {}

    for (opener, response), count in valid_bigram_counts.items():
        opener_total = opener_totals.get(opener, 1)

        # Calculate conditional probability
        probability = count / opener_total if opener_total > 0 else 0.0

        # Create BigramProbability with auto-calculated rarity_score
        prob_map[(opener, response)] = BigramProbability(
            opener=opener,
            response=response,
            count=count,
            probability=probability,
            rarity_score=None,  # Auto-calculated in __post_init__
        )

    logger.info(f"Built probability map for {len(prob_map)} valid bigrams")
    return prob_map


def calculate_ras(interaction_sequence: List[InteractionEvent], valid_bigrams: Optional[set] = None) -> float:
    """
    Calculate the Response Appropriateness Score (RAS) for an interaction sequence.

    RAS measures whether the animal continues the syntactic chain after a system response:
    R = (Number of valid follow-up responses) / (Total system responses)

    An interaction is scored as positive if:
    1. System emits valid bigram response (e.g., 8→12)
    2. Animal responds within timeout window
    3. Animal's response forms valid bigram with system's cluster

    Args:
        interaction_sequence: List of InteractionEvent objects in temporal order
        valid_bigrams: Optional set of valid bigrams for syntax validation

    Returns:
        RAS score between 0 and 1
    """
    if valid_bigrams is None:
        valid_bigrams = set()

    positive_responses = 0
    total_system_responses = 0

    for i, interaction in enumerate(interaction_sequence):
        if interaction.source == "system":
            total_system_responses += 1

            # Check if animal responded with valid bigram
            if i + 1 < len(interaction_sequence):
                next_interaction = interaction_sequence[i + 1]
                if (next_interaction.source == "animal" and
                        (not valid_bigrams or
                         (interaction.cluster_id, next_interaction.cluster_id) in valid_bigrams)):
                    positive_responses += 1

    return positive_responses / max(total_system_responses, 1)


class InteractionAgent:
    """
    Main Interaction Agent for closed-loop communication.

    This agent bridges the Rust Execution Layer (feature extraction, synthesis)
    with the Python Logic Layer (cognitive processing, decision making).

    Usage:
        agent = InteractionAgent()
        agent.start()

        # Agent will automatically:
        # 1. Receive feature events from Rust
        # 2. Process through cognitive layer
        # 3. Send synthesis actions back to Rust

        # ... later ...
        agent.stop()
    """

    def __init__(
        self,
        config: Optional[InteractionAgentConfig] = None,
        on_feature_event: Optional[Callable[[FeatureEvent], None]] = None,
        on_context_change: Optional[Callable[[str, float], None]] = None,
        on_speaker_change: Optional[Callable[[str, float], None]] = None,
    ):
        """
        Initialize the Interaction Agent.

        Args:
            config: Agent configuration
            on_feature_event: Optional callback for feature events
            on_context_change: Optional callback for context changes
            on_speaker_change: Optional callback for speaker changes (Direction 3)
        """
        self.config = config or InteractionAgentConfig()
        self.on_feature_event = on_feature_event
        self.on_context_change = on_context_change
        self.on_speaker_change = on_speaker_change

        # Initialize components
        self.feature_subscriber = FeatureSubscriber(
            config=FeatureSubscriberConfig(
                event_endpoint=self.config.feature_endpoint,
                verbose_logging=self.config.verbose_logging,
            ),
            on_event=self._handle_feature_event,
        )

        self.action_publisher = ActionPublisher(
            config=ActionPublisherConfig(
                action_endpoint=self.config.action_endpoint,
            ),
        )

        # Initialize parsing strategy based on domain_mode (Sprint 1)
        self.parser = self.config.get_parsing_strategy()

        # Initialize ContextClassifier if path provided
        self.context_classifier = None
        if self.config.context_classifier_path:
            try:
                from realtime.context_classifier import ContextClassifier
                self.context_classifier = ContextClassifier.load(
                    self.config.context_classifier_path
                )
                logger.info(
                    f"Loaded ContextClassifier from {self.config.context_classifier_path}"
                )

                # Validate classifier labels against canonical ontology
                self._validate_classifier_labels()
            except Exception as e:
                logger.warning(
                    f"Failed to load ContextClassifier from "
                    f"{self.config.context_classifier_path}: {e}. "
                    f"Falling back to rule-based inference."
                )

        # Initialize SpeakerDatabase for Direction 3 (Speaker Embeddings)
        # Can be attached externally for speaker tracking
        self.speaker_db: Optional["SpeakerDatabase"] = None

        # State management
        self.state = AgentState.IDLE
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Context tracking
        self._current_context: Optional[str] = None
        self._context_confidence: float = 0.0

        # Speaker tracking (Direction 3)
        self._current_speaker: Optional[str] = None
        self._speaker_confidence: float = 0.0

        # v1.2.0: Cluster ID tracking for syntax validation
        self._last_cluster_id: Optional[int] = None

        # v1.3.0: Emitter ID tracking for speaker diarization
        self._last_emitter_id: Optional[int] = None

        self._last_response_time: float = 0.0

        # Statistics
        self._events_processed = 0
        self._responses_sent = 0
        self._start_time: Optional[float] = None

        # Parsing statistics (Sprint 1)
        self._idioms_detected = 0
        self._tokens_parsed = 0

        # v1.5.0: Ethological validation tracking
        self._interaction_history: List[InteractionEvent] = []
        self._session_start_time: Optional[float] = None
        self._session_metrics: Optional[SessionMetrics] = None
        self._last_system_response_cluster: Optional[int] = None
        self._last_system_response_time: Optional[float] = None

        # Generate session_id if not provided
        if self.config.enable_ethological_mode and self.config.session_id is None:
            import uuid
            self.config.session_id = f"session_{uuid.uuid4().hex[:8]}"

        logger.info("InteractionAgent initialized")
        logger.info(f"  Feature endpoint: {self.config.feature_endpoint}")
        logger.info(f"  Action endpoint: {self.config.action_endpoint}")
        logger.info(f"  Domain mode: {self.config.domain_mode}")
        logger.info(f"  Parser: {self.parser.name}")

    def start(self) -> None:
        """Start the interaction agent"""
        if self._running:
            logger.warning("Agent already running")
            return

        logger.info("Starting Interaction Agent...")

        # Connect components
        self.feature_subscriber.connect()
        self.action_publisher.connect()

        # Start subscriber
        self.feature_subscriber.start()

        self._running = True
        self._start_time = time.time()
        self.state = AgentState.LISTENING

        # v1.5.0: Initialize session metrics for ethological mode
        if self.config.enable_ethological_mode:
            self._session_start_time = time.time()
            self._session_metrics = SessionMetrics(
                session_id=self.config.session_id or "unknown",
                duration_seconds=0.0,
                condition=self.config.experimental_condition,
                start_time=self._session_start_time,
            )
            logger.info(f"Ethological validation mode enabled")
            logger.info(f"  Session ID: {self._session_metrics.session_id}")
            logger.info(f"  Condition: {self._session_metrics.condition}")

        logger.info("✓ Interaction Agent started")
        logger.info(f"  State: {self.state.value}")

    def stop(self) -> None:
        """Stop the interaction agent"""
        if not self._running:
            return

        logger.info("Stopping Interaction Agent...")

        self._running = False
        self.state = AgentState.IDLE

        # Stop components
        self.feature_subscriber.stop()
        self.action_publisher.disconnect()

        logger.info("✓ Interaction Agent stopped")

    def _handle_feature_event(self, event: FeatureEvent) -> None:
        """
        Handle a feature event from Rust.

        This is the main event handler that processes incoming features
        and generates synthesis responses.

        Args:
            event: FeatureEvent from Rust
        """
        self._events_processed += 1

        # Update state
        self.state = AgentState.LISTENING

        # Call optional callback
        if self.on_feature_event:
            try:
                self.on_feature_event(event)
            except Exception as e:
                logger.error(f"Error in feature event callback: {e}")

        # Process through cognitive layer
        result = self._process_features(event)

        # Check if we should respond
        if self._should_respond(result):
            self._send_response(result, event)

        # Check for context change
        if result.get("context_state") != self._current_context:
            old_context = self._current_context
            self._current_context = str(result.get("context_state", "unknown"))
            self._context_confidence = result.get("confidence", 0.0)

            if self.on_context_change:
                try:
                    self.on_context_change(self._current_context, self._context_confidence)
                except Exception as e:
                    logger.error(f"Error in context change callback: {e}")

            logger.debug(f"Context changed: {old_context} -> {self._current_context}")

        # Check for speaker change (Direction 3)
        speaker_id = result.get("speaker_id")
        if speaker_id != self._current_speaker:
            old_speaker = self._current_speaker
            self._current_speaker = speaker_id
            self._speaker_confidence = result.get("speaker_confidence", 0.0)

            if self.on_speaker_change and speaker_id is not None:
                try:
                    self.on_speaker_change(speaker_id, self._speaker_confidence)
                except Exception as e:
                    logger.error(f"Error in speaker change callback: {e}")

            if self.config.verbose_logging and speaker_id is not None:
                logger.debug(f"Speaker changed: {old_speaker} -> {speaker_id} (confidence: {self._speaker_confidence:.2f})")

        # v1.2.0: Track last cluster_id for bigram validation
        self._last_cluster_id = event.cluster_id

        # v1.3.0: Track last emitter_id for speaker diarization
        self._last_emitter_id = event.emitter_id

        # v1.5.0: Track animal interaction events for RAS
        if self.config.enable_ethological_mode:
            self._track_animal_event(event)

    def _process_features(self, event: FeatureEvent) -> Dict[str, Any]:
        """
        Process features through cognitive layer.

        This method integrates the parsing strategy (Sprint 1) with
        context detection and synthesis parameter generation.

        Args:
            event: Feature event to process

        Returns:
            Processing result with context, parsed tokens, and synthesis parameters
        """
        # Parse segment sequence using the strategy pattern (Sprint 1)
        # Use cluster_id as the segment ID for parsing (sequence is just an ordering counter)
        parse_result: Optional[ParseResult] = None
        segment_sequence = [event.cluster_id]
        if segment_sequence:
            parse_result = self.parser.parse(segment_sequence)
            self._tokens_parsed += len(parse_result.tokens)
            self._idioms_detected += parse_result.idiom_count

            # Log idiom detection for bat mode
            if parse_result.idiom_count > 0 and self.config.verbose_logging:
                logger.info(f"Detected {parse_result.idiom_count} idiom(s) in sequence")

        # Simplified context inference from 112D features
        # v1.2.0: Now passes cluster_id for cluster-based context inference
        context, inferred_confidence = self._infer_context(
            event.features_112d,
            event.emitter_id,
            cluster_id=event.cluster_id
        )

        # v1.2.0: Use Rust Student confidence if available, otherwise use inferred
        if event.confidence is not None:
            confidence = event.confidence
            confidence_source = "rust_student"
        else:
            confidence = inferred_confidence
            confidence_source = "inferred"

        # v1.2.0: Bigram syntax validation
        bigram_valid = self._validate_bigram(event.cluster_id)

        # v1.3.0: Speaker profile lookup for Level 2 grounding
        speaker_profile = self._get_speaker_profile(event.emitter_id)
        speaker_bias_multiplier = 1.0

        if speaker_profile is not None:
            # Get context-specific response bias
            speaker_bias_multiplier = speaker_profile.get_response_bias(context)
            if self.config.verbose_logging:
                logger.debug(
                    f"Speaker profile: emitter_id={event.emitter_id}, "
                    f"rank={speaker_profile.dominance_rank}, "
                    f"context={context}, bias={speaker_bias_multiplier:.2f}"
                )

        # v1.4.0: Bigram probability lookup for Markov chain weighting
        bigram_probability = self.config.default_bigram_probability
        bigram_rarity_score = 1.0 - bigram_probability
        cognitive_attention = False

        if self.config.enable_probabilistic_weighting and self.config.bigram_probability_map:
            bigram_key = (self._last_cluster_id, event.cluster_id)
            if bigram_key in self.config.bigram_probability_map:
                bp = self.config.bigram_probability_map[bigram_key]
                bigram_probability = bp.probability
                bigram_rarity_score = bp.rarity_score

                # Check if rarity triggers cognitive attention
                if bigram_rarity_score > self.config.rarity_attention_threshold:
                    cognitive_attention = True

                if self.config.verbose_logging:
                    logger.debug(
                        f"Bigram {bigram_key}: prob={bigram_probability:.3f}, "
                        f"rarity={bigram_rarity_score:.3f}, "
                        f"attention={cognitive_attention}"
                    )
            elif self._last_cluster_id is not None:
                # Known bigram but not in map (use default)
                pass
        elif not self.config.enable_probabilistic_weighting:
            # When probabilistic weighting is disabled, use neutral values
            bigram_probability = 1.0
            bigram_rarity_score = 0.0

        # Calculate effective confidence combining all modifiers
        effective_confidence = confidence * speaker_bias_multiplier

        # v1.4.0: Apply bigram probability weighting
        if self.config.enable_probabilistic_weighting and self.config.bigram_probability_map:
            # Common bigrams boost confidence, rare bigrams reduce it
            # Use multiplicative modulation: confidence × (0.5 + probability)
            # This ensures: high prob (>0.5) boosts, low prob (<0.5) reduces
            probability_multiplier = 0.5 + bigram_probability
            effective_confidence = effective_confidence * probability_multiplier

        # Speaker identification if speaker_db is available
        speaker_id = None
        speaker_confidence = None
        if self.speaker_db is not None and event.speaker_embedding is not None:
            try:
                matches = self.speaker_db.identify(event.speaker_embedding, top_k=1)
                if matches:
                    speaker_id, speaker_confidence = matches[0]
                    if self.config.verbose_logging:
                        logger.debug(f"Identified speaker: {speaker_id} (confidence: {speaker_confidence:.2f})")
            except Exception as e:
                logger.warning(f"Speaker identification failed: {e}")

        # Build result with parsed tokens and speaker info
        result = {
            "context_state": context,
            "confidence": confidence,
            "confidence_source": confidence_source,  # v1.2.0
            "cluster_id": event.cluster_id,
            "sequence": event.sequence,
            "timestamp": event.timestamp,
            "features_112d": event.features_112d,
            "emitter_id": event.emitter_id,
            "parse_result": parse_result,  # Sprint 1: Include parsed tokens
            "strategy_used": self.parser.name,  # Sprint 1: Track which strategy was used
            "speaker_id": speaker_id,  # Direction 3: Identified speaker
            "speaker_confidence": speaker_confidence,  # Direction 3: Speaker identification confidence
            "uncertainty": event.uncertainty,  # Module 1: Uncertainty from NBD
            "bigram_valid": bigram_valid,  # v1.2.0: Syntax validation
            # v1.3.0: Level 2 speaker grounding
            "speaker_profile": speaker_profile,
            "speaker_bias_multiplier": speaker_bias_multiplier,
            # v1.4.0: Probabilistic bigram weighting
            "bigram_probability": bigram_probability,
            "bigram_rarity_score": bigram_rarity_score,
            "cognitive_attention": cognitive_attention,
            "effective_confidence": effective_confidence,
        }

        return result

    def _validate_classifier_labels(self) -> None:
        """
        Validate that all classifier labels map to canonical response contexts.

        Raises:
            ValueError: If a classifier label cannot be mapped to a canonical context
        """
        if self.context_classifier is None:
            return

        canonical_contexts = set(self.config.canonical_contexts)
        class_names = set(self.context_classifier.class_names)

        # Check which labels need mapping
        unmapped = class_names - canonical_contexts

        if not unmapped:
            logger.info("All classifier labels are in canonical ontology")
            return

        # Check if we have mappings for the unmapped labels
        if self.config.context_label_mapping:
            mapped_contexts = set(self.config.context_label_mapping.values())
            still_unmapped = unmapped - mapped_contexts - canonical_contexts

            if not still_unmapped:
                logger.info(f"All labels mapped via context_label_mapping")
                return

        # Build default mapping for pseudo-labels (context_0 -> social, etc.)
        # as a fallback warning
        pseudo_labels = {name for name in unmapped if name.startswith("context_")}
        if pseudo_labels:
            logger.warning(
                f"Classifier has {len(pseudo_labels)} pseudo-labels (e.g., 'context_0') "
                f"that are not in canonical ontology: {canonical_contexts}. "
                f"Use 'context_label_mapping' config to map these to canonical contexts. "
                f"Unmapped labels will cause the agent to not respond."
            )

    def _map_to_canonical_context(self, raw_context: str) -> str:
        """
        Map a raw classifier label to a canonical response context.

        Args:
            raw_context: The raw context label from the classifier

        Returns:
            A canonical context string
        """
        # If already canonical, return as-is
        if raw_context in self.config.canonical_contexts:
            return raw_context

        # Try to map using the provided mapping
        if self.config.context_label_mapping:
            mapped = self.config.context_label_mapping.get(raw_context)
            if mapped and mapped in self.config.canonical_contexts:
                logger.debug(f"Mapped '{raw_context}' -> '{mapped}'")
                return mapped

        # No mapping found - return the raw context but this will cause
        # the agent to not respond (as intended - fail safe)
        logger.warning(
            f"No mapping for context '{raw_context}' to canonical ontology. "
            f"Agent will not respond to this context."
        )
        return raw_context

    def _infer_context(self, features_112d: np.ndarray, emitter_id: Optional[int] = None, cluster_id: Optional[int] = None) -> Tuple[str, float]:
        """
        Infer behavioral context from 112D features and emitter identity.

        v1.2.0: Priority order is:
        1. Cluster-based context map (BGMM-distilled archetypes)
        2. ContextClassifier ML model
        3. Rule-based fallback

        Args:
            features_112d: 112D feature vector
            emitter_id: Optional emitter identity from source separation
            cluster_id: Optional cluster ID from Rust Student (0-44)

        Returns:
            Tuple of (context string, confidence score)
        """
        # v1.2.0: Primary - Use pre-computed cluster archetype context
        if cluster_id is not None and self.config.cluster_context_map is not None:
            if cluster_id in self.config.cluster_context_map:
                context = self.config.cluster_context_map[cluster_id]
                # For cluster-based inference, use high confidence since
                # the context is derived from the archetype, not noisy instance
                confidence = 0.85
                logger.debug(f"Cluster-based inference: cluster={cluster_id} → context={context}")
                return context, confidence

        # Use ContextClassifier if available
        if self.context_classifier is not None:
            try:
                raw_context, confidence = self.context_classifier.predict(features_112d)
                # Map to canonical context
                context = self._map_to_canonical_context(raw_context)
                logger.debug(f"ML inference: raw={raw_context}, mapped={context}, confidence={confidence:.2f}")
                return context, confidence
            except Exception as e:
                logger.warning(f"ContextClassifier prediction failed: {e}. Falling back to rules.")

        # Fallback to rule-based inference
        f0 = float(features_112d[0]) if features_112d[0] > 0 else 5000.0
        rms = float(features_112d[1]) if len(features_112d) > 1 else 0.5

        # Simple rule-based context inference
        if f0 > 8000 and rms > 0.6:
            context = "alarm"
        elif f0 > 6000:
            context = "territorial"
        elif f0 < 4000:
            context = "social"
        else:
            context = "contact"

        # Calculate confidence from variance heuristic for rule-based fallback
        confidence = self._calculate_confidence(features_112d, context)
        return context, confidence

    def _calculate_confidence(self, features_112d: np.ndarray, context: str) -> float:
        """Calculate confidence in context detection."""
        # Simple confidence based on feature variance
        variance = np.var(features_112d)
        return min(0.95, max(0.3, 0.5 + variance * 0.1))

    def _validate_bigram(self, current_cluster_id: int) -> bool:
        """
        v1.2.0: Validate that the current cluster follows valid syntax.

        Checks if (last_cluster_id, current_cluster_id) is in the set of
        valid bigrams from LRN-6 analysis.

        Args:
            current_cluster_id: The cluster ID of the current event

        Returns:
            True if the bigram is valid (or no validation configured)
        """
        # If no bigrams configured, skip validation
        if self.config.valid_bigrams is None:
            return True

        # First event (no previous cluster) is always valid
        if self._last_cluster_id is None:
            return True

        # Check if the bigram is in the valid set
        bigram = (self._last_cluster_id, current_cluster_id)
        return bigram in self.config.valid_bigrams

    def _get_speaker_profile(self, emitter_id: Optional[int]) -> Optional[SpeakerProfile]:
        """
        v1.3.0: Get speaker profile for an emitter_id.

        Args:
            emitter_id: The emitter ID from Rust source separation

        Returns:
            SpeakerProfile if found and speaker adaptation enabled, else None
        """
        # If speaker adaptation is disabled, always return None
        if not self.config.enable_speaker_adaptation:
            return None

        # If no speaker profiles configured, return None
        if self.config.speaker_profiles is None:
            return None

        # If emitter_id is None, return None
        if emitter_id is None:
            return None

        # Look up the profile
        return self.config.speaker_profiles.get(emitter_id)

    def _should_respond(self, result: Dict[str, Any]) -> bool:
        """
        Determine if the agent should generate a response.

        v1.2.0: Now checks Rust Student confidence threshold and bigram validity.
        v1.3.0: Now applies speaker-specific response bias.
        v1.4.0: Now uses probability-weighted effective_confidence.

        Args:
            result: Processing result

        Returns:
            True if should respond
        """
        # Check rate limiting
        current_time = time.time()
        time_since_last = (current_time - self._last_response_time) * 1000

        if time_since_last < self.config.response_cooldown_ms:
            return False

        # v1.4.0: Use effective_confidence (includes probability weighting)
        # Fallback to base confidence if effective_confidence not available
        effective_confidence = result.get("effective_confidence", result.get("confidence", 0.0))

        if effective_confidence < self.config.confidence_threshold:
            base_confidence = result.get("confidence", 0.0)
            speaker_bias = result.get("speaker_bias_multiplier", 1.0)
            bigram_prob = result.get("bigram_probability", 1.0)
            logger.debug(
                f"Effective confidence {effective_confidence:.2f} "
                f"(base={base_confidence:.2f} × speaker={speaker_bias:.2f} × bigram={bigram_prob:.2f}) "
                f"< threshold {self.config.confidence_threshold:.2f}"
            )
            return False

        # v1.3.0: Check speaker bias threshold
        speaker_bias_multiplier = result.get("speaker_bias_multiplier", 1.0)
        if speaker_bias_multiplier < self.config.speaker_bias_threshold:
            logger.debug(
                f"Speaker bias {speaker_bias_multiplier:.2f} < threshold {self.config.speaker_bias_threshold:.2f}"
            )
            return False

        # v1.2.0: Check bigram syntax validity
        bigram_valid = result.get("bigram_valid", True)
        if not bigram_valid:
            logger.debug(f"Bigram ({self._last_cluster_id}, {result.get('cluster_id')}) is invalid")
            return False

        # Check uncertainty threshold (Module 1: Uncertainty Quantification)
        uncertainty = result.get("uncertainty", None)
        if uncertainty is not None and uncertainty > self.config.uncertainty_threshold:
            return False

        # Check context - must be in canonical ontology and response-enabled
        context = result.get("context_state", "")
        response_contexts = set(self.config.canonical_contexts) - {"social"}  # social doesn't trigger response

        return context in response_contexts

    def _send_response(self, result: Dict[str, Any], event: FeatureEvent) -> None:
        """
        Send synthesis response to Rust.

        Args:
            result: Processing result
            event: Original feature event
        """
        self.state = AgentState.RESPONDING

        # Create timeline
        cluster_id = result.get("cluster_id", event.cluster_id)
        context = result.get("context_state", "contact")

        timeline = self._create_response_timeline(cluster_id, context)
        deltas = self._create_deltas(context)

        # Send action
        success = self.action_publisher.publish_timeline(
            timeline=timeline,
            deltas=deltas,
            priority="normal",
        )

        if success:
            self._responses_sent += 1
            self._last_response_time = time.time()
            logger.debug(f"Sent response: cluster={cluster_id}, context={context}")

            # v1.5.0: Track system response for RAS
            if self.config.enable_ethological_mode:
                self._track_system_response(cluster_id)

        # Return to listening
        self.state = AgentState.LISTENING

    def _create_response_timeline(
        self,
        cluster_id: int,
        context: str,
    ) -> List[TimelineEvent]:
        """Create synthesis timeline for response."""
        # Adjust timing based on context
        if context == "alarm":
            duration = 100.0
            amplitude = 0.9
        elif context == "territorial":
            duration = 200.0
            amplitude = 0.85
        else:  # contact
            duration = 150.0
            amplitude = 0.75

        return [
            TimelineEvent(
                cluster_id=cluster_id,
                start_time_ms=0.0,
                duration_ms=duration,
                amplitude=amplitude,
            )
        ]

    def _create_deltas(self, context: str) -> Optional[MicroDynamicsDelta]:
        """Create micro-dynamics deltas based on context."""
        if context == "alarm":
            return MicroDynamicsDelta(
                delta_mean_f0_hz=500.0,
                delta_rms_energy=0.2,
            )
        elif context == "territorial":
            return MicroDynamicsDelta(
                delta_mean_f0_hz=200.0,
                delta_duration_ms=20.0,
            )
        elif context == "social":
            return MicroDynamicsDelta(
                delta_mean_f0_hz=-100.0,
                delta_sustain_level=0.1,
            )
        else:
            return None

    # ========================================================================
    # v1.5.0: Ethological Validation Methods
    # ========================================================================

    def _track_animal_event(self, event: FeatureEvent) -> None:
        """
        Track an animal vocalization event for RAS calculation.

        Args:
            event: The animal's feature event
        """
        if not self.config.enable_ethological_mode:
            return

        current_time = time.time()

        # Calculate time since previous event
        time_since_previous = 0.0
        if self._interaction_history:
            time_since_previous = current_time - self._interaction_history[-1].timestamp

        # Create interaction event
        interaction = InteractionEvent(
            timestamp=current_time,
            source="animal",
            cluster_id=event.cluster_id,
            emitter_id=event.emitter_id,
            response_to=self._last_system_response_cluster,
            time_since_previous=time_since_previous,
        )

        # Add to history (with bounded size)
        self._interaction_history.append(interaction)
        if len(self._interaction_history) > self.config.max_interaction_history:
            self._interaction_history.pop(0)

        # Update session metrics
        if self._session_metrics is not None:
            self._session_metrics.total_animal_vocalizations += 1

            # Check if this is a valid follow-up to system response
            if (interaction.response_to is not None and
                    self.config.valid_bigrams is not None):
                bigram = (interaction.response_to, event.cluster_id)
                if bigram in self.config.valid_bigrams:
                    self._session_metrics.positive_responses += 1
                else:
                    self._session_metrics.negative_responses += 1

            # Update RAS score
            self._session_metrics.ras_score = calculate_ras(
                self._interaction_history,
                self.config.valid_bigrams,
            )

            # Update duration
            self._session_metrics.duration_seconds = current_time - (self._session_start_time or current_time)

    def _track_system_response(self, cluster_id: int) -> None:
        """
        Track a system response event for RAS calculation.

        Args:
            cluster_id: The cluster ID of the system's response
        """
        if not self.config.enable_ethological_mode:
            return

        current_time = time.time()

        # Calculate time since previous event
        time_since_previous = 0.0
        if self._interaction_history:
            time_since_previous = current_time - self._interaction_history[-1].timestamp

        # Create interaction event
        interaction = InteractionEvent(
            timestamp=current_time,
            source="system",
            cluster_id=cluster_id,
            emitter_id=None,  # System has no emitter_id
            response_to=self._last_cluster_id,  # System responded to animal
            time_since_previous=time_since_previous,
        )

        # Add to history
        self._interaction_history.append(interaction)
        if len(self._interaction_history) > self.config.max_interaction_history:
            self._interaction_history.pop(0)

        # Track for next animal event
        self._last_system_response_cluster = cluster_id
        self._last_system_response_time = current_time

        # Update session metrics
        if self._session_metrics is not None:
            self._session_metrics.total_system_responses += 1

    def get_session_metrics(self) -> Optional[SessionMetrics]:
        """
        Get current session metrics for ethological validation.

        Returns:
            SessionMetrics if ethological mode is enabled, else None
        """
        if not self.config.enable_ethological_mode:
            return None

        # Update duration before returning
        if self._session_metrics is not None and self._session_start_time is not None:
            self._session_metrics.duration_seconds = time.time() - self._session_start_time

        return self._session_metrics

    def get_interaction_history(self) -> List[InteractionEvent]:
        """
        Get the interaction history for analysis.

        Returns:
            List of InteractionEvent objects
        """
        return self._interaction_history.copy()

    def calculate_current_ras(self) -> float:
        """
        Calculate the current RAS score for this session.

        Returns:
            RAS score between 0 and 1
        """
        if not self.config.enable_ethological_mode:
            return 0.0

        return calculate_ras(self._interaction_history, self.config.valid_bigrams)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        uptime = time.time() - self._start_time if self._start_time else 0.0

        return {
            "state": self.state.value,
            "running": self._running,
            "uptime_seconds": uptime,
            "events_processed": self._events_processed,
            "responses_sent": self._responses_sent,
            "current_context": self._current_context,
            "context_confidence": self._context_confidence,
            "events_per_second": self._events_processed / max(uptime, 1.0),
            "responses_per_second": self._responses_sent / max(uptime, 1.0),
            "feature_subscriber": self.feature_subscriber.get_stats(),
            "action_publisher": self.action_publisher.get_stats(),
            # Sprint 1: Parsing statistics
            "parsing": {
                "strategy": self.parser.name,
                "domain_mode": self.config.domain_mode,
                "is_holophrastic": self.parser.is_holophrastic,
                "tokens_parsed": self._tokens_parsed,
                "idioms_detected": self._idioms_detected,
            },
            # v1.3.0: Speaker tracking statistics
            "speaker_tracking": {
                "last_emitter_id": self._last_emitter_id,
                "speaker_adaptation_enabled": self.config.enable_speaker_adaptation,
                "speaker_profiles_count": len(self.config.speaker_profiles) if self.config.speaker_profiles else 0,
            },
            # v1.5.0: Ethological validation statistics
            "ethological_validation": {
                "enabled": self.config.enable_ethological_mode,
                "session_id": self._session_metrics.session_id if self._session_metrics else None,
                "condition": self._session_metrics.condition if self._session_metrics else None,
                "ras_score": self._session_metrics.ras_score if self._session_metrics else 0.0,
                "total_animal_vocalizations": self._session_metrics.total_animal_vocalizations if self._session_metrics else 0,
                "total_system_responses": self._session_metrics.total_system_responses if self._session_metrics else 0,
                "positive_responses": self._session_metrics.positive_responses if self._session_metrics else 0,
                "negative_responses": self._session_metrics.negative_responses if self._session_metrics else 0,
                "interaction_history_size": len(self._interaction_history),
            },
        }

    def is_running(self) -> bool:
        """Check if agent is running."""
        return self._running

    @property
    def current_state(self) -> AgentState:
        """Get current agent state."""
        return self.state

    @property
    def current_context(self) -> Optional[str]:
        """Get current detected context."""
        return self._current_context

    @property
    def current_speaker(self) -> Optional[str]:
        """Get current identified speaker (Direction 3)."""
        return self._current_speaker


def create_test_agent(
    feature_endpoint: str = "ipc:///tmp/test_features.ipc",
    action_endpoint: str = "ipc:///tmp/test_actions.ipc",
) -> InteractionAgent:
    """
    Create a test agent with custom endpoints.

    Args:
        feature_endpoint: ZeroMQ endpoint for feature events
        action_endpoint: ZeroMQ endpoint for action commands

    Returns:
        Configured InteractionAgent
    """
    config = InteractionAgentConfig(
        feature_endpoint=feature_endpoint,
        action_endpoint=action_endpoint,
        verbose_logging=True,
    )
    return InteractionAgent(config=config)


if __name__ == "__main__":
    # Demo/test mode
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 60)
    print("Interaction Agent - Closed-Loop Cognitive Agent")
    print("=" * 60)

    def on_feature(event: FeatureEvent):
        print(f"[EVENT] Cluster {event.cluster_id}, Seq {event.sequence}")

    def on_context(context: str, confidence: float):
        print(f"[CONTEXT] {context} ({confidence:.2f})")

    agent = InteractionAgent(
        on_feature_event=on_feature,
        on_context_change=on_context,
    )

    print("\nStarting agent (Ctrl+C to stop)...")
    try:
        agent.start()

        # Print stats every 5 seconds
        while True:
            time.sleep(5.0)
            stats = agent.get_stats()
            print("\n--- Stats ---")
            print(f"State: {stats['state']}")
            print(f"Events: {stats['events_processed']}")
            print(f"Responses: {stats['responses_sent']}")
            print(f"Context: {stats['current_context']} ({stats['context_confidence']:.2f})")

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        agent.stop()
        print("\nFinal Stats:")
        for k, v in agent.get_stats().items():
            print(f"  {k}: {v}")
