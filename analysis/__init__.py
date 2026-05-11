"""
Analysis Module

Advanced ethological analysis frameworks enabled by the dual-stream
continuous latent-space architecture.

This module contains six novel analysis frameworks that go beyond
the old discrete clustering paradigm:

1. **Graded Continuum Analysis** (graded_continuum.py)
   - Maps continuous dispute trajectories in 16D affect space
   - Identifies "tipping points" where spatial disputes become physical
   - Replaces discrete GMM buckets with continuous trajectory analysis

2. **Micro-Phonology Discovery** (micro_phonology.py)
   - Discovers sub-50ms phonetic units using CPC/Mamba NBD
   - Analyzes combinatorial phonology (A+B ≠ B+A)
   - Replaces old 50ms debounce that merged rapid trills

3. **Dialect Forcing Protocol** (dialect_forcing.py)
   - Active vocal learning experiments via latent-space interpolation
   - SLERP-like interpolation between dialect prototypes
   - Tests "crowd-based vocal learning" hypothesis

4. **Broadcast/Unicast Classifier** (addressing_classifier.py)
   - Classifies vocalizations as broadcast (colony) or unicast (individual)
   - Uses multi-modal evidence: spatial, syntactic, affective
   - Enables social network analysis

5. **Syntactic Surprise Analysis** (syntactic_surprise.py)
   - Uses autoregressive transformer probabilities to compute surprise
   - Measures information-theoretic surprise (negative log-likelihood)
   - Detects rule-breaking innovation and potential deception

6. **Ethological Turing Test** (turing_test.py)
   - DTW-based comparison of AI-bat vs natural bat-bat conversations
   - Multi-dimensional prosodic comparison (F0, RMS, centroid, affect)
   - Determines if AI passes "naturalistic interaction" test

## Feature Extraction Pipeline

The `feature_pipeline` module provides the data pipeline to extract
112D RosettaFeatures, 16D affect vectors, and VQ-VAE tokens from
raw audio, enabling all analysis frameworks.

Example Usage:
    >>> from analysis import (
    ...     GradedContinuumAnalyzer,
    ...     MicroPhonologyAnalyzer,
    ...     DialectForcer,
    ...     AddressingClassifier,
    ...     SyntacticSurpriseAnalyzer,
    ...     EthologicalTuringTest,
    ...     FeaturePipeline,
    ... )
    >>>
    >>> # Extract features from audio
    >>> pipeline = FeaturePipeline()
    >>> features = pipeline.process_audio_file('bat_call.wav', 'seg_001')
    >>>
    >>> # Analyze with frameworks
    >>> analyzer = GradedContinuumAnalyzer()
    >>> dispute = analyzer.analyze_dispute(...)
    >>>
    >>> # Classify addressing
    >>> classifier = AddressingClassifier()
    >>> result = classifier.classify(
    ...     syntactic_token=features.syntactic_token,
    ...     affect_vector=features.affect_vector_16d,
    ... )

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

# Graded Continuum Analysis
from .graded_continuum import (
    DisputeSegment,
    DisputeTrajectory,
    TippingPointAnalysis,
    GradedContinuumAnalyzer,
    DEFAULT_GRADED_CONTINUUM,
)

# Micro-Phonology Discovery
from .micro_phonology import (
    MicroUnit,
    PhonemeSequence,
    PhonotacticRule,
    MicroPhonologyAnalyzer,
    BAT_MICRO_PHONOLOGY,
    visualize_phonotactic_rules,
)

# Dialect Forcing Protocol
from .dialect_forcing import (
    DialectType,
    DialectDefinition,
    DialectForcingTrial,
    DialectForcer,
    DEFAULT_DIALECT_FORCER,
)

# Broadcast/Unicast Classification
from .addressing_classifier import (
    AddressMode,
    AddressingClassification,
    AddressingPattern,
    AddressingClassifier,
    DEFAULT_ADDRESSING_CLASSIFIER,
)

# Syntactic Surprise Analysis
from .syntactic_surprise import (
    SurpriseEvent,
    SurpriseProfile,
    SyntacticSurpriseAnalyzer,
    DEFAULT_SYNTACTIC_SURPRISE,
)

# Ethological Turing Test
from .turing_test import (
    ConversationType,
    ProsodicTrajectory,
    DTWResult,
    TuringTestResult,
    EthologicalTuringTest,
    dtw_distance,
    DEFAULT_TURING_TEST,
)

# Feature Extraction Pipeline
from .feature_pipeline import (
    AudioSegment,
    ExtractedFeatures,
    RosettaFeatureExtractor,
    AffectiveVAEEncoder,
    SyntacticVQVAEEncoder,
    FeaturePipeline,
    DEFAULT_ROSETTA_EXTRACTOR,
    DEFAULT_VAE_ENCODER,
    DEFAULT_VQVAE_ENCODER,
    DEFAULT_FEATURE_PIPELINE,
)

__all__ = [
    # Graded Continuum
    "DisputeSegment",
    "DisputeTrajectory",
    "TippingPointAnalysis",
    "GradedContinuumAnalyzer",
    "DEFAULT_GRADED_CONTINUUM",
    # Micro-Phonology
    "MicroUnit",
    "PhonemeSequence",
    "PhonotacticRule",
    "MicroPhonologyAnalyzer",
    "BAT_MICRO_PHONOLOGY",
    "visualize_phonotactic_rules",
    # Dialect Forcing
    "DialectType",
    "DialectDefinition",
    "DialectForcingTrial",
    "DialectForcer",
    "DEFAULT_DIALECT_FORCER",
    # Addressing Classification
    "AddressMode",
    "AddressingClassification",
    "AddressingPattern",
    "AddressingClassifier",
    "DEFAULT_ADDRESSING_CLASSIFIER",
    # Syntactic Surprise
    "SurpriseEvent",
    "SurpriseProfile",
    "SyntacticSurpriseAnalyzer",
    "DEFAULT_SYNTACTIC_SURPRISE",
    # Turing Test
    "ConversationType",
    "ProsodicTrajectory",
    "DTWResult",
    "TuringTestResult",
    "EthologicalTuringTest",
    "dtw_distance",
    "DEFAULT_TURING_TEST",
    # Feature Pipeline
    "AudioSegment",
    "ExtractedFeatures",
    "RosettaFeatureExtractor",
    "AffectiveVAEEncoder",
    "SyntacticVQVAEEncoder",
    "FeaturePipeline",
    "DEFAULT_ROSETTA_EXTRACTOR",
    "DEFAULT_VAE_ENCODER",
    "DEFAULT_VQVAE_ENCODER",
    "DEFAULT_FEATURE_PIPELINE",
]

__version__ = "1.0.0"
