"""
Ethological Validation Module

Provides biologically-accurate validation metrics for animal-AI interactions,
replacing naive metrics like "response rate" and "Response Appropriateness Score"
with ethologically-grounded measurements.

Key Components:
- TaxaTemporalProfile: Species-specific temporal constraints
- TemporalGate: Validates response latencies against biological windows
- AcousticConvergenceEngine: Measures vocal dialect matching
- ProsodicDTW: DTW-based prosodic similarity against natural baselines
- MultiFactorAcceptanceScore: Fused metric combining all three

Example Usage:
    >>> from ethological_validation import (
    ...     create_mfas_for_species,
    ...     InteractionEvent,
    ...     get_temporal_gate,
    ... )
    >>>
    >>> # Create MFAS calculator for a species
    >>> mfas = create_mfas_for_species("rousettus_aegyptiacus", baseline_contours)
    >>>
    >>> # Evaluate an interaction
    >>> event = InteractionEvent(
    ...     species="rousettus_aegyptiacus",
    ...     ai_output_state=ai_affect,
    ...     animal_pre_state=animal_pre,
    ...     animal_post_state=animal_post,
    ...     animal_f0_contour=f0_contour,
    ...     ai_end_time_ms=1000,
    ...     animal_response_time_ms=1090,
    ... )
    >>> result = mfas.evaluate_interaction(event)
    >>> print(f"MFAS: {result.mfas_score:.3f}")

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from .acoustic_convergence import (
    AcousticConvergenceEngine,
    ConvergenceResult,
    MultiDimensionalConvergence,
    compute_batch_convergence,
    compute_convergence_from_affect_vectors,
)
from .mfas import (
    InteractionEvent,
    MFASComparator,
    MFASResult,
    MultiFactorAcceptanceScore,
    BAT_MFAS,
    MARMOSET_MFAS,
    create_mfas_for_species,
)
from .prosodic_dtw import (
    DTWResult,
    FastDTW,
    ProsodicDTW,
    ProsodicFeature,
    ProsodicFeatureExtractor,
    DEFAULT_FEATURE_EXTRACTOR,
    DEFAULT_PROSODIC_DTW,
)
from .taxa_profiles import (
    SPECIES_PROFILES,
    TaxaTemporalProfile,
    TemporalGate,
    analyze_corpus_latencies,
    create_custom_profile,
    get_temporal_gate,
)

__all__ = [
    # Taxa Profiles
    "SPECIES_PROFILES",
    "TaxaTemporalProfile",
    "TemporalGate",
    "get_temporal_gate",
    "create_custom_profile",
    "analyze_corpus_latencies",
    # Acoustic Convergence
    "AcousticConvergenceEngine",
    "ConvergenceResult",
    "MultiDimensionalConvergence",
    "compute_convergence_from_affect_vectors",
    "compute_batch_convergence",
    # Prosodic DTW
    "FastDTW",
    "ProsodicDTW",
    "ProsodicFeature",
    "ProsodicFeatureExtractor",
    "DTWResult",
    "DEFAULT_PROSODIC_DTW",
    "DEFAULT_FEATURE_EXTRACTOR",
    # MFAS
    "InteractionEvent",
    "MFASResult",
    "MultiFactorAcceptanceScore",
    "MFASComparator",
    "create_mfas_for_species",
    "BAT_MFAS",
    "MARMOSET_MFAS",
]

__version__ = "1.0.0"
