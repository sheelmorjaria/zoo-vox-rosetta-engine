//! Technical Architecture - Rust Execution Layer
//! ===============================================
//!
//! This crate provides the Rust execution layer for the animal vocalization
//! analysis system. It handles all time-critical operations including:
//!
//! - Source separation using Conv-TasNet (via ONNX/Tract)
//! - Real-time audio synthesis with granular engines
//! - Thermal management and power governance
//! - Safety monitoring with watchdog timers
//! - IEEE 1588 PTP for precision timing
//! - Deterministic provenance logging
//!
//! Architecture Strategy:
//! ----------------------
//! This crate follows the "Execution vs. Logic" split:
//!
//! - **Execution Layer (Rust)**: Signal processing, hardware access, safety
//! - **Logic Layer (Python)**: Cognitive intelligence, decision making, learning
//!
//! The crate exposes a clean PyO3 interface for Python integration.
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

// ═══════════════════════════════════════════════════════════════════════════════
// BORING RUST: Lint Configuration
// ═══════════════════════════════════════════════════════════════════════════════
//
// This crate follows Boring Rust principles with some relaxations for
// scientific computing/audio DSP. See clippy.toml for threshold configuration.
//
// Tier 1 (Default): Strict rules for agent-generated code
// Tier 2 (#[hot_path]): Relaxed complexity for performance-critical code
// Tier 3 (#[human_authored]): Unrestricted, agent treats as black box
// ═══════════════════════════════════════════════════════════════════════════════

// PyO3's #[pymethods] macro generates non-local impl blocks, which triggers warnings.
// JUSTIFICATION: Known false positive from PyO3 macro, safe to suppress.
#![cfg_attr(feature = "python-bindings", allow(non_local_definitions))]
// ─────────────────────────────────────────────────────────────────────────────────
// SAFETY-CRITICAL: These are NEVER allowed
// ─────────────────────────────────────────────────────────────────────────────────
// Note: These are enforced via clippy.toml and Cargo.toml [lints] section.
// - unwrap_used: DENY - Use .context("...")? instead
// - expect_used: DENY - Use .context("...")? instead
// - panic: DENY - No panics in production code
// - indexing_slicing: DENY - Use .get(i) instead of [i]

// ─────────────────────────────────────────────────────────────────────────────────
// ALLOWED WITH JUSTIFICATION: Scientific Computing/DSP-specific relaxations
// ─────────────────────────────────────────────────────────────────────────────────

// JUSTIFICATION: Library crate with many public API items not used internally.
// Items are part of the public API and used by downstream consumers.
#![allow(dead_code)]
// JUSTIFICATION: Signal processing functions legitimately need many parameters
// (sample_rate, window_size, hop_size, n_mels, fmin, fmax, etc.)
// TODO: Consider using config structs for functions with >5 params
#![allow(clippy::too_many_arguments)]
// JUSTIFICATION: Complex numeric types (ndarray::Array2<f32>, etc.) are common
// in DSP and cannot be simplified without loss of clarity.
#![allow(clippy::type_complexity)]
// JUSTIFICATION: Index-based loops are often clearer for DSP algorithms where
// the index has semantic meaning (sample index, frame number, etc.)
#![allow(clippy::needless_range_loop)]
// JUSTIFICATION: Some new() constructors don't need Default impl (e.g., with required params)
#![allow(clippy::new_without_default)]
// JUSTIFICATION: Performance-optimized enums may have size variance (e.g., large vs small variants)
#![allow(clippy::large_enum_variant)]
// ─────────────────────────────────────────────────────────────────────────────────
// DOCUMENTATION STYLE: Minor relaxations for doc comment formatting
// ─────────────────────────────────────────────────────────────────────────────────

// JUSTIFICATION: Doc comment style preferences - these don't affect correctness
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_nested_refdefs)]
// ─────────────────────────────────────────────────────────────────────────────────
// STYLE PREFERENCES: Minor patterns that are acceptable in this codebase
// ─────────────────────────────────────────────────────────────────────────────────

// JUSTIFICATION: vec![] followed by push can be clearer for incremental construction
#![allow(clippy::vec_init_then_push)]
// JUSTIFICATION: min/max pattern sometimes clearer than clamped() for audio bounds
#![allow(clippy::manual_clamp)]
// JUSTIFICATION: Manual counter loops can be clearer for DSP with index semantics
#![allow(clippy::explicit_counter_loop)]
// JUSTIFICATION: Custom NaN handling needed for float comparisons in DSP
#![allow(clippy::non_canonical_partial_ord_impl)]
// JUSTIFICATION: std::i32::MAX style acceptable for compatibility
#![allow(clippy::legacy_numeric_constants)]
// JUSTIFICATION: Some APIs don't need Default trait (e.g., required config)
#![allow(clippy::should_implement_trait)]
// ─────────────────────────────────────────────────────────────────────────────────
// DEVELOPMENT ALLOWS: These should be removed over time
// ─────────────────────────────────────────────────────────────────────────────────

// Note: These suppress warnings that indicate incomplete code. As modules are cleaned
// up, these should be moved to module-level or removed entirely.
#![allow(unused_imports)]
#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_comparisons)]

use anyhow::Result;
use log::{error, info, warn};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// Re-export public types
pub use safety::{SafetyConfig, SafetyMonitor, SafetyStats, SafetyViolation, WatchdogTimer};
pub use source_separation::{ConvTasNetSeparator, SeparatorConfig};
pub use synthesis::{
    generate_dynamic_microharmonic_sample,
    AudioFeatures,
    AudioSegment,
    CombinedSynthesizer,
    ConcatenativeSynthesizer,
    CrossSpeciesAdapter,
    // Dynamic Microharmonic (NEW)
    DynamicMicroharmonicParams,
    DynamicMicroharmonicSynthesizer,
    EnhancedMicroharmonicSynthesizer,
    GranularSynthesizer,
    MicroharmonicConstraints,
    MicroharmonicValidator,
    // Multi-Buffer Sequencer for Corvid Multi-Modal Support (NEW)
    Modality,
    ModalityTimeline,
    MultiBufferGranularSequencer,
    PhraseSegment,
    RealTimeSafetyMonitor,
    SafetyCheck,
    // Semantic Reconstruction (STAGE 4) - SourceMetadata only (112D fields)
    SourceMetadata,
    SpeciesParameters,
    SuperpositionalSynthesizer,
    SynthesisConfig,
    SynthesisMode,
    SynthesisPerformanceStats,
    SynthesisResult,
    TimelineEvent,
    ValidationResult,
};

// Manifest Bridge exports (Rust/Python pipeline communication)
pub use manifest_bridge::{
    ClusterInfo as ManifestClusterInfo, ClustersManifest, ExemplarEntry as ManifestExemplarEntry, ExemplarMetadata,
    PipelineController, SegmentEntry, SegmentsManifest, SynthesisManifest,
};

pub use thermal::{TemperatureReading, ThermalGovernor, ThermalState, ThermalStats};

// Island Hopping Navigation (NEW)
pub use island_hopping::{
    apply_delta_to_granular, AudioIsland, GranularParams, NavigationEngine, NavigationMode, NavigationWaypoint,
    PhraseDatabase, SafetyClamp, TimelineExecutor, Vector30D, VectorDelta,
};

// Metadata-First Synthesizer (NEW - Rust implementation of Python metadata_synthesizer.py)
pub use metadata_synthesizer::{
    MetadataQuery, MetadataSynthesizer, PhraseCandidate, SynthesisRecipe, SynthesisTarget, VectorSpaceQueryEngine,
};

#[cfg(feature = "python-bindings")]
pub use metadata_synthesizer::{PyMetadataQuery, PyMetadataSynthesizer, PyPhraseCandidate, PySynthesisRecipe};

// Micro-dynamics extractor exports (NEW)
pub use micro_dynamics_extractor::{
    FeatureDim, FeatureVector, MicroDynamicsExtractor, MicroDynamicsFeatures, MicroDynamicsFeatures15D,
    MicroDynamicsFeatures37D, MicroDynamicsFeatures39D, MicroDynamicsFeatures45D, MicroDynamicsFeatures56D,
    MultiScaleValue, RosettaFeatures,
};

// Semantic Reconstruction exports (NEW - STAGE 4 of 112D pipeline)
pub use semantic_reconstruction::{
    CachedGranularSynthesizer, ExemplarEntry, ExemplarManager, SemanticTimelineEvent, SourceMetadata112D,
    SynthesisConfig112D, SynthesisTimeline,
};

#[cfg(feature = "python-bindings")]
pub use micro_dynamics_extractor::{PyMicroDynamicsExtractor, PyMicroDynamicsFeatures};

// Pitch detection exports (NEW - YIN and autocorrelation)
pub use pitch::{AutocorrEstimator, F0Estimate, PitchAlgorithm, YinEstimator};

// Delta features exports (NEW - Δ and ΔΔ MFCCs and temporal features)
pub use delta::{DeltaFeatures, DeltaWidth, MfccDeltaComputer, TemporalDeltaComputer, TemporalFeatureType};

// Multi-scale aggregation exports (NEW - Statistical and hierarchical aggregation)
pub use multi_scale::{
    HierarchicalAggregator, HierarchicalConfig, HierarchicalFeatures, MultiScaleFeatures, StatisticalAggregator,
};

// Psychoacoustic features exports (NEW - 37D expansion)
pub use psychoacoustics::{BrightnessCalculator, PitchEntropyCalculator, RoughnessCalculator};

// Temporal features exports (NEW - 37D expansion)
pub use temporal::{RhythmicStabilityCalculator, TemporalCentroidCalculator};

// Advanced spectral features exports (NEW - 37D expansion)
pub use spectral_advanced::{SpectralFlatnessCalculator, SpectralKurtosisCalculator, SpectralTiltCalculator};

// Harmonic analysis exports (NEW - 37D expansion)
pub use harmonics::{HarmonicDeviationCalculator, InharmonicityCalculator};

// Formant analysis exports (NEW - 37D expansion)
pub use formants::{FormantBandwidthCalculator, FormantExtractor};

// Modulation dynamics exports (NEW - 37D expansion)
pub use modulation::{AmDepthCalculator, FmDepthCalculator, FmRateCalculator};

// Benchmark and evaluation exports (NEW - Phase 5)
pub use benchmark::{
    ClassificationMetrics, ClassificationReport, ComparisonReport, ConfusionMatrix, DatasetLoader, DatasetMetadata,
    DatasetType, ExtractionReport, FeatureAblationResults, FeatureEvaluator, Label, MetricCalculator, Recording,
};

// Change point detection exports (NEW - Phase 3)
pub use change_point_detection::{ChangePointError, PeltSegmenter};

// Clustering exports (NEW - Phase 3)
pub use clustering::{ClusterStats, ClusteringError, DbscanClustering, StandardScaler};

// HDBSCAN exports (NEW - Hierarchical DBSCAN for variable density)
pub use hdbscan::{DistanceMetric, HdbscanClustering, HdbscanError, HdbscanStats};

// Acoustic Similarity exports (NEW - Pairwise similarity for continuous manifolds)
pub use acoustic_similarity::{
    AcousticSimilarityEngine, BetweenTypeDistance, ConfusionEntry, DistanceMetric as SimilarityMetric,
    FeatureDiscrimination, FilePair, KnnClassifier, KnnCvResults, KnnNeighbor, KnnResult, NeighborhoodAnalysis,
    SearchResult, SimilarityAnalysis, SimilarityIndex,
};

#[cfg(feature = "python-bindings")]
pub use acoustic_similarity::PyAcousticSimilarityEngine;

// Dynamic Segmenter exports (NEW - Change Point Detection for atomic phrase discovery)
pub use dynamic_segmenter::{
    AtomicPhraseAnalyzer, AtomicPhraseType, ChangePoint, DynamicPhraseCandidate, DynamicSegmenter,
    DynamicSegmenterConfig, EmissionStrategy, SegmentationResult, TypedPhraseCandidate,
};

// Adaptive Segmentation exports (NEW - Onset detection for variable-length phrases)
pub use adaptive_segmentation::{AdaptiveSegmenter, OnsetDetector, SegmentationError};

// Within-Vocalization Analysis exports (NEW - TDD-tested multi-phrase detection)
pub use within_vocalization_analyzer::{
    BoundaryType as WvaBoundaryType, CorpusPhraseAnalyzer, CorpusPhraseStatistics, PhraseBoundary, PhraseSegmentation,
    WithinVocalizationAnalyzer, WithinVocalizationConfig,
};

// GMM exports (NEW - Phoneme discovery approach)
pub use gmm::{GaussianMixtureModel, GmmError};

// HMM exports (NEW - Temporal sequence modeling)
pub use hmm::{HiddenMarkovModel, HmmError};

// DTW exports (NEW - Time-aware clustering)
pub use dtw::{DtwClusterStats, DtwDbscan, DtwError, DtwMetric, FastDtw};

// Vocabulary to Synthesis exports (NEW - Mapping, segmentation, synthesis)
pub use audio_segmenter::{
    AudioGrain, AudioSegmentForSynthesis, AudioSegmenter, GrainEnvelope, SegmentContext, SegmenterError,
};
pub use synthesis_pipeline::{
    ConcatenativeParams, GrainEnvelopeType, GranularSynthesisParams, MetadataDrivenParams, SynthesisAssets,
    SynthesisError, SynthesisPipeline,
};
pub use vocabulary_mapper::{
    AnnotationDataset, DurationStats, VocabularyError, VocabularyItem, VocabularyMapper, VocabularyOccurrence,
    VocabularyStatistics, VocalizationContext,
};

// Graded Phrase Mining exports (NEW - Intensity-based phrase discovery)
pub use graded_phrase_mining::{
    FeatureMode, GradedMiningConfig, GradedMiningThresholds, GradedPhraseMiner, MotifClusterInfo, MotifReport,
    MotifSegment, ProcessingApproach, SpeciesGradingPrediction,
};

// Lexicon to Syntax exports (NEW - Master pipeline)
pub use lexicon_to_syntax::{
    DiscoveryConfig, FeatureDimension, LexiconStatistics, LexiconToSyntaxPipeline, LexiconToSyntaxResult,
    LexiconVocabularyItem, PhonemeModel, PhraseFeatures, PipelineCheckpoint, PipelineError, RefinementConfig,
    SegmentationConfig, SegmentedPhrase, VectorizationConfig,
};

// Parallel extraction exports (NEW - Phase 3)
pub use parallel_extraction::{
    analyze_context,
    analyze_social_network,
    analyze_turn_taking,
    batch_process_and_cluster,
    calculate_inter_cluster_similarity,
    calculate_intra_cluster_similarity,
    // DBSCAN Clustering for Phrase Discovery (NEW)
    cluster_phrase_candidates,
    // Synthesis Output (NEW - JSON Export & Audio Segmentation)
    export_phrases_for_synthesis,
    load_annotations_from_csv,
    AnnotationEntry,
    AtomicPhraseWithUsage,
    ClusterInfo,
    ClusteredPhrase,
    CommunicationEfficiency,
    CompositionalityStats,
    ContextAnalysis,
    ContextTurnStats,
    ConversationStats,
    // Annotation and Turn-Taking Analysis (NEW)
    EmitterAnnotation,
    ExtractionConfig,
    ExtractionError,
    ForbiddenReason,
    ForbiddenTransition,
    GapAnalysis,
    GrammarRule,
    InteractionPair,
    LibraryStatistics,
    LinguisticAnalysis,
    OverlapAnalysis,
    ParallelExtractionPipeline,
    PhonotacticsAnalysis,
    // Phrase Audio Library (NEW)
    PhraseAudioLibrary,
    PhraseAudioSegment,
    PhraseUsageStats,
    PipelineResult,
    PragmaticsAnalysis,
    PragmaticsAnalysisWithEmitter,
    ProsodyAnalysis,
    ResponseTimeStats,
    Rhythmicity,
    SentenceSegment,
    SocialNetworkAnalysis,
    SynthesisMetadata,
    SynthesisOutput,
    SynthesisPhrase,
    TurnTakingAnalysis,
    TurnTakingPattern,
    VocalizationResult,
    VocalizationWithEmitter,
    ZipfAnalysis,
};

// Rename to avoid conflict with metadata_synthesizer::PhraseCandidate
pub use parallel_extraction::PhraseCandidate as ExtractionPhraseCandidate;

// Corpus Analysis exports (NEW - Phrase X discovery)
pub use corpus_analysis::{
    CorpusError, CorpusStatistics, NGram, NGramMiner, PMICalculator, PhraseX, PhraseXDiscoveryEngine,
    Result as CorpusResult, SuffixEntropyCalculator,
};

// Neural Boundary Detection exports (NEW - Stage 1 of synthesis pipeline)
pub use neural_boundary::{segment_into_phrases, BoundaryDetectorConfig, BoundaryType, NeuralBoundaryDetector};
// Rename to avoid conflict with within_vocalization_analyzer::PhraseBoundary
pub use neural_boundary::{BoundaryType as NbdBoundaryType, PhraseBoundary as NbdPhraseBoundary};

// Zoo Vox Rosetta Engine v2.0 exports (NEW - Multi-modality species adaptation)
pub use sequence::{Motif, NgramStats, SequenceAnalysis, SequenceModule};
pub use species::{
    AnalysisModality, AnalysisModule, AtomicGranularity, ContextRules, DecodingMethod, EncodingStrategy, FeatureParams,
    HierarchicalThresholds, SpeciesConfig, SpeciesConfigFactory,
};
pub use spectral::{ContourConfig, ContourFeatures, FMType, FrequencyContour, SpectralModule};

// Zoo Vox Rosetta v2.0 - Phrase Data Preparation System exports
pub use zoo_vox_data_models::{
    AcousticFeatures30D, AcousticFeatures45D, BehaviorAnnotation, ContextAssociation, CrossSpeciesPhraseDatabase,
    PhrasePrototype, SpeciesPhraseLibrary,
};
pub use zoo_vox_features::{FeatureError, ZooVoxFeatureExtractor};

#[cfg(feature = "python-bindings")]
pub use zoo_vox_features::PyZooVoxFeatureExtractor;

pub use zoo_vox_extraction::{ZooVoxExtractionConfig, ZooVoxExtractionError, ZooVoxPhraseExtractor};
pub use zoo_vox_library::{create_sample_libraries, LibraryError, ZooVoxLibraryBuilder};

// Zoo Vox Rosetta v2.0 - Within-Call Phrase Discovery (Acoustic Similarity)
pub use zoo_vox_within_call::{
    DiscoveredPhraseType, PhraseInstance, PhraseMotif, SimilarityBasedLibraryBuilder, WithinCallAnalysisResult,
    WithinCallAnalyzer, WithinCallConfig,
};

// Phrase Discovery Pipeline (Unified Segmentation + Similarity)
pub use phrase_discovery::{
    PhraseDiscoveryConfig, PhraseDiscoveryPipeline, PhraseDiscoveryResult, PipelinePhraseType, PipelineStats,
};

/// Zoo Vox Rosetta result type
pub type ZooVoxResult<T> = std::result::Result<T, ZooVoxError>;

/// Zoo Vox Rosetta error type
#[derive(Debug)]
pub enum ZooVoxError {
    /// IO error
    Io(std::io::Error),
    /// JSON serialization error
    Json(serde_json::Error),
    /// Feature extraction error
    Feature(FeatureError),
    /// Extraction error
    Extraction(ZooVoxExtractionError),
    /// Library error
    Library(LibraryError),
}

impl From<std::io::Error> for ZooVoxError {
    fn from(e: std::io::Error) -> Self {
        ZooVoxError::Io(e)
    }
}

impl From<serde_json::Error> for ZooVoxError {
    fn from(e: serde_json::Error) -> Self {
        ZooVoxError::Json(e)
    }
}

impl From<FeatureError> for ZooVoxError {
    fn from(e: FeatureError) -> Self {
        ZooVoxError::Feature(e)
    }
}

impl From<ZooVoxExtractionError> for ZooVoxError {
    fn from(e: ZooVoxExtractionError) -> Self {
        ZooVoxError::Extraction(e)
    }
}

impl From<LibraryError> for ZooVoxError {
    fn from(e: LibraryError) -> Self {
        ZooVoxError::Library(e)
    }
}

impl std::fmt::Display for ZooVoxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZooVoxError::Io(e) => write!(f, "IO error: {}", e),
            ZooVoxError::Json(e) => write!(f, "JSON error: {}", e),
            ZooVoxError::Feature(e) => write!(f, "Feature error: {}", e),
            ZooVoxError::Extraction(e) => write!(f, "Extraction error: {}", e),
            ZooVoxError::Library(e) => write!(f, "Library error: {}", e),
        }
    }
}

impl std::error::Error for ZooVoxError {}

pub use logging::ProvenanceLogger;
pub use master_controller::{
    detect_fpga, Action, AtomicParameters, CognitiveProcessor, ExecutionReceipt, HealthStatus, IntentPriority,
    IntentToken, RejectionReason, SessionProfile, SharedMemoryConfig, SharedMemoryRingBuffer, SynthesisComplexity,
    WatchdogConfig,
};
pub use ptp::{PtpClock, PtpTimestamp};

#[cfg(feature = "python-bindings")]
pub use master_controller::PyCognitiveProcessor;

// Peer controller exports
pub use peer_controller::{AudioMuteState, HeartbeatMessage, OperationMode, PeerController, PeerControllerConfig};

// Acoustic simulator exports (for TDD testing)
pub use acoustic_simulator::{
    AcousticEnvironment, AcousticSimulator, EnvironmentType, NoiseMixture, NoiseProfile, SpectralColor,
    TemporalCharacteristics,
};

// Environmental monitor exports
pub use environmental_monitor::{
    EnvironmentalConditions, EnvironmentalMonitor, EnvironmentalMonitorConfig, LightLevel, RainIntensity,
    SensorReading, SessionViability, SolarForecast, TemperatureClassification,
};

// Power manager exports
pub use power_manager::{
    BatteryState, PowerBudget, PowerManager, PowerManagerConfig, PowerMode, SolarPrediction, ThrottleState,
};

// Wildlife sentry exports
pub use wildlife_sentry::{
    DetectionEvent, SpeciesSignature, TriggerUrgency, WakeTrigger, WildlifeSentry, WildlifeSentryConfig,
};

// Data synchronizer exports
pub use data_synchronizer::{
    DataSynchronizer, LogEntry, QueuedEntry, StorageBackend, StorageType, SyncConfig, SyncPriority, SyncStatus,
};

// Visual recording exports (for context verification in post-processing)
pub use visual_recording::{
    AudioEventType, AudioSyncEvent, ContextAnnotation, FrameQueue, RecordingState, RecordingStatistics, VisualMetadata,
    VisualRecorder, VisualRecorderConfig,
};

// IACUC compliance exports
pub use iacuc_compliance::{
    ComplianceCheck, ComplianceState, DailyLimits, EmergencyContact, IacucComplianceEngine, IacucIntent,
    IacucIntentType, IacucProtocol, PolicyViolation, SpeciesLimit, TimeWindow, ViolationType, Weekday,
};

// Time-series archive exports
pub use time_series_archive::{
    ParquetCompression, ParquetExportConfig, RetentionPolicy, StorageQuota, StorageStats, TimeSeriesArchiver,
    TimeSeriesBatch, TimeSeriesConfig, TimeSeriesPoint,
};

// Auto-calibration exports
pub use auto_calibration::{
    CalibrationConfig, CalibrationEngine, CalibrationHealthStatus, CalibrationResult, CalibrationTone,
    FrequencyResponsePoint, GainAdjustment, SignalType, SpeakerImpedance,
};

// Shadow model monitor exports
pub use shadow_model_monitor::{
    AlertLevel, DriftAlert, DriftSample, InferenceModel, InputFeatures, MockActiveModel, MockShadowModel,
    ModelComparison, ModelPrediction, ShadowModelConfig, ShadowModelMonitor,
};

// Web dashboard exports
pub use web_dashboard::{
    AuthToken, CalibrationDashboardStatus, CommandAuditLog, CommandResult, DashboardCommand, DashboardConfig,
    DashboardOperationMode, DashboardState, GaugeValue, IacucStatus, WebDashboard, WsMessage,
};

// Multi-node coordination exports
pub use multi_node_coordination::{
    ClockAccuracy, ClockClass, ClusterConfig, ClusterId, ElectionResult, FusedDetectionData, LocationEstimate,
    MultiNodeCoordinator, NodeCapabilities, NodeId, NodeInfo, TdmaSchedule, TdmaSlot,
};

// Performance testing exports
pub use peer_controller_performance::{
    benchmark_concurrent_processing, benchmark_memory_allocation, benchmark_message_processing,
    benchmark_mode_switching, benchmark_serialization_throughput, benchmark_timeout_detection, format_metrics,
    run_all_benchmarks, PeerControllerSimulator, PerformanceMetrics,
};

// Import modules
mod acoustic_simulator;
mod auto_calibration;
mod data_synchronizer;
mod environmental_monitor;
mod iacuc_compliance;
pub mod island_hopping; // Make public for integration tests
mod logging;
pub mod manifest_bridge; // Rust/Python pipeline communication
mod master_controller;
mod multi_node_coordination;
mod peer_controller;
pub mod peer_controller_performance;
mod power_manager;
mod ptp;
mod safety;
mod shadow_model_monitor;
mod source_separation;
pub mod synthesis; // Make public for integration tests
mod thermal;
mod time_series_archive;
mod visual_recording;
mod web_dashboard;
mod wildlife_sentry;

// Taxonomic-aware routing for hybrid expert architecture
pub mod taxonomic_router;

// Voting Ensemble for Species Classification (NN + RF)
pub mod voting_ensemble;

// Taxonomic-Aware Feature Gating (Dynamic Feature Reweighting)
pub mod feature_gating;

// Metadata-first synthesizer (30D vector space queries)
mod metadata_synthesizer;

// Micro-dynamics extractor (NEW - 30D feature extraction)
pub mod micro_dynamics_extractor;

// Semantic Reconstruction (STAGE 4) - 112D metadata
mod semantic_reconstruction;

// Semantic Reconstruction (STAGE 4) - 112D metadata forpub mod semantic_reconstruction;

// Pitch detection (NEW - YIN and autocorrelation algorithms)
pub mod pitch;

// Delta features (NEW - Δ and ΔΔ MFCCs and temporal features)
pub mod delta;

// Multi-scale aggregation (NEW - Statistical and hierarchical aggregation)
pub mod multi_scale;

// Benchmark and evaluation (NEW - Phase 5)
pub mod benchmark;

// Change point detection (NEW - PELT algorithm for Phase 3)
mod change_point_detection;

// Clustering (NEW - DBSCAN algorithm for Phase 3)
pub mod clustering;

// HDBSCAN (NEW - Hierarchical DBSCAN for variable-density clustering)
pub mod hdbscan;

// Adaptive Segmentation (NEW - Onset detection for variable-length phrases)
mod adaptive_segmentation;

// GMM + HMM (NEW - Phoneme discovery approach)
mod gmm;
mod hmm;

// DTW (NEW - Dynamic Time Warping for time-aware clustering)
mod dtw;

// Vocabulary to Synthesis (NEW - Mapping, segmentation, and synthesis)
mod audio_segmenter;
mod synthesis_pipeline;
mod vocabulary_mapper;

// Lexicon to Syntax (NEW - Master pipeline: Segmentation → Vectorization → Discovery → Refinement)
pub mod lexicon_to_syntax;

// Parallel extraction (NEW - Main pipeline for Phase 3)
mod parallel_extraction;

// Corpus Analysis (NEW - Phrase X discovery for corpus analysis)
mod corpus_analysis;
mod neural_boundary;

// Within-Vocalization Analysis (NEW - Multi-phrase detection within vocalizations)
pub mod within_vocalization_analyzer;

// Phrase Sequence Analysis (NEW - Syntactic structure discovery)
pub mod phrase_sequence_analyzer;

// Spectral Analysis (NEW - Zoo Vox Rosetta v2.0: FM whistle analysis for dolphins)
pub mod spectral;

// Sequence Analysis (NEW - Zoo Vox Rosetta v2.0: N-gram analysis for combinatorial syntax)
pub mod sequence;

// Species Configuration (NEW - Zoo Vox Rosetta v2.0: Species-specific adaptation layer)
pub mod species;

// Zoo Vox Rosetta v2.0 - Phrase Data Preparation System
// 30D/45D acoustic feature extraction, phrase segmentation, and library management
pub mod zoo_vox_data_models;
pub mod zoo_vox_extraction;
pub mod zoo_vox_features;
pub mod zoo_vox_library;
pub mod zoo_vox_within_call;

// Phrase Discovery Pipeline (NEW - Unified segmentation + similarity pipeline)
pub mod phrase_discovery;

// Trajectory Analysis Module (NEW - Continuous manifold analysis)
pub mod trajectory_analysis;

// Graded Phrase Mining (NEW - Intensity-based phrase discovery)
pub mod graded_phrase_mining;

// Synthetic Gap Analysis (NEW - Inter-type discriminability)
pub mod synthetic_gap_analysis;

// Arousal-Based Source Selection (NEW - Match synthesis to emotional intensity)
pub mod arousal_based_selection;

// Rhythm Sequencer (NEW - IPIs as first-class objects for species-typical timing)
pub mod rhythm_sequencer;

// Species-Specific Deep Dive Modules (NEW - Macaque spectral derivative, Dolphin bispectrum)
pub mod species_deep_dive;

// Psychoacoustic features (NEW - 37D expansion: pitch_entropy, roughness, brightness)
pub mod psychoacoustics;

// Temporal features (NEW - 37D expansion: rhythmic_stability)
pub mod temporal;

// Advanced spectral features (NEW - 37D expansion: spectral_tilt)
pub mod spectral_advanced;

// Harmonic analysis (NEW - 37D expansion: harmonic_deviation)
pub mod harmonics;

// Formant analysis (NEW - 37D expansion: formant_freqs)
pub mod formants;

// Modulation dynamics (NEW - 37D expansion: fm_depth, fm_rate)
pub mod modulation;

// Sequence analysis for combinatorial syntax testing
pub mod sequence_analysis;

// Acoustic Similarity Engine (NEW - Pairwise similarity instead of clustering)
pub mod acoustic_similarity;

// Dynamic Phrase Segmentation (NEW - Change Point Detection for atomic phrase discovery)
pub mod dynamic_segmenter;

// Human-Guided Context Discovery (Annotation Alignment for semantic grounding)
pub mod annotation_aligner;

// Rosetta Pipeline (Integrated Zoo Vox Rosetta Engine)
pub mod rosetta_pipeline;

// Acoustic Algebra (Vector spaces for acoustic features)
pub mod acoustic_algebra_105d;
pub mod acoustic_algebra_45d;

// Bio-Acoustic Interaction Agent (Synthesis Integration)
pub mod bio_acoustic_agent;

// Dictionary Loader (Field Deployment Persistence)
pub mod dictionary_loader;

// Async Semiotic State (Python-Rust Decoupling)
pub mod async_semiotic_state;

// Computational Ethology (Linguistic Structure Validation)
pub mod computational_ethology;

/// Configuration for the Technical Architect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechArchConfig {
    /// Source separation configuration
    pub separator: SeparatorConfig,
    /// Thermal configuration
    pub thermal: thermal::ThermalConfig,
    /// Safety configuration
    pub safety: SafetyConfig,
    /// Synthesis configuration
    pub synthesis: SynthesisConfig,
    /// PTP configuration
    pub ptp: ptp::PtpConfig,
    /// Logging configuration
    pub logging: logging::LoggingConfig,
    /// Target latency budget in milliseconds
    pub target_latency_ms: f64,
}

impl Default for TechArchConfig {
    fn default() -> Self {
        Self {
            separator: SeparatorConfig::default(),
            thermal: thermal::ThermalConfig::default(),
            safety: SafetyConfig::default(),
            synthesis: SynthesisConfig::default(),
            ptp: ptp::PtpConfig::default(),
            logging: logging::LoggingConfig::default(),
            target_latency_ms: 100.0, // 100ms budget
        }
    }
}

/// Technical Architect - Main entry point for the Rust execution layer
///
/// This struct coordinates all time-critical operations and provides
/// a clean API for both Rust and Python consumers.
pub struct TechnicalArchitect {
    /// Configuration
    config: TechArchConfig,
    /// Source separator
    separator: Arc<RwLock<ConvTasNetSeparator>>,
    /// Thermal governor
    thermal: Arc<ThermalGovernor>,
    /// Safety monitor
    safety: Arc<SafetyMonitor>,
    /// Synthesis engine
    synthesizer: Arc<RwLock<GranularSynthesizer>>,
    /// PTP clock
    ptp_clock: Arc<PtpClock>,
    /// Provenance logger
    logger: Arc<ProvenanceLogger>,
    /// Performance statistics
    stats: Arc<Mutex<PerformanceStats>>,
    /// System state
    state: Arc<RwLock<SystemState>>,
}

/// System state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemState {
    /// Whether the system is operational
    pub is_operational: bool,
    /// Current thermal state
    pub thermal_state: ThermalState,
    /// Number of safety violations since start
    pub safety_violations: u64,
    /// Last heartbeat timestamp
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    /// Current latency in milliseconds
    pub current_latency_ms: f64,
}

/// Performance statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Total audio frames processed
    pub frames_processed: u64,
    /// Total source separations performed
    pub separations: u64,
    /// Average processing time per frame (ms)
    pub avg_frame_time_ms: f64,
    /// Maximum processing time (ms)
    pub max_frame_time_ms: f64,
    /// Number of thermal throttling events
    pub thermal_throttle_count: u64,
    /// Number of safety interventions
    pub safety_interventions: u64,
    /// System uptime in seconds
    pub uptime_seconds: u64,
}

impl TechnicalArchitect {
    /// Create a new Technical Architect
    pub async fn new(config: TechArchConfig) -> Result<Self> {
        info!("Initializing Technical Architect with config: {:?}", config);

        // Initialize separator
        let separator = Arc::new(RwLock::new(ConvTasNetSeparator::new(config.separator.clone()).await?));

        // Initialize thermal governor
        let thermal = Arc::new(ThermalGovernor::new(config.thermal.clone()).await?);

        // Initialize safety monitor
        let safety = Arc::new(SafetyMonitor::new(config.safety.clone()).await?);

        // Initialize synthesizer
        let synthesizer = Arc::new(RwLock::new(GranularSynthesizer::new(config.synthesis.clone()).await?));

        // Initialize PTP clock
        let ptp_clock = Arc::new(PtpClock::new(config.ptp.clone()).await?);

        // Initialize logger
        let logger = Arc::new(ProvenanceLogger::new(config.logging.clone()).await?);

        let start_time = chrono::Utc::now();

        let architect = Self {
            config,
            separator,
            thermal,
            safety,
            synthesizer,
            ptp_clock,
            logger,
            stats: Arc::new(Mutex::new(PerformanceStats::default())),
            state: Arc::new(RwLock::new(SystemState {
                is_operational: true,
                thermal_state: ThermalState::Normal,
                safety_violations: 0,
                last_heartbeat: start_time,
                current_latency_ms: 0.0,
            })),
        };

        // Start background tasks
        architect.start_background_tasks().await?;

        info!("Technical Architect initialized successfully");
        Ok(architect)
    }

    /// Start background monitoring tasks
    async fn start_background_tasks(&self) -> Result<()> {
        let thermal = self.thermal.clone();
        let safety = self.safety.clone();
        let state = self.state.clone();
        let _stats = self.stats.clone();

        // Thermal monitoring task
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
            loop {
                interval.tick().await;
                if let Err(e) = thermal.monitor().await {
                    error!("Thermal monitoring error: {}", e);
                }
            }
        });

        // Safety monitoring task
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(100));
            loop {
                interval.tick().await;
                if let Err(e) = safety.monitor().await {
                    error!("Safety monitoring error: {}", e);
                }
            }
        });

        // Heartbeat task
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
            loop {
                interval.tick().await;
                let mut state = state.write().await;
                state.last_heartbeat = chrono::Utc::now();
            }
        });

        Ok(())
    }

    /// Process an audio frame (main entry point)
    ///
    /// This method processes a noisy audio frame and returns the cleaned audio.
    /// It checks safety and thermal conditions before processing.
    pub async fn process_audio_frame(&self, audio: Vec<f32>) -> Result<Vec<f32>> {
        let start = std::time::Instant::now();

        // Update heartbeat
        {
            let mut state = self.state.write().await;
            state.last_heartbeat = chrono::Utc::now();
        }

        // Check safety
        let safety_check = self.safety.check_safety().await?;
        if !safety_check.is_safe {
            let violation = SafetyViolation {
                violation_type: "SAFETY_CHECK_FAILED".to_string(),
                severity: "CRITICAL".to_string(),
                timestamp: chrono::Utc::now(),
            };
            self.safety.trigger_shutdown(violation).await?;
            return Err(anyhow::anyhow!("Safety check failed"));
        }

        // Check thermal state
        let thermal_state = self.thermal.get_state().await;
        let mut state = self.state.write().await;
        state.thermal_state = thermal_state;

        // If throttling, return simplified processing
        if matches!(thermal_state, ThermalState::Critical | ThermalState::Throttling) {
            warn!("Thermal throttling active, simplifying processing");
            self.stats.lock().thermal_throttle_count += 1;
            return Ok(audio); // Return raw audio
        }

        // Log provenance
        let timestamp = self.ptp_clock.get_timestamp().await?;
        let _ = self.logger.log_decision("process_audio_frame", timestamp).await;

        // Run source separation
        let clean_audio = {
            let separator = self.separator.read().await;
            separator.separate(&audio).await?
        };

        // Update statistics
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        {
            let mut stats = self.stats.lock();
            stats.frames_processed += 1;
            stats.separations += 1;
            stats.avg_frame_time_ms = elapsed;
            stats.max_frame_time_ms = stats.max_frame_time_ms.max(elapsed);
        }

        // Update state
        {
            let mut state = self.state.write().await;
            state.current_latency_ms = elapsed;
        }

        // Check latency budget
        if elapsed > self.config.target_latency_ms {
            warn!(
                "Latency budget exceeded: {:.2}ms > {:.2}ms",
                elapsed, self.config.target_latency_ms
            );
        }

        Ok(clean_audio)
    }

    /// Get current performance statistics
    pub async fn get_stats(&self) -> PerformanceStats {
        self.stats.lock().clone()
    }

    /// Get current system state
    pub async fn get_state(&self) -> SystemState {
        self.state.read().await.clone()
    }

    /// Get thermal state
    pub async fn get_thermal_state(&self) -> ThermalState {
        self.thermal.get_state().await
    }

    /// Get thermal statistics
    pub async fn get_thermal_stats(&self) -> thermal::ThermalStats {
        self.thermal.get_stats().await
    }

    /// Get safety statistics
    pub async fn get_safety_stats(&self) -> SafetyStats {
        self.safety.get_stats().await
    }

    /// Get PTP timestamp
    pub async fn get_ptp_timestamp(&self) -> Result<PtpTimestamp> {
        self.ptp_clock.get_timestamp().await
    }

    /// Get reference to thermal governor (for master controller)
    pub fn get_thermal_governor(&self) -> &Arc<ThermalGovernor> {
        &self.thermal
    }

    /// Get reference to safety monitor (for master controller)
    pub fn get_safety_monitor(&self) -> &Arc<SafetyMonitor> {
        &self.safety
    }

    /// Get reference to PTP clock (for master controller)
    pub fn get_ptp_clock(&self) -> &Arc<PtpClock> {
        &self.ptp_clock
    }

    /// Shutdown the system gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Technical Architect");

        {
            let mut state = self.state.write().await;
            state.is_operational = false;
        }

        self.logger.flush().await?;
        self.synthesizer.write().await.shutdown().await?;
        self.ptp_clock.shutdown().await?;

        info!("Technical Architect shutdown complete");
        Ok(())
    }

    /// Emergency mute - immediately silence all audio output
    ///
    /// This is a safety-critical function that:
    /// 1. Immediately sets output gain to zero
    /// 2. Stops any ongoing synthesis
    /// 3. Logs the event with PTP timestamp
    ///
    /// This function must complete in < 1ms to be effective for safety.
    pub async fn emergency_mute(&self) -> Result<()> {
        error!("EMERGENCY MUTE activated");

        // Get PTP timestamp for logging
        let timestamp = self.ptp_clock.get_timestamp().await?;

        // Immediately stop synthesis
        {
            let mut synthesizer = self.synthesizer.write().await;
            synthesizer.emergency_stop()?;
        }

        // Update system state to reflect muted status
        {
            let mut state = self.state.write().await;
            state.current_latency_ms = 0.0; // Reset latency
        }

        // Log the emergency mute event with provenance
        self.logger.log_emergency_event("emergency_mute", timestamp).await?;

        error!("Emergency mute completed at PTP timestamp: {:?}", timestamp);
        Ok(())
    }

    /// Generate a response audio segment
    pub async fn generate_response(&self, features: &synthesis::AudioFeatures) -> Result<Vec<f32>> {
        let synthesizer = self.synthesizer.read().await;
        synthesizer.generate(features).await
    }

    // ========================================================================
    // Enhanced Synthesis Methods
    // ========================================================================

    /// Create an enhanced microharmonic synthesizer for the given species
    pub fn create_microharmonic_synthesizer(
        &self,
        species: String,
        phrase_segments: HashMap<String, synthesis::PhraseSegment>,
    ) -> EnhancedMicroharmonicSynthesizer {
        EnhancedMicroharmonicSynthesizer::new(species, phrase_segments, self.config.synthesis.sample_rate)
    }

    /// Synthesize in horizontal mode (sequential concatenation)
    pub async fn synthesize_horizontal(
        &self,
        synthesizer: &EnhancedMicroharmonicSynthesizer,
        phrase_keys: Vec<String>,
        constraints: Option<&MicroharmonicConstraints>,
    ) -> Result<SynthesisResult> {
        let default_constraints = MicroharmonicConstraints::default();
        let constraints = constraints.unwrap_or(&default_constraints);
        synthesizer.synthesize_horizontal(&phrase_keys, constraints).await
    }

    /// Synthesize in vertical mode (simultaneous layering)
    pub async fn synthesize_vertical(
        &self,
        synthesizer: &EnhancedMicroharmonicSynthesizer,
        phrase_keys: Vec<String>,
        constraints: Option<&MicroharmonicConstraints>,
    ) -> Result<SynthesisResult> {
        let default_constraints = MicroharmonicConstraints::default();
        let constraints = constraints.unwrap_or(&default_constraints);
        synthesizer.synthesize_vertical(&phrase_keys, constraints).await
    }

    /// Synthesize in combined mode (mixed encoding)
    pub async fn synthesize_combined(
        &self,
        synthesizer: &EnhancedMicroharmonicSynthesizer,
        synthesis_plan: Vec<(SynthesisMode, Vec<String>)>,
        constraints: Option<&MicroharmonicConstraints>,
    ) -> Result<SynthesisResult> {
        let default_constraints = MicroharmonicConstraints::default();
        let constraints = constraints.unwrap_or(&default_constraints);
        synthesizer.synthesize_combined(&synthesis_plan, constraints).await
    }
}

// PyO3 Python bindings (when feature is enabled)
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

/// Python wrapper for TechnicalArchitect
#[cfg(feature = "python-bindings")]
#[pyclass(name = "TechnicalArchitect")]
pub struct PyTechnicalArchitect {
    inner: Arc<TechnicalArchitect>,
}

/// Python wrapper for Dynamic Microharmonic Synthesizer
#[cfg(feature = "python-bindings")]
#[pyclass(name = "DynamicMicroharmonicSynthesizer")]
pub struct PyDynamicMicroharmonicSynthesizer {
    inner: synthesis::DynamicMicroharmonicSynthesizer,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyDynamicMicroharmonicSynthesizer {
    /// Create a new Dynamic Microharmonic Synthesizer
    #[new]
    fn new(sample_rate: usize) -> Self {
        Self {
            inner: synthesis::DynamicMicroharmonicSynthesizer::new(sample_rate),
        }
    }

    /// Synthesize a single phrase with given parameters
    ///
    /// Parameters:
    /// - f0_base: Fundamental frequency in Hz
    /// - duration_ms: Duration in milliseconds
    /// - attack_ms: Attack time in milliseconds
    /// - decay_ms: Decay time in milliseconds
    /// - sustain_level: Sustain amplitude (0.0 to 1.0)
    /// - vibrato_rate_hz: Vibrato rate in Hz
    /// - vibrato_depth_cents: Vibrato depth in cents
    /// - jitter_amount: Jitter amount (0.0 to 0.1)
    /// - shimmer_amount: Shimmer amount (0.0 to 0.1)
    /// - spectral_tilt: Spectral tilt in dB/octave (negative values)
    /// - hnr_db: Harmonic-to-noise ratio in dB
    ///
    /// Returns: List of audio samples
    #[allow(clippy::too_many_arguments)]
    fn synthesize_phrase(
        &self,
        f0_base: f32,
        duration_ms: f32,
        attack_ms: f32,
        decay_ms: f32,
        sustain_level: f32,
        vibrato_rate_hz: f32,
        vibrato_depth_cents: f32,
        jitter_amount: f32,
        shimmer_amount: f32,
        spectral_tilt: f32,
        hnr_db: f32,
    ) -> Vec<f32> {
        let params = synthesis::DynamicMicroharmonicParams {
            f0_base,
            duration_ms,
            attack_ms,
            decay_ms,
            sustain_level,
            vibrato_rate_hz,
            vibrato_depth_cents,
            jitter_amount,
            shimmer_amount,
            spectral_tilt,
            hnr_db,
        };

        self.inner.synthesize_phrase(&params)
    }

    /// Synthesize a sequence of phrases (sentence)
    ///
    /// Parameters:
    /// - phrase_params_json: JSON string of list of phrase parameter dicts
    /// - crossfade_ms: Crossfade duration between phrases
    ///
    /// Returns: List of audio samples for the entire sequence
    fn synthesize_sequence(&self, phrase_params_json: String, crossfade_ms: f32) -> PyResult<Vec<f32>> {
        let phrase_params: Vec<synthesis::DynamicMicroharmonicParams> = serde_json::from_str(&phrase_params_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid parameters JSON: {}", e)))?;

        Ok(self.inner.synthesize_sequence(&phrase_params, crossfade_ms))
    }

    /// Generate random micro-dynamics parameters for exploration
    ///
    /// Parameters:
    /// - f0_base: Target fundamental frequency
    /// - duration_ms: Target duration
    /// - variability: Randomness amount (0.0 to 1.0)
    ///
    /// Returns: JSON string of parameters
    fn generate_random_params(&self, f0_base: f32, duration_ms: f32, variability: f32) -> PyResult<String> {
        let params = self.inner.generate_random_params(f0_base, duration_ms, variability);

        serde_json::to_string(&params)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization failed: {}", e)))
    }

    /// Get default parameters for marmoset vocalizations
    ///
    /// Parameters:
    /// - f0_base: Fundamental frequency in Hz
    /// - duration_ms: Duration in milliseconds
    ///
    /// Returns: JSON string of default marmoset parameters
    fn marmoset_default(&self, f0_base: f32, duration_ms: f32) -> PyResult<String> {
        let params = synthesis::DynamicMicroharmonicParams::marmoset_default(f0_base, duration_ms);

        serde_json::to_string(&params)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization failed: {}", e)))
    }

    /// Get default parameters for bat vocalizations
    ///
    /// Parameters:
    /// - f0_base: Fundamental frequency in Hz
    /// - duration_ms: Duration in milliseconds
    ///
    /// Returns: JSON string of default bat parameters
    fn bat_default(&self, f0_base: f32, duration_ms: f32) -> PyResult<String> {
        let params = synthesis::DynamicMicroharmonicParams::bat_default(f0_base, duration_ms);

        serde_json::to_string(&params)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization failed: {}", e)))
    }
}

/// Python wrapper for 17D SourceMetadata
///
/// This provides Python access to the full 17-dimensional micro-dynamics
/// metadata for delta-based synthesis.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "SourceMetadata")]
#[derive(Clone, Copy)]
pub struct PySourceMetadata {
    // === Fundamental (3 features) ===
    pub mean_f0_hz: f32,
    pub duration_ms: f32,
    pub f0_range_hz: f32,

    // === Grit Factors (3 features) ===
    pub harmonic_to_noise_ratio: f32,
    pub spectral_flatness: f32,
    pub harmonicity: f32,

    // === Motion Factors (7 features) ===
    pub attack_time_ms: f32,
    pub decay_time_ms: f32,
    pub sustain_level: f32,
    pub vibrato_rate_hz: f32,
    pub vibrato_depth: f32,
    pub jitter: f32,
    pub shimmer: f32,

    // === Fingerprint Factors (14 features) ===
    pub mfcc_1: f32,
    pub mfcc_2: f32,
    pub mfcc_3: f32,
    pub mfcc_4: f32,
    pub mfcc_5: f32,
    pub mfcc_6: f32,
    pub mfcc_7: f32,
    pub mfcc_8: f32,
    pub mfcc_9: f32,
    pub mfcc_10: f32,
    pub mfcc_11: f32,
    pub mfcc_12: f32,
    pub mfcc_13: f32,
    pub spectral_flux: f32,

    // === Rhythm Factors (3 features) ===
    pub median_ici_ms: f32,
    pub onset_rate_hz: f32,
    pub ici_coefficient_of_variation: f32,

    // === Resonance Factors (6 features) - 45D Expansion ===
    pub formant_1_hz: f32,
    pub formant_2_hz: f32,
    pub formant_3_hz: f32,
    pub formant_1_bandwidth: f32,
    pub formant_2_bandwidth: f32,
    pub formant_dispersion: f32,

    // === Spectral Shape Factors (4 features) - 45D Expansion ===
    pub spectral_centroid: f32,
    pub spectral_spread: f32,
    pub spectral_skewness: f32,
    pub spectral_kurtosis: f32,

    // === Modulation Factors (3 features) - 45D Expansion ===
    pub spectral_tilt: f32,
    pub fm_slope: f32,
    pub am_depth: f32,

    // === Non-Linear Factors (2 features) - 45D Expansion ===
    pub subharmonic_ratio: f32,
    pub spectral_entropy: f32,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PySourceMetadata {
    /// Create 45D SourceMetadata (simplified constructor for common use)
    ///
    /// For full control, use the builder() method instead.
    /// The 45D expansion fields use sensible defaults if not specified.
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (
        // Fundamental (3)
        mean_f0_hz, duration_ms, f0_range_hz,
        // Grit Factors (3)
        harmonic_to_noise_ratio=20.0, spectral_flatness=0.3, harmonicity=0.8,
        // Motion Factors (7)
        attack_time_ms=5.0, decay_time_ms=20.0, sustain_level=0.7,
        vibrato_rate_hz=7.0, vibrato_depth=50.0, jitter=0.01, shimmer=0.03,
        // Fingerprint Factors (13 MFCCs)
        mfcc_1=0.0, mfcc_2=0.0, mfcc_3=0.0, mfcc_4=0.0, mfcc_5=0.0,
        mfcc_6=0.0, mfcc_7=0.0, mfcc_8=0.0, mfcc_9=0.0, mfcc_10=0.0,
        mfcc_11=0.0, mfcc_12=0.0, mfcc_13=0.0,
        // Spectral Dynamics (1)
        spectral_flux=0.5,
        // Rhythm Factors (3)
        median_ici_ms=15.0, onset_rate_hz=8.0, ici_coefficient_of_variation=0.3,
        // 45D Expansion - Resonance (6)
        formant_1_hz=1000.0, formant_2_hz=2000.0, formant_3_hz=3000.0,
        formant_1_bandwidth=150.0, formant_2_bandwidth=240.0, formant_dispersion=1000.0,
        // 45D Expansion - Spectral Shape (4)
        spectral_centroid=5000.0, spectral_spread=2000.0, spectral_skewness=0.0, spectral_kurtosis=3.0,
        // 45D Expansion - Modulation (3)
        spectral_tilt=-6.0, fm_slope=0.0, am_depth=0.0,
        // 45D Expansion - Non-Linear (2)
        subharmonic_ratio=0.0, spectral_entropy=0.3
    ))]
    fn new(
        // Fundamental (3)
        mean_f0_hz: f32,
        duration_ms: f32,
        f0_range_hz: f32,
        // Grit Factors (3)
        harmonic_to_noise_ratio: f32,
        spectral_flatness: f32,
        harmonicity: f32,
        // Motion Factors (7)
        attack_time_ms: f32,
        decay_time_ms: f32,
        sustain_level: f32,
        vibrato_rate_hz: f32,
        vibrato_depth: f32,
        jitter: f32,
        shimmer: f32,
        // Fingerprint Factors (13 MFCCs)
        mfcc_1: f32,
        mfcc_2: f32,
        mfcc_3: f32,
        mfcc_4: f32,
        mfcc_5: f32,
        mfcc_6: f32,
        mfcc_7: f32,
        mfcc_8: f32,
        mfcc_9: f32,
        mfcc_10: f32,
        mfcc_11: f32,
        mfcc_12: f32,
        mfcc_13: f32,
        // Spectral Dynamics (1)
        spectral_flux: f32,
        // Rhythm Factors (3)
        median_ici_ms: f32,
        onset_rate_hz: f32,
        ici_coefficient_of_variation: f32,
        // 45D Expansion - Resonance (6)
        formant_1_hz: f32,
        formant_2_hz: f32,
        formant_3_hz: f32,
        formant_1_bandwidth: f32,
        formant_2_bandwidth: f32,
        formant_dispersion: f32,
        // 45D Expansion - Spectral Shape (4)
        spectral_centroid: f32,
        spectral_spread: f32,
        spectral_skewness: f32,
        spectral_kurtosis: f32,
        // 45D Expansion - Modulation (3)
        spectral_tilt: f32,
        fm_slope: f32,
        am_depth: f32,
        // 45D Expansion - Non-Linear (2)
        subharmonic_ratio: f32,
        spectral_entropy: f32,
    ) -> Self {
        Self {
            mean_f0_hz,
            duration_ms,
            f0_range_hz,
            harmonic_to_noise_ratio,
            spectral_flatness,
            harmonicity,
            attack_time_ms,
            decay_time_ms,
            sustain_level,
            vibrato_rate_hz,
            vibrato_depth,
            jitter,
            shimmer,
            mfcc_1,
            mfcc_2,
            mfcc_3,
            mfcc_4,
            mfcc_5,
            mfcc_6,
            mfcc_7,
            mfcc_8,
            mfcc_9,
            mfcc_10,
            mfcc_11,
            mfcc_12,
            mfcc_13,
            spectral_flux,
            median_ici_ms,
            onset_rate_hz,
            ici_coefficient_of_variation,
            formant_1_hz,
            formant_2_hz,
            formant_3_hz,
            formant_1_bandwidth,
            formant_2_bandwidth,
            formant_dispersion,
            spectral_centroid,
            spectral_spread,
            spectral_skewness,
            spectral_kurtosis,
            spectral_tilt,
            fm_slope,
            am_depth,
            subharmonic_ratio,
            spectral_entropy,
        }
    }

    /// Create a SourceMetadata builder for partial construction
    ///
    /// Example:
    /// ```python
    /// from technical_architecture import SourceMetadata
    ///
    /// metadata = SourceMetadata.builder() \\
    ///     .mean_f0_hz(7000.0) \\
    ///     .duration_ms(50.0) \\
    ///     .f0_range_hz(400.0) \\
    ///     .harmonic_to_noise_ratio(20.0) \\
    ///     .build()
    /// ```
    #[staticmethod]
    fn builder() -> PySourceMetadataBuilder {
        PySourceMetadataBuilder::create()
    }

    // === Fundamental Getters/Setters ===
    fn get_mean_f0_hz(&self) -> f32 {
        self.mean_f0_hz
    }
    fn set_mean_f0_hz(&mut self, value: f32) {
        self.mean_f0_hz = value;
    }

    fn get_duration_ms(&self) -> f32 {
        self.duration_ms
    }
    fn set_duration_ms(&mut self, value: f32) {
        self.duration_ms = value;
    }

    fn get_f0_range_hz(&self) -> f32 {
        self.f0_range_hz
    }
    fn set_f0_range_hz(&mut self, value: f32) {
        self.f0_range_hz = value;
    }

    // === Grit Factor Getters/Setters ===
    fn get_harmonic_to_noise_ratio(&self) -> f32 {
        self.harmonic_to_noise_ratio
    }
    fn set_harmonic_to_noise_ratio(&mut self, value: f32) {
        self.harmonic_to_noise_ratio = value;
    }

    fn get_spectral_flatness(&self) -> f32 {
        self.spectral_flatness
    }
    fn set_spectral_flatness(&mut self, value: f32) {
        self.spectral_flatness = value;
    }

    // === Motion Factor Getters/Setters ===
    fn get_attack_time_ms(&self) -> f32 {
        self.attack_time_ms
    }
    fn set_attack_time_ms(&mut self, value: f32) {
        self.attack_time_ms = value;
    }

    fn get_decay_time_ms(&self) -> f32 {
        self.decay_time_ms
    }
    fn set_decay_time_ms(&mut self, value: f32) {
        self.decay_time_ms = value;
    }

    fn get_sustain_level(&self) -> f32 {
        self.sustain_level
    }
    fn set_sustain_level(&mut self, value: f32) {
        self.sustain_level = value;
    }

    fn get_vibrato_rate_hz(&self) -> f32 {
        self.vibrato_rate_hz
    }
    fn set_vibrato_rate_hz(&mut self, value: f32) {
        self.vibrato_rate_hz = value;
    }

    fn get_vibrato_depth(&self) -> f32 {
        self.vibrato_depth
    }
    fn set_vibrato_depth(&mut self, value: f32) {
        self.vibrato_depth = value;
    }

    fn get_jitter(&self) -> f32 {
        self.jitter
    }
    fn set_jitter(&mut self, value: f32) {
        self.jitter = value;
    }

    fn get_shimmer(&self) -> f32 {
        self.shimmer
    }
    fn set_shimmer(&mut self, value: f32) {
        self.shimmer = value;
    }

    // === Grit Factor Getters/Setters (continued) ===
    fn get_harmonicity(&self) -> f32 {
        self.harmonicity
    }
    fn set_harmonicity(&mut self, value: f32) {
        self.harmonicity = value;
    }

    // === Fingerprint Factor Getters/Setters ===
    fn get_mfcc_1(&self) -> f32 {
        self.mfcc_1
    }
    fn set_mfcc_1(&mut self, value: f32) {
        self.mfcc_1 = value;
    }

    fn get_mfcc_2(&self) -> f32 {
        self.mfcc_2
    }
    fn set_mfcc_2(&mut self, value: f32) {
        self.mfcc_2 = value;
    }

    fn get_mfcc_3(&self) -> f32 {
        self.mfcc_3
    }
    fn set_mfcc_3(&mut self, value: f32) {
        self.mfcc_3 = value;
    }

    fn get_mfcc_4(&self) -> f32 {
        self.mfcc_4
    }
    fn set_mfcc_4(&mut self, value: f32) {
        self.mfcc_4 = value;
    }

    fn get_mfcc_5(&self) -> f32 {
        self.mfcc_5
    }
    fn set_mfcc_5(&mut self, value: f32) {
        self.mfcc_5 = value;
    }

    fn get_mfcc_6(&self) -> f32 {
        self.mfcc_6
    }
    fn set_mfcc_6(&mut self, value: f32) {
        self.mfcc_6 = value;
    }

    fn get_mfcc_7(&self) -> f32 {
        self.mfcc_7
    }
    fn set_mfcc_7(&mut self, value: f32) {
        self.mfcc_7 = value;
    }

    fn get_mfcc_8(&self) -> f32 {
        self.mfcc_8
    }
    fn set_mfcc_8(&mut self, value: f32) {
        self.mfcc_8 = value;
    }

    fn get_mfcc_9(&self) -> f32 {
        self.mfcc_9
    }
    fn set_mfcc_9(&mut self, value: f32) {
        self.mfcc_9 = value;
    }

    fn get_mfcc_10(&self) -> f32 {
        self.mfcc_10
    }
    fn set_mfcc_10(&mut self, value: f32) {
        self.mfcc_10 = value;
    }

    fn get_mfcc_11(&self) -> f32 {
        self.mfcc_11
    }
    fn set_mfcc_11(&mut self, value: f32) {
        self.mfcc_11 = value;
    }

    fn get_mfcc_12(&self) -> f32 {
        self.mfcc_12
    }
    fn set_mfcc_12(&mut self, value: f32) {
        self.mfcc_12 = value;
    }

    fn get_mfcc_13(&self) -> f32 {
        self.mfcc_13
    }
    fn set_mfcc_13(&mut self, value: f32) {
        self.mfcc_13 = value;
    }

    fn get_spectral_flux(&self) -> f32 {
        self.spectral_flux
    }
    fn set_spectral_flux(&mut self, value: f32) {
        self.spectral_flux = value;
    }

    // === Rhythm Factor Getters/Setters ===
    fn get_median_ici_ms(&self) -> f32 {
        self.median_ici_ms
    }
    fn set_median_ici_ms(&mut self, value: f32) {
        self.median_ici_ms = value;
    }

    fn get_onset_rate_hz(&self) -> f32 {
        self.onset_rate_hz
    }
    fn set_onset_rate_hz(&mut self, value: f32) {
        self.onset_rate_hz = value;
    }

    fn get_ici_coefficient_of_variation(&self) -> f32 {
        self.ici_coefficient_of_variation
    }
    fn set_ici_coefficient_of_variation(&mut self, value: f32) {
        self.ici_coefficient_of_variation = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "SourceMetadata(F0={}Hz, Dur={}ms, Range={}Hz, HNR={}dB, Flat={}, Attack={}ms, Decay={}ms, Jitter={})",
            self.mean_f0_hz as i32,
            self.duration_ms as i32,
            self.f0_range_hz as i32,
            (self.harmonic_to_noise_ratio * 10.0) as i32 / 10,
            (self.spectral_flatness * 100.0) as i32 / 100,
            (self.attack_time_ms * 10.0) as i32 / 10,
            (self.decay_time_ms * 10.0) as i32 / 10,
            (self.jitter * 1000.0) as i32 / 1000
        )
    }
}

#[cfg(feature = "python-bindings")]
impl From<PySourceMetadata> for synthesis::SourceMetadata {
    fn from(py: PySourceMetadata) -> Self {
        Self {
            mean_f0_hz: py.mean_f0_hz,
            duration_ms: py.duration_ms,
            f0_range_hz: py.f0_range_hz,
            harmonic_to_noise_ratio: py.harmonic_to_noise_ratio,
            spectral_flatness: py.spectral_flatness,
            harmonicity: py.harmonicity,
            attack_time_ms: py.attack_time_ms,
            decay_time_ms: py.decay_time_ms,
            sustain_level: py.sustain_level,
            vibrato_rate_hz: py.vibrato_rate_hz,
            vibrato_depth: py.vibrato_depth,
            jitter: py.jitter,
            shimmer: py.shimmer,
            mfcc_1: py.mfcc_1,
            mfcc_2: py.mfcc_2,
            mfcc_3: py.mfcc_3,
            mfcc_4: py.mfcc_4,
            mfcc_5: py.mfcc_5,
            mfcc_6: py.mfcc_6,
            mfcc_7: py.mfcc_7,
            mfcc_8: py.mfcc_8,
            mfcc_9: py.mfcc_9,
            mfcc_10: py.mfcc_10,
            mfcc_11: py.mfcc_11,
            mfcc_12: py.mfcc_12,
            mfcc_13: py.mfcc_13,
            spectral_flux: py.spectral_flux,
            median_ici_ms: py.median_ici_ms,
            onset_rate_hz: py.onset_rate_hz,
            ici_coefficient_of_variation: py.ici_coefficient_of_variation,
            // 45D Expansion - Resonance
            formant_1_hz: py.formant_1_hz,
            formant_2_hz: py.formant_2_hz,
            formant_3_hz: py.formant_3_hz,
            formant_1_bandwidth: py.formant_1_bandwidth,
            formant_2_bandwidth: py.formant_2_bandwidth,
            formant_dispersion: py.formant_dispersion,
            // 45D Expansion - Spectral Shape
            spectral_centroid: py.spectral_centroid,
            spectral_spread: py.spectral_spread,
            spectral_skewness: py.spectral_skewness,
            spectral_kurtosis: py.spectral_kurtosis,
            // 45D Expansion - Modulation
            spectral_tilt: py.spectral_tilt,
            fm_slope: py.fm_slope,
            am_depth: py.am_depth,
            // 45D Expansion - Non-Linear
            subharmonic_ratio: py.subharmonic_ratio,
            spectral_entropy: py.spectral_entropy,
        }
    }
}

#[cfg(feature = "python-bindings")]
impl From<synthesis::SourceMetadata> for PySourceMetadata {
    fn from(rust: synthesis::SourceMetadata) -> Self {
        Self {
            mean_f0_hz: rust.mean_f0_hz,
            duration_ms: rust.duration_ms,
            f0_range_hz: rust.f0_range_hz,
            harmonic_to_noise_ratio: rust.harmonic_to_noise_ratio,
            spectral_flatness: rust.spectral_flatness,
            harmonicity: rust.harmonicity,
            attack_time_ms: rust.attack_time_ms,
            decay_time_ms: rust.decay_time_ms,
            sustain_level: rust.sustain_level,
            vibrato_rate_hz: rust.vibrato_rate_hz,
            vibrato_depth: rust.vibrato_depth,
            jitter: rust.jitter,
            shimmer: rust.shimmer,
            mfcc_1: rust.mfcc_1,
            mfcc_2: rust.mfcc_2,
            mfcc_3: rust.mfcc_3,
            mfcc_4: rust.mfcc_4,
            mfcc_5: rust.mfcc_5,
            mfcc_6: rust.mfcc_6,
            mfcc_7: rust.mfcc_7,
            mfcc_8: rust.mfcc_8,
            mfcc_9: rust.mfcc_9,
            mfcc_10: rust.mfcc_10,
            mfcc_11: rust.mfcc_11,
            mfcc_12: rust.mfcc_12,
            mfcc_13: rust.mfcc_13,
            spectral_flux: rust.spectral_flux,
            median_ici_ms: rust.median_ici_ms,
            onset_rate_hz: rust.onset_rate_hz,
            ici_coefficient_of_variation: rust.ici_coefficient_of_variation,
            // 45D Expansion - Resonance
            formant_1_hz: rust.formant_1_hz,
            formant_2_hz: rust.formant_2_hz,
            formant_3_hz: rust.formant_3_hz,
            formant_1_bandwidth: rust.formant_1_bandwidth,
            formant_2_bandwidth: rust.formant_2_bandwidth,
            formant_dispersion: rust.formant_dispersion,
            // 45D Expansion - Spectral Shape
            spectral_centroid: rust.spectral_centroid,
            spectral_spread: rust.spectral_spread,
            spectral_skewness: rust.spectral_skewness,
            spectral_kurtosis: rust.spectral_kurtosis,
            // 45D Expansion - Modulation
            spectral_tilt: rust.spectral_tilt,
            fm_slope: rust.fm_slope,
            am_depth: rust.am_depth,
            // 45D Expansion - Non-Linear
            subharmonic_ratio: rust.subharmonic_ratio,
            spectral_entropy: rust.spectral_entropy,
        }
    }
}

/// Python wrapper for SourceMetadataBuilder
///
/// Provides fluent builder API for constructing SourceMetadata
/// with only the features you know, using defaults for the rest.
///
/// **Note**: This is a private class (_SourceMetadataBuilder) intended for internal use.
/// Users should access it via SourceMetadata.builder() method.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "_SourceMetadataBuilder")]
#[derive(Clone)]
pub struct PySourceMetadataBuilder {
    metadata: PySourceMetadata,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PySourceMetadataBuilder {
    /// Create a new builder with Rust default values
    #[staticmethod]
    fn create() -> Self {
        let rust_default = synthesis::SourceMetadata::default();
        Self {
            metadata: PySourceMetadata::from(rust_default),
        }
    }

    // === Fundamental ===
    fn mean_f0_hz(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.mean_f0_hz = value;
        new
    }

    fn duration_ms(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.duration_ms = value;
        new
    }

    fn f0_range_hz(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.f0_range_hz = value;
        new
    }

    // === Grit Factors ===
    fn harmonic_to_noise_ratio(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.harmonic_to_noise_ratio = value;
        new
    }

    fn spectral_flatness(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.spectral_flatness = value;
        new
    }

    // === Motion Factors ===
    fn attack_time_ms(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.attack_time_ms = value;
        new
    }

    fn decay_time_ms(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.decay_time_ms = value;
        new
    }

    fn sustain_level(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.sustain_level = value;
        new
    }

    fn vibrato_rate_hz(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.vibrato_rate_hz = value;
        new
    }

    fn vibrato_depth(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.vibrato_depth = value;
        new
    }

    fn jitter(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.jitter = value;
        new
    }

    // === Fingerprint Factors ===
    fn mfcc(&self, mfcc_1: f32, mfcc_2: f32, mfcc_3: f32, mfcc_4: f32) -> Self {
        let mut new = self.clone();
        new.metadata.mfcc_1 = mfcc_1;
        new.metadata.mfcc_2 = mfcc_2;
        new.metadata.mfcc_3 = mfcc_3;
        new.metadata.mfcc_4 = mfcc_4;
        new
    }

    fn spectral_flux(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.spectral_flux = value;
        new
    }

    // === Rhythm Factors ===
    fn rhythm(&self, median_ici_ms: f32, onset_rate_hz: f32, ici_cv: f32) -> Self {
        let mut new = self.clone();
        new.metadata.median_ici_ms = median_ici_ms;
        new.metadata.onset_rate_hz = onset_rate_hz;
        new.metadata.ici_coefficient_of_variation = ici_cv;
        new
    }

    // === 45D Expansion - Resonance Factors ===
    fn formant_1_hz(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.formant_1_hz = value;
        new
    }

    fn formant_2_hz(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.formant_2_hz = value;
        new
    }

    fn formant_3_hz(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.formant_3_hz = value;
        new
    }

    fn formant_1_bandwidth(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.formant_1_bandwidth = value;
        new
    }

    fn formant_2_bandwidth(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.formant_2_bandwidth = value;
        new
    }

    fn formant_dispersion(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.formant_dispersion = value;
        new
    }

    // === 45D Expansion - Spectral Shape Factors ===
    fn spectral_centroid(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.spectral_centroid = value;
        new
    }

    fn spectral_spread(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.spectral_spread = value;
        new
    }

    fn spectral_skewness(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.spectral_skewness = value;
        new
    }

    fn spectral_kurtosis(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.spectral_kurtosis = value;
        new
    }

    // === 45D Expansion - Modulation Factors ===
    fn spectral_tilt(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.spectral_tilt = value;
        new
    }

    fn fm_slope(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.fm_slope = value;
        new
    }

    fn am_depth(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.am_depth = value;
        new
    }

    // === 45D Expansion - Non-Linear Factors ===
    fn subharmonic_ratio(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.subharmonic_ratio = value;
        new
    }

    fn spectral_entropy(&self, value: f32) -> Self {
        let mut new = self.clone();
        new.metadata.spectral_entropy = value;
        new
    }

    /// Build the SourceMetadata
    fn build(&self) -> PySourceMetadata {
        self.metadata
    }
}

/// PyO3 bindings for Granular Concatenative Synthesizer
///
/// High-fidelity synthesizer that preserves formant structure
/// by manipulating real audio samples.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "GranularConcatenativeSynthesizer")]
pub struct PyGranularConcatenativeSynthesizer {
    inner: synthesis::GranularConcatenativeSynthesizer,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
#[allow(non_local_definitions)]
impl PyGranularConcatenativeSynthesizer {
    /// Create a new Granular Concatenative Synthesizer
    ///
    /// Parameters:
    /// - sample_rate: Audio sample rate (e.g., 22050)
    #[new]
    fn new(sample_rate: usize) -> Self {
        Self {
            inner: synthesis::GranularConcatenativeSynthesizer::new(sample_rate),
        }
    }

    /// Load source audio buffer (real recording)
    ///
    /// Parameters:
    /// - source: List of audio samples (f32 values)
    fn load_source(&mut self, source: Vec<f32>) {
        self.inner.load_source(source);
    }

    /// Load source audio buffer with metadata (for delta-based synthesis)
    ///
    /// **VECTOR DELTA SUPPORT**: This enables delta commands like "shift pitch by +50Hz"
    /// instead of absolute commands like "set pitch to 7000Hz".
    ///
    /// Parameters:
    /// - source: List of audio samples (f32 values)
    /// - metadata: SourceMetadata object with F0, duration, range info
    ///
    /// Example:
    /// ```python
    /// from technical_architecture import GranularConcatenativeSynthesizer, SourceMetadata
    ///
    /// metadata = SourceMetadata(
    ///     mean_f0_hz=6800.0,
    ///     duration_ms=50.0,
    ///     f0_range_hz=400.0
    /// )
    /// synthesizer.load_source_with_metadata(audio_buffer, metadata)
    ///
    /// # Now we can use delta commands!
    /// synthesizer.shift_pitch_by_hz(200.0)  # 6800 + 200 = 7000Hz
    /// synthesizer.shift_duration_by_ms(-10.0)  # 50 - 10 = 40ms
    /// ```
    fn load_source_with_metadata(&mut self, source: Vec<f32>, metadata: PySourceMetadata) {
        let rust_metadata: synthesis::SourceMetadata = metadata.into();
        self.inner.load_source_with_metadata(source, rust_metadata);
    }

    /// Set source metadata (call after load_source() if metadata known)
    ///
    /// Parameters:
    /// - metadata: SourceMetadata object with F0, duration, range info
    fn set_source_metadata(&mut self, metadata: PySourceMetadata) {
        let rust_metadata: synthesis::SourceMetadata = metadata.into();
        self.inner.set_source_metadata(rust_metadata);
    }

    /// Shift pitch by absolute Hz amount (VECTOR DELTA COMMAND)
    ///
    /// **GOOD**: "Shift pitch by +50Hz relative to source"
    /// **BAD**: "Set pitch to 7000Hz" (ignores source F0)
    ///
    /// This requires source metadata to be set (via load_source_with_metadata or set_source_metadata).
    ///
    /// Parameters:
    /// - delta_hz: Pitch shift in Hz (positive = higher, negative = lower)
    ///
    /// Example:
    /// ```python
    /// # Source F0 = 6800Hz
    /// synthesizer.shift_pitch_by_hz(200.0)   # Result: 7000Hz
    /// synthesizer.shift_pitch_by_hz(-300.0)  # Result: 6500Hz
    /// ```
    fn shift_pitch_by_hz(&mut self, delta_hz: f32) {
        self.inner.shift_pitch_by_hz(delta_hz);
    }

    /// Shift duration by absolute ms amount (VECTOR DELTA COMMAND)
    ///
    /// **GOOD**: "Shift duration by -10ms relative to source"
    /// **BAD**: "Set duration to 40ms" (ignores source duration)
    ///
    /// This requires source metadata to be set (via load_source_with_metadata or set_source_metadata).
    ///
    /// Parameters:
    /// - delta_ms: Duration shift in ms (positive = longer, negative = shorter)
    ///
    /// Example:
    /// ```python
    /// # Source duration = 50ms
    /// synthesizer.shift_duration_by_ms(-10.0)  # Result: 40ms
    /// synthesizer.shift_duration_by_ms(20.0)   # Result: 70ms
    /// ```
    fn shift_duration_by_ms(&mut self, delta_ms: f32) {
        self.inner.shift_duration_by_ms(delta_ms);
    }

    /// Apply Vector Delta (17D feature shift)
    ///
    /// **PRIMARY INTEGRATION POINT FOR ACOUSTIC ALGEBRA**
    ///
    /// Applies multiple shifts simultaneously from a delta vector.
    /// This is how you connect Acoustic Algebra output to Rust synthesis.
    ///
    /// From acoustic algebra: delta = virtual_phrase - nearest_real_phrase
    ///
    /// Parameters:
    /// - delta_f0_hz: Pitch shift in Hz
    /// - delta_duration_ms: Duration shift in ms
    /// - delta_f0_range_hz: F0 range shift in Hz
    ///
    /// Example:
    /// ```python
    /// from analysis.rosetta_stone.contextual_map import ContextualMap
    /// from technical_architecture import GranularConcatenativeSynthesizer, SourceMetadata
    ///
    /// # 1. Generate virtual phrase (30% aggressive)
    /// virtual = map_obj.generate_graded_phrase('aggression', intensity=0.3)
    ///
    /// # 2. Find nearest real phrase
    /// nearest_key, nearest_vec, distance = map_obj.find_nearest_real_phrase(virtual, phrase_vectors)
    ///
    /// # 3. Calculate delta
    /// delta_f0 = virtual.mean_f0_hz - nearest_vec.mean_f0_hz
    /// delta_dur = virtual.duration_ms - nearest_vec.duration_ms
    /// delta_range = virtual.f0_range_hz - nearest_vec.f0_range_hz
    ///
    /// # 4. Load source with metadata
    /// metadata = SourceMetadata(
    ///     mean_f0_hz=nearest_vec.mean_f0_hz,
    ///     duration_ms=nearest_vec.duration_ms,
    ///     f0_range_hz=nearest_vec.f0_range_hz
    /// )
    /// synthesizer.load_source_with_metadata(audio_buffer, metadata)
    ///
    /// # 5. Apply delta (VECTOR DELTA COMMAND!)
    /// synthesizer.apply_vector_delta(delta_f0, delta_dur, delta_range)
    ///
    /// # 6. Synthesize
    /// output = synthesizer.synthesize(duration_ms=virtual.duration_ms)
    /// ```
    fn apply_vector_delta(&mut self, delta_f0_hz: f32, delta_duration_ms: f32, delta_f0_range_hz: f32) {
        self.inner
            .apply_vector_delta(delta_f0_hz, delta_duration_ms, delta_f0_range_hz);
    }

    /// Set pitch shift ratio
    ///
    /// Parameters:
    /// - ratio: Pitch shift ratio (0.5 = octave down, 1.0 = natural, 2.0 = octave up)
    fn set_pitch_shift(&mut self, ratio: f32) {
        self.inner.set_pitch_shift(ratio);
    }

    /// Set grain size in milliseconds
    ///
    /// Parameters:
    /// - size_ms: Grain window size (typically 10-50ms)
    fn set_grain_size_ms(&mut self, size_ms: f32) {
        self.inner.set_grain_size_ms(size_ms);
    }

    /// Synthesize audio with specified duration
    ///
    /// This manipulates the loaded source audio using granular synthesis,
    /// preserving formant structure while allowing pitch/time flexibility.
    ///
    /// Parameters:
    /// - duration_ms: Output duration in milliseconds
    ///
    /// Returns: List of synthesized audio samples
    fn synthesize(&mut self, duration_ms: f32) -> Vec<f32> {
        self.inner.synthesize(duration_ms)
    }

    /// Convenience method: Synthesize from file path
    ///
    /// Loads audio from file and synthesizes with given parameters.
    ///
    /// Parameters:
    /// - file_path: Path to audio file (WAV)
    /// - duration_ms: Output duration in milliseconds
    /// - pitch_shift: Pitch shift ratio (default 1.0)
    /// - grain_size_ms: Grain size in milliseconds (default 20.0)
    ///
    /// Returns: List of synthesized audio samples
    fn synthesize_from_file(
        &mut self,
        file_path: String,
        _duration_ms: f32,
        _pitch_shift: Option<f32>,
        _grain_size_ms: Option<f32>,
    ) -> PyResult<Vec<f32>> {
        // Read audio file using soundfile
        use std::path::Path;
        let path = Path::new(&file_path);

        if !path.exists() {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "Audio file not found: {}",
                file_path
            )));
        }

        // For now, return error - we'll need to add proper audio file loading
        // This is a placeholder for the actual implementation
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "synthesize_from_file not yet implemented - use load_source() with pre-loaded audio",
        ))
    }
}

#[cfg(feature = "python-bindings")]
#[pymethods]
#[allow(non_local_definitions)]
impl PyTechnicalArchitect {
    /// Create a new Technical Architect from Python
    #[new]
    fn new(config_json: String) -> PyResult<Self> {
        let config: TechArchConfig = serde_json::from_str(&config_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid config: {}", e)))?;

        // Use tokio runtime
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        let inner = rt
            .block_on(async { TechnicalArchitect::new(config).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to initialize: {}", e)))?;

        Ok(Self { inner: Arc::new(inner) })
    }

    /// Process an audio frame from Python
    fn process_audio_frame(&self, audio: Vec<f32>) -> PyResult<Vec<f32>> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        rt.block_on(async { self.inner.process_audio_frame(audio).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Processing failed: {}", e)))
    }

    /// Get thermal state as string
    fn get_thermal_state(&self) -> PyResult<String> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        let state = rt.block_on(async { self.inner.get_thermal_state().await });

        Ok(format!("{:?}", state))
    }

    /// Get statistics as JSON string
    fn get_stats(&self) -> PyResult<String> {
        let stats = self.inner.stats.lock();
        serde_json::to_string(&*stats)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to serialize: {}", e)))
    }
}

// ============================================================================
// Visual Recording Python Bindings
// ============================================================================

/// Python wrapper for AudioSyncEvent
#[cfg(feature = "python-bindings")]
#[pyclass(name = "AudioSyncEvent")]
#[derive(Clone)]
pub struct PyAudioSyncEvent {
    #[pyo3(get, set)]
    pub timestamp_ns: u64,
    #[pyo3(get, set)]
    pub event_type: String,
    #[pyo3(get, set)]
    pub phrase_key: Option<String>,
    #[pyo3(get, set)]
    pub context: Option<String>,
    #[pyo3(get, set)]
    pub individual_id: Option<String>,
    #[pyo3(get, set)]
    pub frame_index: Option<usize>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
#[allow(non_local_definitions)]
impl PyAudioSyncEvent {
    #[new]
    fn new(
        timestamp_ns: u64,
        event_type: String,
        phrase_key: Option<String>,
        context: Option<String>,
        individual_id: Option<String>,
        frame_index: Option<usize>,
    ) -> Self {
        Self {
            timestamp_ns,
            event_type,
            phrase_key,
            context,
            individual_id,
            frame_index,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioSyncEvent(timestamp_ns={}, event_type={}, phrase={:?}, context={:?})",
            self.timestamp_ns, self.event_type, self.phrase_key, self.context
        )
    }
}

#[cfg(feature = "python-bindings")]
impl From<PyAudioSyncEvent> for visual_recording::AudioSyncEvent {
    fn from(py_event: PyAudioSyncEvent) -> Self {
        visual_recording::AudioSyncEvent {
            timestamp_ns: py_event.timestamp_ns,
            event_type: match py_event.event_type.as_str() {
                "vocalization_detected" => visual_recording::AudioEventType::VocalizationDetected,
                "response_generated" => visual_recording::AudioEventType::ResponseGenerated,
                "phrase_discovered" => visual_recording::AudioEventType::PhraseDiscovered,
                "context_switch" => visual_recording::AudioEventType::ContextSwitch,
                _ => visual_recording::AudioEventType::VocalizationDetected,
            },
            phrase_key: py_event.phrase_key,
            context: py_event.context,
            individual_id: py_event.individual_id,
            frame_index: py_event.frame_index,
        }
    }
}

// ============================================================================
// PyO3 Bindings for Safety-Critical Components
// ============================================================================

/// Python wrapper for OperationMode
#[cfg(feature = "python-bindings")]
#[pyclass(name = "OperationMode")]
#[derive(Clone, Copy)]
pub struct PyOperationMode {
    inner: peer_controller::OperationMode,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyOperationMode {
    #[staticmethod]
    fn passthrough() -> Self {
        Self {
            inner: peer_controller::OperationMode::Passthrough,
        }
    }

    #[staticmethod]
    fn interactive() -> Self {
        Self {
            inner: peer_controller::OperationMode::Interactive,
        }
    }

    fn is_passthrough(&self) -> bool {
        matches!(self.inner, peer_controller::OperationMode::Passthrough)
    }

    fn is_interactive(&self) -> bool {
        matches!(self.inner, peer_controller::OperationMode::Interactive)
    }

    fn __repr__(&self) -> String {
        match self.inner {
            peer_controller::OperationMode::Passthrough => "OperationMode.Passthrough".to_string(),
            peer_controller::OperationMode::Interactive => "OperationMode.Interactive".to_string(),
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __eq__(&self, other: &PyOperationMode) -> bool {
        self.inner == other.inner
    }
}

/// Python wrapper for PeerControllerConfig
#[cfg(feature = "python-bindings")]
#[pyclass(name = "PeerControllerConfig")]
#[derive(Clone)]
pub struct PyPeerControllerConfig {
    #[pyo3(get, set)]
    pub heartbeat_endpoint: String,

    #[pyo3(get, set)]
    pub heartbeat_timeout_ms: u64,

    #[pyo3(get, set)]
    pub poll_interval_ms: u64,

    #[pyo3(get, set)]
    pub verbose_logging: bool,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
#[allow(non_local_definitions)]
impl PyPeerControllerConfig {
    #[new]
    #[pyo3(signature = (
        heartbeat_endpoint=None,
        heartbeat_timeout_ms=100,
        poll_interval_ms=10,
        verbose_logging=false
    ))]
    fn new(
        heartbeat_endpoint: Option<String>,
        heartbeat_timeout_ms: u64,
        poll_interval_ms: u64,
        verbose_logging: bool,
    ) -> Self {
        Self {
            heartbeat_endpoint: heartbeat_endpoint.unwrap_or_else(|| "ipc:///tmp/cognitive_heartbeat.ipc".to_string()),
            heartbeat_timeout_ms,
            poll_interval_ms,
            verbose_logging,
        }
    }

    #[staticmethod]
    fn default() -> Self {
        let rust_config = peer_controller::PeerControllerConfig::default();
        Self {
            heartbeat_endpoint: rust_config.heartbeat_endpoint,
            heartbeat_timeout_ms: rust_config.heartbeat_timeout_ms,
            poll_interval_ms: rust_config.poll_interval_ms,
            verbose_logging: rust_config.verbose_logging,
        }
    }
}

#[cfg(feature = "python-bindings")]
impl From<PyPeerControllerConfig> for peer_controller::PeerControllerConfig {
    fn from(py_config: PyPeerControllerConfig) -> Self {
        peer_controller::PeerControllerConfig {
            heartbeat_endpoint: py_config.heartbeat_endpoint,
            heartbeat_timeout_ms: py_config.heartbeat_timeout_ms,
            poll_interval_ms: py_config.poll_interval_ms,
            verbose_logging: py_config.verbose_logging,
        }
    }
}

/// Python wrapper for PeerController
#[cfg(feature = "python-bindings")]
#[pyclass(name = "PeerController")]
pub struct PyPeerController {
    inner: std::sync::Mutex<peer_controller::PeerController>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyPeerController {
    /// Create a new PeerController
    #[new]
    fn new(config: PyPeerControllerConfig) -> PyResult<Self> {
        let rust_config: peer_controller::PeerControllerConfig = config.into();
        peer_controller::PeerController::new(rust_config)
            .map(|controller| Self {
                inner: std::sync::Mutex::new(controller),
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create PeerController: {}", e)))
    }

    /// Tick the controller (check for heartbeat and update mode)
    /// Returns the current operation mode
    fn tick(&self) -> PyResult<PyOperationMode> {
        let mut controller = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock failed: {}", e)))?;
        controller
            .tick()
            .map(|mode| PyOperationMode { inner: mode })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Tick failed: {}", e)))
    }

    /// Get the configuration
    fn get_config(&self) -> PyPeerControllerConfig {
        let controller = self.inner.lock().unwrap();
        let rust_config = controller.get_config();
        PyPeerControllerConfig {
            heartbeat_endpoint: rust_config.heartbeat_endpoint.clone(),
            heartbeat_timeout_ms: rust_config.heartbeat_timeout_ms,
            poll_interval_ms: rust_config.poll_interval_ms,
            verbose_logging: rust_config.verbose_logging,
        }
    }

    /// Check if currently in Interactive mode
    fn is_interactive(&self) -> PyResult<bool> {
        let mut controller = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock failed: {}", e)))?;
        let mode = controller
            .tick()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Tick failed: {}", e)))?;
        Ok(matches!(mode, peer_controller::OperationMode::Interactive))
    }

    /// Check if currently in Passthrough mode
    fn is_passthrough(&self) -> PyResult<bool> {
        let mut controller = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock failed: {}", e)))?;
        let mode = controller
            .tick()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Tick failed: {}", e)))?;
        Ok(matches!(mode, peer_controller::OperationMode::Passthrough))
    }

    fn __repr__(&self) -> String {
        let mut controller = self.inner.lock().unwrap();
        let mode = controller.tick().unwrap_or(peer_controller::OperationMode::Passthrough);
        format!("PeerController(mode={:?})", mode)
    }
}

// ============================================================================
// Environmental Monitor Python Bindings
// ============================================================================

/// Python wrapper for SessionViability
#[cfg(feature = "python-bindings")]
#[pyclass(name = "SessionViability")]
#[derive(Clone, Copy)]
pub struct PySessionViability {
    inner: environmental_monitor::SessionViability,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PySessionViability {
    #[staticmethod]
    fn viable() -> Self {
        Self {
            inner: environmental_monitor::SessionViability::Viable,
        }
    }

    #[staticmethod]
    fn marginal() -> Self {
        Self {
            inner: environmental_monitor::SessionViability::Marginal,
        }
    }

    #[staticmethod]
    fn infeasible() -> Self {
        Self {
            inner: environmental_monitor::SessionViability::Infeasible,
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __str__(&self) -> String {
        match self.inner {
            environmental_monitor::SessionViability::Viable => "Viable".to_string(),
            environmental_monitor::SessionViability::Marginal => "Marginal".to_string(),
            environmental_monitor::SessionViability::Infeasible => "Infeasible".to_string(),
        }
    }

    fn __eq__(&self, other: &PySessionViability) -> bool {
        std::mem::discriminant(&self.inner) == std::mem::discriminant(&other.inner)
    }
}

/// Python wrapper for RainIntensity
#[cfg(feature = "python-bindings")]
#[pyclass(name = "RainIntensity")]
#[derive(Clone, Copy)]
pub struct PyRainIntensity {
    inner: environmental_monitor::RainIntensity,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyRainIntensity {
    #[staticmethod]
    fn none() -> Self {
        Self {
            inner: environmental_monitor::RainIntensity::None,
        }
    }

    #[staticmethod]
    fn light() -> Self {
        Self {
            inner: environmental_monitor::RainIntensity::Light,
        }
    }

    #[staticmethod]
    fn moderate() -> Self {
        Self {
            inner: environmental_monitor::RainIntensity::Moderate,
        }
    }

    #[staticmethod]
    fn heavy() -> Self {
        Self {
            inner: environmental_monitor::RainIntensity::Heavy,
        }
    }

    #[staticmethod]
    fn storm() -> Self {
        Self {
            inner: environmental_monitor::RainIntensity::Storm,
        }
    }

    #[staticmethod]
    fn from_mm_h(mm_h: f32) -> Self {
        Self {
            inner: environmental_monitor::RainIntensity::from_mm_h(mm_h),
        }
    }

    fn forces_passthrough(&self) -> bool {
        self.inner.forces_passthrough()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __str__(&self) -> String {
        match self.inner {
            environmental_monitor::RainIntensity::None => "None".to_string(),
            environmental_monitor::RainIntensity::Light => "Light".to_string(),
            environmental_monitor::RainIntensity::Moderate => "Moderate".to_string(),
            environmental_monitor::RainIntensity::Heavy => "Heavy".to_string(),
            environmental_monitor::RainIntensity::Storm => "Storm".to_string(),
        }
    }

    fn __eq__(&self, other: &PyRainIntensity) -> bool {
        std::mem::discriminant(&self.inner) == std::mem::discriminant(&other.inner)
    }
}

/// Python wrapper for TemperatureClassification
#[cfg(feature = "python-bindings")]
#[pyclass(name = "TemperatureClassification")]
#[derive(Clone, Copy)]
pub struct PyTemperatureClassification {
    inner: environmental_monitor::TemperatureClassification,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyTemperatureClassification {
    #[staticmethod]
    fn freezing() -> Self {
        Self {
            inner: environmental_monitor::TemperatureClassification::Freezing,
        }
    }

    #[staticmethod]
    fn cold() -> Self {
        Self {
            inner: environmental_monitor::TemperatureClassification::Cold,
        }
    }

    #[staticmethod]
    fn mild() -> Self {
        Self {
            inner: environmental_monitor::TemperatureClassification::Mild,
        }
    }

    #[staticmethod]
    fn hot() -> Self {
        Self {
            inner: environmental_monitor::TemperatureClassification::Hot,
        }
    }

    #[staticmethod]
    fn extreme() -> Self {
        Self {
            inner: environmental_monitor::TemperatureClassification::Extreme,
        }
    }

    #[staticmethod]
    fn from_celsius(celsius: f32) -> Self {
        Self {
            inner: environmental_monitor::TemperatureClassification::from_celsius(celsius),
        }
    }

    fn forces_passthrough(&self) -> bool {
        self.inner.forces_passthrough()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __str__(&self) -> String {
        match self.inner {
            environmental_monitor::TemperatureClassification::Freezing => "Freezing".to_string(),
            environmental_monitor::TemperatureClassification::Cold => "Cold".to_string(),
            environmental_monitor::TemperatureClassification::Mild => "Mild".to_string(),
            environmental_monitor::TemperatureClassification::Hot => "Hot".to_string(),
            environmental_monitor::TemperatureClassification::Extreme => "Extreme".to_string(),
        }
    }

    fn __eq__(&self, other: &PyTemperatureClassification) -> bool {
        std::mem::discriminant(&self.inner) == std::mem::discriminant(&other.inner)
    }
}

/// Python wrapper for EnvironmentalConditions
#[cfg(feature = "python-bindings")]
#[pyclass(name = "EnvironmentalConditions")]
#[derive(Clone)]
pub struct PyEnvironmentalConditions {
    #[pyo3(get, set)]
    pub temperature_celsius: f32,
    #[pyo3(get, set)]
    pub humidity_percent: f32,
    #[pyo3(get, set)]
    pub light_lux: f32,
    #[pyo3(get, set)]
    pub rain_intensity_mm_h: f32,
    #[pyo3(get, set)]
    pub wind_speed_m_s: f32,
}

#[cfg(feature = "python-bindings")]
impl From<environmental_monitor::EnvironmentalConditions> for PyEnvironmentalConditions {
    fn from(rust: environmental_monitor::EnvironmentalConditions) -> Self {
        Self {
            temperature_celsius: rust.temperature_celsius,
            humidity_percent: rust.humidity_percent,
            light_lux: rust.light_lux,
            rain_intensity_mm_h: rust.rain_intensity_mm_h,
            wind_speed_m_s: rust.wind_speed_m_s,
        }
    }
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyEnvironmentalConditions {
    #[new]
    #[pyo3(signature = (
        temperature_celsius=22.0,
        humidity_percent=60.0,
        light_lux=500.0,
        rain_intensity_mm_h=0.0,
        wind_speed_m_s=2.0
    ))]
    fn new(
        temperature_celsius: f32,
        humidity_percent: f32,
        light_lux: f32,
        rain_intensity_mm_h: f32,
        wind_speed_m_s: f32,
    ) -> Self {
        Self {
            temperature_celsius,
            humidity_percent,
            light_lux,
            rain_intensity_mm_h,
            wind_speed_m_s,
        }
    }

    fn rain_intensity(&self) -> PyRainIntensity {
        PyRainIntensity {
            inner: environmental_monitor::RainIntensity::from_mm_h(self.rain_intensity_mm_h),
        }
    }

    fn temperature_classification(&self) -> PyTemperatureClassification {
        PyTemperatureClassification {
            inner: environmental_monitor::TemperatureClassification::from_celsius(self.temperature_celsius),
        }
    }

    fn assess_viability(&self) -> PySessionViability {
        // Create a temporary EnvironmentalConditions to use its method
        let rust_conditions = environmental_monitor::EnvironmentalConditions {
            timestamp: ptp::PtpTimestamp::new(0, 0),
            temperature_celsius: self.temperature_celsius,
            humidity_percent: self.humidity_percent,
            light_lux: self.light_lux,
            rain_intensity_mm_h: self.rain_intensity_mm_h,
            wind_speed_m_s: self.wind_speed_m_s,
            atmospheric_pressure_hpa: 1013.25,
            battery_temperature_celsius: 25.0,
        };
        PySessionViability {
            inner: rust_conditions.assess_viability(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "EnvironmentalConditions(temp={}°C, rain={}mm/h, light={}lux)",
            self.temperature_celsius, self.rain_intensity_mm_h, self.light_lux
        )
    }
}

/// Python wrapper for EnvironmentalMonitorConfig
#[cfg(feature = "python-bindings")]
#[pyclass(name = "EnvironmentalMonitorConfig")]
#[derive(Clone)]
pub struct PyEnvironmentalMonitorConfig {
    #[pyo3(get, set)]
    pub poll_interval_ms: u64,
    #[pyo3(get, set)]
    pub sensor_timeout_ms: u64,
    #[pyo3(get, set)]
    pub enable_rain_detection: bool,
    #[pyo3(get, set)]
    pub enable_solar_forecast: bool,
    #[pyo3(get, set)]
    pub mock_mode: bool,
}

#[cfg(feature = "python-bindings")]
impl From<PyEnvironmentalMonitorConfig> for environmental_monitor::EnvironmentalMonitorConfig {
    fn from(py_config: PyEnvironmentalMonitorConfig) -> Self {
        Self {
            poll_interval_ms: py_config.poll_interval_ms,
            sensor_timeout_ms: py_config.sensor_timeout_ms,
            enable_rain_detection: py_config.enable_rain_detection,
            enable_solar_forecast: py_config.enable_solar_forecast,
            mock_mode: py_config.mock_mode,
        }
    }
}

#[cfg(feature = "python-bindings")]
#[pymethods]
#[allow(non_local_definitions)]
impl PyEnvironmentalMonitorConfig {
    #[new]
    #[pyo3(signature = (
        poll_interval_ms=5000,
        sensor_timeout_ms=1000,
        enable_rain_detection=true,
        enable_solar_forecast=true,
        mock_mode=false
    ))]
    fn new(
        poll_interval_ms: u64,
        sensor_timeout_ms: u64,
        enable_rain_detection: bool,
        enable_solar_forecast: bool,
        mock_mode: bool,
    ) -> Self {
        Self {
            poll_interval_ms,
            sensor_timeout_ms,
            enable_rain_detection,
            enable_solar_forecast,
            mock_mode,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "EnvironmentalMonitorConfig(poll={}ms, mock={})",
            self.poll_interval_ms, self.mock_mode
        )
    }
}

/// Python wrapper for EnvironmentalMonitor
#[cfg(feature = "python-bindings")]
#[pyclass(name = "EnvironmentalMonitor")]
pub struct PyEnvironmentalMonitor {
    inner: std::sync::Mutex<environmental_monitor::EnvironmentalMonitor>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyEnvironmentalMonitor {
    #[new]
    fn new(config: PyEnvironmentalMonitorConfig) -> Self {
        let rust_config: environmental_monitor::EnvironmentalMonitorConfig = config.into();
        Self {
            inner: std::sync::Mutex::new(environmental_monitor::EnvironmentalMonitor::new(rust_config)),
        }
    }

    #[staticmethod]
    fn with_defaults() -> Self {
        Self {
            inner: std::sync::Mutex::new(environmental_monitor::EnvironmentalMonitor::with_defaults()),
        }
    }

    #[staticmethod]
    fn for_testing() -> Self {
        Self {
            inner: std::sync::Mutex::new(environmental_monitor::EnvironmentalMonitor::for_testing()),
        }
    }

    fn poll_sensors(&self) -> PyResult<PyEnvironmentalConditions> {
        let mut monitor = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock failed: {}", e)))?;
        monitor
            .poll_sensors()
            .map(PyEnvironmentalConditions::from)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Poll failed: {}", e)))
    }

    fn current_conditions(&self) -> PyEnvironmentalConditions {
        let monitor = self.inner.lock().unwrap();
        PyEnvironmentalConditions::from(monitor.current_conditions().clone())
    }

    fn assess_session_viability(&self) -> PySessionViability {
        let monitor = self.inner.lock().unwrap();
        PySessionViability {
            inner: monitor.assess_session_viability(),
        }
    }

    fn forces_passthrough(&self) -> bool {
        let monitor = self.inner.lock().unwrap();
        monitor.forces_passthrough()
    }

    fn set_conditions(&self, conditions: PyEnvironmentalConditions) -> PyResult<()> {
        let mut monitor = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock failed: {}", e)))?;
        // Convert Python conditions to Rust conditions
        let rust_conditions = environmental_monitor::EnvironmentalConditions {
            timestamp: ptp::PtpTimestamp::new(0, 0),
            temperature_celsius: conditions.temperature_celsius,
            humidity_percent: conditions.humidity_percent,
            light_lux: conditions.light_lux,
            rain_intensity_mm_h: conditions.rain_intensity_mm_h,
            wind_speed_m_s: conditions.wind_speed_m_s,
            atmospheric_pressure_hpa: 1013.25,
            battery_temperature_celsius: 25.0,
        };
        (*monitor).set_conditions(rust_conditions);
        Ok(())
    }

    fn __repr__(&self) -> String {
        let monitor = self.inner.lock().unwrap();
        let conditions = monitor.current_conditions();
        format!(
            "EnvironmentalMonitor(temp={}°C, rain={}mm/h)",
            conditions.temperature_celsius, conditions.rain_intensity_mm_h
        )
    }
}

// ============================================================================
// Thermal State Python Bindings
// ============================================================================

/// Python wrapper for ThermalState
#[cfg(feature = "python-bindings")]
#[pyclass(name = "ThermalState")]
#[derive(Clone, Copy)]
pub struct PyThermalState {
    inner: thermal::ThermalState,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyThermalState {
    #[staticmethod]
    fn normal() -> Self {
        Self {
            inner: thermal::ThermalState::Normal,
        }
    }

    #[staticmethod]
    fn warning() -> Self {
        Self {
            inner: thermal::ThermalState::Warning,
        }
    }

    #[staticmethod]
    fn throttling() -> Self {
        Self {
            inner: thermal::ThermalState::Throttling,
        }
    }

    #[staticmethod]
    fn critical() -> Self {
        Self {
            inner: thermal::ThermalState::Critical,
        }
    }

    fn requires_throttling(&self) -> bool {
        self.inner.requires_throttling()
    }

    fn is_critical(&self) -> bool {
        self.inner.is_critical()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __str__(&self) -> String {
        match self.inner {
            thermal::ThermalState::Normal => "Normal".to_string(),
            thermal::ThermalState::Warning => "Warning".to_string(),
            thermal::ThermalState::Throttling => "Throttling".to_string(),
            thermal::ThermalState::Critical => "Critical".to_string(),
        }
    }

    fn __eq__(&self, other: &PyThermalState) -> bool {
        std::mem::discriminant(&self.inner) == std::mem::discriminant(&other.inner)
    }
}

/// Python wrapper for VisualRecorderConfig
#[cfg(feature = "python-bindings")]
#[pyclass(name = "VisualRecorderConfig")]
#[derive(Clone)]
pub struct PyVisualRecorderConfig {
    #[pyo3(get, set)]
    pub camera_id: u32,
    #[pyo3(get, set)]
    pub resolution: (u32, u32),
    #[pyo3(get, set)]
    pub fps: f32,
    #[pyo3(get, set)]
    pub codec: String,
    #[pyo3(get, set)]
    pub compression_quality: u8,
    #[pyo3(get, set)]
    pub max_queue_size: usize,
    #[pyo3(get, set)]
    pub recording_dir: String,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
#[allow(non_local_definitions)]
impl PyVisualRecorderConfig {
    #[new]
    #[pyo3(signature = (
        camera_id=0,
        resolution=(1280, 720),
        fps=30.0,
        codec="mp4v".to_string(),
        compression_quality=75,
        max_queue_size=100,
        recording_dir="./recordings/visual".to_string()
    ))]
    fn new(
        camera_id: u32,
        resolution: (u32, u32),
        fps: f32,
        codec: String,
        compression_quality: u8,
        max_queue_size: usize,
        recording_dir: String,
    ) -> Self {
        Self {
            camera_id,
            resolution,
            fps,
            codec,
            compression_quality,
            max_queue_size,
            recording_dir,
        }
    }

    #[staticmethod]
    fn default() -> Self {
        Self {
            camera_id: 0,
            resolution: (1280, 720),
            fps: 30.0,
            codec: "mp4v".to_string(),
            compression_quality: 75,
            max_queue_size: 100,
            recording_dir: "./recordings/visual".to_string(),
        }
    }
}

#[cfg(feature = "python-bindings")]
impl From<PyVisualRecorderConfig> for visual_recording::VisualRecorderConfig {
    fn from(py_config: PyVisualRecorderConfig) -> Self {
        visual_recording::VisualRecorderConfig {
            camera_id: py_config.camera_id,
            resolution: py_config.resolution,
            fps: py_config.fps,
            codec: py_config.codec,
            compression_quality: py_config.compression_quality,
            max_queue_size: py_config.max_queue_size,
            recording_dir: py_config.recording_dir,
        }
    }
}

/// Python wrapper for RecordingStatistics
#[cfg(feature = "python-bindings")]
#[pyclass(name = "RecordingStatistics")]
#[derive(Clone)]
pub struct PyRecordingStatistics {
    #[pyo3(get, set)]
    pub state: String,
    #[pyo3(get, set)]
    pub frames_recorded: usize,
    #[pyo3(get, set)]
    pub dropped_frames: usize,
    #[pyo3(get, set)]
    pub current_session_id: Option<String>,
    #[pyo3(get, set)]
    pub duration_seconds: f64,
    #[pyo3(get, set)]
    pub pending_events: usize,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
#[allow(non_local_definitions)]
impl PyRecordingStatistics {
    #[new]
    #[pyo3(signature = (state, frames_recorded, dropped_frames, current_session_id, duration_seconds, pending_events))]
    fn new(
        state: String,
        frames_recorded: usize,
        dropped_frames: usize,
        current_session_id: Option<String>,
        duration_seconds: f64,
        pending_events: usize,
    ) -> Self {
        Self {
            state,
            frames_recorded,
            dropped_frames,
            current_session_id,
            duration_seconds,
            pending_events,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RecordingStatistics(state={}, frames={}, dropped={}, session={:?}, duration={:.2}s)",
            self.state, self.frames_recorded, self.dropped_frames, self.current_session_id, self.duration_seconds
        )
    }
}

#[cfg(feature = "python-bindings")]
impl From<visual_recording::RecordingStatistics> for PyRecordingStatistics {
    fn from(stats: visual_recording::RecordingStatistics) -> Self {
        Self {
            state: format!("{:?}", stats.state),
            frames_recorded: stats.frames_recorded,
            dropped_frames: stats.dropped_frames,
            current_session_id: stats.current_session_id,
            duration_seconds: stats.duration_seconds,
            pending_events: stats.pending_events,
        }
    }
}

/// Python wrapper for VisualMetadata
#[cfg(feature = "python-bindings")]
#[pyclass(name = "VisualMetadata")]
#[derive(Clone)]
pub struct PyVisualMetadata {
    #[pyo3(get, set)]
    pub session_id: String,
    #[pyo3(get, set)]
    pub camera_id: u32,
    #[pyo3(get, set)]
    pub resolution: (u32, u32),
    #[pyo3(get, set)]
    pub fps: f32,
    #[pyo3(get, set)]
    pub start_time_ns: u64,
    #[pyo3(get, set)]
    pub end_time_ns: Option<u64>,
    #[pyo3(get, set)]
    pub state: String,
    #[pyo3(get, set)]
    pub audio_sync_events: Vec<PyAudioSyncEvent>,
    #[pyo3(get, set)]
    pub storage_path: Option<String>,
    #[pyo3(get, set)]
    pub file_size_bytes: Option<u64>,
    #[pyo3(get, set)]
    pub duration_seconds: Option<f64>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
#[allow(non_local_definitions)]
impl PyVisualMetadata {
    fn __repr__(&self) -> String {
        format!(
            "VisualMetadata(session_id={}, camera={}, resolution={:?}, fps={}, state={}, events={})",
            self.session_id,
            self.camera_id,
            self.resolution,
            self.fps,
            self.state,
            self.audio_sync_events.len()
        )
    }

    fn calculate_duration_seconds(&self) -> Option<f64> {
        self.duration_seconds
    }

    fn sync_timestamp_to_frame(&self, timestamp_ns: u64) -> Option<usize> {
        if timestamp_ns < self.start_time_ns {
            return None;
        }
        let elapsed_ns = timestamp_ns - self.start_time_ns;
        let elapsed_seconds = elapsed_ns as f64 / 1_000_000_000.0;
        Some((elapsed_seconds * self.fps as f64) as usize)
    }
}

#[cfg(feature = "python-bindings")]
impl From<visual_recording::VisualMetadata> for PyVisualMetadata {
    fn from(metadata: visual_recording::VisualMetadata) -> Self {
        // Calculate duration before moving fields
        let duration_seconds = metadata.calculate_duration_seconds();

        Self {
            session_id: metadata.session_id,
            camera_id: metadata.camera_id,
            resolution: metadata.resolution,
            fps: metadata.fps,
            start_time_ns: metadata.start_time_ns,
            end_time_ns: metadata.end_time_ns,
            state: format!("{:?}", metadata.state),
            audio_sync_events: metadata
                .audio_sync_events
                .into_iter()
                .map(|e| PyAudioSyncEvent {
                    timestamp_ns: e.timestamp_ns,
                    event_type: format!("{:?}", e.event_type),
                    phrase_key: e.phrase_key,
                    context: e.context,
                    individual_id: e.individual_id,
                    frame_index: e.frame_index,
                })
                .collect(),
            storage_path: metadata.storage_path,
            file_size_bytes: metadata.file_size_bytes,
            duration_seconds,
        }
    }
}

/// Python wrapper for VisualRecorder
#[cfg(feature = "python-bindings")]
#[pyclass(name = "VisualRecorder")]
pub struct PyVisualRecorder {
    inner: Arc<parking_lot::Mutex<visual_recording::VisualRecorder>>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
#[allow(non_local_definitions)]
impl PyVisualRecorder {
    #[new]
    fn new(config: PyVisualRecorderConfig) -> Self {
        let rust_config: visual_recording::VisualRecorderConfig = config.into();
        Self {
            inner: Arc::new(parking_lot::Mutex::new(visual_recording::VisualRecorder::new(
                rust_config,
            ))),
        }
    }

    #[staticmethod]
    fn with_default_config(recording_dir: Option<String>) -> Self {
        let mut config = visual_recording::VisualRecorderConfig::default();
        if let Some(dir) = recording_dir {
            config.recording_dir = dir;
        }
        Self {
            inner: Arc::new(parking_lot::Mutex::new(visual_recording::VisualRecorder::new(config))),
        }
    }

    /// Start a new recording session
    fn start_session(&self, session_id: String) -> PyResult<String> {
        let mut recorder = self.inner.lock();
        recorder
            .start_session(&session_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to start session: {}", e)))
    }

    /// Stop current recording session
    fn stop_session(&self) -> PyResult<PyVisualMetadata> {
        let mut recorder = self.inner.lock();
        let metadata = recorder
            .stop_session()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to stop session: {}", e)))?;
        Ok(metadata.into())
    }

    /// Register audio event for synchronization
    fn register_audio_event(&self, event: PyAudioSyncEvent) -> PyResult<()> {
        let recorder = self.inner.lock();
        let rust_event: visual_recording::AudioSyncEvent = event.into();
        recorder.register_audio_event(rust_event);
        Ok(())
    }

    /// Get recording statistics
    fn get_statistics(&self) -> PyRecordingStatistics {
        let recorder = self.inner.lock();
        recorder.get_statistics().into()
    }

    /// Get current recording state
    fn get_state(&self) -> String {
        let recorder = self.inner.lock();
        format!("{:?}", recorder.state())
    }

    /// Check if currently recording
    fn is_recording(&self) -> bool {
        let recorder = self.inner.lock();
        recorder.is_recording()
    }

    /// Get current session ID
    fn get_session_id(&self) -> Option<String> {
        let recorder = self.inner.lock();
        recorder.session_id()
    }

    /// Get number of pending events
    fn get_pending_event_count(&self) -> usize {
        let recorder = self.inner.lock();
        recorder.pending_event_count()
    }

    /// Resolve video file path for a session
    fn resolve_video_path(&self, session_id: String) -> String {
        let recorder = self.inner.lock();
        recorder.resolve_video_path(&session_id)
    }

    /// Resolve metadata file path for a session
    fn resolve_metadata_path(&self, session_id: String) -> String {
        let recorder = self.inner.lock();
        recorder.resolve_metadata_path(&session_id)
    }

    fn __repr__(&self) -> String {
        let recorder = self.inner.lock();
        let state_str = format!("{:?}", recorder.state());
        format!(
            "VisualRecorder(state={}, session_id={:?})",
            state_str,
            recorder.session_id()
        )
    }
}

// ============================================================================
// PyO3 Bindings for Island Hopping Navigation
// ============================================================================

/// Python wrapper for Vector30D
#[cfg(feature = "python-bindings")]
#[pyclass(name = "Vector30D")]
#[derive(Clone, Copy)]
pub struct PyVector30D {
    inner: island_hopping::Vector30D,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyVector30D {
    /// Create a new Vector30D with all 30 dimensions
    #[allow(clippy::too_many_arguments)]
    #[new]
    fn new(
        // Fundamental (3)
        mean_f0_hz: f32,
        f0_range_hz: f32,
        duration_ms: f32,
        // Grit Factors (3)
        harmonic_to_noise_ratio: f32,
        spectral_flatness: f32,
        harmonicity: f32,
        // Motion Factors (7)
        attack_time_ms: f32,
        decay_time_ms: f32,
        sustain_level: f32,
        vibrato_rate_hz: f32,
        vibrato_depth: f32,
        jitter: f32,
        shimmer: f32,
        // Fingerprint Factors (13 MFCCs)
        mfcc_1: f32,
        mfcc_2: f32,
        mfcc_3: f32,
        mfcc_4: f32,
        mfcc_5: f32,
        mfcc_6: f32,
        mfcc_7: f32,
        mfcc_8: f32,
        mfcc_9: f32,
        mfcc_10: f32,
        mfcc_11: f32,
        mfcc_12: f32,
        mfcc_13: f32,
        // Spectral Dynamics (1)
        spectral_flux: f32,
        // Rhythm Factors (3)
        median_ici_ms: f32,
        onset_rate_hz: f32,
        ici_coefficient_of_variation: f32,
    ) -> Self {
        Self {
            inner: island_hopping::Vector30D::new(
                // Fundamental
                mean_f0_hz,
                f0_range_hz,
                duration_ms,
                // Grit Factors
                harmonic_to_noise_ratio,
                spectral_flatness,
                harmonicity,
                // Motion Factors
                attack_time_ms,
                decay_time_ms,
                sustain_level,
                vibrato_rate_hz,
                vibrato_depth,
                jitter,
                shimmer,
                // Fingerprint Factors
                mfcc_1,
                mfcc_2,
                mfcc_3,
                mfcc_4,
                mfcc_5,
                mfcc_6,
                mfcc_7,
                mfcc_8,
                mfcc_9,
                mfcc_10,
                mfcc_11,
                mfcc_12,
                mfcc_13,
                // Spectral Dynamics
                spectral_flux,
                // Rhythm Factors
                median_ici_ms,
                onset_rate_hz,
                ici_coefficient_of_variation,
            ),
        }
    }

    /// Create a Vector30D with default values
    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: island_hopping::Vector30D::default(),
        }
    }

    /// Calculate distance to another vector
    fn distance_to(&self, other: &PyVector30D) -> f32 {
        self.inner.distance_to(&other.inner)
    }

    /// Interpolate between two vectors (Bridge Builder - SAFE)
    fn interpolate(&self, other: &PyVector30D, alpha: f32) -> PyVector30D {
        PyVector30D {
            inner: self.inner.interpolate(&other.inner, alpha),
        }
    }

    /// Add two vectors
    fn add(&self, other: &PyVector30D) -> PyVector30D {
        PyVector30D {
            inner: self.inner.add(&other.inner),
        }
    }

    /// Subtract two vectors
    fn sub(&self, other: &PyVector30D) -> PyVector30D {
        PyVector30D {
            inner: self.inner.sub(&other.inner),
        }
    }

    /// Scale vector by factor
    fn scale(&self, factor: f32) -> PyVector30D {
        PyVector30D {
            inner: self.inner.scale(factor),
        }
    }

    /// Get magnitude
    fn magnitude(&self) -> f32 {
        self.inner.magnitude()
    }

    /// Normalize to unit vector
    fn normalized(&self) -> PyVector30D {
        PyVector30D {
            inner: self.inner.normalized(),
        }
    }

    // Getters for all 30 dimensions
    fn get_mean_f0_hz(&self) -> f32 {
        self.inner.mean_f0_hz
    }
    fn get_duration_ms(&self) -> f32 {
        self.inner.duration_ms
    }
    fn get_f0_range_hz(&self) -> f32 {
        self.inner.f0_range_hz
    }
    fn get_harmonic_to_noise_ratio(&self) -> f32 {
        self.inner.harmonic_to_noise_ratio
    }
    fn get_spectral_flatness(&self) -> f32 {
        self.inner.spectral_flatness
    }
    fn get_harmonicity(&self) -> f32 {
        self.inner.harmonicity
    }
    fn get_attack_time_ms(&self) -> f32 {
        self.inner.attack_time_ms
    }
    fn get_decay_time_ms(&self) -> f32 {
        self.inner.decay_time_ms
    }
    fn get_sustain_level(&self) -> f32 {
        self.inner.sustain_level
    }
    fn get_vibrato_rate_hz(&self) -> f32 {
        self.inner.vibrato_rate_hz
    }
    fn get_vibrato_depth(&self) -> f32 {
        self.inner.vibrato_depth
    }
    fn get_jitter(&self) -> f32 {
        self.inner.jitter
    }
    fn get_shimmer(&self) -> f32 {
        self.inner.shimmer
    }
    fn get_mfcc_1(&self) -> f32 {
        self.inner.mfcc_1
    }
    fn get_mfcc_2(&self) -> f32 {
        self.inner.mfcc_2
    }
    fn get_mfcc_3(&self) -> f32 {
        self.inner.mfcc_3
    }
    fn get_mfcc_4(&self) -> f32 {
        self.inner.mfcc_4
    }
    fn get_mfcc_5(&self) -> f32 {
        self.inner.mfcc_5
    }
    fn get_mfcc_6(&self) -> f32 {
        self.inner.mfcc_6
    }
    fn get_mfcc_7(&self) -> f32 {
        self.inner.mfcc_7
    }
    fn get_mfcc_8(&self) -> f32 {
        self.inner.mfcc_8
    }
    fn get_mfcc_9(&self) -> f32 {
        self.inner.mfcc_9
    }
    fn get_mfcc_10(&self) -> f32 {
        self.inner.mfcc_10
    }
    fn get_mfcc_11(&self) -> f32 {
        self.inner.mfcc_11
    }
    fn get_mfcc_12(&self) -> f32 {
        self.inner.mfcc_12
    }
    fn get_mfcc_13(&self) -> f32 {
        self.inner.mfcc_13
    }
    fn get_spectral_flux(&self) -> f32 {
        self.inner.spectral_flux
    }
    fn get_median_ici_ms(&self) -> f32 {
        self.inner.median_ici_ms
    }
    fn get_onset_rate_hz(&self) -> f32 {
        self.inner.onset_rate_hz
    }
    fn get_ici_coefficient_of_variation(&self) -> f32 {
        self.inner.ici_coefficient_of_variation
    }

    fn __repr__(&self) -> String {
        format!(
            "Vector30D(F0={}Hz, Dur={}ms, Range={}Hz, HNR={}dB)",
            self.inner.mean_f0_hz as i32,
            self.inner.duration_ms as i32,
            self.inner.f0_range_hz as i32,
            self.inner.harmonic_to_noise_ratio as i32
        )
    }
}

/// Python wrapper for NavigationEngine
#[cfg(feature = "python-bindings")]
#[pyclass(name = "NavigationEngine")]
pub struct PyNavigationEngine {
    inner: island_hopping::NavigationEngine,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyNavigationEngine {
    /// Create a new navigation engine
    #[new]
    fn new() -> Self {
        Self {
            inner: island_hopping::NavigationEngine::new(),
        }
    }

    /// Create with custom max warp distance
    #[staticmethod]
    fn with_max_warp(max_safe_warp: f32) -> Self {
        Self {
            inner: island_hopping::NavigationEngine::with_max_warp(max_safe_warp),
        }
    }

    /// Interpolate between two vectors (Bridge Builder - SAFE)
    fn interpolate(&self, start: &PyVector30D, end: &PyVector30D, alpha: f32) -> PyVector30D {
        PyVector30D {
            inner: self.inner.interpolate(&start.inner, &end.inner, alpha),
        }
    }

    /// Apply safety clamping to target
    fn clamp_to_safe_distance(
        &self,
        target: &PyVector30D,
        anchor: &PyVector30D,
        anchor_island: Option<String>,
    ) -> PyResult<PyNavigationWaypoint> {
        let waypoint = self
            .inner
            .clamp_to_safe_distance(&target.inner, &anchor.inner, anchor_island);
        Ok(PyNavigationWaypoint { inner: waypoint })
    }

    /// Add an island to the database
    fn add_island(&mut self, key: String, features: PyVector30D, species: String) {
        let island = island_hopping::AudioIsland {
            key,
            features: features.inner,
            audio: None,
            species,
            metadata: std::collections::HashMap::new(),
        };
        self.inner.database_mut().add_island(island);
    }

    /// Find nearest island to target vector
    fn find_nearest_island(&self, target: &PyVector30D) -> PyResult<Option<PyAudioIsland>> {
        if let Some(island) = self.inner.find_nearest_island(&target.inner) {
            Ok(Some(PyAudioIsland {
                key: island.key.clone(),
                features: PyVector30D { inner: island.features },
                species: island.species.clone(),
            }))
        } else {
            Ok(None)
        }
    }

    fn __repr__(&self) -> String {
        "NavigationEngine()".to_string()
    }
}

/// Python wrapper for NavigationWaypoint
#[cfg(feature = "python-bindings")]
#[pyclass(name = "NavigationWaypoint")]
#[derive(Clone)]
pub struct PyNavigationWaypoint {
    inner: island_hopping::NavigationWaypoint,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyNavigationWaypoint {
    /// Get the (possibly clamped) target vector
    fn get_target(&self) -> PyVector30D {
        PyVector30D {
            inner: self.inner.target,
        }
    }

    /// Get navigation mode
    fn get_mode(&self) -> String {
        match self.inner.mode {
            island_hopping::NavigationMode::Interpolation => "Interpolation".to_string(),
            island_hopping::NavigationMode::Extrapolation => "Extrapolation".to_string(),
            island_hopping::NavigationMode::ExtrapolationClamped => "ExtrapolationClamped".to_string(),
        }
    }

    /// Get anchor island key
    fn get_anchor_island(&self) -> Option<String> {
        self.inner.anchor_island.clone()
    }

    /// Get distance to anchor
    fn get_distance_to_anchor(&self) -> f32 {
        self.inner.distance_to_anchor
    }

    /// Check if clamping was applied
    fn was_clamped(&self) -> bool {
        self.inner.was_clamped
    }

    fn __repr__(&self) -> String {
        format!(
            "NavigationWaypoint(mode={}, clamped={}, distance={})",
            self.get_mode(),
            self.inner.was_clamped,
            self.inner.distance_to_anchor
        )
    }
}

/// Python wrapper for AudioIsland
#[cfg(feature = "python-bindings")]
#[pyclass(name = "AudioIsland")]
#[derive(Clone)]
pub struct PyAudioIsland {
    pub key: String,
    pub features: PyVector30D,
    pub species: String,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyAudioIsland {
    #[new]
    fn new(key: String, features: PyVector30D, species: String) -> Self {
        Self { key, features, species }
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioIsland(key={}, species={}, F0={}Hz)",
            self.key, self.species, self.features.inner.mean_f0_hz as i32
        )
    }
}

#[cfg(feature = "python-bindings")]
#[pymodule]
fn technical_architecture(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTechnicalArchitect>()?;
    m.add_class::<PyDynamicMicroharmonicSynthesizer>()?;
    m.add_class::<PyGranularConcatenativeSynthesizer>()?;
    m.add_class::<PySourceMetadata>()?; // For 17D delta-based synthesis
    m.add_class::<PySourceMetadataBuilder>()?; // For building partial metadata
                                               // Safety-critical components
    m.add_class::<PyOperationMode>()?;
    m.add_class::<PyPeerController>()?;
    m.add_class::<PyPeerControllerConfig>()?;
    // Thermal state
    m.add_class::<PyThermalState>()?;
    // Environmental monitoring classes
    m.add_class::<PySessionViability>()?;
    m.add_class::<PyRainIntensity>()?;
    m.add_class::<PyTemperatureClassification>()?;
    m.add_class::<PyEnvironmentalConditions>()?;
    m.add_class::<PyEnvironmentalMonitor>()?;
    m.add_class::<PyEnvironmentalMonitorConfig>()?;
    // Visual recording classes
    m.add_class::<PyVisualRecorder>()?;
    m.add_class::<PyVisualRecorderConfig>()?;
    m.add_class::<PyVisualMetadata>()?;
    m.add_class::<PyRecordingStatistics>()?;
    m.add_class::<PyAudioSyncEvent>()?;
    // Island Hopping Navigation classes (NEW)
    m.add_class::<PyVector30D>()?;
    m.add_class::<PyNavigationEngine>()?;
    m.add_class::<PyNavigationWaypoint>()?;
    m.add_class::<PyAudioIsland>()?;
    // Micro-dynamics extractor classes (NEW - 30D feature extraction for BEANS)
    m.add_class::<PyMicroDynamicsExtractor>()?;
    m.add_class::<PyMicroDynamicsFeatures>()?;
    // ZooVox 45D feature extractor and Acoustic Similarity Engine (NEW - BEANS benchmark integration)
    m.add_class::<PyZooVoxFeatureExtractor>()?;
    m.add_class::<PyAcousticSimilarityEngine>()?;
    Ok(())
}

// Re-export for use in other Rust modules
pub use TechArchConfig as Config;

impl TechArchConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| anyhow::anyhow!("Failed to parse TechArchConfig from JSON: {}", e))
    }
}
