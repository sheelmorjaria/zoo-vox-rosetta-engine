// =============================================================================
// Species Configuration Module - Species-Specific Adaptation Layer
// =============================================================================
//
// Provides species-specific configurations for the Zoo Vox Rosetta Engine.
// Each species has different encoding strategies, modalities, and required modules.

use serde::{Deserialize, Serialize};

/// Atomic granularity - defines which hierarchical level carries semantic meaning
///
/// Different species encode meaning at different levels of their vocal hierarchy:
/// - Zebra Finch: Meaning is in the MOTIF (song pattern, ~350ms)
/// - Egyptian Bat: Meaning is in the SYLLABLE (chirp type, ~32ms)
/// - Dolphin: Meaning is in the CONTOUR (whistle shape, ~500ms+)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AtomicGranularity {
    /// Motif-level is the carrier of meaning (songbirds)
    /// Motifs are sequences of syllables that form complete songs
    Motif,

    /// Syllable-level is the carrier of meaning (bats, many mammals)
    /// Syllables are discrete acoustic units with distinct meaning
    Syllable,

    /// Note-level is the carrier of meaning (some insects, simple calls)
    /// Individual notes have semantic content
    Note,

    /// Contour-level is the carrier of meaning (dolphins, whales)
    /// Frequency-modulated contours carry identity/context
    Contour,
}

impl std::fmt::Display for AtomicGranularity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtomicGranularity::Motif => write!(f, "Motif"),
            AtomicGranularity::Syllable => write!(f, "Syllable"),
            AtomicGranularity::Note => write!(f, "Note"),
            AtomicGranularity::Contour => write!(f, "Contour"),
        }
    }
}

/// Encoding strategy for context decoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncodingStrategy {
    /// Combinatorial syntax (zebra finch, orcas)
    Combinatorial,

    /// Quantitative encoding by count (meerkats)
    Quantitative,

    /// Coda-type encoding (sperm whales)
    CodaType,

    /// Frequency-modulated contours (dolphins)
    FrequencyModulated,

    /// Duration-mediated encoding (bats)
    DurationMediated,

    /// Phrase type selection (marmosets)
    PhraseType,

    /// Minimal encoding (macaques, giant otters)
    Minimal,
}

impl std::fmt::Display for EncodingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncodingStrategy::Combinatorial => write!(f, "Combinatorial"),
            EncodingStrategy::Quantitative => write!(f, "Quantitative"),
            EncodingStrategy::CodaType => write!(f, "Coda-Type"),
            EncodingStrategy::FrequencyModulated => write!(f, "Frequency-Modulated"),
            EncodingStrategy::DurationMediated => write!(f, "Duration-Mediated"),
            EncodingStrategy::PhraseType => write!(f, "Phrase-Type"),
            EncodingStrategy::Minimal => write!(f, "Minimal"),
        }
    }
}

/// Primary analysis modality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisModality {
    /// Temporal phrase analysis
    Temporal,

    /// Spectral frequency analysis
    Spectral,

    /// Hybrid temporal + spectral
    Hybrid,
}

impl std::fmt::Display for AnalysisModality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnalysisModality::Temporal => write!(f, "Temporal"),
            AnalysisModality::Spectral => write!(f, "Spectral"),
            AnalysisModality::Hybrid => write!(f, "Hybrid"),
        }
    }
}

/// Required analysis module
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnalysisModule {
    /// Temporal phrase analysis
    Temporal,

    /// Spectral frequency analysis
    Spectral,

    /// N-gram sequence analysis
    Sequence,

    /// Duration-based analysis
    Duration,

    /// Phrase count analysis
    Count,
}

/// Hierarchical thresholds for Motif → Syllable → Note segmentation
///
/// These thresholds should scale with species' typical tempo and sample rate.
/// The "tempo_factor" adjusts thresholds for species with faster/slower vocalization rates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalThresholds {
    /// Motif-level thresholds
    pub motif_min_ms: f32,
    pub motif_max_ms: f32,
    pub motif_change_threshold: f32,

    /// Syllable-level thresholds
    pub syllable_min_ms: f32,
    pub syllable_max_ms: f32,
    pub syllable_change_threshold: f32,

    /// Note-level thresholds
    pub note_min_ms: f32,
    pub note_max_ms: f32,
    pub note_change_threshold: f32,

    /// Tempo factor: 1.0 = bird tempo (22kHz reference)
    /// Bats (250kHz) have tempo_factor ~0.3 (events happen 3x faster)
    /// Dolphins (slow whistles) have tempo_factor ~2.0
    pub tempo_factor: f32,
}

impl HierarchicalThresholds {
    /// Create thresholds for songbirds (zebra finch)
    /// Motifs (~350ms) are the atomic unit of meaning
    pub fn zebra_finch() -> Self {
        Self {
            motif_min_ms: 50.0,
            motif_max_ms: 500.0,
            motif_change_threshold: 0.30,
            syllable_min_ms: 20.0,
            syllable_max_ms: 150.0,
            syllable_change_threshold: 0.21,
            note_min_ms: 10.0,
            note_max_ms: 50.0,
            note_change_threshold: 0.15,
            tempo_factor: 1.0,
        }
    }

    /// Create thresholds for Egyptian fruit bats
    /// Syllables (~32ms) are the atomic unit of meaning
    pub fn bat() -> Self {
        Self {
            // Bats: events happen ~3x faster due to ultrasonic sample rate
            motif_min_ms: 100.0, // Multi-syllable sequences
            motif_max_ms: 400.0,
            motif_change_threshold: 0.40,
            syllable_min_ms: 15.0, // ATOMIC LEVEL for bats
            syllable_max_ms: 100.0,
            syllable_change_threshold: 0.28,
            note_min_ms: 5.0, // FM sweep components
            note_max_ms: 40.0,
            note_change_threshold: 0.20,
            tempo_factor: 0.3,
        }
    }

    /// Create thresholds for dolphins
    /// Contours (~500ms+) are the atomic unit of meaning
    pub fn dolphin() -> Self {
        Self {
            motif_min_ms: 300.0,
            motif_max_ms: 3000.0,
            motif_change_threshold: 0.20,
            syllable_min_ms: 100.0,
            syllable_max_ms: 800.0,
            syllable_change_threshold: 0.14,
            note_min_ms: 50.0,
            note_max_ms: 300.0,
            note_change_threshold: 0.10,
            tempo_factor: 2.0,
        }
    }

    /// Create thresholds for marmosets
    pub fn marmoset() -> Self {
        Self {
            motif_min_ms: 80.0,
            motif_max_ms: 800.0,
            motif_change_threshold: 0.35,
            syllable_min_ms: 30.0,
            syllable_max_ms: 250.0,
            syllable_change_threshold: 0.25,
            note_min_ms: 15.0,
            note_max_ms: 80.0,
            note_change_threshold: 0.18,
            tempo_factor: 1.0,
        }
    }

    /// Default thresholds
    pub fn default_thresholds() -> Self {
        Self {
            motif_min_ms: 50.0,
            motif_max_ms: 500.0,
            motif_change_threshold: 0.25,
            syllable_min_ms: 20.0,
            syllable_max_ms: 200.0,
            syllable_change_threshold: 0.18,
            note_min_ms: 10.0,
            note_max_ms: 80.0,
            note_change_threshold: 0.12,
            tempo_factor: 1.0,
        }
    }
}

/// Feature extraction parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureParams {
    /// Minimum phrase duration in ms
    pub phrase_min_ms: f64,

    /// Maximum phrase duration in ms
    pub phrase_max_ms: f64,

    /// Similarity threshold for phrase typing
    pub similarity_threshold: f64,

    /// Feature dimension
    pub feature_dim: usize,
}

impl Default for FeatureParams {
    fn default() -> Self {
        Self {
            phrase_min_ms: 30.0,
            phrase_max_ms: 500.0,
            similarity_threshold: 0.75,
            feature_dim: 30,
        }
    }
}

/// Feature weights for the 45D feature vector
///
/// The 45D vector is organized as:
/// - D0-D4: Spectral (centroid, spread, skewness, kurtosis, flatness)
/// - D5-D9: Harmonic (f0, harmonics, inharmonicity)
/// - D10-D14: Temporal (rms, zcr, attack, decay, sustain)
/// - D15-D19: Modulation (am_rate, am_depth, fm_rate, fm_depth, fm_slope)
/// - D20-D24: Cepstral (c0-c4)
/// - D25-D29: Formant (f1, f2, f3, b1, b2)
/// - D30-D34: Micro-dynamics (onset_rate, median_ici, ici_cv, burst_rate, gap_rate)
/// - D35-D39: Psychoacoustic (loudness, sharpness, roughness, fluctuation, tonality)
/// - D40-D44: Temporal fine structure (acf_peak, acf_strength, sfm, periodicity, entropy)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureWeights {
    /// Weight for spectral features (D0-D4)
    pub spectral: f32,
    /// Weight for harmonic features (D5-D9)
    pub harmonic: f32,
    /// Weight for temporal features (D10-D14)
    pub temporal: f32,
    /// Weight for modulation features (D15-D19)
    pub modulation: f32,
    /// Weight for cepstral features (D20-D24)
    pub cepstral: f32,
    /// Weight for formant features (D25-D29)
    pub formant: f32,
    /// Weight for micro-dynamics (D30-D34)
    pub micro_dynamics: f32,
    /// Weight for psychoacoustic features (D35-D39)
    pub psychoacoustic: f32,
    /// Weight for temporal fine structure (D40-D44)
    pub tfs: f32,
    /// Individual feature overrides (index, weight)
    pub overrides: Vec<(usize, f32)>,
}

impl Default for FeatureWeights {
    fn default() -> Self {
        Self {
            spectral: 1.0,
            harmonic: 1.0,
            temporal: 1.0,
            modulation: 1.0,
            cepstral: 1.0,
            formant: 1.0,
            micro_dynamics: 1.0,
            psychoacoustic: 1.0,
            tfs: 1.0,
            overrides: Vec::new(),
        }
    }
}

impl FeatureWeights {
    /// Create weights optimized for dolphin whistles (FM contours)
    pub fn dolphin() -> Self {
        Self {
            spectral: 1.5, // Important for whistle frequency
            harmonic: 0.5, // Less harmonic content
            temporal: 1.0,
            modulation: 2.5, // FM slope is CRITICAL for whistles
            cepstral: 0.8,
            formant: 0.5,        // Not relevant for dolphins
            micro_dynamics: 0.5, // Long contours, less micro-structure
            psychoacoustic: 1.2, // Pitch perception important
            tfs: 0.8,
            overrides: vec![
                (18, 3.0), // D18: fm_slope - highest weight for whistle contours
                (17, 2.0), // D17: fm_rate - FM rate important
                (0, 1.8),  // D0: spectral_centroid - frequency tracking
                (12, 1.5), // D12: attack - onset of whistle
            ],
        }
    }

    /// Create weights optimized for macaque coos (fine spectral discrimination)
    pub fn macaque() -> Self {
        Self {
            spectral: 2.0, // Fine spectral discrimination needed
            harmonic: 1.5, // Harmonic structure in coos
            temporal: 1.2,
            modulation: 0.8,
            cepstral: 1.8, // Good for voice quality
            formant: 2.0,  // Formants carry information
            micro_dynamics: 1.0,
            psychoacoustic: 1.5, // Pitch perception
            tfs: 1.2,
            overrides: vec![
                (3, 2.0),  // D3: spectral_kurtosis - fine discrimination
                (4, 1.8),  // D4: spectral_tilt - voice quality
                (25, 1.8), // D25: f1 (first formant)
                (26, 1.8), // D26: f2 (second formant)
                (5, 1.5),  // D5: f0 - fundamental frequency
            ],
        }
    }

    /// Create weights optimized for Egyptian fruit bats (ultrasonic FM sweeps)
    pub fn bat() -> Self {
        Self {
            spectral: 1.8,   // High-frequency content
            harmonic: 1.2,   // Some harmonic structure
            temporal: 1.5,   // Timing critical for echolocation
            modulation: 2.0, // FM sweeps are key
            cepstral: 0.8,
            formant: 0.3,        // Not relevant for ultrasonic
            micro_dynamics: 2.0, // Rapid changes in echolocation calls
            psychoacoustic: 0.8,
            tfs: 1.5, // Fine timing for echolocation
            overrides: vec![
                (18, 2.5), // D18: fm_slope - FM sweep direction
                (13, 2.0), // D13: decay - call termination
                (14, 1.8), // D14: sustain - call duration
                (30, 1.8), // D30: onset_rate - rapid call sequences
                (33, 1.5), // D33: burst_rate - echolocation pattern
            ],
        }
    }

    /// Create weights optimized for zebra finch song (combinatorial syntax)
    pub fn zebra_finch() -> Self {
        Self {
            spectral: 1.2, // Moderate spectral importance
            harmonic: 1.8, // Harmonic stack structure in song
            temporal: 1.5, // Syllable timing important
            modulation: 1.0,
            cepstral: 1.2,
            formant: 0.8,
            micro_dynamics: 1.5, // Syllable transitions
            psychoacoustic: 1.0,
            tfs: 1.3, // Temporal fine structure for song
            overrides: vec![
                (5, 1.8),  // D5: f0 - pitch of syllables
                (10, 1.5), // D10: rms - amplitude envelope
                (12, 1.5), // D12: attack - syllable onset
                (31, 1.5), // D31: median_ici - inter-call interval
            ],
        }
    }

    /// Create weights optimized for marmoset phee calls (discrete types)
    pub fn marmoset() -> Self {
        Self {
            spectral: 1.5, // Spectral shape distinguishes call types
            harmonic: 1.3, // Harmonic structure in phee calls
            temporal: 1.2,
            modulation: 1.0,
            cepstral: 1.0,
            formant: 1.2, // Formants in primate vocalizations
            micro_dynamics: 1.0,
            psychoacoustic: 1.2,
            tfs: 1.0,
            overrides: vec![
                (0, 1.5),  // D0: spectral_centroid - frequency center
                (5, 1.3),  // D5: f0 - fundamental
                (25, 1.3), // D25: f1 - first formant
                (14, 1.2), // D14: sustain - call length
            ],
        }
    }

    /// Create weights optimized for sperm whale codas (rhythm patterns)
    pub fn sperm_whale() -> Self {
        Self {
            spectral: 0.8, // Less spectral variation
            harmonic: 0.5, // Clicks are broadband
            temporal: 2.5, // TIMING IS EVERYTHING for codas
            modulation: 0.5,
            cepstral: 0.8,
            formant: 0.3,
            micro_dynamics: 2.0, // Click patterns
            psychoacoustic: 0.8,
            tfs: 1.8, // Temporal structure critical
            overrides: vec![
                (30, 2.5), // D30: onset_rate - click rate
                (31, 2.5), // D31: median_ici - inter-click interval
                (32, 2.0), // D32: ici_cv - rhythm variation
                (43, 2.0), // D43: periodicity - rhythmic pattern
                (44, 1.8), // D44: entropy - pattern regularity
            ],
        }
    }

    /// Create weights optimized for orca calls (hybrid tonal/pulsed)
    pub fn orca() -> Self {
        Self {
            spectral: 1.5, // Both tonal and pulsed calls
            harmonic: 1.2,
            temporal: 1.5,   // Call timing
            modulation: 1.8, // FM in tonal calls
            cepstral: 1.0,
            formant: 0.8,
            micro_dynamics: 1.5, // Pulsed call patterns
            psychoacoustic: 1.2,
            tfs: 1.3,
            overrides: vec![
                (18, 2.0), // D18: fm_slope - FM in tonal calls
                (0, 1.5),  // D0: spectral_centroid
                (30, 1.5), // D30: onset_rate - pulsed patterns
            ],
        }
    }

    /// Create weights optimized for meerkat alarm calls (quantitative)
    pub fn meerkat() -> Self {
        Self {
            spectral: 1.3,
            harmonic: 1.0,
            temporal: 2.0, // Call count and timing
            modulation: 1.0,
            cepstral: 0.8,
            formant: 1.0,
            micro_dynamics: 1.5,
            psychoacoustic: 1.0,
            tfs: 1.2,
            overrides: vec![
                (10, 1.8), // D10: rms - call amplitude
                (30, 1.8), // D30: onset_rate - call rate
                (12, 1.5), // D12: attack - urgency
            ],
        }
    }

    /// Get the weight for a specific feature index
    pub fn get_weight(&self, index: usize) -> f32 {
        // Check overrides first
        for (override_idx, weight) in &self.overrides {
            if *override_idx == index {
                return *weight;
            }
        }

        // Otherwise use group weight
        match index {
            0..=4 => self.spectral,
            5..=9 => self.harmonic,
            10..=14 => self.temporal,
            15..=19 => self.modulation,
            20..=24 => self.cepstral,
            25..=29 => self.formant,
            30..=34 => self.micro_dynamics,
            35..=39 => self.psychoacoustic,
            40..=44 => self.tfs,
            _ => 1.0,
        }
    }

    /// Convert to a full 45D weight vector
    pub fn to_weight_vector(&self) -> Vec<f32> {
        (0..45).map(|i| self.get_weight(i)).collect()
    }

    /// Convert to a 30D weight vector (first 6 feature groups)
    ///
    /// Used for WithinCallAnalyzer which operates on 30D features:
    /// - D0-D4: Spectral
    /// - D5-D9: Harmonic
    /// - D10-D14: Temporal
    /// - D15-D19: Modulation
    /// - D20-D24: Cepstral
    /// - D25-D29: Formant
    pub fn to_weight_vector_30d(&self) -> Vec<f32> {
        (0..30).map(|i| self.get_weight(i)).collect()
    }
}

/// Decoding method for context prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecodingMethod {
    /// Context = which phrase type selected
    PhraseTypeSelection,

    /// Context = number of phrases
    PhraseCount,

    /// Context = duration threshold
    DurationThreshold,

    /// Context = frequency contour shape
    ContourShape,

    /// Context = phrase sequence pattern
    SequencePattern,
}

/// Context decoding rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextRules {
    /// Decoding method
    pub decoding_method: DecodingMethod,

    /// Context labels
    pub context_labels: Vec<String>,
}

impl Default for ContextRules {
    fn default() -> Self {
        Self {
            decoding_method: DecodingMethod::PhraseTypeSelection,
            context_labels: Vec::new(),
        }
    }
}

/// Species-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesConfig {
    /// Species name
    species: String,

    /// Encoding strategy
    encoding_strategy: EncodingStrategy,

    /// Primary modality
    modality: AnalysisModality,

    /// Required analysis modules
    required_modules: Vec<AnalysisModule>,

    /// Feature extraction parameters
    feature_params: FeatureParams,

    /// Context decoding rules
    context_rules: ContextRules,

    /// Which hierarchical level carries semantic meaning
    atomic_granularity: AtomicGranularity,

    /// Hierarchical thresholds for segmentation
    hierarchical_thresholds: HierarchicalThresholds,

    /// Feature weights for 45D vector similarity
    feature_weights: FeatureWeights,
}

impl SpeciesConfig {
    /// Get species name
    pub fn species(&self) -> &str {
        &self.species
    }

    /// Get encoding strategy
    pub fn encoding_strategy(&self) -> EncodingStrategy {
        self.encoding_strategy
    }

    /// Get modality
    pub fn modality(&self) -> AnalysisModality {
        self.modality
    }

    /// Get required modules
    pub fn required_modules(&self) -> &[AnalysisModule] {
        &self.required_modules
    }

    /// Get feature parameters
    pub fn feature_params(&self) -> &FeatureParams {
        &self.feature_params
    }

    /// Get context labels
    pub fn context_labels(&self) -> &[String] {
        &self.context_rules.context_labels
    }

    /// Get decoding method
    pub fn decoding_method(&self) -> &DecodingMethod {
        &self.context_rules.decoding_method
    }

    /// Get atomic granularity (which level carries meaning)
    pub fn atomic_granularity(&self) -> AtomicGranularity {
        self.atomic_granularity
    }

    /// Get hierarchical thresholds
    pub fn hierarchical_thresholds(&self) -> &HierarchicalThresholds {
        &self.hierarchical_thresholds
    }

    /// Check if a module is required
    pub fn requires_module(&self, module: AnalysisModule) -> bool {
        self.required_modules.contains(&module)
    }

    /// Get feature weights
    pub fn feature_weights(&self) -> &FeatureWeights {
        &self.feature_weights
    }
}

/// Factory for creating species-specific configurations
pub struct SpeciesConfigFactory;

impl SpeciesConfigFactory {
    /// Create configuration for a species
    pub fn create(species: &str) -> SpeciesConfig {
        match species.to_lowercase().as_str() {
            "sperm_whale" | "dominica" | "spermwhale" => Self::sperm_whale_config(),
            "meerkat" | "meerkats" => Self::meerkat_config(),
            "zebra_finch" | "zebrafinch" | "finch" => Self::zebra_finch_config(),
            "dolphin" | "dolphins" | "whistle_signals" | "bottlenose" => Self::dolphin_config(),
            "bat" | "egyptian_bat" | "egyptianbat" | "fruit_bat" => Self::bat_config(),
            "orca" | "orcas" | "killer_whale" => Self::orca_config(),
            "marmoset" | "marmosets" | "common_marmoset" => Self::marmoset_config(),
            "macaque" | "macaques" => Self::macaque_config(),
            "giant_otter" | "giantotter" | "otter" => Self::giant_otter_config(),
            _ => Self::default_config(),
        }
    }

    /// Sperm whale configuration
    fn sperm_whale_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Sperm Whale".to_string(),
            encoding_strategy: EncodingStrategy::CodaType,
            modality: AnalysisModality::Temporal,
            required_modules: vec![AnalysisModule::Temporal],
            feature_params: FeatureParams {
                phrase_min_ms: 10.0,
                phrase_max_ms: 100.0,
                similarity_threshold: 0.80,
                feature_dim: 30,
            },
            context_rules: ContextRules {
                decoding_method: DecodingMethod::PhraseTypeSelection,
                context_labels: vec![
                    "foraging".to_string(),
                    "social".to_string(),
                    "communication".to_string(),
                ],
            },
            atomic_granularity: AtomicGranularity::Contour,
            hierarchical_thresholds: HierarchicalThresholds::dolphin(), // Similar to dolphins
            feature_weights: FeatureWeights::sperm_whale(),
        }
    }

    /// Meerkat configuration
    fn meerkat_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Meerkat".to_string(),
            encoding_strategy: EncodingStrategy::Quantitative,
            modality: AnalysisModality::Temporal,
            required_modules: vec![AnalysisModule::Temporal, AnalysisModule::Count],
            feature_params: FeatureParams {
                phrase_min_ms: 30.0,
                phrase_max_ms: 500.0,
                similarity_threshold: 0.75,
                feature_dim: 30,
            },
            context_rules: ContextRules {
                decoding_method: DecodingMethod::PhraseCount,
                context_labels: vec![
                    "close_call".to_string(),
                    "alarm".to_string(),
                    "social".to_string(),
                    "sentinel".to_string(),
                ],
            },
            atomic_granularity: AtomicGranularity::Syllable,
            hierarchical_thresholds: HierarchicalThresholds::default_thresholds(),
            feature_weights: FeatureWeights::meerkat(),
        }
    }

    /// Zebra finch configuration
    fn zebra_finch_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Zebra Finch".to_string(),
            encoding_strategy: EncodingStrategy::Combinatorial,
            modality: AnalysisModality::Temporal,
            required_modules: vec![AnalysisModule::Temporal, AnalysisModule::Sequence],
            feature_params: FeatureParams {
                phrase_min_ms: 20.0,
                phrase_max_ms: 200.0,
                similarity_threshold: 0.75,
                feature_dim: 30,
            },
            context_rules: ContextRules {
                decoding_method: DecodingMethod::SequencePattern,
                context_labels: vec!["directed".to_string(), "undirected".to_string()],
            },
            atomic_granularity: AtomicGranularity::Motif, // MOTIFS carry meaning for songbirds
            hierarchical_thresholds: HierarchicalThresholds::zebra_finch(),
            feature_weights: FeatureWeights::zebra_finch(),
        }
    }

    /// Dolphin configuration
    fn dolphin_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Dolphin".to_string(),
            encoding_strategy: EncodingStrategy::FrequencyModulated,
            modality: AnalysisModality::Spectral,
            required_modules: vec![AnalysisModule::Spectral],
            feature_params: FeatureParams {
                phrase_min_ms: 500.0,
                phrase_max_ms: 2000.0,
                similarity_threshold: 0.70,
                feature_dim: 56,
            },
            context_rules: ContextRules {
                decoding_method: DecodingMethod::ContourShape,
                context_labels: vec![
                    "signature".to_string(),
                    "social".to_string(),
                    "food".to_string(),
                    "alarm".to_string(),
                ],
            },
            atomic_granularity: AtomicGranularity::Contour,
            hierarchical_thresholds: HierarchicalThresholds::dolphin(),
            feature_weights: FeatureWeights::dolphin(),
        }
    }

    /// Egyptian fruit bat configuration
    fn bat_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Egyptian Fruit Bat".to_string(),
            encoding_strategy: EncodingStrategy::DurationMediated,
            modality: AnalysisModality::Temporal,
            required_modules: vec![AnalysisModule::Temporal, AnalysisModule::Duration],
            feature_params: FeatureParams {
                phrase_min_ms: 30.0,
                phrase_max_ms: 500.0,
                similarity_threshold: 0.75,
                feature_dim: 30,
            },
            context_rules: ContextRules {
                decoding_method: DecodingMethod::DurationThreshold,
                context_labels: vec!["feeding".to_string(), "mating".to_string(), "landing".to_string()],
            },
            atomic_granularity: AtomicGranularity::Syllable, // SYLLABLES carry meaning for bats
            hierarchical_thresholds: HierarchicalThresholds::bat(),
            feature_weights: FeatureWeights::bat(),
        }
    }

    /// Orca configuration
    fn orca_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Orca".to_string(),
            encoding_strategy: EncodingStrategy::Combinatorial,
            modality: AnalysisModality::Hybrid,
            required_modules: vec![
                AnalysisModule::Temporal,
                AnalysisModule::Sequence,
                AnalysisModule::Spectral,
            ],
            feature_params: FeatureParams {
                phrase_min_ms: 50.0,
                phrase_max_ms: 1000.0,
                similarity_threshold: 0.75,
                feature_dim: 56,
            },
            context_rules: ContextRules {
                decoding_method: DecodingMethod::SequencePattern,
                context_labels: vec!["hunting".to_string(), "social".to_string(), "travel".to_string()],
            },
            atomic_granularity: AtomicGranularity::Contour,
            hierarchical_thresholds: HierarchicalThresholds::dolphin(),
            feature_weights: FeatureWeights::orca(),
        }
    }

    /// Marmoset configuration
    fn marmoset_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Common Marmoset".to_string(),
            encoding_strategy: EncodingStrategy::PhraseType,
            modality: AnalysisModality::Temporal,
            required_modules: vec![AnalysisModule::Temporal],
            feature_params: FeatureParams {
                phrase_min_ms: 50.0,
                phrase_max_ms: 500.0,
                similarity_threshold: 0.75,
                feature_dim: 30,
            },
            context_rules: ContextRules {
                decoding_method: DecodingMethod::PhraseTypeSelection,
                context_labels: vec!["phee".to_string(), "tsik".to_string(), "trill".to_string()],
            },
            atomic_granularity: AtomicGranularity::Syllable,
            hierarchical_thresholds: HierarchicalThresholds::marmoset(),
            feature_weights: FeatureWeights::marmoset(),
        }
    }

    /// Macaque configuration
    fn macaque_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Macaque".to_string(),
            encoding_strategy: EncodingStrategy::Minimal,
            modality: AnalysisModality::Temporal,
            required_modules: vec![AnalysisModule::Temporal, AnalysisModule::Spectral],
            feature_params: FeatureParams {
                phrase_min_ms: 100.0,
                phrase_max_ms: 500.0,
                similarity_threshold: 0.90,
                feature_dim: 56,
            },
            context_rules: ContextRules {
                decoding_method: DecodingMethod::PhraseTypeSelection,
                context_labels: vec!["coo".to_string()],
            },
            atomic_granularity: AtomicGranularity::Syllable,
            hierarchical_thresholds: HierarchicalThresholds::default_thresholds(),
            feature_weights: FeatureWeights::macaque(),
        }
    }

    /// Giant otter configuration
    fn giant_otter_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Giant Otter".to_string(),
            encoding_strategy: EncodingStrategy::Minimal,
            modality: AnalysisModality::Temporal,
            required_modules: vec![AnalysisModule::Temporal, AnalysisModule::Spectral],
            feature_params: FeatureParams {
                phrase_min_ms: 50.0,
                phrase_max_ms: 500.0,
                similarity_threshold: 0.80,
                feature_dim: 56,
            },
            context_rules: ContextRules {
                decoding_method: DecodingMethod::PhraseTypeSelection,
                context_labels: vec!["contact".to_string(), "alarm".to_string()],
            },
            atomic_granularity: AtomicGranularity::Syllable,
            hierarchical_thresholds: HierarchicalThresholds::default_thresholds(),
            feature_weights: FeatureWeights::default(),
        }
    }

    /// Default configuration for unknown species
    fn default_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Unknown".to_string(),
            encoding_strategy: EncodingStrategy::PhraseType,
            modality: AnalysisModality::Temporal,
            required_modules: vec![AnalysisModule::Temporal],
            feature_params: FeatureParams::default(),
            context_rules: ContextRules::default(),
            atomic_granularity: AtomicGranularity::Syllable,
            hierarchical_thresholds: HierarchicalThresholds::default_thresholds(),
            feature_weights: FeatureWeights::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sperm_whale_config() {
        let config = SpeciesConfigFactory::create("sperm_whale");

        assert_eq!(config.species(), "Sperm Whale");
        assert_eq!(config.encoding_strategy(), EncodingStrategy::CodaType);
        assert_eq!(config.modality(), AnalysisModality::Temporal);
        assert!(config.requires_module(AnalysisModule::Temporal));
    }

    #[test]
    fn test_dolphin_config() {
        let config = SpeciesConfigFactory::create("dolphin");

        assert_eq!(config.species(), "Dolphin");
        assert_eq!(config.encoding_strategy(), EncodingStrategy::FrequencyModulated);
        assert_eq!(config.modality(), AnalysisModality::Spectral);
        assert!(config.requires_module(AnalysisModule::Spectral));
    }

    #[test]
    fn test_zebra_finch_config() {
        let config = SpeciesConfigFactory::create("zebra_finch");

        assert_eq!(config.species(), "Zebra Finch");
        assert_eq!(config.encoding_strategy(), EncodingStrategy::Combinatorial);
        assert!(config.requires_module(AnalysisModule::Sequence));
    }

    #[test]
    fn test_case_insensitive() {
        let config1 = SpeciesConfigFactory::create("SPERM_WHALE");
        let config2 = SpeciesConfigFactory::create("sperm_whale");

        assert_eq!(config1.species(), config2.species());
    }

    #[test]
    fn test_unknown_species_defaults() {
        let config = SpeciesConfigFactory::create("unknown_species_xyz");

        assert_eq!(config.species(), "Unknown");
        assert_eq!(config.encoding_strategy(), EncodingStrategy::PhraseType);
    }

    #[test]
    fn test_dolphin_feature_weights() {
        let config = SpeciesConfigFactory::create("dolphin");
        let weights = config.feature_weights();

        // FM slope (D18) should have highest weight for dolphins
        assert!(weights.get_weight(18) > 2.0);
        // Modulation group should be elevated
        assert!(weights.modulation > 2.0);
    }

    #[test]
    fn test_macaque_feature_weights() {
        let config = SpeciesConfigFactory::create("macaque");
        let weights = config.feature_weights();

        // Spectral kurtosis (D3) should be elevated for fine discrimination
        assert!(weights.get_weight(3) > 1.5);
        // Formants should be important for macaques
        assert!(weights.formant > 1.5);
    }

    #[test]
    fn test_bat_feature_weights() {
        let config = SpeciesConfigFactory::create("bat");
        let weights = config.feature_weights();

        // FM slope (D18) should be elevated for FM sweeps
        assert!(weights.get_weight(18) > 2.0);
        // Micro-dynamics should be elevated for rapid echolocation
        assert!(weights.micro_dynamics > 1.5);
    }

    #[test]
    fn test_sperm_whale_feature_weights() {
        let config = SpeciesConfigFactory::create("sperm_whale");
        let weights = config.feature_weights();

        // Temporal features should dominate for rhythm-based codas
        assert!(weights.temporal > 2.0);
        // ICI (D31) should be critical for coda patterns
        assert!(weights.get_weight(31) > 2.0);
    }

    #[test]
    fn test_feature_weight_vector() {
        let weights = FeatureWeights::dolphin();
        let vector = weights.to_weight_vector();

        assert_eq!(vector.len(), 45);
        // FM slope (index 18) should have highest weight
        assert!(vector[18] > vector[0]); // fm_slope > spectral centroid
    }

    #[test]
    fn test_feature_weight_overrides() {
        let weights = FeatureWeights::dolphin();

        // Check that overrides take precedence
        // D18 is in modulation group (15-19), but override should apply
        let override_weight = weights.get_weight(18);
        let group_weight = weights.modulation;

        // Override should be different from group weight
        assert_ne!(override_weight, group_weight);
        assert_eq!(override_weight, 3.0); // Explicit override value
    }
}
