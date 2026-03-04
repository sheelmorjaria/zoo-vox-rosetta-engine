//! Taxonomic-Aware Weight Routing for Hybrid Expert Architecture
//! =============================================================
//!
//! Implements biological priors by weighting features based on taxonomic class.
//! This "Biologically-Guided Attention" improves benchmark performance by
//! emphasizing features known to have high discriminatory power for specific clades.
//!
//! # Feature Stack Architecture (112D)
//! - Layer 1 (0-45): Base Physics - Duration, F0, Resonance, Spectral
//! - Layer 2 (46-75): Macro Texture - Harmonic Density, GLCM Roughness
//! - Layer 3 (76-111): Micro Texture - FM Bins, ICI Bins, Dynamics, Rhythm
//!
//! # Divide and Conquer Architecture
//! - Gatekeeper Input (76D): Base Physics (46D) + Macro Texture (30D)
//! - Species Expert Input (82D): Base Physics (46D) + Micro Texture (36D)
//!
//! # Usage
//! ```rust
//! use technical_architecture::taxonomic_router::{Taxon, get_taxonomic_weights, apply_taxonomic_mask};
//!
//! let features = vec![1.0; 112];
//! let weights = get_taxonomic_weights(Taxon::Cetacean);
//! let masked = apply_taxonomic_mask(&features, Taxon::Cetacean);
//! ```

use serde::{Deserialize, Serialize};

/// High-level taxonomic groups in bioacoustic benchmarks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Taxon {
    /// Toothed whales (dolphins, porpoises) - clicks and whistles
    Cetacean,
    /// Baleen whales (humpback, blue) - songs and moans
    Mysticete,
    /// Songbirds (passerines) - complex syntax
    Songbird,
    /// Non-passerine birds (parrots, owls) - simple calls
    NonPasserine,
    /// Frogs and toads - pulse trains and trills
    Amphibian,
    /// Seals and sea lions - grunts and barks
    Pinniped,
    /// Insects - rigid tempo patterns
    Insect,
    /// General mammals (bats, primates) - FM sweeps and formants
    Mammal,
    /// Unknown or unclassified
    Unknown,
}

/// Consolidated taxonomic groups for Gatekeeper RF (6 classes)
///
/// Groups rare marine classes together to improve RF accuracy:
/// - Marine_Mammal: Cetacean + Mysticete + Pinniped (all marine)
/// - Bird: Songbird + NonPasserine (all birds)
/// - Mammal: Terrestrial mammals (bats, primates)
/// - Insect, Amphibian, Unknown: kept as-is
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsolidatedTaxon {
    /// All birds (Songbirds + Non-Passerines)
    Bird,
    /// Terrestrial mammals (bats, primates)
    Mammal,
    /// Marine mammals (Cetaceans + Mysticetes + Pinnipeds)
    MarineMammal,
    /// Insects (crickets, mosquitoes, cicadas)
    Insect,
    /// Amphibians (frogs, toads)
    Amphibian,
    /// Unknown or unclassified
    Unknown,
}

/// Number of consolidated taxonomic classes
pub const CONSOLIDATED_TAXON_COUNT: usize = 6;

/// Maps detailed Taxon to ConsolidatedTaxon for Gatekeeper RF
pub fn consolidate_taxon(taxon: Taxon) -> ConsolidatedTaxon {
    match taxon {
        Taxon::Songbird | Taxon::NonPasserine => ConsolidatedTaxon::Bird,
        Taxon::Cetacean | Taxon::Mysticete | Taxon::Pinniped => ConsolidatedTaxon::MarineMammal,
        Taxon::Mammal => ConsolidatedTaxon::Mammal,
        Taxon::Insect => ConsolidatedTaxon::Insect,
        Taxon::Amphibian => ConsolidatedTaxon::Amphibian,
        Taxon::Unknown => ConsolidatedTaxon::Unknown,
    }
}

/// Maps ConsolidatedTaxon to index (for RF training/inference)
pub fn consolidated_taxon_to_idx(taxon: ConsolidatedTaxon) -> usize {
    match taxon {
        ConsolidatedTaxon::Bird => 0,
        ConsolidatedTaxon::Mammal => 1,
        ConsolidatedTaxon::MarineMammal => 2,
        ConsolidatedTaxon::Insect => 3,
        ConsolidatedTaxon::Amphibian => 4,
        ConsolidatedTaxon::Unknown => 5,
    }
}

/// Maps index to ConsolidatedTaxon
pub fn idx_to_consolidated_taxon(idx: usize) -> ConsolidatedTaxon {
    match idx {
        0 => ConsolidatedTaxon::Bird,
        1 => ConsolidatedTaxon::Mammal,
        2 => ConsolidatedTaxon::MarineMammal,
        3 => ConsolidatedTaxon::Insect,
        4 => ConsolidatedTaxon::Amphibian,
        _ => ConsolidatedTaxon::Unknown,
    }
}

/// Get label names for consolidated taxonomic classes
pub fn consolidated_taxon_labels() -> Vec<String> {
    vec![
        "Bird".to_string(),
        "Mammal".to_string(),
        "MarineMammal".to_string(),
        "Insect".to_string(),
        "Amphibian".to_string(),
        "Unknown".to_string(),
    ]
}

/// Feature dimension constants
pub const FEATURE_DIM: usize = 112;
pub const PHYSICS_DIM: usize = 46;  // Layer 1: indices 0-45
pub const TEXTURE_DIM: usize = 66;  // Layers 2-3: indices 46-111
pub const MACRO_TEXTURE_DIM: usize = 30;  // Layer 2: indices 46-75
pub const MICRO_TEXTURE_DIM: usize = 36;  // Layer 3: indices 76-111

/// Gatekeeper input dimension (Base + Macro = 46 + 30 = 76D)
pub const GATEKEEPER_DIM: usize = PHYSICS_DIM + MACRO_TEXTURE_DIM;
/// Species Expert input dimension (Base + Micro = 46 + 36 = 82D)
pub const SPECIES_EXPERT_DIM: usize = PHYSICS_DIM + MICRO_TEXTURE_DIM;

/// Feature index ranges based on 112D Stack Architecture
pub mod feature_indices {
    // Layer 1: Base Physics (0-45)
    pub const F0: usize = 0;
    pub const DURATION: usize = 1;
    pub const RMS_ENERGY: usize = 2;
    pub const ZERO_CROSSING_RATE: usize = 3;
    pub const HARMONICITY: usize = 4;
    pub const ATTACK_TIME: usize = 6;
    pub const SPECTRAL_CENTROID: usize = 28;

    // Layer 2: Macro Texture (46-75)
    pub const HARMONIC_TEXTURE_START: usize = 46;
    pub const HARMONIC_TEXTURE_END: usize = 54;
    pub const PITCH_GEOMETRY_START: usize = 54;
    pub const PITCH_GEOMETRY_END: usize = 61;
    pub const GLCM_START: usize = 61;
    pub const GLCM_END: usize = 71;

    // Layer 3: Micro Texture (76-111)
    pub const SPECTRAL_DERIVATIVE_START: usize = 76;
    pub const SPECTRAL_DERIVATIVE_END: usize = 81;
    pub const FM_BINS_START: usize = 81;
    pub const FM_BINS_END: usize = 86;
    pub const DYNAMICS_BINS_START: usize = 86;
    pub const DYNAMICS_BINS_END: usize = 91;
    pub const ICI_BINS_START: usize = 94;
    pub const ICI_BINS_END: usize = 99;
    pub const RHYTHM_START: usize = 96;
    pub const RHYTHM_END: usize = 106;
}

// =============================================================================
// Taxonomic Weight Functions
// =============================================================================

/// Generates a 112D weight vector based on the taxonomic strategy.
///
/// Returns a vector of weights where:
/// - 1.0 = no change (baseline)
/// - >1.0 = emphasize this feature
/// - <1.0 = suppress this feature
pub fn get_taxonomic_weights(taxon: Taxon) -> Vec<f32> {
    let mut weights = vec![1.0; FEATURE_DIM];

    use feature_indices::*;

    match taxon {
        // Odontocetes (Dolphins/Toothed Whales): ICI, FM Slope, Centroid
        Taxon::Cetacean => {
            // ICI is critical for click classification
            for i in ICI_BINS_START..ICI_BINS_END {
                weights[i] = 3.0;
            }
            // FM slope for whistles
            for i in FM_BINS_START..FM_BINS_END {
                weights[i] = 2.5;
            }
            // Spectral centroid for click characterization
            weights[SPECTRAL_CENTROID] = 2.0;
        }

        // Baleen Whales (Humpback/Blue): Duration, Harmonics, Low F0
        Taxon::Mysticete => {
            // Long duration is distinctive
            weights[DURATION] = 3.0;
            // Rich harmonic structure in songs
            for i in HARMONIC_TEXTURE_START..HARMONIC_TEXTURE_END {
                weights[i] = 2.5;
            }
            // Low fundamental frequency
            weights[F0] = 2.0;
            // Pitch geometry for melodic patterns
            for i in PITCH_GEOMETRY_START..PITCH_GEOMETRY_END {
                weights[i] = 1.8;
            }
        }

        // Songbirds (Passerines): F0, Harmonics, Spectral complexity
        Taxon::Songbird => {
            // Fundamental frequency precision
            weights[F0] = 1.8;
            // Harmonic structure
            weights[HARMONICITY] = 1.5;
            // Spectral region (MFCCs/Formants)
            for i in 14..28 {
                weights[i] = 1.5;
            }
            // Pitch geometry for melodic patterns
            for i in PITCH_GEOMETRY_START..PITCH_GEOMETRY_END {
                weights[i] = 1.5;
            }
        }

        // Non-Passerine Birds (Parrots/Owls): Formants, Attack
        Taxon::NonPasserine => {
            // Spectral shape for squawks/hoots
            for i in SPECTRAL_CENTROID..GLCM_START {
                weights[i] = 2.5;
            }
            // Sharp onsets
            weights[ATTACK_TIME] = 1.8;
        }

        // Anurans (Frogs/Toads): AM Pulse, Tempo, ICI
        Taxon::Amphibian => {
            // Amplitude modulation (pulse trains)
            for i in DYNAMICS_BINS_START..DYNAMICS_BINS_END {
                weights[i] = 3.0;
            }
            // Tempo/rhythm for trills
            for i in RHYTHM_START..RHYTHM_END {
                weights[i] = 2.5;
            }
            // Inter-call interval
            for i in ICI_BINS_START..ICI_BINS_END {
                weights[i] = 2.0;
            }
            weights[F0] = 2.0;
        }

        // Pinnipeds (Seals/Sea Lions): Roughness, Rhythm, Spectral Tilt
        Taxon::Pinniped => {
            // GLCM texture for roughness
            for i in GLCM_START..GLCM_END {
                weights[i] = 2.5;
            }
            // Pulse trains in knocking sounds
            for i in RHYTHM_START..RHYTHM_END {
                weights[i] = 2.0;
            }
            // Spectral tilt for broadband barks
            for i in 35..45 {
                weights[i] = 1.8;
            }
        }

        // Insects: Tempo, Centroid
        Taxon::Insect => {
            // Rigid tempo patterns
            for i in RHYTHM_START..RHYTHM_END {
                weights[i] = 3.5;
            }
            // Spectral centroid for buzzy sounds
            weights[SPECTRAL_CENTROID] = 2.5;
            // AM patterns
            for i in DYNAMICS_BINS_START..DYNAMICS_BINS_END {
                weights[i] = 2.0;
            }
        }

        // General Mammals (Bats, Primates): Formants, Spectral Tilt, FM
        Taxon::Mammal => {
            // Formant structure
            for i in 28..45 {
                weights[i] = 2.0;
            }
            // FM sweeps (bats)
            for i in FM_BINS_START..FM_BINS_END {
                weights[i] = 2.5;
            }
            // Pitch modulation
            weights[F0] = 1.5;
        }

        // Unknown: Keep uniform weights
        Taxon::Unknown => {}
    }

    weights
}

/// Applies the taxonomic mask to the feature vector.
///
/// Element-wise multiplication of features by taxonomic weights.
pub fn apply_taxonomic_mask(features: &[f32], taxon: Taxon) -> Vec<f32> {
    let weights = get_taxonomic_weights(taxon);
    features.iter()
        .zip(weights.iter())
        .map(|(f, w)| f * w)
        .collect()
}

// =============================================================================
// Feature Slicing Functions
// =============================================================================

/// Extracts the Physics subvector (Layer 1, 46D) from 112D features.
pub fn slice_physics(features: &[f32]) -> Vec<f32> {
    features[0..PHYSICS_DIM].to_vec()
}

/// Extracts the Texture subvector (Layers 2-3, 66D) from 112D features.
pub fn slice_texture(features: &[f32]) -> Vec<f32> {
    features[PHYSICS_DIM..FEATURE_DIM].to_vec()
}

/// Slice Macro Texture Features (30D) - Layer 2 only (without Base)
pub fn slice_macro_texture_only(features: &[f32]) -> Vec<f32> {
    features[46..76].to_vec()
}

/// Slice Micro Texture Features (36D) - Layer 3 only (without Base)
pub fn slice_micro_texture_only(features: &[f32]) -> Vec<f32> {
    features[76..112].to_vec()
}

// =============================================================================
// Divide and Conquer Feature Slicing
// =============================================================================

/// Slice Gatekeeper Input (76D): Base Physics (46D) + Macro Texture (30D)
/// Used by the Random Forest for taxonomic classification
pub fn slice_gatekeeper_input(features: &[f32]) -> Vec<f32> {
    assert!(features.len() == FEATURE_DIM, "Invalid feature dimension: expected {}, got {}", FEATURE_DIM, features.len());
    let mut result = vec![0.0f32; GATEKEEPER_DIM];
    // Base Physics (46D)
    result[..PHYSICS_DIM].copy_from_slice(&features[..PHYSICS_DIM]);
    // Macro Texture (30D)
    result[PHYSICS_DIM..GATEKEEPER_DIM].copy_from_slice(&features[46..76]);
    result
}

/// Slice Species Expert Input (82D): Base Physics (46D) + Micro Texture (36D)
/// Used by the Neural Network for species classification
pub fn slice_species_expert_input(features: &[f32]) -> Vec<f32> {
    assert!(features.len() == FEATURE_DIM, "Invalid feature dimension");
    let mut result = vec![0.0f32; SPECIES_EXPERT_DIM];
    // Base Physics (46D)
    result[..PHYSICS_DIM].copy_from_slice(&features[..PHYSICS_DIM]);
    // Micro Texture (36D)
    result[PHYSICS_DIM..SPECIES_EXPERT_DIM].copy_from_slice(&features[76..112]);
    result
}

/// Get feature slice bounds for documentation
pub fn get_feature_slices() -> (String, String, String) {
    (
        "Base Physics (46D): Duration, Harmonics, F0, Pitch".to_string(),
        "Macro Texture (30D): Harmonic, Spectral, Rhythm, Dynamics".to_string(),
        "Micro Texture (36D): FM, ICI, Pitch Modulation, Jitter/Shimmer".to_string(),
    )
}

// =============================================================================
// Species/Task to Taxon Mapping
// =============================================================================

/// Maps a species name to a taxonomic group.
///
/// Uses common patterns in species naming to determine the clade.
pub fn map_species_to_taxon(species: &str) -> Taxon {
    let species_lower = species.to_lowercase();

    // Cetaceans (toothed whales)
    if species_lower.contains("dolphin")
        || species_lower.contains("porpoise")
        || species_lower.contains("orca")
        || species_lower.contains("sperm whale")
        || species_lower.contains("beaked")
        || species_lower.contains("delphinid")
        || species_lower.contains("phocoen")
    {
        return Taxon::Cetacean;
    }

    // Mysticetes (baleen whales)
    if species_lower.contains("humpback")
        || species_lower.contains("blue whale")
        || species_lower.contains("fin whale")
        || species_lower.contains("minke")
        || species_lower.contains("gray whale")
        || species_lower.contains("right whale")
        || species_lower.contains("bowhead")
        || species_lower.contains("balaenopter")
    {
        return Taxon::Mysticete;
    }

    // Pinnipeds
    if species_lower.contains("seal")
        || species_lower.contains("sea lion")
        || species_lower.contains("walrus")
        || species_lower.contains("phocid")
        || species_lower.contains("otariid")
    {
        return Taxon::Pinniped;
    }

    // Songbirds (passerines)
    if species_lower.contains("sparrow")
        || species_lower.contains("finch")
        || species_lower.contains("warbler")
        || species_lower.contains("thrush")
        || species_lower.contains("robin")
        || species_lower.contains("cardinal")
        || species_lower.contains("towhee")
        || species_lower.contains("ovenbird")
        || species_lower.contains("wren")
        || species_lower.contains("tit")
        || species_lower.contains("swainson")
    {
        return Taxon::Songbird;
    }

    // Non-passerine birds
    if species_lower.contains("parrot")
        || species_lower.contains("owl")
        || species_lower.contains("hawk")
        || species_lower.contains("eagle")
        || species_lower.contains("duck")
        || species_lower.contains("goose")
        || species_lower.contains("gull")
        || species_lower.contains("crow")
        || species_lower.contains("raven")
        || species_lower.contains("penguin")
        || species_lower.contains("psittacid")
        || species_lower.contains("strigid")
    {
        return Taxon::NonPasserine;
    }

    // Anurans (frogs/toads)
    if species_lower.contains("frog")
        || species_lower.contains("toad")
        || species_lower.contains("ranid")
        || species_lower.contains("bufonid")
        || species_lower.contains("hylid")
        || species_lower.contains("peeper")
    {
        return Taxon::Amphibian;
    }

    // Insects
    if species_lower.contains("cricket")
        || species_lower.contains("mosquito")
        || species_lower.contains("cicada")
        || species_lower.contains("grasshopper")
        || species_lower.contains("katydid")
        || species_lower.contains("bee")
        || species_lower.contains("fly")
        || species_lower.contains("anopheles")
        || species_lower.contains("aedes")
        || species_lower.contains("culex")
        || species_lower.contains("culicid")
    {
        return Taxon::Insect;
    }

    // Bats (mammals with FM)
    if species_lower.contains("bat")
        || species_lower.contains("pteropodid")
        || species_lower.contains("vesper")
        || species_lower.contains("phyllostomid")
    {
        return Taxon::Mammal;
    }

    // Primates and other mammals
    if species_lower.contains("monkey")
        || species_lower.contains("ape")
        || species_lower.contains("gibbon")
        || species_lower.contains("chimp")
        || species_lower.contains("gorilla")
        || species_lower.contains("primate")
    {
        return Taxon::Mammal;
    }

    // Default to Unknown
    Taxon::Unknown
}

/// Maps a BEANS task name to a taxonomic group.
///
/// BEANS tasks often have species groups embedded in the task name.
pub fn map_task_to_taxon(task: &str) -> Taxon {
    let task_lower = task.to_lowercase();

    // Check for specific task patterns
    if task_lower.contains("gibbon") {
        return Taxon::Mammal;
    }
    if task_lower.contains("dcase") || task_lower.contains("bird") {
        return Taxon::Songbird;
    }
    if task_lower.contains("rfcx") {
        return Taxon::Mammal; // Rainforest Connection includes various species
    }
    if task_lower.contains("watkins") {
        return Taxon::Cetacean; // Watkins marine mammal database
    }
    if task_lower.contains("humbug") {
        return Taxon::Insect; // Mosquito database
    }
    if task_lower.contains("cbi") {
        return Taxon::Songbird; // Cornell Bird Identifier
    }

    // Default
    Taxon::Unknown
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TDD Test Suite: Divide and Conquer Feature Slicing
    // =========================================================================

    #[test]
    fn test_slice_gatekeeper_input_dimensions() {
        // Create a mock 112D feature vector
        let features: Vec<f32> = (0..112).map(|i| i as f32).collect();
        let result = slice_gatekeeper_input(&features);

        assert_eq!(result.len(), GATEKEEPER_DIM);

        // Verify Base Physics (46D) is copied correctly
        for i in 0..PHYSICS_DIM {
            assert!((result[i] - features[i]).abs() < 1e-6);
        }

        // Verify Macro Texture (30D) is copied correctly
        for i in 0..MACRO_TEXTURE_DIM {
            assert!((result[PHYSICS_DIM + i] - features[46 + i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_slice_species_expert_input_dimensions() {
        // Create a mock 112D feature vector
        let features: Vec<f32> = (0..112).map(|i| i as f32).collect();
        let result = slice_species_expert_input(&features);

        assert_eq!(result.len(), SPECIES_EXPERT_DIM);

        // Verify Base Physics (46D) is copied correctly
        for i in 0..PHYSICS_DIM {
            assert!((result[i] - features[i]).abs() < 1e-6);
        }

        // Verify Micro Texture (36D) is copied correctly
        for i in 0..MICRO_TEXTURE_DIM {
            assert!((result[PHYSICS_DIM + i] - features[76 + i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_feature_slices_distinct() {
        // Verify that Gatekeeper and Expert inputs are different
        let features: Vec<f32> = (0..112).map(|i| i as f32).collect();

        let gatekeeper = slice_gatekeeper_input(&features);
        let expert = slice_species_expert_input(&features);

        // Gatekeeper should have Macro Texture (features[46..76])
        // Expert should have Micro Texture (features[76..112])
        assert_ne!(gatekeeper[46..76], expert[46..82]);
    }

    #[test]
    fn test_feature_slices_share_base() {
        // Verify that both inputs share the same Base Physics
        let features: Vec<f32> = (0..112).map(|i| i as f32).collect();

        let gatekeeper = slice_gatekeeper_input(&features);
        let expert = slice_species_expert_input(&features);

        // Both should have the same Base Physics
        for i in 0..PHYSICS_DIM {
            assert!((gatekeeper[i] - expert[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gatekeeper_input_constant_values() {
        // Test with constant values to verify correct slicing
        let mut features = vec![0.0f32; FEATURE_DIM];

        // Set Base Physics to 1.0
        for i in 0..46 { features[i] = 1.0; }
        // Set Macro Texture to 2.0
        for i in 46..76 { features[i] = 2.0; }
        // Set Micro Texture to 3.0
        for i in 76..112 { features[i] = 3.0; }

        let gatekeeper = slice_gatekeeper_input(&features);

        // Gatekeeper should have 1.0 for base, 2.0 for macro
        for i in 0..46 { assert!((gatekeeper[i] - 1.0).abs() < 1e-6); }
        for i in 46..76 { assert!((gatekeeper[i] - 2.0).abs() < 1e-6); }
    }

    #[test]
    fn test_species_expert_input_constant_values() {
        // Test with constant values to verify correct slicing
        let mut features = vec![0.0f32; FEATURE_DIM];

        // Set Base Physics to 1.0
        for i in 0..46 { features[i] = 1.0; }
        // Set Macro Texture to 2.0
        for i in 46..76 { features[i] = 2.0; }
        // Set Micro Texture to 3.0
        for i in 76..112 { features[i] = 3.0; }

        let expert = slice_species_expert_input(&features);

        // Expert should have 1.0 for base, 3.0 for micro
        for i in 0..46 { assert!((expert[i] - 1.0).abs() < 1e-6); }
        for i in 46..82 { assert!((expert[i] - 3.0).abs() < 1e-6); }
    }

    #[test]
    fn test_feature_slices_constants() {
        // Verify constants are correctly defined
        assert_eq!(GATEKEEPER_DIM, 76);
        assert_eq!(SPECIES_EXPERT_DIM, 82);
        assert_eq!(PHYSICS_DIM, 46);
        assert_eq!(MACRO_TEXTURE_DIM, 30);
        assert_eq!(MICRO_TEXTURE_DIM, 36);
    }

    // =========================================================================
    // Consolidated Taxon Tests (TDD)
    // =========================================================================

    #[test]
    fn test_consolidate_taxon_birds() {
        // Both bird types should map to Bird
        assert_eq!(consolidate_taxon(Taxon::Songbird), ConsolidatedTaxon::Bird);
        assert_eq!(consolidate_taxon(Taxon::NonPasserine), ConsolidatedTaxon::Bird);
    }

    #[test]
    fn test_consolidate_taxon_marine_mammals() {
        // All marine mammals should map to MarineMammal
        assert_eq!(consolidate_taxon(Taxon::Cetacean), ConsolidatedTaxon::MarineMammal);
        assert_eq!(consolidate_taxon(Taxon::Mysticete), ConsolidatedTaxon::MarineMammal);
        assert_eq!(consolidate_taxon(Taxon::Pinniped), ConsolidatedTaxon::MarineMammal);
    }

    #[test]
    fn test_consolidate_taxon_terrestrial_mammal() {
        // Terrestrial mammals stay as Mammal
        assert_eq!(consolidate_taxon(Taxon::Mammal), ConsolidatedTaxon::Mammal);
    }

    #[test]
    fn test_consolidate_taxon_others() {
        // Other taxa remain distinct
        assert_eq!(consolidate_taxon(Taxon::Insect), ConsolidatedTaxon::Insect);
        assert_eq!(consolidate_taxon(Taxon::Amphibian), ConsolidatedTaxon::Amphibian);
        assert_eq!(consolidate_taxon(Taxon::Unknown), ConsolidatedTaxon::Unknown);
    }

    #[test]
    fn test_consolidated_taxon_index_roundtrip() {
        // Verify index roundtrip for all consolidated taxa
        for expected in [
            ConsolidatedTaxon::Bird,
            ConsolidatedTaxon::Mammal,
            ConsolidatedTaxon::MarineMammal,
            ConsolidatedTaxon::Insect,
            ConsolidatedTaxon::Amphibian,
            ConsolidatedTaxon::Unknown,
        ] {
            let idx = consolidated_taxon_to_idx(expected);
            let actual = idx_to_consolidated_taxon(idx);
            assert_eq!(expected, actual, "Roundtrip failed for {:?}", expected);
        }
    }

    #[test]
    fn test_consolidated_taxon_count() {
        // Verify we have exactly 6 classes
        assert_eq!(CONSOLIDATED_TAXON_COUNT, 6);
        assert_eq!(consolidated_taxon_labels().len(), 6);
    }

    #[test]
    fn test_consolidated_taxon_labels() {
        let labels = consolidated_taxon_labels();
        assert_eq!(labels[0], "Bird");
        assert_eq!(labels[1], "Mammal");
        assert_eq!(labels[2], "MarineMammal");
        assert_eq!(labels[3], "Insect");
        assert_eq!(labels[4], "Amphibian");
        assert_eq!(labels[5], "Unknown");
    }

    // =========================================================================
    // Taxonomic Weight Tests
    // =========================================================================

    #[test]
    fn test_weight_vector_dimensions() {
        // All taxonomic weights should be 112D
        for taxon in [
            Taxon::Cetacean, Taxon::Mysticete, Taxon::Songbird,
            Taxon::NonPasserine, Taxon::Amphibian, Taxon::Pinniped,
            Taxon::Insect, Taxon::Mammal, Taxon::Unknown,
        ] {
            let weights = get_taxonomic_weights(taxon);
            assert_eq!(weights.len(), FEATURE_DIM,
                "Weight vector for {:?} should be {}D", taxon, FEATURE_DIM);
        }
    }

    #[test]
    fn test_unknown_weights_are_uniform() {
        let weights = get_taxonomic_weights(Taxon::Unknown);
        for (i, &w) in weights.iter().enumerate() {
            assert!((w - 1.0).abs() < 1e-6,
                "Unknown weight at index {} should be 1.0, got {}", i, w);
        }
    }

    #[test]
    fn test_cetacean_weights_emphasize_ici() {
        let weights = get_taxonomic_weights(Taxon::Cetacean);

        // ICI bins should be heavily weighted
        for i in feature_indices::ICI_BINS_START..feature_indices::ICI_BINS_END {
            assert!(weights[i] > 2.0,
                "Cetacean weight at ICI index {} should be > 2.0, got {}", i, weights[i]);
        }
    }

    #[test]
    fn test_mysticete_weights_emphasize_duration() {
        let weights = get_taxonomic_weights(Taxon::Mysticete);

        // Duration should be heavily weighted for baleen whales
        assert!(weights[feature_indices::DURATION] > 2.0,
            "Mysticete duration weight should be > 2.0, got {}",
            weights[feature_indices::DURATION]);

        // Harmonic texture should also be emphasized
        for i in feature_indices::HARMONIC_TEXTURE_START..feature_indices::HARMONIC_TEXTURE_END {
            assert!(weights[i] > 2.0,
                "Mysticete harmonic weight at {} should be > 2.0, got {}", i, weights[i]);
        }
    }

    #[test]
    fn test_insect_weights_emphasize_tempo() {
        let weights = get_taxonomic_weights(Taxon::Insect);

        // Rhythm/tempo should be heavily weighted for insects
        for i in feature_indices::RHYTHM_START..feature_indices::RHYTHM_END {
            assert!(weights[i] > 3.0,
                "Insect rhythm weight at {} should be > 3.0, got {}", i, weights[i]);
        }
    }

    #[test]
    fn test_amphibian_weights_emphasize_dynamics() {
        let weights = get_taxonomic_weights(Taxon::Amphibian);

        // Dynamics bins (AM) should be heavily weighted for frogs
        for i in feature_indices::DYNAMICS_BINS_START..feature_indices::DYNAMICS_BINS_END {
            assert!(weights[i] > 2.0,
                "Amphibian dynamics weight at {} should be > 2.0, got {}", i, weights[i]);
        }
    }

    #[test]
    fn test_apply_mask_dimensions() {
        let features = vec![1.0; FEATURE_DIM];
        let masked = apply_taxonomic_mask(&features, Taxon::Cetacean);
        assert_eq!(masked.len(), FEATURE_DIM);
    }

    #[test]
    fn test_apply_mask_unknown_is_identity() {
        let features: Vec<f32> = (0..FEATURE_DIM).map(|i| i as f32).collect();
        let masked = apply_taxonomic_mask(&features, Taxon::Unknown);

        for (i, (f, m)) in features.iter().zip(masked.iter()).enumerate() {
            assert!((f - m).abs() < 1e-6,
                "Unknown mask at {} should be identity: {} vs {}", i, f, m);
        }
    }

    #[test]
    fn test_apply_mask_modifies_correct_indices() {
        let features = vec![1.0; FEATURE_DIM];
        let masked = apply_taxonomic_mask(&features, Taxon::Cetacean);

        // ICI bins should be multiplied by 3.0
        for i in feature_indices::ICI_BINS_START..feature_indices::ICI_BINS_END {
            assert!((masked[i] - 3.0).abs() < 1e-6,
                "Masked ICI at {} should be 3.0, got {}", i, masked[i]);
        }
    }

    #[test]
    fn test_slice_physics_dimensions() {
        let features = vec![1.0; FEATURE_DIM];
        let physics = slice_physics(&features);
        assert_eq!(physics.len(), PHYSICS_DIM);
    }

    #[test]
    fn test_slice_texture_dimensions() {
        let features = vec![1.0; FEATURE_DIM];
        let texture = slice_texture(&features);
        assert_eq!(texture.len(), TEXTURE_DIM);
    }

    #[test]
    fn test_slices_sum_to_full() {
        let features: Vec<f32> = (0..FEATURE_DIM).map(|i| i as f32).collect();
        let physics = slice_physics(&features);
        let texture = slice_texture(&features);

        assert_eq!(physics.len() + texture.len(), FEATURE_DIM);
        assert_eq!(physics[0], 0.0); // First element
        assert_eq!(texture[0], PHYSICS_DIM as f32); // First texture element
    }

    #[test]
    fn test_map_species_cetacean() {
        assert_eq!(map_species_to_taxon("Common Dolphin"), Taxon::Cetacean);
        assert_eq!(map_species_to_taxon("Harbor Porpoise"), Taxon::Cetacean);
        assert_eq!(map_species_to_taxon("Sperm Whale"), Taxon::Cetacean);
        assert_eq!(map_species_to_taxon("delphinidae"), Taxon::Cetacean);
    }

    #[test]
    fn test_map_species_mysticete() {
        assert_eq!(map_species_to_taxon("Humpback Whale"), Taxon::Mysticete);
        assert_eq!(map_species_to_taxon("Blue Whale"), Taxon::Mysticete);
        assert_eq!(map_species_to_taxon("Minke Whale"), Taxon::Mysticete);
        assert_eq!(map_species_to_taxon("Balaenoptera musculus"), Taxon::Mysticete);
    }

    #[test]
    fn test_map_species_songbird() {
        assert_eq!(map_species_to_taxon("Eastern Towhee"), Taxon::Songbird);
        assert_eq!(map_species_to_taxon("Northern Cardinal"), Taxon::Songbird);
        assert_eq!(map_species_to_taxon("Zebra Finch"), Taxon::Songbird);
        assert_eq!(map_species_to_taxon("American Robin"), Taxon::Songbird);
        assert_eq!(map_species_to_taxon("Swainson's Thrush"), Taxon::Songbird);
        assert_eq!(map_species_to_taxon("Ovenbird"), Taxon::Songbird);
        assert_eq!(map_species_to_taxon("Great Tit"), Taxon::Songbird);
    }

    #[test]
    fn test_map_species_insect() {
        assert_eq!(map_species_to_taxon("Anopheles gambiae"), Taxon::Insect);
        assert_eq!(map_species_to_taxon("Aedes aegypti"), Taxon::Insect);
        assert_eq!(map_species_to_taxon("Mosquito"), Taxon::Insect);
        assert_eq!(map_species_to_taxon("Cricket"), Taxon::Insect);
        assert_eq!(map_species_to_taxon("Cicada"), Taxon::Insect);
        assert_eq!(map_species_to_taxon("non-mosquito"), Taxon::Insect);
    }

    #[test]
    fn test_map_species_amphibian() {
        assert_eq!(map_species_to_taxon("Spring Peeper"), Taxon::Amphibian);
        assert_eq!(map_species_to_taxon("Tree Frog"), Taxon::Amphibian);
        assert_eq!(map_species_to_taxon("American Toad"), Taxon::Amphibian);
    }

    #[test]
    fn test_map_species_mammal() {
        assert_eq!(map_species_to_taxon("Egyptian Fruit Bat"), Taxon::Mammal);
        assert_eq!(map_species_to_taxon("Gibbon"), Taxon::Mammal);
        assert_eq!(map_species_to_taxon("Lar Gibbon"), Taxon::Mammal);
    }

    #[test]
    fn test_map_species_pinniped() {
        assert_eq!(map_species_to_taxon("Harbor Seal"), Taxon::Pinniped);
        assert_eq!(map_species_to_taxon("California Sea Lion"), Taxon::Pinniped);
        assert_eq!(map_species_to_taxon("Walrus"), Taxon::Pinniped);
    }

    #[test]
    fn test_map_task_to_taxon() {
        assert_eq!(map_task_to_taxon("task_gibbons"), Taxon::Mammal);
        assert_eq!(map_task_to_taxon("task_dcase"), Taxon::Songbird);
        assert_eq!(map_task_to_taxon("task_watkins"), Taxon::Cetacean);
        assert_eq!(map_task_to_taxon("task_humbugdb"), Taxon::Insect);
    }

    #[test]
    fn test_weights_are_positive() {
        // All weights should be positive to avoid flipping feature signs
        for taxon in [
            Taxon::Cetacean, Taxon::Mysticete, Taxon::Songbird,
            Taxon::NonPasserine, Taxon::Amphibian, Taxon::Pinniped,
            Taxon::Insect, Taxon::Mammal, Taxon::Unknown,
        ] {
            let weights = get_taxonomic_weights(taxon);
            for (i, &w) in weights.iter().enumerate() {
                assert!(w > 0.0,
                    "Weight for {:?} at index {} should be positive, got {}", taxon, i, w);
            }
        }
    }

    #[test]
    fn test_mask_preserves_sign() {
        // Negative features should stay negative after masking
        let features: Vec<f32> = (0..FEATURE_DIM).map(|i| -(i as f32)).collect();
        let masked = apply_taxonomic_mask(&features, Taxon::Cetacean);

        for (i, m) in masked.iter().enumerate() {
            assert!(*m <= 0.0,
                "Masked value at {} should be <= 0 (sign preserved), got {}", i, m);
        }
    }
}
