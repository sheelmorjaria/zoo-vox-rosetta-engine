//! BEANS-Zero Taxonomic Weight Router
//! ====================================
//!
//! Dynamically selects feature weights based on species/taxonomic group.
//! This optimizes the distance metric for zero-shot classification by
//! emphasizing the most discriminative features for each taxonomic group.
//!
//! Key Insight:
//! A frog's call is defined by PULSE RATE, not FM slope.
//! A whale's whistle is defined by FM CONTOUR, not formants.
//! A bird's song is defined by PITCH + SPECTRAL ENVELOPE.
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use serde::{Deserialize, Serialize};

/// Feature weights for 45D acoustic distance computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureWeights {
    // Fundamental (3 features)
    pub f0_weight: f32,
    pub duration_weight: f32,
    pub f0_range_weight: f32,

    // Grit (3 features)
    pub hnr_weight: f32,
    pub spectral_flatness_weight: f32,
    pub harmonicity_weight: f32,

    // Motion (7 features)
    pub attack_weight: f32,
    pub decay_weight: f32,
    pub sustain_weight: f32,
    pub vibrato_rate_weight: f32,
    pub vibrato_depth_weight: f32,
    pub jitter_weight: f32,
    pub shimmer_weight: f32,

    // Fingerprint/MFCC (14 features)
    pub mfcc_weight: f32,

    // Rhythm (3 features)
    pub tempo_weight: f32,
    pub pulse_clarity_weight: f32,
    pub rhythm_regularity_weight: f32,

    // Resonance (6 features)
    pub formant_weight: f32,

    // Spectral Shape (4 features)
    pub spectral_centroid_weight: f32,
    pub spectral_spread_weight: f32,
    pub spectral_skewness_weight: f32,
    pub spectral_kurtosis_weight: f32,

    // Modulation (3 features)
    pub spectral_tilt_weight: f32,
    pub fm_slope_weight: f32,
    pub am_depth_weight: f32,

    // Non-Linear (2 features)
    pub subharmonic_weight: f32,
    pub spectral_entropy_weight: f32,
}

impl Default for FeatureWeights {
    fn default() -> Self {
        Self {
            // All weights at 1.0 (uniform)
            f0_weight: 1.0,
            duration_weight: 1.0,
            f0_range_weight: 1.0,
            hnr_weight: 1.0,
            spectral_flatness_weight: 1.0,
            harmonicity_weight: 1.0,
            attack_weight: 1.0,
            decay_weight: 1.0,
            sustain_weight: 1.0,
            vibrato_rate_weight: 1.0,
            vibrato_depth_weight: 1.0,
            jitter_weight: 1.0,
            shimmer_weight: 1.0,
            mfcc_weight: 1.0,
            tempo_weight: 1.0,
            pulse_clarity_weight: 1.0,
            rhythm_regularity_weight: 1.0,
            formant_weight: 1.0,
            spectral_centroid_weight: 1.0,
            spectral_spread_weight: 1.0,
            spectral_skewness_weight: 1.0,
            spectral_kurtosis_weight: 1.0,
            spectral_tilt_weight: 1.0,
            fm_slope_weight: 1.0,
            am_depth_weight: 1.0,
            subharmonic_weight: 1.0,
            spectral_entropy_weight: 1.0,
        }
    }
}

impl FeatureWeights {
    /// Convert to array of 45 weights matching Vector45D layout
    pub fn to_array(&self) -> [f32; 45] {
        [
            // Fundamental (3)
            self.f0_weight,
            self.duration_weight,
            self.f0_range_weight,
            // Grit (3)
            self.hnr_weight,
            self.spectral_flatness_weight,
            self.harmonicity_weight,
            // Motion (7)
            self.attack_weight,
            self.decay_weight,
            self.sustain_weight,
            self.vibrato_rate_weight,
            self.vibrato_depth_weight,
            self.jitter_weight,
            self.shimmer_weight,
            // Fingerprint/MFCC (14) - all use same weight
            self.mfcc_weight,
            self.mfcc_weight,
            self.mfcc_weight,
            self.mfcc_weight,
            self.mfcc_weight,
            self.mfcc_weight,
            self.mfcc_weight,
            self.mfcc_weight,
            self.mfcc_weight,
            self.mfcc_weight,
            self.mfcc_weight,
            self.mfcc_weight,
            self.mfcc_weight,
            self.mfcc_weight,
            // Rhythm (3)
            self.tempo_weight,
            self.pulse_clarity_weight,
            self.rhythm_regularity_weight,
            // Resonance (6)
            self.formant_weight,
            self.formant_weight,
            self.formant_weight,
            self.formant_weight,
            self.formant_weight,
            self.formant_weight,
            // Spectral Shape (4)
            self.spectral_centroid_weight,
            self.spectral_spread_weight,
            self.spectral_skewness_weight,
            self.spectral_kurtosis_weight,
            // Modulation (3)
            self.spectral_tilt_weight,
            self.fm_slope_weight,
            self.am_depth_weight,
            // Non-Linear (2)
            self.subharmonic_weight,
            self.spectral_entropy_weight,
        ]
    }

    /// Cetacean weights (whales, dolphins, porpoises)
    /// Focus: ICI (rhythm), FM Slope, Spectral Centroid
    pub fn cetacean() -> Self {
        Self {
            // Rhythm is CRITICAL for click-based species
            tempo_weight: 3.0,
            pulse_clarity_weight: 3.5,
            rhythm_regularity_weight: 3.0,
            // FM Slope for whistles
            fm_slope_weight: 2.5,
            am_depth_weight: 2.0,
            // Spectral for tonal content
            spectral_centroid_weight: 2.0,
            spectral_spread_weight: 1.5,
            // Reduced importance
            f0_weight: 1.0,
            formant_weight: 0.5, // Less relevant underwater
            ..Default::default()
        }
    }

    /// Sperm Whale weights (rhythm-dominant)
    pub fn sperm_whale() -> Self {
        Self {
            // Rhythm is THE identifier for sperm whale codas
            tempo_weight: 4.0,
            pulse_clarity_weight: 4.5,
            rhythm_regularity_weight: 4.0,
            // ICI pattern recognition
            attack_weight: 2.5,
            decay_weight: 2.5,
            // Reduced - clicks don't have much FM
            fm_slope_weight: 0.5,
            f0_weight: 0.5,
            formant_weight: 0.3,
            ..Default::default()
        }
    }

    /// Songbird weights (passerines)
    /// Focus: F0, Harmonics, Spectral Flux
    pub fn songbird() -> Self {
        Self {
            // Pitch structure
            f0_weight: 1.8,
            f0_range_weight: 1.5,
            // Harmonic content
            harmonicity_weight: 1.5,
            hnr_weight: 1.5,
            // Spectral envelope (brightness)
            spectral_centroid_weight: 1.5,
            spectral_spread_weight: 1.3,
            // MFCC for timbre
            mfcc_weight: 1.5,
            // Temporal dynamics
            attack_weight: 1.3,
            decay_weight: 1.2,
            // Formants less critical for small birds
            formant_weight: 0.8,
            ..Default::default()
        }
    }

    /// Amphibian weights (frogs, toads)
    /// Focus: ICI (pulse rate), F0 band
    pub fn amphibian() -> Self {
        Self {
            // Pulse rate is THE identifier
            tempo_weight: 3.5,
            pulse_clarity_weight: 3.5,
            rhythm_regularity_weight: 3.0,
            // Pitch band
            f0_weight: 2.0,
            f0_range_weight: 1.5,
            // Onset/attack for call detection
            attack_weight: 2.0,
            decay_weight: 1.5,
            // Reduced - frogs have simple calls
            fm_slope_weight: 0.3,
            vibrato_rate_weight: 0.5,
            formant_weight: 0.5,
            mfcc_weight: 0.8,
            ..Default::default()
        }
    }

    /// Insect weights (cicadas, crickets, katydids)
    /// Focus: Onset rate, Spectral centroid (high freq)
    pub fn insect() -> Self {
        Self {
            // Buzzing rate
            tempo_weight: 3.5,
            pulse_clarity_weight: 3.0,
            // High frequency content
            spectral_centroid_weight: 2.5,
            spectral_spread_weight: 2.0,
            // Texture/micro-dynamics
            spectral_flatness_weight: 1.8,
            spectral_entropy_weight: 1.5,
            // Temporal
            attack_weight: 2.0,
            decay_weight: 1.5,
            // Reduced - insects are harmonic-weak
            harmonicity_weight: 0.5,
            f0_weight: 1.0, // Often ultrasonic
            formant_weight: 0.3,
            ..Default::default()
        }
    }

    /// Mammal weights (primates, carnivores)
    /// Focus: Formants, Spectral Tilt
    pub fn mammal() -> Self {
        Self {
            // Vocal tract shape (formants)
            formant_weight: 2.0,
            // Spectral tilt (effort/arousal)
            spectral_tilt_weight: 1.8,
            // General spectral shape
            spectral_centroid_weight: 1.5,
            spectral_spread_weight: 1.3,
            // Pitch range
            f0_weight: 1.5,
            f0_range_weight: 1.3,
            // Temporal for some species
            attack_weight: 1.3,
            decay_weight: 1.2,
            // MFCC for voice quality
            mfcc_weight: 1.5,
            ..Default::default()
        }
    }

    /// Bat weights (ultrasonic)
    /// Focus: FM Slope, Decay, High-frequency features
    pub fn bat() -> Self {
        Self {
            // FM sweep is key
            fm_slope_weight: 3.0,
            // Call envelope
            attack_weight: 2.0,
            decay_weight: 2.0,
            // High frequency
            spectral_centroid_weight: 2.5,
            spectral_spread_weight: 2.0,
            // Duration often diagnostic
            duration_weight: 1.8,
            // Pulse rate for feeding buzzes
            tempo_weight: 2.0,
            pulse_clarity_weight: 2.0,
            // Reduced
            formant_weight: 0.5,
            mfcc_weight: 1.0,
            ..Default::default()
        }
    }

    /// Gibbon weights (primate with complex songs)
    pub fn gibbon() -> Self {
        Self {
            // Great call structure
            f0_weight: 2.0,
            f0_range_weight: 2.0,
            // Formants for individual ID
            formant_weight: 1.8,
            // Spectral features
            spectral_tilt_weight: 1.5,
            spectral_centroid_weight: 1.3,
            // Temporal structure
            tempo_weight: 1.5,
            rhythm_regularity_weight: 1.5,
            // MFCC for voice
            mfcc_weight: 1.5,
            ..Default::default()
        }
    }

    /// Mosquito weights (wing beat frequency)
    pub fn mosquito() -> Self {
        Self {
            // Wing beat frequency is THE identifier
            f0_weight: 3.5,
            // Harmonics from wing beat
            harmonicity_weight: 2.0,
            hnr_weight: 2.0,
            // Very steady
            jitter_weight: 0.5,
            shimmer_weight: 0.5,
            // Reduced
            formant_weight: 0.2,
            mfcc_weight: 0.5,
            ..Default::default()
        }
    }
}

/// Router that maps BEANS-Zero species labels to optimized feature weights
pub struct BeansZeroWeightRouter;

impl BeansZeroWeightRouter {
    /// Get optimal feature weights for a given species label
    pub fn get_weights(species_label: &str) -> FeatureWeights {
        let label = species_label.to_lowercase();

        // ================================================================
        // 1. CETACEANS (Whales & Dolphins)
        // ================================================================
        if Self::is_cetacean(&label) {
            // Sperm Whale - rhythm dominant (codas)
            if label.contains("sperm") {
                return FeatureWeights::sperm_whale();
            }
            // Other cetaceans - FM/tonal dominant
            return FeatureWeights::cetacean();
        }

        // ================================================================
        // 2. BATS (Ultrasonic)
        // ================================================================
        if label.contains("bat") {
            return FeatureWeights::bat();
        }

        // ================================================================
        // 3. AMPHIBIANS (Frogs & Toads)
        // ================================================================
        if Self::is_amphibian(&label) {
            return FeatureWeights::amphibian();
        }

        // ================================================================
        // 4. INSECTS/ARTHROPODS
        // ================================================================
        if Self::is_insect(&label) {
            // Mosquitoes - wing beat frequency
            if label.contains("mosquito")
                || label.contains("aedes")
                || label.contains("anopheles")
                || label.contains("culex")
                || label.contains("ae aegypti")
                || label.contains("an arabiensis")
            {
                return FeatureWeights::mosquito();
            }
            return FeatureWeights::insect();
        }

        // ================================================================
        // 5. PRIMATES (Gibbons, Monkeys)
        // ================================================================
        if Self::is_primate(&label) {
            if label.contains("gibbon") {
                return FeatureWeights::gibbon();
            }
            return FeatureWeights::mammal();
        }

        // ================================================================
        // 6. OTHER MAMMALS
        // ================================================================
        if Self::is_mammal(&label) {
            return FeatureWeights::mammal();
        }

        // ================================================================
        // 7. SONGBIRDS (Passerines and other birds)
        // ================================================================
        if Self::is_bird(&label) {
            return FeatureWeights::songbird();
        }

        // ================================================================
        // 8. FALLBACK (Uniform weights)
        // ================================================================
        FeatureWeights::default()
    }

    fn is_cetacean(label: &str) -> bool {
        label.contains("whale")
            || label.contains("dolphin")
            || label.contains("porpoise")
            || label.contains("cetacean")
            || label.contains("orca")
            || label.contains("cachalot")
    }

    fn is_amphibian(label: &str) -> bool {
        label.contains("frog")
            || label.contains("toad")
            || label.contains("peeper")
            || label.contains("coqui")
            || label.contains("salamander")
            || label.contains("newt")
            || label.contains("treefrog")
            || label.contains("chorus")
    }

    fn is_insect(label: &str) -> bool {
        label.contains("cicada")
            || label.contains("cricket")
            || label.contains("katydid")
            || label.contains("grasshopper")
            || label.contains("mosquito")
            || label.contains("aedes")
            || label.contains("anopheles")
            || label.contains("culex")
            || label.contains("an arabiensis")
            || label.contains("an gambiae")
            || label.contains("an funestus")
            || label.contains("ae aegypti")
            || label.contains("bee")
            || label.contains("wasp")
            || label.contains("fly")
            || label.contains("beetle")
            || label.contains("moth")
            || label.contains("arthropod")
    }

    fn is_primate(label: &str) -> bool {
        label.contains("gibbon")
            || label.contains("monkey")
            || label.contains("ape")
            || label.contains("chimpanzee")
            || label.contains("gorilla")
            || label.contains("orangutan")
            || label.contains("lemur")
            || label.contains("marmoset")
            || label.contains("macaque")
    }

    fn is_mammal(label: &str) -> bool {
        label.contains("meerkat")
            || label.contains("hyena")
            || label.contains("coyote")
            || label.contains("wolf")
            || label.contains("fox")
            || label.contains("dog")
            || label.contains("cat")
            || label.contains("lion")
            || label.contains("tiger")
            || label.contains("leopard")
            || label.contains("bear")
            || label.contains("elephant")
            || label.contains("seal")
            || label.contains("sea lion")
            || label.contains("hog")
            || label.contains("pig")
            || label.contains("deer")
            || label.contains("elk")
            || label.contains("moose")
            || label.contains("beaver")
            || label.contains("squirrel")
            || label.contains("rodent")
    }

    fn is_bird(label: &str) -> bool {
        // Songbirds
        label.contains("sparrow") || label.contains("finch") ||
        label.contains("wren") || label.contains("thrush") ||
        label.contains("warbler") || label.contains("blackbird") ||
        label.contains("robin") || label.contains("bluebird") ||
        label.contains("towhee") || label.contains("cardinal") ||
        label.contains("tananger") || label.contains("bunting") ||
        label.contains("grosbeak") || label.contains("oriole") ||
        label.contains("blackbird") || label.contains("cowbird") ||
        // Other birds
        label.contains("jay") || label.contains("crow") ||
        label.contains("raven") || label.contains("magpie") ||
        label.contains("chickadee") || label.contains("titmouse") ||
        label.contains("nuthatch") || label.contains("creeper") ||
        label.contains("mockingbird") || label.contains("catbird") ||
        label.contains("thrasher") || label.contains("starling") ||
        label.contains("woodpecker") || label.contains("flicker") ||
        label.contains("owl") || label.contains("hawk") ||
        label.contains("eagle") || label.contains("falcon") ||
        label.contains("dove") || label.contains("pigeon") ||
        label.contains("quail") || label.contains("pheasant") ||
        label.contains("duck") || label.contains("goose") ||
        label.contains("swan") || label.contains("heron") ||
        label.contains("egret") || label.contains("crane") ||
        label.contains("gull") || label.contains("tern") ||
        label.contains("parrot") || label.contains("parakeet") ||
        label.contains("cuckoo") || label.contains("nightjar") ||
        label.contains("swift") || label.contains("swallow") ||
        label.contains("martin") || label.contains("lark") ||
        label.contains("flycatcher") || label.contains("vireo") ||
        label.contains("kinglet") || label.contains("gnatcatcher") ||
        label.contains("shrike") || label.contains("kingbird") ||
        // Taxonomic
        label.contains("aves") || label.contains("passeriformes") ||
        label.contains("bird")
    }
}

/// Taxonomic group detection for logging/analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaxonomicGroup {
    Cetacean,
    Bat,
    Amphibian,
    Insect,
    Primate,
    Mammal,
    Bird,
    Unknown,
}

impl BeansZeroWeightRouter {
    /// Detect the taxonomic group for a species label
    pub fn detect_group(species_label: &str) -> TaxonomicGroup {
        let label = species_label.to_lowercase();

        if Self::is_cetacean(&label) {
            return TaxonomicGroup::Cetacean;
        }
        if label.contains("bat") {
            return TaxonomicGroup::Bat;
        }
        if Self::is_amphibian(&label) {
            return TaxonomicGroup::Amphibian;
        }
        if Self::is_insect(&label) {
            return TaxonomicGroup::Insect;
        }
        if Self::is_primate(&label) {
            return TaxonomicGroup::Primate;
        }
        if Self::is_mammal(&label) {
            return TaxonomicGroup::Mammal;
        }
        if Self::is_bird(&label) {
            return TaxonomicGroup::Bird;
        }

        TaxonomicGroup::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cetacean_detection() {
        assert_eq!(
            BeansZeroWeightRouter::detect_group("Minke whale"),
            TaxonomicGroup::Cetacean
        );
        assert_eq!(
            BeansZeroWeightRouter::detect_group("Spinner Dolphin"),
            TaxonomicGroup::Cetacean
        );
        assert_eq!(
            BeansZeroWeightRouter::detect_group("Harbor Porpoise"),
            TaxonomicGroup::Cetacean
        );
        assert_eq!(
            BeansZeroWeightRouter::detect_group("Sperm Whale"),
            TaxonomicGroup::Cetacean
        );
    }

    #[test]
    fn test_bird_detection() {
        assert_eq!(
            BeansZeroWeightRouter::detect_group("Eastern Towhee"),
            TaxonomicGroup::Bird
        );
        assert_eq!(
            BeansZeroWeightRouter::detect_group("Northern Cardinal"),
            TaxonomicGroup::Bird
        );
        assert_eq!(
            BeansZeroWeightRouter::detect_group("Swainson's Thrush"),
            TaxonomicGroup::Bird
        );
        assert_eq!(
            BeansZeroWeightRouter::detect_group("Blue Jay"),
            TaxonomicGroup::Bird
        );
    }

    #[test]
    fn test_amphibian_detection() {
        assert_eq!(
            BeansZeroWeightRouter::detect_group("Spring Peeper"),
            TaxonomicGroup::Amphibian
        );
        assert_eq!(
            BeansZeroWeightRouter::detect_group("American Toad"),
            TaxonomicGroup::Amphibian
        );
        assert_eq!(
            BeansZeroWeightRouter::detect_group("Gray Treefrog"),
            TaxonomicGroup::Amphibian
        );
    }

    #[test]
    fn test_insect_detection() {
        assert_eq!(
            BeansZeroWeightRouter::detect_group("Ash Cicada"),
            TaxonomicGroup::Insect
        );
        assert_eq!(
            BeansZeroWeightRouter::detect_group("Meadow Grasshopper"),
            TaxonomicGroup::Insect
        );
        assert_eq!(
            BeansZeroWeightRouter::detect_group("an arabiensis"),
            TaxonomicGroup::Insect
        );
    }

    #[test]
    fn test_mammal_detection() {
        assert_eq!(
            BeansZeroWeightRouter::detect_group("Meerkat close call"),
            TaxonomicGroup::Mammal
        );
        assert_eq!(
            BeansZeroWeightRouter::detect_group("Hyena groan and moo"),
            TaxonomicGroup::Mammal
        );
        assert_eq!(
            BeansZeroWeightRouter::detect_group("Multiple pulse gibbon call"),
            TaxonomicGroup::Primate
        );
    }

    #[test]
    fn test_sperm_whale_weights() {
        let weights = BeansZeroWeightRouter::get_weights("Sperm Whale");
        assert!(
            weights.tempo_weight > 3.0,
            "Sperm whale should have high tempo weight"
        );
        assert!(
            weights.pulse_clarity_weight > 4.0,
            "Sperm whale should have high pulse clarity weight"
        );
    }

    #[test]
    fn test_frog_weights() {
        let weights = BeansZeroWeightRouter::get_weights("Spring Peeper");
        assert!(
            weights.tempo_weight > 3.0,
            "Frog should have high tempo weight"
        );
        assert!(
            weights.f0_weight > 1.5,
            "Frog should have elevated F0 weight"
        );
    }

    #[test]
    fn test_bird_weights() {
        let weights = BeansZeroWeightRouter::get_weights("Eastern Towhee");
        assert!(
            weights.f0_weight > 1.5,
            "Bird should have elevated F0 weight"
        );
        assert!(
            weights.mfcc_weight > 1.0,
            "Bird should have elevated MFCC weight"
        );
    }

    #[test]
    fn test_default_weights() {
        let weights = BeansZeroWeightRouter::get_weights("Unknown species");
        assert_eq!(weights.f0_weight, 1.0);
        assert_eq!(weights.tempo_weight, 1.0);
    }

    #[test]
    fn test_weights_array_length() {
        let weights = FeatureWeights::default();
        let arr = weights.to_array();
        assert_eq!(arr.len(), 45);
    }
}
