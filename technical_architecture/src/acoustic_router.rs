//! Acoustic Router for Detection Pipeline
//! ==============================================================
//!
//! Routes species to acoustic specialists based on acoustic coherence
//! rather than biological taxonomy. This ensures specialists handle
//! similar feature distributions.
//!
//! Acoustic Groups (13 total):
//! - UltrasonicMammal: Bats (20-80kHz, 5-50ms, FM sweeps)
//! - SonicLongMammal: Baleen whales (20-5000Hz, 0.5-5s, low moans)
//! - SonicShortMammal: Primates, land mammals (mid F0, variable)
//! - InsectWingbeat: Mosquitoes, flies, bees (steady F0, pure tones)
//! - InsectStridulation: Crickets, cicadas (broadband, impulsive)
//! - BirdHighFreq: Songbirds (4-8kHz, fast modulation)
//! - BirdLowFreq: Doves, owls (200-1000Hz, long duration)
//! - BirdMechanical: Hummingbirds (broadband, pulse-like)
//! - MarineWhistle: Dolphins, orcas (FM sweeps, harmonic, 2-24kHz)
//! - MarineClick: Porpoises, sperm whales (impulsive, broadband)
//! - MarineMoan: Baleen whales fallback (low F0, long duration)
//! - Amphibian: Frogs, toads (500-5000Hz, pulsed)
//! - Pinniped: Seals, sea lions (100-5000Hz, varied)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Acoustic Group Enum
// =============================================================================

/// 13 acoustic groups based on frequency, duration, and modulation patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AcousticGroup {
    // Mammals (3-way split by frequency/duration)
    UltrasonicMammal, // Bats: 20-80kHz, 5-50ms
    SonicLongMammal,  // Baleen whales: 20-5000Hz, 0.5-5s
    SonicShortMammal, // Primates: mid F0, variable

    // Insects (2-way split by production mechanism)
    InsectWingbeat,     // Mosquitoes, flies, bees: steady F0, pure tones
    InsectStridulation, // Crickets, cicadas: broadband, impulsive

    // Birds (3-way split by frequency/production)
    BirdHighFreq,   // Songbirds: 4-8kHz, fast modulation
    BirdLowFreq,    // Doves, owls: 200-1000Hz, long duration
    BirdMechanical, // Hummingbirds: broadband, pulse-like

    // Marine Mammals (3-way split by sound type)
    MarineWhistle, // Dolphins, orcas: FM sweeps, harmonic
    MarineClick,   // Porpoises, sperm whales: impulsive, broadband
    MarineMoan,    // Baleen whales: low F0, long duration

    // Other
    Amphibian, // Frogs, toads: 500-5000Hz, pulsed
    Pinniped,  // Seals, sea lions: 100-5000Hz
}

impl AcousticGroup {
    /// Get the filename suffix for this acoustic group
    pub fn filename_suffix(&self) -> &'static str {
        match self {
            AcousticGroup::UltrasonicMammal => "ultrasonic_mammal",
            AcousticGroup::SonicLongMammal => "sonic_long_mammal",
            AcousticGroup::SonicShortMammal => "sonic_short_mammal",
            AcousticGroup::InsectWingbeat => "insect_wingbeat",
            AcousticGroup::InsectStridulation => "insect_stridulation",
            AcousticGroup::BirdHighFreq => "bird_high_freq",
            AcousticGroup::BirdLowFreq => "bird_low_freq",
            AcousticGroup::BirdMechanical => "bird_mechanical",
            AcousticGroup::MarineWhistle => "marine_whistle",
            AcousticGroup::MarineClick => "marine_click",
            AcousticGroup::MarineMoan => "marine_moan",
            AcousticGroup::Amphibian => "amphibian",
            AcousticGroup::Pinniped => "pinniped",
        }
    }

    /// Get all acoustic group variants
    pub fn all() -> &'static [AcousticGroup; 13] {
        &[
            AcousticGroup::UltrasonicMammal,
            AcousticGroup::SonicLongMammal,
            AcousticGroup::SonicShortMammal,
            AcousticGroup::InsectWingbeat,
            AcousticGroup::InsectStridulation,
            AcousticGroup::BirdHighFreq,
            AcousticGroup::BirdLowFreq,
            AcousticGroup::BirdMechanical,
            AcousticGroup::MarineWhistle,
            AcousticGroup::MarineClick,
            AcousticGroup::MarineMoan,
            AcousticGroup::Amphibian,
            AcousticGroup::Pinniped,
        ]
    }

    /// Get expected acoustic characteristics for this group
    pub fn characteristics(&self) -> AcousticCharacteristics {
        match self {
            AcousticGroup::UltrasonicMammal => AcousticCharacteristics {
                freq_range_hz: (20000, 80000),
                typical_duration_ms: (5.0, 50.0),
                description: "Ultrasonic FM sweeps (bats)",
            },
            AcousticGroup::SonicLongMammal => AcousticCharacteristics {
                freq_range_hz: (20, 5000),
                typical_duration_ms: (500.0, 5000.0),
                description: "Low-frequency moans (baleen whales)",
            },
            AcousticGroup::SonicShortMammal => AcousticCharacteristics {
                freq_range_hz: (100, 8000),
                typical_duration_ms: (50.0, 1000.0),
                description: "Mid-frequency calls (primates)",
            },
            AcousticGroup::InsectWingbeat => AcousticCharacteristics {
                freq_range_hz: (100, 1000),
                typical_duration_ms: (10.0, 500.0),
                description: "Steady pure tones (mosquitoes)",
            },
            AcousticGroup::InsectStridulation => AcousticCharacteristics {
                freq_range_hz: (2000, 10000),
                typical_duration_ms: (10.0, 200.0),
                description: "Broadband pulses (crickets)",
            },
            AcousticGroup::BirdHighFreq => AcousticCharacteristics {
                freq_range_hz: (4000, 8000),
                typical_duration_ms: (50.0, 500.0),
                description: "High-frequency modulated calls (songbirds)",
            },
            AcousticGroup::BirdLowFreq => AcousticCharacteristics {
                freq_range_hz: (200, 1000),
                typical_duration_ms: (100.0, 1000.0),
                description: "Low-frequency calls (doves, owls)",
            },
            AcousticGroup::BirdMechanical => AcousticCharacteristics {
                freq_range_hz: (100, 10000),
                typical_duration_ms: (10.0, 100.0),
                description: "Broadband mechanical sounds (hummingbirds)",
            },
            AcousticGroup::MarineWhistle => AcousticCharacteristics {
                freq_range_hz: (2000, 24000),
                typical_duration_ms: (100.0, 1000.0),
                description: "FM whistles (dolphins)",
            },
            AcousticGroup::MarineClick => AcousticCharacteristics {
                freq_range_hz: (1000, 150000),
                typical_duration_ms: (0.05, 1.0),
                description: "Impulsive clicks (porpoises)",
            },
            AcousticGroup::MarineMoan => AcousticCharacteristics {
                freq_range_hz: (20, 500),
                typical_duration_ms: (1000.0, 5000.0),
                description: "Low moans (baleen whales)",
            },
            AcousticGroup::Amphibian => AcousticCharacteristics {
                freq_range_hz: (500, 5000),
                typical_duration_ms: (50.0, 500.0),
                description: "Pulsed calls (frogs)",
            },
            AcousticGroup::Pinniped => AcousticCharacteristics {
                freq_range_hz: (100, 5000),
                typical_duration_ms: (100.0, 1000.0),
                description: "Varied calls (seals)",
            },
        }
    }
}

/// Acoustic characteristics for a group
#[derive(Debug, Clone)]
pub struct AcousticCharacteristics {
    pub freq_range_hz: (u32, u32),
    pub typical_duration_ms: (f32, f32),
    pub description: &'static str,
}

// =============================================================================
// Species to Acoustic Group Mapping
// =============================================================================

/// Map a species name to its acoustic group based on acoustic characteristics
pub fn map_species_to_acoustic_group(species: &str) -> AcousticGroup {
    let s = species.to_lowercase();

    // === ULTRASONIC MAMMALS (Bats) ===
    if s.contains("bat")
        || s.contains("pteropodid")
        || s.contains("vesper")
        || s.contains("phyllostomid")
        || s.contains("rhinolophus")
        || s.contains("myotis")
        || s.contains("nyctalus")
        || s.contains("pipistrellus")
        || s.contains("eptesicus")
        || s.contains("plecotus")
        || s.contains("miniopterus")
        || s.contains("tadarida")
        || s.contains("molossid")
        || s.contains("vespertilion")
        || s.contains("noctilio")
        || s.contains("hypsignathus")
        || s.contains("pteropus")
        || s.contains("nyctinomops")
        || s.contains("molossus")
    {
        return AcousticGroup::UltrasonicMammal;
    }

    // === SONIC LONG MAMMALS (Baleen Whales) ===
    if s.contains("humpback")
        || s.contains("blue whale")
        || s.contains("fin whale")
        || s.contains("minke")
        || s.contains("gray whale")
        || s.contains("grey whale")
        || s.contains("right whale")
        || s.contains("bowhead")
        || s.contains("balaenopter")
        || s.contains("megaptera")
        || s.contains("eschrichtius")
        || s.contains("balaena")
    {
        return AcousticGroup::SonicLongMammal;
    }

    // === MARINE WHISTLE (Dolphins, Orcas) ===
    if s.contains("dolphin")
        || s.contains("delphin")
        || s.contains("orca")
        || s.contains("killer whale")
        || s.contains("pilot whale")
        || s.contains("tursiops")
        || s.contains("grampus")
        || s.contains("stenella")
        || s.contains("lagenorhynchus")
        || s.contains("delphinapterus")
    {
        return AcousticGroup::MarineWhistle;
    }

    // === MARINE CLICK (Porpoises, Sperm Whales) ===
    if s.contains("porpoise")
        || s.contains("phocoen")
        || s.contains("sperm whale")
        || s.contains("physeter")
        || s.contains("beaked whale")
        || s.contains("ziphius")
        || s.contains("mesoplodon")
        || s.contains("kogia")
    {
        return AcousticGroup::MarineClick;
    }

    // === MARINE MOAN (Baleen whales - fallback) ===
    if s.contains("whale") && !s.contains("killer") {
        return AcousticGroup::MarineMoan;
    }

    // === PINNIPEDS ===
    if s.contains("seal")
        || s.contains("sea lion")
        || s.contains("walrus")
        || s.contains("phocid")
        || s.contains("otariid")
        || s.contains("otary")
    {
        return AcousticGroup::Pinniped;
    }

    // === INSECT WINGBEAT (Pure tones) ===
    if s.contains("mosquito")
        || s.contains("aedes")
        || s.contains("anopheles")
        || s.contains("culex")
        || s.contains("culicid")
        || s.contains("fly")
        || s.contains("muscidae")
        || s.contains("bee")
        || s.contains("apis")
        || s.contains("bombus")
        || s.contains("wasp")
        || s.contains("syrphid")
    {
        return AcousticGroup::InsectWingbeat;
    }

    // === INSECT STRIDULATION (Broadband pulses) ===
    if s.contains("cricket")
        || s.contains("cicada")
        || s.contains("grasshopper")
        || s.contains("katydid")
        || s.contains("tettigoniid")
        || s.contains("gryllid")
        || s.contains("acridid")
        || s.contains("orthoptera")
    {
        return AcousticGroup::InsectStridulation;
    }

    // === BIRD HIGH FREQ (Songbirds) ===
    if s.contains("sparrow")
        || s.contains("finch")
        || s.contains("warbler")
        || s.contains("thrush")
        || s.contains("robin")
        || s.contains("cardinal")
        || s.contains("towhee")
        || s.contains("ovenbird")
        || s.contains("wren")
        || s.contains("tit")
        || s.contains("swainson")
        || s.contains("junco")
        || s.contains("bunting")
        || s.contains("blackbird")
        || s.contains("meadowlark")
        || s.contains("cowbird")
        || s.contains("oriole")
        || s.contains("grackle")
        || s.contains("bobolink")
        || s.contains("lark")
        || s.contains("pipit")
        || s.contains("longspur")
        || s.contains("bluebird")
        || s.contains("solitaire")
        || s.contains("passerine")
        || s.contains("passer")
    {
        return AcousticGroup::BirdHighFreq;
    }

    // === BIRD LOW FREQ (Non-passerines with low calls) ===
    if s.contains("dove")
        || s.contains("pigeon")
        || s.contains("owl")
        || s.contains("cuckoo")
        || s.contains("quail")
        || s.contains("grouse")
        || s.contains("turkey")
        || s.contains("goose")
        || s.contains("swan")
        || s.contains("heron")
        || s.contains("stork")
        || s.contains("crane")
        || s.contains("columb")
        || s.contains("strigid")
    {
        return AcousticGroup::BirdLowFreq;
    }

    // === BIRD MECHANICAL (Hummingbirds, snipe) ===
    if s.contains("hummingbird")
        || s.contains("trochilid")
        || s.contains("snipe")
        || s.contains("gallinago")
        || s.contains("woodpecker")
        || s.contains("picid")
    {
        return AcousticGroup::BirdMechanical;
    }

    // === OTHER NON-PASSERINE BIRDS ===
    if s.contains("parrot")
        || s.contains("hawk")
        || s.contains("eagle")
        || s.contains("duck")
        || s.contains("gull")
        || s.contains("crow")
        || s.contains("raven")
        || s.contains("penguin")
        || s.contains("psittacid")
        || s.contains("bird")
    {
        return AcousticGroup::BirdLowFreq;
    }

    // === AMPHIBIANS ===
    if s.contains("frog")
        || s.contains("toad")
        || s.contains("ranid")
        || s.contains("bufonid")
        || s.contains("hylid")
        || s.contains("peeper")
        || s.contains("anuran")
    {
        return AcousticGroup::Amphibian;
    }

    // === SONIC SHORT MAMMALS (Primates, land mammals) ===
    if s.contains("monkey")
        || s.contains("ape")
        || s.contains("gibbon")
        || s.contains("chimp")
        || s.contains("gorilla")
        || s.contains("primate")
        || s.contains("marmoset")
        || s.contains("lemur")
        || s.contains("tamarin")
        || s.contains("capuchin")
        || s.contains("macaque")
        || s.contains("howler")
    {
        return AcousticGroup::SonicShortMammal;
    }

    // Default to SonicShortMammal (most common fallback)
    AcousticGroup::SonicShortMammal
}

/// Build canonical label map for species name normalization
pub fn build_canonical_label_map() -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mappings = [
        ("Dall's Porpoise", "Phocoenoides dalli"),
        ("Harbor Porpoise", "Phocoena phocoena"),
        ("Finless Porpoise", "Neophocaena phocaenoides"),
        ("Bottlenose Dolphin", "Tursiops truncatus"),
        ("Common Dolphin", "Delphinus delphis"),
        ("Killer Whale", "Orcinus orca"),
        ("Humpback Whale", "Megaptera novaeangliae"),
        ("Minke Whale", "Balaenoptera acutorostrata"),
        ("Minke whale", "Balaenoptera acutorostrata"),
        ("Blue Whale", "Balaenoptera musculus"),
        ("Fin Whale", "Balaenoptera physalus"),
        ("Gray Whale", "Eschrichtius robustus"),
    ];
    for (common, scientific) in &mappings {
        map.insert(common.to_lowercase(), scientific.to_string());
        map.insert(scientific.to_lowercase(), scientific.to_string());
    }
    map
}

/// Normalize a label to canonical form
pub fn normalize_label(label: &str, canonical_map: &HashMap<String, String>) -> String {
    canonical_map
        .get(&label.to_lowercase())
        .cloned()
        .unwrap_or_else(|| label.to_string())
}

// =============================================================================
// Feature Dimension Constant
// =============================================================================

/// Feature dimension for acoustic specialists
pub const FEATURE_DIM: usize = 112;

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Phase 1: Routing Logic Tests (Red -> Green)
    // =========================================================================

    #[test]
    fn test_humpback_whale_maps_to_sonic_long_mammal() {
        // Humpback whales produce low-frequency moans (20-5000Hz, 0.5-5s)
        // Should map to SonicLongMammal, NOT Mysticete (taxonomic)
        assert_eq!(
            map_species_to_acoustic_group("Humpback Whale"),
            AcousticGroup::SonicLongMammal
        );
        assert_eq!(
            map_species_to_acoustic_group("Megaptera novaeangliae"),
            AcousticGroup::SonicLongMammal
        );
    }

    #[test]
    fn test_bat_maps_to_ultrasonic_mammal() {
        // Bats use ultrasonic FM sweeps (20-80kHz, 5-50ms)
        // Should map to UltrasonicMammal, NOT Mammal (taxonomic)
        assert_eq!(
            map_species_to_acoustic_group("Vespertilionidae"),
            AcousticGroup::UltrasonicMammal
        );
        assert_eq!(
            map_species_to_acoustic_group("Little Brown Bat"),
            AcousticGroup::UltrasonicMammal
        );
        assert_eq!(
            map_species_to_acoustic_group("Myotis lucifugus"),
            AcousticGroup::UltrasonicMammal
        );
    }

    #[test]
    fn test_dolphin_maps_to_marine_whistle() {
        // Dolphins produce FM whistles
        assert_eq!(
            map_species_to_acoustic_group("Bottlenose Dolphin"),
            AcousticGroup::MarineWhistle
        );
        assert_eq!(
            map_species_to_acoustic_group("Tursiops truncatus"),
            AcousticGroup::MarineWhistle
        );
        assert_eq!(
            map_species_to_acoustic_group("Common Dolphin"),
            AcousticGroup::MarineWhistle
        );
    }

    #[test]
    fn test_porpoise_maps_to_marine_click() {
        // Porpoises produce impulsive clicks
        assert_eq!(
            map_species_to_acoustic_group("Harbor Porpoise"),
            AcousticGroup::MarineClick
        );
        assert_eq!(
            map_species_to_acoustic_group("Phocoena phocoena"),
            AcousticGroup::MarineClick
        );
    }

    #[test]
    fn test_songbird_maps_to_bird_high_freq() {
        // Songbirds produce high-frequency modulated calls (4-8kHz)
        assert_eq!(
            map_species_to_acoustic_group("Zebra Finch"),
            AcousticGroup::BirdHighFreq
        );
        assert_eq!(
            map_species_to_acoustic_group("Song Sparrow"),
            AcousticGroup::BirdHighFreq
        );
        assert_eq!(
            map_species_to_acoustic_group("American Robin"),
            AcousticGroup::BirdHighFreq
        );
    }

    #[test]
    fn test_dove_maps_to_bird_low_freq() {
        // Doves produce low-frequency calls (200-1000Hz)
        assert_eq!(
            map_species_to_acoustic_group("Mourning Dove"),
            AcousticGroup::BirdLowFreq
        );
    }

    #[test]
    fn test_hummingbird_maps_to_bird_mechanical() {
        // Hummingbirds produce broadband mechanical sounds
        assert_eq!(
            map_species_to_acoustic_group("Ruby-throated Hummingbird"),
            AcousticGroup::BirdMechanical
        );
    }

    #[test]
    fn test_mosquito_maps_to_insect_wingbeat() {
        // Mosquitoes produce steady pure tones from wingbeats
        assert_eq!(
            map_species_to_acoustic_group("Aedes aegypti"),
            AcousticGroup::InsectWingbeat
        );
        assert_eq!(map_species_to_acoustic_group("Mosquito"), AcousticGroup::InsectWingbeat);
    }

    #[test]
    fn test_cricket_maps_to_insect_stridulation() {
        // Crickets produce broadband pulses via stridulation
        assert_eq!(
            map_species_to_acoustic_group("Field Cricket"),
            AcousticGroup::InsectStridulation
        );
        assert_eq!(
            map_species_to_acoustic_group("Cicada"),
            AcousticGroup::InsectStridulation
        );
    }

    #[test]
    fn test_frog_maps_to_amphibian() {
        // Frogs produce pulsed calls (500-5000Hz)
        assert_eq!(map_species_to_acoustic_group("Spring Peeper"), AcousticGroup::Amphibian);
        assert_eq!(map_species_to_acoustic_group("American Toad"), AcousticGroup::Amphibian);
    }

    #[test]
    fn test_seal_maps_to_pinniped() {
        // Seals produce varied calls (100-5000Hz)
        assert_eq!(map_species_to_acoustic_group("Harbor Seal"), AcousticGroup::Pinniped);
        assert_eq!(
            map_species_to_acoustic_group("California Sea Lion"),
            AcousticGroup::Pinniped
        );
    }

    #[test]
    fn test_primate_maps_to_sonic_short_mammal() {
        // Primates produce mid-frequency calls
        assert_eq!(
            map_species_to_acoustic_group("Common Marmoset"),
            AcousticGroup::SonicShortMammal
        );
        assert_eq!(map_species_to_acoustic_group("Gibbon"), AcousticGroup::SonicShortMammal);
    }

    #[test]
    fn test_acoustic_group_all_returns_13_groups() {
        let groups = AcousticGroup::all();
        assert_eq!(groups.len(), 13);
    }

    #[test]
    fn test_acoustic_group_filename_suffix() {
        assert_eq!(AcousticGroup::UltrasonicMammal.filename_suffix(), "ultrasonic_mammal");
        assert_eq!(AcousticGroup::BirdHighFreq.filename_suffix(), "bird_high_freq");
        assert_eq!(AcousticGroup::MarineWhistle.filename_suffix(), "marine_whistle");
    }

    #[test]
    fn test_acoustic_characteristics() {
        let chars = AcousticGroup::UltrasonicMammal.characteristics();
        assert_eq!(chars.freq_range_hz, (20000, 80000));
        assert!(chars.description.contains("bat"));
    }

    #[test]
    fn test_canonical_label_normalization() {
        let map = build_canonical_label_map();

        // Common name should map to scientific
        assert_eq!(normalize_label("Bottlenose Dolphin", &map), "Tursiops truncatus");
        assert_eq!(normalize_label("bottlenose dolphin", &map), "Tursiops truncatus");

        // Scientific name should stay as-is
        assert_eq!(normalize_label("Tursiops truncatus", &map), "Tursiops truncatus");

        // Unknown species should pass through
        assert_eq!(normalize_label("Unknown Species", &map), "Unknown Species");
    }

    #[test]
    fn test_sperm_whale_maps_to_marine_click() {
        // Sperm whales produce impulsive clicks for echolocation
        assert_eq!(map_species_to_acoustic_group("Sperm Whale"), AcousticGroup::MarineClick);
        assert_eq!(
            map_species_to_acoustic_group("Physeter macrocephalus"),
            AcousticGroup::MarineClick
        );
    }

    #[test]
    fn test_killer_whale_maps_to_marine_whistle() {
        // Killer whales produce FM whistles
        assert_eq!(
            map_species_to_acoustic_group("Killer Whale"),
            AcousticGroup::MarineWhistle
        );
        assert_eq!(
            map_species_to_acoustic_group("Orcinus orca"),
            AcousticGroup::MarineWhistle
        );
    }

    #[test]
    fn test_generic_whale_fallback() {
        // Generic "whale" (without "killer") should map to MarineMoan
        assert_eq!(
            map_species_to_acoustic_group("Some Unknown Whale"),
            AcousticGroup::MarineMoan
        );
    }
}
