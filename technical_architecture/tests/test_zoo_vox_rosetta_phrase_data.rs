// =============================================================================
// Zoo Vox Rosetta Engine v2.0 - Phrase Data Preparation Test Suite
// =============================================================================
//
// Tests for the 30D acoustic feature extraction, phrase segmentation,
// and library management modules.

#[cfg(test)]
mod test_acoustic_features_30d {
    use technical_architecture::zoo_vox_data_models::AcousticFeatures30D;

    #[test]
    fn test_features_creation() {
        let features = AcousticFeatures30D::new();
        assert_eq!(features.mean_f0_hz, 0.0);
        assert_eq!(features.duration_ms, 0.0);
    }

    #[test]
    fn test_features_to_vector() {
        let features = AcousticFeatures30D {
            mean_f0_hz: 6800.0,
            duration_ms: 65.0,
            f0_range_hz: 300.0,
            harmonic_to_noise_ratio: 15.0,
            spectral_flatness: 0.25,
            harmonicity: 0.75,
            ..Default::default()
        };

        let vec = features.to_vector();
        assert_eq!(vec.len(), 30);
        assert!((vec[0] - 6800.0).abs() < 1e-10);
        assert!((vec[1] - 65.0).abs() < 1e-10);
        assert!((vec[2] - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_features_from_vector() {
        let mut vec = [0.0; 30];
        vec[0] = 7000.0;
        vec[1] = 50.0;
        vec[13] = -500.0; // MFCC 1

        let features = AcousticFeatures30D::from_vector(vec);
        assert!((features.mean_f0_hz - 7000.0).abs() < 1e-10);
        assert!((features.duration_ms - 50.0).abs() < 1e-10);
        assert!((features.mfcc_1 - (-500.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let f1 = AcousticFeatures30D {
            mean_f0_hz: 6800.0,
            duration_ms: 65.0,
            ..Default::default()
        };

        let f2 = f1.clone();

        // Identical features should have similarity 1.0
        let sim = f1.cosine_similarity(&f2);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance() {
        let f1 = AcousticFeatures30D {
            mean_f0_hz: 6800.0,
            duration_ms: 65.0,
            ..Default::default()
        };

        let f2 = AcousticFeatures30D {
            mean_f0_hz: 7000.0,
            duration_ms: 65.0,
            ..Default::default()
        };

        let dist = f1.distance(&f2);
        assert!(dist > 0.0);
    }
}

#[cfg(test)]
mod test_phrase_prototype {
    use technical_architecture::zoo_vox_data_models::{ContextAssociation, PhrasePrototype};

    #[test]
    fn test_phrase_creation() {
        let phrase = PhrasePrototype::new("marmoset_001", "F0_6800_DUR_65", "marmoset");
        assert_eq!(phrase.phrase_id, "marmoset_001");
        assert_eq!(phrase.phrase_key, "F0_6800_DUR_65");
        assert_eq!(phrase.species, "marmoset");
    }

    #[test]
    fn test_phrase_generate_key() {
        let phrase = PhrasePrototype::new("test_001", "F0_0_DUR_0", "marmoset");
        let mut phrase = phrase;
        phrase.features_30d.mean_f0_hz = 6800.0;
        phrase.features_30d.duration_ms = 65.0;

        let key = phrase.generate_key(200.0, 10.0);
        assert!(key.starts_with("F0_"));
        assert!(key.contains("_DUR_"));
    }

    #[test]
    fn test_context_association() {
        let ctx = ContextAssociation::new("alarm", "defensive");
        assert_eq!(ctx.context_label, "alarm");
        assert_eq!(ctx.context_category, "defensive");
    }
}

#[cfg(test)]
mod test_species_phrase_library {
    use technical_architecture::zoo_vox_data_models::{PhrasePrototype, SpeciesPhraseLibrary};

    #[test]
    fn test_library_creation() {
        let library = SpeciesPhraseLibrary::new("marmoset");
        assert_eq!(library.species, "marmoset");
        assert_eq!(library.total_phrases, 0);
    }

    #[test]
    fn test_library_add_phrase() {
        let mut library = SpeciesPhraseLibrary::new("marmoset");

        let phrase = PhrasePrototype::new("marmoset_001", "F0_6800_DUR_65", "marmoset");
        library.add_phrase(phrase);

        assert_eq!(library.total_phrases, 1);
    }

    #[test]
    fn test_library_statistics() {
        let mut library = SpeciesPhraseLibrary::new("marmoset");

        let mut phrase1 = PhrasePrototype::new("marmoset_001", "F0_6800_DUR_65", "marmoset");
        phrase1.occurrence_count = 100;
        phrase1.primary_context = Some("contact".to_string());

        let mut phrase2 = PhrasePrototype::new("marmoset_002", "F0_7200_DUR_80", "marmoset");
        phrase2.occurrence_count = 50;
        phrase2.primary_context = Some("alarm".to_string());

        library.add_phrase(phrase1);
        library.add_phrase(phrase2);
        library.recalculate_statistics();

        assert_eq!(library.total_occurrences, 150);
        assert!(library.type_entropy > 0.0);
    }
}

#[cfg(test)]
mod test_cross_species_database {
    use technical_architecture::zoo_vox_data_models::{CrossSpeciesPhraseDatabase, SpeciesPhraseLibrary};

    #[test]
    fn test_database_creation() {
        let database = CrossSpeciesPhraseDatabase::new();
        assert_eq!(database.database_version, "2.0");
        assert_eq!(database.species_libraries.len(), 0);
    }

    #[test]
    fn test_database_add_library() {
        let mut database = CrossSpeciesPhraseDatabase::new();
        let library = SpeciesPhraseLibrary::new("marmoset");

        database.add_library(library);

        assert_eq!(database.species_libraries.len(), 1);
        assert!(database.total_phrases() == 0);
    }
}

#[cfg(test)]
mod test_feature_extractor {
    use technical_architecture::zoo_vox_features::ZooVoxFeatureExtractor;

    #[test]
    fn test_extractor_creation() {
        let extractor = ZooVoxFeatureExtractor::new(48000);
        assert_eq!(extractor.sample_rate(), 48000);
    }

    #[test]
    fn test_extract_empty_audio() {
        let mut extractor = ZooVoxFeatureExtractor::new(48000);
        let result = extractor.extract(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_sine_wave() {
        let mut extractor = ZooVoxFeatureExtractor::new(48000);

        // Generate 440 Hz sine wave
        let sample_rate = 48000.0;
        let frequency = 440.0;
        let duration = 0.5; // 500ms
        let n_samples = (sample_rate * duration) as usize;

        let audio: Vec<f64> = (0..n_samples)
            .map(|i| (2.0 * std::f64::consts::PI * frequency * i as f64 / sample_rate).sin() * 0.5)
            .collect();

        let features = extractor.extract(&audio).unwrap();

        // Check duration
        assert!((features.duration_ms - 500.0).abs() < 10.0);

        // Check F0 is approximately correct (within 20%)
        let f0_ratio = features.mean_f0_hz / frequency;
        assert!(
            f0_ratio > 0.8 && f0_ratio < 1.2,
            "F0 estimate: {} Hz, expected: {} Hz",
            features.mean_f0_hz,
            frequency
        );
    }
}

#[cfg(test)]
mod test_phrase_extractor {
    use technical_architecture::zoo_vox_extraction::{ZooVoxExtractionConfig, ZooVoxPhraseExtractor};

    #[test]
    fn test_extraction_config_default() {
        let config = ZooVoxExtractionConfig::default();
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.min_phrase_duration_ms, 10.0);
    }

    #[test]
    fn test_extractor_creation() {
        let config = ZooVoxExtractionConfig::new(44100);
        let extractor = ZooVoxPhraseExtractor::new(config);
        assert_eq!(extractor.sample_rate(), 44100);
    }

    #[test]
    fn test_extractor_for_species() {
        let extractor = ZooVoxPhraseExtractor::for_species("sperm_whale", 48000);
        assert_eq!(extractor.sample_rate(), 48000);
    }

    #[test]
    fn test_extract_phrases_sine_wave() {
        let config = ZooVoxExtractionConfig::new(48000);
        let mut extractor = ZooVoxPhraseExtractor::new(config);

        // Generate 1 second of 1000 Hz sine wave
        let audio: Vec<f64> = (0..48000)
            .map(|i| (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / 48000.0).sin() * 0.5)
            .collect();

        let phrases = extractor.extract_phrases(&audio, "marmoset", None).unwrap();

        // Should extract at least one phrase
        assert!(!phrases.is_empty());
    }
}

#[cfg(test)]
mod test_library_builder {
    use technical_architecture::zoo_vox_data_models::PhrasePrototype;
    use technical_architecture::zoo_vox_library::ZooVoxLibraryBuilder;

    #[test]
    fn test_builder_creation() {
        let builder = ZooVoxLibraryBuilder::new();
        assert_eq!(builder.similarity_threshold(), 0.85);
    }

    #[test]
    fn test_build_empty_library() {
        let builder = ZooVoxLibraryBuilder::new();
        let library = builder.build_library(Vec::new(), "marmoset", None).unwrap();

        assert_eq!(library.species, "marmoset");
        assert_eq!(library.total_phrases, 0);
    }

    #[test]
    fn test_build_library_with_phrases() {
        let builder = ZooVoxLibraryBuilder::new();

        let mut phrase = PhrasePrototype::new("marmoset_001", "F0_6800_DUR_65", "marmoset");
        phrase.occurrence_count = 100;
        phrase.primary_context = Some("contact".to_string());

        let library = builder.build_library(vec![phrase], "marmoset", None).unwrap();

        assert_eq!(library.total_phrases, 1);
        assert_eq!(library.total_occurrences, 100);
        assert!(library.context_labels.contains(&"contact".to_string()));
    }
}

#[cfg(test)]
mod test_integration {
    use technical_architecture::species::SpeciesConfigFactory;
    use technical_architecture::{
        ZooVoxExtractionConfig, ZooVoxFeatureExtractor, ZooVoxLibraryBuilder, ZooVoxPhraseExtractor,
    };

    #[test]
    fn test_full_pipeline() {
        // Generate synthetic audio (1 second of 6800 Hz marmoset-like call)
        let sample_rate = 48000;
        let audio: Vec<f64> = (0..sample_rate)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                // AM-modulated tone to simulate marmoset call
                let carrier = (2.0 * std::f64::consts::PI * 6800.0 * t).sin();
                let modulation = 0.3 * (2.0 * std::f64::consts::PI * 15.0 * t).sin() + 0.7;
                carrier * modulation * 0.5
            })
            .collect();

        // Extract 30D features
        let mut feature_extractor = ZooVoxFeatureExtractor::new(sample_rate as u32);
        let features = feature_extractor.extract(&audio).unwrap();

        assert!(features.mean_f0_hz > 0.0);
        assert!((features.duration_ms - 1000.0).abs() < 10.0);

        // Extract phrases
        let config = ZooVoxExtractionConfig::for_species("marmoset", sample_rate as u32);
        let mut phrase_extractor = ZooVoxPhraseExtractor::new(config);
        let phrases = phrase_extractor.extract_phrases(&audio, "marmoset", None).unwrap();

        assert!(!phrases.is_empty());

        // Build library
        let builder = ZooVoxLibraryBuilder::new();
        let library = builder.build_library(phrases, "marmoset", None).unwrap();

        assert!(library.total_phrases > 0);
    }

    #[test]
    fn test_species_config_integration() {
        let config = SpeciesConfigFactory::create("marmoset");

        assert_eq!(config.species(), "Common Marmoset");
        assert!(!config.context_labels().is_empty());
    }
}
