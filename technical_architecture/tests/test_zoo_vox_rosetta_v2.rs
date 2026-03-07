// =============================================================================
// Zoo Vox Rosetta Engine v2.0 - TDD Test Suite
// =============================================================================
//
// Tests for the multi-modality species adaptation framework including:
// - SpectralModule (dolphin FM whistles)
// - SequenceModule (zebra finch n-gram syntax)
// - SpeciesConfigFactory (species-specific configurations)

#[cfg(test)]
mod test_spectral_module {
    use technical_architecture::spectral::{ContourConfig, FMType, SpectralModule};

    #[test]
    fn test_spectral_module_creation() {
        let config = ContourConfig {
            min_sweep_range: 500.0,
            min_duration_ms: 100.0,
            frequency_bins: 8,
            time_bins: 10,
        };

        let module = SpectralModule::new(config);

        assert_eq!(module.frequency_resolution(), 10.0);
        assert_eq!(module.time_resolution(), 1.0);
    }

    #[test]
    fn test_fm_type_classification_rising() {
        let config = ContourConfig::default();
        let module = SpectralModule::new(config);

        // Rising sweep: 5kHz to 15kHz
        let frequencies = vec![5000.0, 7500.0, 10000.0, 12500.0, 15000.0];
        let fm_type = module.classify_fm_type(&frequencies);

        assert_eq!(fm_type, FMType::Rising);
    }

    #[test]
    fn test_fm_type_classification_falling() {
        let config = ContourConfig::default();
        let module = SpectralModule::new(config);

        // Falling sweep: 15kHz to 5kHz
        let frequencies = vec![15000.0, 12500.0, 10000.0, 7500.0, 5000.0];
        let fm_type = module.classify_fm_type(&frequencies);

        assert_eq!(fm_type, FMType::Falling);
    }

    #[test]
    fn test_fm_type_classification_u_shaped() {
        let config = ContourConfig::default();
        let module = SpectralModule::new(config);

        // U-shaped: down then up
        let frequencies = vec![10000.0, 7500.0, 5000.0, 7500.0, 10000.0];
        let fm_type = module.classify_fm_type(&frequencies);

        assert_eq!(fm_type, FMType::UShaped);
    }

    #[test]
    fn test_fm_type_classification_inverted_u() {
        let config = ContourConfig::default();
        let module = SpectralModule::new(config);

        // Inverted U: up then down
        let frequencies = vec![5000.0, 10000.0, 15000.0, 10000.0, 5000.0];
        let fm_type = module.classify_fm_type(&frequencies);

        assert_eq!(fm_type, FMType::InvertedU);
    }

    #[test]
    fn test_fm_type_classification_complex() {
        let config = ContourConfig::default();
        let module = SpectralModule::new(config);

        // Complex: multiple inflections
        let frequencies = vec![5000.0, 10000.0, 5000.0, 15000.0, 8000.0];
        let fm_type = module.classify_fm_type(&frequencies);

        assert_eq!(fm_type, FMType::Complex);
    }

    #[test]
    fn test_fm_type_classification_flat() {
        let config = ContourConfig::default();
        let module = SpectralModule::new(config);

        // Flat: minimal variation
        let frequencies = vec![10000.0, 10050.0, 10020.0, 10080.0, 10040.0];
        let fm_type = module.classify_fm_type(&frequencies);

        assert_eq!(fm_type, FMType::Flat);
    }

    #[test]
    fn test_contour_discretization() {
        let config = ContourConfig {
            min_sweep_range: 500.0,
            min_duration_ms: 100.0,
            frequency_bins: 8,
            time_bins: 10,
        };
        let module = SpectralModule::new(config);

        // Frequency contour from 5kHz to 15kHz
        let frequencies: Vec<f64> = (0..10).map(|i| 5000.0 + i as f64 * 1000.0).collect();
        let signature = module.discretize_contour(&frequencies);

        assert_eq!(signature.len(), 10);
        // First bin should be low, last bin should be high
        assert!(signature[0] < signature[9]);
    }

    #[test]
    fn test_contour_features_extraction() {
        let config = ContourConfig::default();
        let module = SpectralModule::new(config);

        let frequencies = vec![5000.0, 7500.0, 10000.0, 12500.0, 15000.0];
        let features = module.extract_contour_features(&frequencies, 192000);

        assert_eq!(features.f_start, 5000.0);
        assert_eq!(features.f_end, 15000.0);
        assert_eq!(features.f_min, 5000.0);
        assert_eq!(features.f_max, 15000.0);
        assert_eq!(features.f_range, 10000.0);
        assert_eq!(features.inflections, 0);
        assert_eq!(features.fm_type, FMType::Rising);
    }

    #[test]
    fn test_analyze_dolphin_whistle() {
        let config = ContourConfig {
            min_sweep_range: 1000.0,
            min_duration_ms: 200.0,
            frequency_bins: 8,
            time_bins: 10,
        };
        let module = SpectralModule::new(config);

        // Generate synthetic FM whistle (rising sweep)
        let sample_rate = 192000u32;
        let duration_ms = 500.0;
        let n_samples = (sample_rate as f64 * duration_ms / 1000.0) as usize;

        let audio: Vec<f32> = (0..n_samples)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                // Frequency rises from 5kHz to 15kHz over 500ms
                let freq = 5000.0 + 10000.0 * (i as f64 / n_samples as f64);
                (2.0 * std::f64::consts::PI * freq * t).sin() as f32 * 0.5
            })
            .collect();

        let contours = module.analyze(&audio, sample_rate);

        assert!(!contours.is_empty(), "Should detect at least one contour");
    }
}

#[cfg(test)]
mod test_sequence_module {
    use technical_architecture::sequence::SequenceModule;

    #[test]
    fn test_sequence_module_creation() {
        let module = SequenceModule::new(3);

        assert_eq!(module.max_ngram_order(), 3);
        assert_eq!(module.min_occurrence(), 2);
    }

    #[test]
    fn test_find_motifs_basic() {
        let module = SequenceModule::new(3);

        // Sequence with repeated pattern [0, 1, 2]
        let sequence = vec![0, 1, 2, 3, 0, 1, 2, 4, 0, 1, 2];

        let motifs = module.find_motifs(&sequence);

        // Should find [0, 1, 2] pattern appearing 3 times
        assert!(motifs.iter().any(|m| m.pattern == vec![0, 1, 2] && m.occurrences >= 3));
    }

    #[test]
    fn test_find_motifs_length_variations() {
        let module = SequenceModule::new(5);

        // Sequence with patterns of different lengths
        let sequence = vec![0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 3];

        let motifs = module.find_motifs(&sequence);

        // Should find [0, 1] appearing multiple times
        assert!(!motifs.is_empty());
    }

    #[test]
    fn test_compute_bigram_stats() {
        let module = SequenceModule::new(3);

        let sequence = vec![0, 1, 0, 1, 2, 0, 1];

        let stats = module.compute_ngram_stats(&sequence);

        // Bigram (0, 1) appears 3 times
        assert!(stats.unique_bigrams > 0);
        assert_eq!(stats.most_common_bigram, (0, 1));
    }

    #[test]
    fn test_compute_trigram_stats() {
        let module = SequenceModule::new(3);

        // Sequence with clear trigram pattern
        let sequence = vec![0, 1, 2, 3, 0, 1, 2, 4, 0, 1, 2];

        let stats = module.compute_ngram_stats(&sequence);

        // Trigram (0, 1, 2) appears 3 times
        assert!(stats.unique_trigrams > 0);
        assert_eq!(stats.most_common_trigram, (0, 1, 2));
    }

    #[test]
    fn test_compute_perplexity_low() {
        let module = SequenceModule::new(3);

        // Repetitive sequence - low perplexity
        let sequence = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let perplexity = module.compute_perplexity(&sequence);

        // Low perplexity for repetitive sequence
        assert!(
            perplexity < 2.0,
            "Perplexity should be low for repetitive sequence, got {}",
            perplexity
        );
    }

    #[test]
    fn test_compute_perplexity_high() {
        let module = SequenceModule::new(3);

        // Diverse sequence - high perplexity
        let sequence = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        let perplexity = module.compute_perplexity(&sequence);

        // High perplexity for diverse sequence (should be ~log2(10) ≈ 3.32)
        assert!(
            perplexity > 3.0,
            "Perplexity should be high for diverse sequence, got {}",
            perplexity
        );
    }

    #[test]
    fn test_bigram_entropy() {
        let module = SequenceModule::new(3);

        // Uniform distribution of bigrams
        let sequence = vec![0, 1, 2, 3, 0, 1, 2, 3];

        let stats = module.compute_ngram_stats(&sequence);

        // Entropy should be positive
        assert!(stats.bigram_entropy > 0.0);
    }

    #[test]
    fn test_full_sequence_analysis() {
        let module = SequenceModule::new(3);

        // Zebra finch-like sequence
        let sequence = vec![0, 1, 2, 3, 0, 1, 2, 4, 5, 0, 1, 2, 3];

        let analysis = module.analyze(&sequence);

        assert_eq!(analysis.sequence.len(), 13);
        assert!(!analysis.motifs.is_empty());
        assert!(analysis.perplexity > 0.0);
    }

    #[test]
    fn test_transition_matrix() {
        let module = SequenceModule::new(3);

        let sequence = vec![0, 1, 0, 1, 2, 0, 1];

        let transitions = module.compute_transition_matrix(&sequence);

        // Type 0 should always transition to type 1 (probability 1.0)
        assert!(transitions.contains_key(&0));
        assert_eq!(transitions[&0].get(&1), Some(&1.0));
    }
}

#[cfg(test)]
mod test_species_config {
    use technical_architecture::species::{AnalysisModality, AnalysisModule, EncodingStrategy, SpeciesConfigFactory};

    #[test]
    fn test_sperm_whale_config() {
        let config = SpeciesConfigFactory::create("sperm_whale");

        assert_eq!(config.species(), "Sperm Whale");
        assert_eq!(config.encoding_strategy(), EncodingStrategy::CodaType);
        assert_eq!(config.modality(), AnalysisModality::Temporal);
        assert!(config.required_modules().contains(&AnalysisModule::Temporal));
        assert!(config.feature_params().similarity_threshold >= 0.75);
    }

    #[test]
    fn test_dolphin_config() {
        let config = SpeciesConfigFactory::create("dolphin");

        assert_eq!(config.species(), "Dolphin");
        assert_eq!(config.encoding_strategy(), EncodingStrategy::FrequencyModulated);
        assert_eq!(config.modality(), AnalysisModality::Spectral);
        assert!(config.required_modules().contains(&AnalysisModule::Spectral));
    }

    #[test]
    fn test_zebra_finch_config() {
        let config = SpeciesConfigFactory::create("zebra_finch");

        assert_eq!(config.species(), "Zebra Finch");
        assert_eq!(config.encoding_strategy(), EncodingStrategy::Combinatorial);
        assert_eq!(config.modality(), AnalysisModality::Temporal);
        assert!(config.required_modules().contains(&AnalysisModule::Sequence));
    }

    #[test]
    fn test_meerkat_config() {
        let config = SpeciesConfigFactory::create("meerkat");

        assert_eq!(config.species(), "Meerkat");
        assert_eq!(config.encoding_strategy(), EncodingStrategy::Quantitative);
        assert!(config.required_modules().contains(&AnalysisModule::Count));
    }

    #[test]
    fn test_bat_config() {
        let config = SpeciesConfigFactory::create("bat");

        assert_eq!(config.species(), "Egyptian Fruit Bat");
        assert_eq!(config.encoding_strategy(), EncodingStrategy::DurationMediated);
        assert!(config.required_modules().contains(&AnalysisModule::Duration));
    }

    #[test]
    fn test_orca_config() {
        let config = SpeciesConfigFactory::create("orca");

        assert_eq!(config.species(), "Orca");
        assert_eq!(config.modality(), AnalysisModality::Hybrid);
        assert!(config.required_modules().contains(&AnalysisModule::Sequence));
        assert!(config.required_modules().contains(&AnalysisModule::Spectral));
    }

    #[test]
    fn test_marmoset_config() {
        let config = SpeciesConfigFactory::create("marmoset");

        assert_eq!(config.species(), "Common Marmoset");
        assert_eq!(config.encoding_strategy(), EncodingStrategy::PhraseType);
    }

    #[test]
    fn test_unknown_species_defaults() {
        let config = SpeciesConfigFactory::create("unknown_species_xyz");

        assert_eq!(config.species(), "Unknown");
        assert_eq!(config.encoding_strategy(), EncodingStrategy::PhraseType);
        assert_eq!(config.modality(), AnalysisModality::Temporal);
    }

    #[test]
    fn test_case_insensitive_species_names() {
        let config1 = SpeciesConfigFactory::create("SPERM_WHALE");
        let config2 = SpeciesConfigFactory::create("Sperm_Whale");
        let config3 = SpeciesConfigFactory::create("sperm_whale");

        assert_eq!(config1.species(), config2.species());
        assert_eq!(config2.species(), config3.species());
    }

    #[test]
    fn test_context_labels_exist() {
        let sperm_whale = SpeciesConfigFactory::create("sperm_whale");
        assert!(!sperm_whale.context_labels().is_empty());

        let dolphin = SpeciesConfigFactory::create("dolphin");
        assert!(!dolphin.context_labels().is_empty());

        let zebra_finch = SpeciesConfigFactory::create("zebra_finch");
        assert!(!zebra_finch.context_labels().is_empty());
    }
}

#[cfg(test)]
mod test_integration {
    use technical_architecture::sequence::SequenceModule;
    use technical_architecture::species::AnalysisModality;
    use technical_architecture::species::AnalysisModule;
    use technical_architecture::species::SpeciesConfigFactory;
    use technical_architecture::spectral::{ContourConfig, SpectralModule};

    #[test]
    fn test_zebra_finch_pipeline() {
        // Get species config
        let config = SpeciesConfigFactory::create("zebra_finch");

        // Verify Sequence module is needed
        assert!(config.required_modules().contains(&AnalysisModule::Sequence));

        // Create sequence module
        let seq_module = SequenceModule::new(3);

        // Analyze a zebra finch-like sequence
        let sequence = vec![0, 1, 2, 3, 0, 1, 2, 4, 5, 0, 1, 2, 3, 6, 7];
        let analysis = seq_module.analyze(&sequence);

        // Verify analysis produces expected output
        assert!(
            !analysis.motifs.is_empty(),
            "Should find motifs in zebra finch sequence"
        );
        assert!(analysis.ngram_stats.unique_bigrams > 0);
    }

    #[test]
    fn test_dolphin_pipeline() {
        // Get species config
        let config = SpeciesConfigFactory::create("dolphin");

        // Verify Spectral module is needed
        assert!(config.required_modules().contains(&AnalysisModule::Spectral));

        // Create spectral module
        let spec_config = ContourConfig {
            min_sweep_range: 1000.0,
            min_duration_ms: 200.0,
            frequency_bins: 8,
            time_bins: 10,
        };
        let spec_module = SpectralModule::new(spec_config);

        // Generate synthetic FM whistle
        let sample_rate = 192000u32;
        let duration_ms = 500.0;
        let n_samples = (sample_rate as f64 * duration_ms / 1000.0) as usize;

        let audio: Vec<f32> = (0..n_samples)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                let freq = 5000.0 + 10000.0 * (i as f64 / n_samples as f64);
                (2.0 * std::f64::consts::PI * freq * t).sin() as f32 * 0.5
            })
            .collect();

        let contours = spec_module.analyze(&audio, sample_rate);

        // Verify contours detected
        assert!(!contours.is_empty(), "Should detect FM contours in dolphin whistle");
    }

    #[test]
    fn test_sperm_whale_temporal_pipeline() {
        // Get species config
        let config = SpeciesConfigFactory::create("sperm_whale");

        // Verify Temporal module is primary
        assert!(config.required_modules().contains(&AnalysisModule::Temporal));
        assert_eq!(config.modality(), AnalysisModality::Temporal);

        // Verify high similarity threshold for stereotyped codas
        assert!(config.feature_params().similarity_threshold >= 0.80);
    }
}

#[cfg(test)]
mod test_contour_features {
    use technical_architecture::spectral::{ContourConfig, SpectralModule};

    #[test]
    fn test_contour_features_slope() {
        let config = ContourConfig::default();
        let module = SpectralModule::new(config);

        // Rising from 5kHz to 10kHz over 100ms
        let frequencies: Vec<f64> = (0..10).map(|i| 5000.0 + i as f64 * 555.55).collect();
        let features = module.extract_contour_features(&frequencies, 192000);

        // Slope should be positive for rising sweep
        assert!(features.slope > 0.0);
    }

    #[test]
    fn test_contour_features_inflection_count() {
        let config = ContourConfig::default();
        let module = SpectralModule::new(config);

        // U-shaped contour has 1 inflection
        let frequencies = vec![10000.0, 7500.0, 5000.0, 7500.0, 10000.0];
        let features = module.extract_contour_features(&frequencies, 192000);

        assert_eq!(features.inflections, 1);
    }

    #[test]
    fn test_contour_duration_calculation() {
        let config = ContourConfig {
            min_sweep_range: 100.0,
            min_duration_ms: 10.0,
            frequency_bins: 8,
            time_bins: 100,
        };
        let module = SpectralModule::new(config);

        // 100 samples at 192kHz = ~0.52ms per sample
        let frequencies: Vec<f64> = (0..100).map(|i| 5000.0 + i as f64 * 50.0).collect();
        let features = module.extract_contour_features(&frequencies, 192000);

        assert!(features.duration_ms > 0.0);
    }
}
