//! Hierarchical Veto Ensemble for Bioacoustic Classification
//! ==========================================================
//!
//! Implements a two-stage classification system that mimics biological taxonomy:
//!
//! 1. **Taxonomy Gatekeeper (RF on 46D Physics)**: Predicts broad taxonomic group
//!    - Uses "Gross Physics" features (Duration, Pitch, Tempo)
//!    - High accuracy for broad groups (96.9% for Mammals)
//!
//! 2. **Species Expert (NN on 66D Texture)**: Predicts Top-5 species candidates
//!    - Uses "Fine Texture" features (Spectral shape, Modulation)
//!    - Distinguishes similar species within the same taxonomic group
//!
//! 3. **Veto Mechanism**: Ensures NN respects RF's taxonomic decision
//!    - Eliminates "cross-clade errors" (e.g., Bird confused for Insect)
//!    - Selects first Top-5 candidate that matches RF's taxonomy
//!
//! # Scientific Justification
//!
//! This approach is biologically valid because:
//! - Broad classification requires only physics features
//! - Fine classification requires texture features
//! - Forcing taxonomic consistency eliminates embarrassing errors

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::taxonomic_router::{Taxon, map_species_to_taxon, PHYSICS_DIM, TEXTURE_DIM};

// =============================================================================
// Core Data Structures
// =============================================================================

/// Represents a species prediction with confidence score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesPrediction {
    /// Species label (e.g., "Eastern Towhee")
    pub label: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Taxonomic group of this species
    pub taxon: Taxon,
}

/// Represents a taxonomic prediction from the Gatekeeper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxonPrediction {
    /// Predicted taxonomic group
    pub taxon: Taxon,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
}

/// Result of the Hierarchical Veto Ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VetoResult {
    /// Final species prediction after veto
    pub species: String,
    /// Confidence of the final prediction
    pub confidence: f32,
    /// Taxonomic group of the prediction
    pub taxon: Taxon,
    /// Gatekeeper's taxonomic prediction
    pub gatekeeper_taxon: Taxon,
    /// Gatekeeper's confidence
    pub gatekeeper_confidence: f32,
    /// Index of the selected candidate (0 = first choice accepted)
    pub selected_rank: usize,
    /// Total candidates considered
    pub total_candidates: usize,
    /// Whether veto was applied (first candidate rejected)
    pub veto_applied: bool,
}

/// Configuration for the Veto Mechanism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VetoConfig {
    /// Minimum confidence for gatekeeper to override NN
    pub gatekeeper_min_confidence: f32,
    /// Maximum candidates to consider from NN
    pub max_candidates: usize,
    /// Taxon to use when gatekeeper is uncertain
    pub fallback_taxon: Taxon,
}

impl Default for VetoConfig {
    fn default() -> Self {
        Self {
            gatekeeper_min_confidence: 0.5,
            max_candidates: 5,
            fallback_taxon: Taxon::Unknown,
        }
    }
}

// =============================================================================
// Veto Mechanism Implementation
// =============================================================================

/// The Hierarchical Veto Ensemble classifier
pub struct HierarchicalVetoEnsemble {
    /// Configuration
    pub config: VetoConfig,
}

impl HierarchicalVetoEnsemble {
    /// Create a new ensemble with default configuration
    pub fn new() -> Self {
        Self {
            config: VetoConfig::default(),
        }
    }

    /// Create a new ensemble with custom configuration
    pub fn with_config(config: VetoConfig) -> Self {
        Self { config }
    }

    /// Apply the veto mechanism to select the best species prediction
    ///
    /// # Algorithm
    /// 1. Check if gatekeeper confidence meets minimum threshold
    /// 2. Iterate through Top-N candidates from NN
    /// 3. Select first candidate whose taxonomy matches gatekeeper
    /// 4. If no match found, fall back to first NN candidate
    pub fn apply_veto(
        &self,
        gatekeeper_pred: &TaxonPrediction,
        nn_candidates: &[SpeciesPrediction],
    ) -> VetoResult {
        let max_candidates = self.config.max_candidates.min(nn_candidates.len());
        
        // Check if gatekeeper is confident enough
        let effective_taxon = if gatekeeper_pred.confidence >= self.config.gatekeeper_min_confidence {
            gatekeeper_pred.taxon
        } else {
            // Gatekeeper uncertain - use fallback or first candidate's taxon
            if nn_candidates.is_empty() {
                self.config.fallback_taxon
            } else {
                nn_candidates[0].taxon
            }
        };

        // If no candidates, return empty result
        if nn_candidates.is_empty() {
            return VetoResult {
                species: "<unknown>".to_string(),
                confidence: 0.0,
                taxon: effective_taxon,
                gatekeeper_taxon: gatekeeper_pred.taxon,
                gatekeeper_confidence: gatekeeper_pred.confidence,
                selected_rank: 0,
                total_candidates: 0,
                veto_applied: false,
            };
        }

        // Search for first matching candidate
        for (rank, candidate) in nn_candidates.iter().take(max_candidates).enumerate() {
            if candidate.taxon == effective_taxon {
                return VetoResult {
                    species: candidate.label.clone(),
                    confidence: candidate.confidence,
                    taxon: candidate.taxon,
                    gatekeeper_taxon: gatekeeper_pred.taxon,
                    gatekeeper_confidence: gatekeeper_pred.confidence,
                    selected_rank: rank,
                    total_candidates: max_candidates,
                    veto_applied: rank > 0,
                };
            }
        }

        // No match found - fall back to first candidate with penalty
        let first = &nn_candidates[0];
        VetoResult {
            species: first.label.clone(),
            confidence: first.confidence * 0.5, // Penalize mismatched prediction
            taxon: first.taxon,
            gatekeeper_taxon: gatekeeper_pred.taxon,
            gatekeeper_confidence: gatekeeper_pred.confidence,
            selected_rank: 0,
            total_candidates: max_candidates,
            veto_applied: false, // No veto, but taxonomic mismatch
        }
    }

    /// Create species predictions with taxonomic labels
    pub fn create_species_predictions(
        labels: &[String],
        confidences: &[f32],
    ) -> Vec<SpeciesPrediction> {
        labels.iter()
            .zip(confidences.iter())
            .map(|(label, &conf)| SpeciesPrediction {
                label: label.clone(),
                confidence: conf,
                taxon: map_species_to_taxon(label),
            })
            .collect()
    }

    /// Apply taxonomic masking at inference time (Separation of Concerns)
    ///
    /// This function implements the Router's masking logic:
    /// - Takes raw logits from NN (all species)
    /// - Takes predicted taxonomy from RF Gatekeeper
    /// - Zeros out logits for species outside the predicted taxonomy
    /// - Returns masked logits for Top-5 selection
    ///
    /// # Arguments
    /// * `raw_logits` - Raw logits from NN (one per species)
    /// * `species_labels` - Labels corresponding to each logit
    /// * `predicted_taxon` - Taxonomy predicted by RF Gatekeeper
    /// * `label_to_taxon` - Mapping from species label to taxonomic group
    ///
    /// # Returns
    /// * Masked logits with non-taxonomy logits set to -inf
    pub fn apply_inference_mask(
        raw_logits: &[f32],
        species_labels: &[String],
        predicted_taxon: Taxon,
        label_to_taxon: &HashMap<String, Taxon>,
    ) -> Vec<f32> {
        raw_logits.iter()
            .enumerate()
            .map(|(i, &logit)| {
                let species_taxon = label_to_taxon
                    .get(&species_labels[i])
                    .copied()
                    .unwrap_or(Taxon::Unknown);

                // If taxonomy matches, keep logit; otherwise, set to -inf
                if species_taxon == predicted_taxon || predicted_taxon == Taxon::Unknown {
                    logit
                } else {
                    f32::NEG_INFINITY // Mask out non-taxonomy species
                }
            })
            .collect()
    }

    /// Get Top-N candidates from masked logits (Separation of Concerns)
    ///
    /// This implements the inference pipeline:
    /// 1. Apply taxonomic mask to raw logits
    /// 2. Convert to softmax probabilities
    /// 3. Return top-N species with confidence scores
    pub fn get_top_n_candidates(
        raw_logits: &[f32],
        species_labels: &[String],
        predicted_taxon: Taxon,
        label_to_taxon: &HashMap<String, Taxon>,
        n: usize,
    ) -> Vec<SpeciesPrediction> {
        // Apply inference mask
        let masked_logits = Self::apply_inference_mask(
            raw_logits,
            species_labels,
            predicted_taxon,
            label_to_taxon,
        );

        // Convert to softmax probabilities (numerically stable)
        let max_logit = masked_logits.iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let exp_sum: f32 = masked_logits.iter()
            .map(|&x| (x - max_logit).exp())
            .sum();

        let probs: Vec<f32> = masked_logits.iter()
            .map(|&x| (x - max_logit).exp() / exp_sum)
            .collect();

        // Get indices sorted by probability (descending), filtering zeros
        let mut indexed: Vec<(usize, f32)> = probs.iter()
            .cloned()
            .enumerate()
            .filter(|(_, prob)| *prob > 1e-10) // Filter out zero probabilities
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-N candidates
        indexed.iter()
            .take(n)
            .map(|(idx, prob)| SpeciesPrediction {
                label: species_labels[*idx].clone(),
                confidence: *prob,
                taxon: label_to_taxon
                    .get(&species_labels[*idx])
                    .copied()
                    .unwrap_or(Taxon::Unknown),
            })
            .collect()
    }

    /// Calculate accuracy metrics for a batch of predictions
    pub fn evaluate_batch(results: &[VetoResult], true_labels: &[String]) -> VetoMetrics {
        let mut correct = 0usize;
        let mut veto_correct = 0usize;
        let mut veto_total = 0usize;
        let mut taxonomic_correct = 0usize;
        let mut total = results.len();

        for (result, true_label) in results.iter().zip(true_labels.iter()) {
            if result.species == *true_label {
                correct += 1;
            }

            // Check taxonomic accuracy
            let true_taxon = map_species_to_taxon(true_label);
            if result.taxon == true_taxon {
                taxonomic_correct += 1;
            }

            // Track veto effectiveness
            if result.veto_applied {
                veto_total += 1;
                if result.species == *true_label {
                    veto_correct += 1;
                }
            }
        }

        VetoMetrics {
            species_accuracy: if total > 0 { correct as f32 / total as f32 * 100.0 } else { 0.0 },
            taxonomic_accuracy: if total > 0 { taxonomic_correct as f32 / total as f32 * 100.0 } else { 0.0 },
            total_samples: total,
            correct_predictions: correct,
            veto_applications: veto_total,
            veto_improvements: veto_correct,
        }
    }
}

impl Default for HierarchicalVetoEnsemble {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Metrics
// =============================================================================

/// Metrics for evaluating the Hierarchical Veto Ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VetoMetrics {
    /// Species classification accuracy (%)
    pub species_accuracy: f32,
    /// Taxonomic group accuracy (%)
    pub taxonomic_accuracy: f32,
    /// Total number of samples
    pub total_samples: usize,
    /// Number of correct species predictions
    pub correct_predictions: usize,
    /// Number of times veto was applied
    pub veto_applications: usize,
    /// Number of times veto improved the prediction
    pub veto_improvements: usize,
}

impl VetoMetrics {
    /// Calculate veto effectiveness (how often veto improves accuracy)
    pub fn veto_effectiveness(&self) -> f32 {
        if self.veto_applications > 0 {
            self.veto_improvements as f32 / self.veto_applications as f32 * 100.0
        } else {
            0.0
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TDD Test Suite: Hierarchical Veto Ensemble
    // =========================================================================

    #[test]
    fn test_veto_ensemble_creation() {
        let ensemble = HierarchicalVetoEnsemble::new();
        assert_eq!(ensemble.config.gatekeeper_min_confidence, 0.5);
        assert_eq!(ensemble.config.max_candidates, 5);
    }

    #[test]
    fn test_veto_config_default() {
        let config = VetoConfig::default();
        assert_eq!(config.gatekeeper_min_confidence, 0.5);
        assert_eq!(config.max_candidates, 5);
        assert_eq!(config.fallback_taxon, Taxon::Unknown);
    }

    #[test]
    fn test_species_prediction_creation() {
        let pred = SpeciesPrediction {
            label: "Eastern Towhee".to_string(),
            confidence: 0.85,
            taxon: Taxon::Songbird,
        };
        assert_eq!(pred.label, "Eastern Towhee");
        assert!((pred.confidence - 0.85).abs() < 1e-6);
        assert_eq!(pred.taxon, Taxon::Songbird);
    }

    #[test]
    fn test_taxon_prediction_creation() {
        let pred = TaxonPrediction {
            taxon: Taxon::Mammal,
            confidence: 0.969,
        };
        assert_eq!(pred.taxon, Taxon::Mammal);
        assert!((pred.confidence - 0.969).abs() < 1e-6);
    }

    #[test]
    fn test_veto_first_candidate_matches() {
        let ensemble = HierarchicalVetoEnsemble::new();
        
        let gatekeeper = TaxonPrediction {
            taxon: Taxon::Mammal,
            confidence: 0.97,
        };
        
        let candidates = vec![
            SpeciesPrediction {
                label: "Lar Gibbon".to_string(),
                confidence: 0.85,
                taxon: Taxon::Mammal,
            },
            SpeciesPrediction {
                label: "Eastern Towhee".to_string(),
                confidence: 0.10,
                taxon: Taxon::Songbird,
            },
        ];
        
        let result = ensemble.apply_veto(&gatekeeper, &candidates);
        
        assert_eq!(result.species, "Lar Gibbon");
        assert!((result.confidence - 0.85).abs() < 1e-6);
        assert_eq!(result.taxon, Taxon::Mammal);
        assert_eq!(result.selected_rank, 0);
        assert!(!result.veto_applied);
    }

    #[test]
    fn test_veto_first_candidate_rejected() {
        let ensemble = HierarchicalVetoEnsemble::new();
        
        let gatekeeper = TaxonPrediction {
            taxon: Taxon::Mammal,
            confidence: 0.97,
        };
        
        // NN's first choice is a bird (wrong clade), second is mammal (correct)
        let candidates = vec![
            SpeciesPrediction {
                label: "Eastern Towhee".to_string(),
                confidence: 0.45,
                taxon: Taxon::Songbird,
            },
            SpeciesPrediction {
                label: "Lar Gibbon".to_string(),
                confidence: 0.40,
                taxon: Taxon::Mammal,
            },
            SpeciesPrediction {
                label: "Humpback Whale".to_string(),
                confidence: 0.15,
                taxon: Taxon::Mysticete,
            },
        ];
        
        let result = ensemble.apply_veto(&gatekeeper, &candidates);
        
        // First candidate rejected, second accepted
        assert_eq!(result.species, "Lar Gibbon");
        assert_eq!(result.taxon, Taxon::Mammal);
        assert_eq!(result.selected_rank, 1);
        assert!(result.veto_applied);
    }

    #[test]
    fn test_veto_no_matching_candidate() {
        let ensemble = HierarchicalVetoEnsemble::new();
        
        let gatekeeper = TaxonPrediction {
            taxon: Taxon::Cetacean,
            confidence: 0.90,
        };
        
        // No cetaceans in candidates
        let candidates = vec![
            SpeciesPrediction {
                label: "Eastern Towhee".to_string(),
                confidence: 0.50,
                taxon: Taxon::Songbird,
            },
            SpeciesPrediction {
                label: "Lar Gibbon".to_string(),
                confidence: 0.30,
                taxon: Taxon::Mammal,
            },
        ];
        
        let result = ensemble.apply_veto(&gatekeeper, &candidates);
        
        // Falls back to first candidate with penalty
        assert_eq!(result.species, "Eastern Towhee");
        assert!((result.confidence - 0.25).abs() < 1e-6); // 0.50 * 0.5 penalty
        assert!(!result.veto_applied);
    }

    #[test]
    fn test_veto_empty_candidates() {
        let ensemble = HierarchicalVetoEnsemble::new();
        
        let gatekeeper = TaxonPrediction {
            taxon: Taxon::Mammal,
            confidence: 0.97,
        };
        
        let candidates: Vec<SpeciesPrediction> = vec![];
        let result = ensemble.apply_veto(&gatekeeper, &candidates);
        
        assert_eq!(result.species, "<unknown>");
        assert_eq!(result.confidence, 0.0);
        assert_eq!(result.total_candidates, 0);
    }

    #[test]
    fn test_veto_low_gatekeeper_confidence() {
        let config = VetoConfig {
            gatekeeper_min_confidence: 0.7,
            ..Default::default()
        };
        let ensemble = HierarchicalVetoEnsemble::with_config(config);
        
        let gatekeeper = TaxonPrediction {
            taxon: Taxon::Mammal,
            confidence: 0.50, // Below threshold
        };
        
        let candidates = vec![
            SpeciesPrediction {
                label: "Eastern Towhee".to_string(),
                confidence: 0.80,
                taxon: Taxon::Songbird,
            },
        ];
        
        let result = ensemble.apply_veto(&gatekeeper, &candidates);
        
        // Gatekeeper uncertain, so NN's first choice is accepted regardless of taxonomy
        assert_eq!(result.species, "Eastern Towhee");
        assert_eq!(result.taxon, Taxon::Songbird); // Uses NN's taxon, not gatekeeper's
    }

    #[test]
    fn test_create_species_predictions() {
        let labels = vec![
            "Eastern Towhee".to_string(),
            "Lar Gibbon".to_string(),
            "Humpback Whale".to_string(),
        ];
        let confidences = vec![0.5, 0.3, 0.2];
        
        let predictions = HierarchicalVetoEnsemble::create_species_predictions(
            &labels, &confidences
        );
        
        assert_eq!(predictions.len(), 3);
        assert_eq!(predictions[0].taxon, Taxon::Songbird);
        assert_eq!(predictions[1].taxon, Taxon::Mammal);
        assert_eq!(predictions[2].taxon, Taxon::Mysticete);
    }

    #[test]
    fn test_evaluate_batch() {
        let ensemble = HierarchicalVetoEnsemble::new();
        
        let results = vec![
            VetoResult {
                species: "Lar Gibbon".to_string(),
                confidence: 0.85,
                taxon: Taxon::Mammal,
                gatekeeper_taxon: Taxon::Mammal,
                gatekeeper_confidence: 0.97,
                selected_rank: 0,
                total_candidates: 3,
                veto_applied: false,
            },
            VetoResult {
                species: "Lar Gibbon".to_string(),
                confidence: 0.40,
                taxon: Taxon::Mammal,
                gatekeeper_taxon: Taxon::Mammal,
                gatekeeper_confidence: 0.97,
                selected_rank: 1, // Veto applied
                total_candidates: 3,
                veto_applied: true,
            },
        ];
        
        let true_labels = vec![
            "Lar Gibbon".to_string(),
            "Lar Gibbon".to_string(),
        ];
        
        let metrics = HierarchicalVetoEnsemble::evaluate_batch(&results, &true_labels);
        
        assert_eq!(metrics.total_samples, 2);
        assert_eq!(metrics.correct_predictions, 2);
        assert!((metrics.species_accuracy - 100.0).abs() < 1e-6);
        assert_eq!(metrics.veto_applications, 1);
        assert_eq!(metrics.veto_improvements, 1);
    }

    #[test]
    fn test_veto_metrics_effectiveness() {
        let metrics = VetoMetrics {
            species_accuracy: 65.0,
            taxonomic_accuracy: 95.0,
            total_samples: 100,
            correct_predictions: 65,
            veto_applications: 20,
            veto_improvements: 15,
        };
        
        // 15 improvements out of 20 applications = 75%
        assert!((metrics.veto_effectiveness() - 75.0).abs() < 1e-6);
    }

    #[test]
    fn test_veto_cross_clade_error_prevention() {
        // This is the key test: NN confuses bird for insect, RF prevents it
        let ensemble = HierarchicalVetoEnsemble::new();
        
        let gatekeeper = TaxonPrediction {
            taxon: Taxon::Insect,
            confidence: 0.92,
        };
        
        // NN thinks it's a bird (cross-clade error), but also has insect candidates
        let candidates = vec![
            SpeciesPrediction {
                label: "Eastern Towhee".to_string(), // Wrong!
                confidence: 0.55,
                taxon: Taxon::Songbird,
            },
            SpeciesPrediction {
                label: "Cricket".to_string(), // Correct!
                confidence: 0.35,
                taxon: Taxon::Insect,
            },
            SpeciesPrediction {
                label: "Tree Frog".to_string(),
                confidence: 0.10,
                taxon: Taxon::Amphibian,
            },
        ];
        
        let result = ensemble.apply_veto(&gatekeeper, &candidates);
        
        // Veto prevents the embarrassing cross-clade error
        assert_eq!(result.species, "Cricket");
        assert_eq!(result.taxon, Taxon::Insect);
        assert!(result.veto_applied);
        assert_eq!(result.selected_rank, 1);
    }

    #[test]
    fn test_veto_respects_max_candidates() {
        let config = VetoConfig {
            max_candidates: 3,
            ..Default::default()
        };
        let ensemble = HierarchicalVetoEnsemble::with_config(config);
        
        let gatekeeper = TaxonPrediction {
            taxon: Taxon::Mammal,
            confidence: 0.90,
        };
        
        // 5 candidates, but we only consider top 3
        let candidates = vec![
            SpeciesPrediction {
                label: "Bird1".to_string(),
                confidence: 0.30,
                taxon: Taxon::Songbird,
            },
            SpeciesPrediction {
                label: "Bird2".to_string(),
                confidence: 0.25,
                taxon: Taxon::Songbird,
            },
            SpeciesPrediction {
                label: "Bird3".to_string(),
                confidence: 0.20,
                taxon: Taxon::Songbird,
            },
            SpeciesPrediction {
                label: "Gibbon".to_string(), // 4th place - not considered!
                confidence: 0.15,
                taxon: Taxon::Mammal,
            },
            SpeciesPrediction {
                label: "Bat".to_string(),
                confidence: 0.10,
                taxon: Taxon::Mammal,
            },
        ];
        
        let result = ensemble.apply_veto(&gatekeeper, &candidates);

        // No mammal in top 3, falls back to first with penalty
        assert_eq!(result.species, "Bird1");
        assert_eq!(result.total_candidates, 3); // Only 3 considered
    }

    // =========================================================================
    // TDD Tests: Inference-Time Masking (Separation of Concerns)
    // =========================================================================

    #[test]
    fn test_apply_inference_mask_basic() {
        let raw_logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let species_labels = vec![
            "Gibbon".to_string(),      // Mammal
            "Towhee".to_string(),      // Songbird
            "Bat".to_string(),         // Mammal
            "Sparrow".to_string(),     // Songbird
            "Whale".to_string(),       // Mysticete
        ];

        let mut label_to_taxon = HashMap::new();
        label_to_taxon.insert("Gibbon".to_string(), Taxon::Mammal);
        label_to_taxon.insert("Towhee".to_string(), Taxon::Songbird);
        label_to_taxon.insert("Bat".to_string(), Taxon::Mammal);
        label_to_taxon.insert("Sparrow".to_string(), Taxon::Songbird);
        label_to_taxon.insert("Whale".to_string(), Taxon::Mysticete);

        // Mask for Mammal - should keep only indices 0 and 2
        let masked = HierarchicalVetoEnsemble::apply_inference_mask(
            &raw_logits,
            &species_labels,
            Taxon::Mammal,
            &label_to_taxon,
        );

        assert!((masked[0] - 1.0).abs() < 1e-6); // Mammal - kept
        assert!(masked[1].is_infinite() && masked[1].is_sign_negative()); // Songbird - masked
        assert!((masked[2] - 3.0).abs() < 1e-6); // Mammal - kept
        assert!(masked[3].is_infinite() && masked[3].is_sign_negative()); // Songbird - masked
        assert!(masked[4].is_infinite() && masked[4].is_sign_negative()); // Mysticete - masked
    }

    #[test]
    fn test_apply_inference_mask_unknown_keeps_all() {
        let raw_logits = vec![1.0, 2.0, 3.0];
        let species_labels = vec![
            "Gibbon".to_string(),
            "Towhee".to_string(),
            "Bat".to_string(),
        ];

        let mut label_to_taxon = HashMap::new();
        label_to_taxon.insert("Gibbon".to_string(), Taxon::Mammal);
        label_to_taxon.insert("Towhee".to_string(), Taxon::Songbird);
        label_to_taxon.insert("Bat".to_string(), Taxon::Mammal);

        // Unknown taxon - should keep all
        let masked = HierarchicalVetoEnsemble::apply_inference_mask(
            &raw_logits,
            &species_labels,
            Taxon::Unknown,
            &label_to_taxon,
        );

        // All should be kept
        for (masked, original) in masked.iter().zip(raw_logits.iter()) {
            assert!((masked - original).abs() < 1e-6);
        }
    }

    #[test]
    fn test_get_top_n_candidates_basic() {
        // Raw logits for 5 species (higher = more confident)
        let raw_logits = vec![
            5.0,  // Gibbon - highest
            3.0,  // Towhee
            4.0,  // Bat
            1.0,  // Sparrow
            2.0,  // Whale
        ];
        let species_labels = vec![
            "Gibbon".to_string(),      // Mammal
            "Towhee".to_string(),      // Songbird
            "Bat".to_string(),         // Mammal
            "Sparrow".to_string(),     // Songbird
            "Whale".to_string(),       // Mysticete
        ];

        let mut label_to_taxon = HashMap::new();
        label_to_taxon.insert("Gibbon".to_string(), Taxon::Mammal);
        label_to_taxon.insert("Towhee".to_string(), Taxon::Songbird);
        label_to_taxon.insert("Bat".to_string(), Taxon::Mammal);
        label_to_taxon.insert("Sparrow".to_string(), Taxon::Songbird);
        label_to_taxon.insert("Whale".to_string(), Taxon::Mysticete);

        // Get top 3 candidates for Mammal
        let candidates = HierarchicalVetoEnsemble::get_top_n_candidates(
            &raw_logits,
            &species_labels,
            Taxon::Mammal,
            &label_to_taxon,
            3,
        );

        // Should only return mammals (Gibbon and Bat)
        assert_eq!(candidates.len(), 2); // Only 2 mammals in list
        assert_eq!(candidates[0].label, "Gibbon"); // Highest mammal
        assert_eq!(candidates[1].label, "Bat");     // Second mammal
        assert!(candidates[0].confidence > candidates[1].confidence);
    }

    #[test]
    fn test_get_top_n_candidates_cross_clade_prevention() {
        // Simulate NN incorrectly thinking a bird is most likely
        let raw_logits = vec![
            10.0, // Towhee - bird (highest raw logit)
            8.0,  // Sparrow - bird
            5.0,  // Gibbon - mammal
            3.0,  // Bat - mammal
            1.0,  // Cricket - insect
        ];
        let species_labels = vec![
            "Towhee".to_string(),
            "Sparrow".to_string(),
            "Gibbon".to_string(),
            "Bat".to_string(),
            "Cricket".to_string(),
        ];

        let mut label_to_taxon = HashMap::new();
        label_to_taxon.insert("Towhee".to_string(), Taxon::Songbird);
        label_to_taxon.insert("Sparrow".to_string(), Taxon::Songbird);
        label_to_taxon.insert("Gibbon".to_string(), Taxon::Mammal);
        label_to_taxon.insert("Bat".to_string(), Taxon::Mammal);
        label_to_taxon.insert("Cricket".to_string(), Taxon::Insect);

        // RF Gatekeeper says "Mammal" - masking should zero out birds
        let candidates = HierarchicalVetoEnsemble::get_top_n_candidates(
            &raw_logits,
            &species_labels,
            Taxon::Mammal,
            &label_to_taxon,
            5,
        );

        // All returned candidates should be mammals
        for candidate in &candidates {
            assert_eq!(candidate.taxon, Taxon::Mammal);
        }

        // First should be Gibbon (highest mammal logit)
        assert_eq!(candidates[0].label, "Gibbon");
        assert_eq!(candidates[1].label, "Bat");
    }
}
