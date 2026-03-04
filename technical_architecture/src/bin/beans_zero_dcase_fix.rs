//! BEANS-Zero DCASE Multi-Label Evaluation Fix
//!
//! DCASE 2021 Task 5 uses multi-label classification:
//! - 88% of samples are "None" (no bird detected)
//! - Multi-label samples: "Ovenbird, Swainson's Thrush"
//!
//! This binary evaluates with proper multi-label handling:
//! - Exact match: predicted == reference
//! - Partial match: predicted is one of the reference labels
//! - "None" handling: separate accuracy for detection vs classification

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Multi-Label Comparison Helpers
// ============================================================================

/// Check if prediction matches reference (handles multi-label)
///
/// Examples:
/// - predict("Ovenbird"), ref("Ovenbird") -> ExactMatch
/// - predict("Ovenbird"), ref("Ovenbird, Swainson's Thrush") -> PartialMatch
/// - predict("Ovenbird"), ref("None") -> NoMatch
/// - predict("None"), ref("None") -> ExactMatch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchType {
    /// Exact match (single label or identical multi-label)
    ExactMatch,
    /// Partial match (predicted is one of the reference labels)
    PartialMatch,
    /// No match
    NoMatch,
}

/// Compare predicted label to reference label with multi-label support
pub fn compare_labels(predicted: &str, reference: &str) -> MatchType {
    // Normalize whitespace
    let predicted = predicted.trim();
    let reference = reference.trim();

    // Exact match
    if predicted == reference {
        return MatchType::ExactMatch;
    }

    // Split reference into labels (for multi-label)
    let ref_labels: Vec<&str> = reference.split(',').map(|s| s.trim()).collect();

    // Single label reference
    if ref_labels.len() == 1 {
        return MatchType::NoMatch;
    }

    // Multi-label reference: check if prediction matches any label
    for label in &ref_labels {
        if predicted == *label {
            return MatchType::PartialMatch;
        }
    }

    MatchType::NoMatch
}

/// Calculate DCASE-specific metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DcaseMetrics {
    /// Total samples
    pub total: usize,
    /// "None" samples
    pub none_samples: usize,
    /// Species samples (non-"None")
    pub species_samples: usize,
    /// Exact matches (full credit)
    pub exact_matches: usize,
    /// Partial matches (half credit)
    pub partial_matches: usize,
    /// None correctly predicted as None
    pub none_correct: usize,
    /// Species correctly predicted (exact or partial)
    pub species_correct: usize,
    /// False positives (predicted species when it was None)
    pub false_positives: usize,
    /// False negatives (predicted None when it was species)
    pub false_negatives: usize,
}

impl DcaseMetrics {
    /// Calculate exact accuracy (only exact matches count)
    pub fn exact_accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.exact_matches as f64 / self.total as f64
        }
    }

    /// Calculate partial accuracy (exact + 0.5 * partial)
    pub fn partial_accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.exact_matches as f64 + 0.5 * self.partial_matches as f64) / self.total as f64
        }
    }

    /// Calculate detection accuracy (None vs Species)
    pub fn detection_accuracy(&self) -> f64 {
        let correct_detections = self.none_correct + self.species_correct;
        if self.total == 0 {
            0.0
        } else {
            correct_detections as f64 / self.total as f64
        }
    }

    /// Calculate F1 score for species detection
    pub fn species_f1(&self) -> f64 {
        let tp = self.species_correct as f64;
        let fp = self.false_positives as f64;
        let fn_ = self.false_negatives as f64;

        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };

        if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        }
    }
}

// ============================================================================
// Main Evaluation
// ============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║       DCASE MULTI-LABEL EVALUATION FIX                                ║");
    println!("║       Handling 88% 'None' class + multi-label format                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Load manifest
    let manifest_path = "beans_zero_cache/beans_audio_manifest.json";
    println!("Loading manifest from: {:?}", manifest_path);

    let manifest_data = std::fs::read_to_string(manifest_path)?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_data)?;

    // Extract DCASE samples
    let samples = manifest
        .get("samples")
        .and_then(|s| s.as_array())
        .ok_or_else(|| anyhow::anyhow!("No samples in manifest"))?;

    let dcase_samples: Vec<_> = samples
        .iter()
        .filter(|s| {
            s.get("labels")
                .and_then(|l| l.get("dataset_name"))
                .and_then(|d| d.as_str())
                .map(|d| d == "dcase")
                .unwrap_or(false)
        })
        .collect();

    println!("DCASE samples: {}", dcase_samples.len());

    // Analyze label distribution
    let mut label_counts: HashMap<String, usize> = HashMap::new();
    for sample in &dcase_samples {
        let output = sample
            .get("labels")
            .and_then(|l| l.get("output"))
            .and_then(|o| o.as_str())
            .unwrap_or("Unknown");
        *label_counts.entry(output.to_string()).or_default() += 1;
    }

    println!("\nLabel distribution (top 10):");
    let mut sorted_labels: Vec<_> = label_counts.iter().collect();
    sorted_labels.sort_by(|a, b| b.1.cmp(a.1));
    for (label, count) in sorted_labels.iter().take(10) {
        println!(
            "  {}: {} ({:.1}%)",
            label,
            count,
            **count as f64 / dcase_samples.len() as f64 * 100.0
        );
    }

    // Count multi-label samples
    let multi_label_count = dcase_samples
        .iter()
        .filter(|s| {
            let output = s
                .get("labels")
                .and_then(|l| l.get("output"))
                .and_then(|o| o.as_str())
                .unwrap_or("");
            output.contains(',') && output != "None"
        })
        .count();

    println!(
        "\nMulti-label samples: {} ({:.1}%)",
        multi_label_count,
        multi_label_count as f64 / dcase_samples.len() as f64 * 100.0
    );

    // Test comparison function
    println!("\n{}", "=".repeat(70));
    println!("TESTING MULTI-LABEL COMPARISON");
    println!("{}", "=".repeat(70));

    let test_cases = vec![
        ("Ovenbird", "Ovenbird", MatchType::ExactMatch),
        (
            "Ovenbird",
            "Ovenbird, Swainson's Thrush",
            MatchType::PartialMatch,
        ),
        (
            "Swainson's Thrush",
            "Ovenbird, Swainson's Thrush",
            MatchType::PartialMatch,
        ),
        ("None", "None", MatchType::ExactMatch),
        ("Ovenbird", "None", MatchType::NoMatch),
        ("None", "Ovenbird", MatchType::NoMatch),
        (
            "Unknown Bird",
            "Ovenbird, Swainson's Thrush",
            MatchType::NoMatch,
        ),
    ];

    for (predicted, reference, expected) in test_cases {
        let result = compare_labels(predicted, reference);
        let status = if result == expected { "✓" } else { "✗" };
        println!(
            "  {} predict({:?}), ref({:?}) -> {:?} (expected {:?})",
            status, predicted, reference, result, expected
        );
    }

    // Simulate evaluation with proper multi-label handling
    println!("\n{}", "=".repeat(70));
    println!("SIMULATED DCASE EVALUATION");
    println!("{}", "=".repeat(70));

    let mut metrics = DcaseMetrics::default();
    metrics.total = dcase_samples.len();

    for sample in &dcase_samples {
        let reference = sample
            .get("labels")
            .and_then(|l| l.get("output"))
            .and_then(|o| o.as_str())
            .unwrap_or("Unknown");

        // Simulate prediction (in reality, this would come from the model)
        // For demonstration, we'll simulate a model that:
        // - Predicts "None" 90% of the time (matching the distribution)
        // - Has 30% accuracy on actual species

        let predicted = if reference == "None" {
            // 90% of None samples are correctly predicted as None
            if rand::random::<f64>() < 0.9 {
                "None"
            } else {
                // False positive: predict a random species
                "Ovenbird"
            }
        } else {
            // For species samples
            if rand::random::<f64>() < 0.3 {
                // Correct prediction (use first label for multi-label)
                reference.split(',').next().unwrap_or("Unknown").trim()
            } else {
                // Wrong prediction
                "None"
            }
        };

        // Update metrics based on match type
        let match_type = compare_labels(predicted, reference);

        match match_type {
            MatchType::ExactMatch => {
                metrics.exact_matches += 1;
                if reference == "None" {
                    metrics.none_samples += 1;
                    metrics.none_correct += 1;
                } else {
                    metrics.species_samples += 1;
                    metrics.species_correct += 1;
                }
            }
            MatchType::PartialMatch => {
                metrics.partial_matches += 1;
                metrics.species_samples += 1;
                metrics.species_correct += 1;
            }
            MatchType::NoMatch => {
                if reference == "None" {
                    metrics.none_samples += 1;
                    metrics.false_positives += 1; // Predicted species when it was None
                } else if predicted == "None" {
                    metrics.species_samples += 1;
                    metrics.false_negatives += 1; // Predicted None when it was species
                } else {
                    metrics.species_samples += 1;
                    // Wrong species prediction (not None)
                }
            }
        }
    }

    println!("\nDCASE Metrics:");
    println!("  Total samples: {}", metrics.total);
    println!(
        "  Exact matches: {} ({:.1}%)",
        metrics.exact_matches,
        metrics.exact_accuracy() * 100.0
    );
    println!("  Partial matches: {}", metrics.partial_matches);
    println!(
        "  Partial accuracy: {:.1}%",
        metrics.partial_accuracy() * 100.0
    );
    println!(
        "  Detection accuracy: {:.1}%",
        metrics.detection_accuracy() * 100.0
    );
    println!("  Species F1: {:.1}%", metrics.species_f1() * 100.0);
    println!("  False positives: {}", metrics.false_positives);
    println!("  False negatives: {}", metrics.false_negatives);

    println!("\n{}", "=".repeat(70));
    println!("RECOMMENDATIONS FOR DCASE IMPROVEMENT:");
    println!("{}", "=".repeat(70));
    println!("1. Use class-balanced training (oversample species, undersample 'None')");
    println!("2. Train separate binary detector (Species vs None) first");
    println!("3. Use multi-label loss function (BCE with sigmoid) for multi-species");
    println!("4. Consider threshold tuning: adjust 'None' threshold based on validation F1");

    Ok(())
}
