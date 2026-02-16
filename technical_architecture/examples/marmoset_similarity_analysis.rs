// Acoustic Similarity Analysis for Marmoset Vocalizations
//
// This example demonstrates a PAIRWISE SIMILARITY approach instead of clustering.
// The key insight is that animal vocalizations form CONTINUOUS MANIFOLDS, not
// discrete clusters.
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │                                                                             │
// │   Marmoset calls exist on a gradient:                                      │
// │                                                                             │
// │      Phee ←───────→ Trill ←───────→ Twitter ←──────→ Tsik                  │
// │         (continuous acoustic transitions, not separate islands)            │
// │                                                                             │
// │   HDBSCAN expects ISLANDS. You have a CONTINENT.                           │
// │                                                                             │
// └─────────────────────────────────────────────────────────────────────────────┘
//
// Usage:
//   cargo run --release --example marmoset_similarity_analysis
//   cargo run --release --example marmoset_similarity_analysis -- --limit 100

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use technical_architecture::{
    AcousticSimilarityEngine,
    KnnClassifier,
    SimilarityAnalysis,
    SimilarityIndex,
    KnnCvResults,
};

// =============================================================================
// Data Structures
// =============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializableFeatures {
    file_name: String,
    call_type: String,
    phrase_index: usize,
    features: Vec<f64>,
    duration_ms: f64,
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║      Acoustic Similarity Analysis for Marmoset Vocalizations         ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                      ║");
    println!("║  🎯 GOAL: Measure acoustic similarity between phrases               ║");
    println!("║         WITHOUT forcing discrete clusters                           ║");
    println!("║                                                                      ║");
    println!("║  🔑 INSIGHT: Animal vocalizations form CONTINUOUS MANIFOLDS         ║");
    println!("║             HDBSCAN expects ISLANDS, but we have a CONTINENT        ║");
    println!("║                                                                      ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Parse arguments
    let args: Vec<String> = std::env::args().collect();
    let results_dir = PathBuf::from("/mnt/c/Users/sheel/Desktop/src/marmoset_phase0_30d_results");
    let output_dir = PathBuf::from("/mnt/c/Users/sheel/Desktop/src/marmoset_similarity_results");

    // Check for existing features
    let features_path = results_dir.join("marmoset_30d_features.bincode");

    if !features_path.exists() {
        println!("❌ No features found at {}", features_path.display());
        println!();
        println!("Please run feature extraction first:");
        println!("  cargo run --release --example phase0_symbolic_stream_marmoset_30d");
        return Err("Features not found".into());
    }

    // Load features
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Loading Features                                             │");
    println!("└──────────────────────────────────────────────────────────────────────┘");
    println!();

    let load_start = Instant::now();
    let feature_data = fs::read(&features_path)?;
    let features: Vec<SerializableFeatures> = bincode::deserialize(&feature_data)?;

    println!("   ✅ Loaded {} phrase features", features.len());
    println!("      Time: {:.2}s", load_start.elapsed().as_secs_f64());
    println!();

    // Convert to arrays
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Converting to Arrays                                         │");
    println!("└──────────────────────────────────────────────────────────────────────┘");
    println!();

    let convert_start = Instant::now();

    let n_samples = features.len();
    let n_dims = features[0].features.len();

    let mut feature_matrix = Array2::zeros((n_samples, n_dims));
    let mut file_names: Vec<String> = Vec::with_capacity(n_samples);
    let mut call_types: Vec<String> = Vec::with_capacity(n_samples);

    for (i, feat) in features.iter().enumerate() {
        for (j, &val) in feat.features.iter().enumerate() {
            feature_matrix[[i, j]] = val;
        }
        file_names.push(feat.file_name.clone());
        call_types.push(feat.call_type.clone());
    }

    println!("   ✅ Created {}x{} feature matrix", n_samples, n_dims);
    println!("      Time: {:.2}s", convert_start.elapsed().as_secs_f64());
    println!();

    // Call type distribution
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    for ct in &call_types {
        *type_counts.entry(ct.clone()).or_insert(0) += 1;
    }

    println!("   📊 Call Type Distribution:");
    for (ct, count) in type_counts.iter() {
        let pct = *count as f64 / n_samples as f64 * 100.0;
        println!("      • {}: {} ({:.1}%)", ct, count, pct);
    }
    println!();

    // Feature names (30D layout)
    let feature_names: Vec<&str> = vec![
        "attack_time_ms",
        "decay_time_ms",
        "sustain_level",
        "vibrato_rate_hz",
        "vibrato_depth",
        "jitter",
        "shimmer",
        "harmonicity",
        "spectral_flatness",
        "hnr",
        "mfcc_0",
        "mfcc_1",
        "mfcc_2",
        "mfcc_3",
        "mfcc_4",
        "mfcc_5",
        "mfcc_6",
        "mfcc_7",
        "mfcc_8",
        "mfcc_9",
        "mfcc_10",
        "mfcc_11",
        "mfcc_12",
        "spectral_flux",
        "median_ici_ms",
        "onset_rate_hz",
        "ici_cv",
        "unknown_27",
        "unknown_28",
        "unknown_29",
    ];

    // ===========================================================================
    // Similarity Analysis
    // ===========================================================================

    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Computing Similarity Analysis                                │");
    println!("└──────────────────────────────────────────────────────────────────────┘");
    println!();

    let analysis_start = Instant::now();

    let analysis = SimilarityAnalysis::analyze(
        &feature_matrix,
        &call_types,
        &file_names,
        &feature_names,
    );

    println!("   ✅ Analysis complete in {:.2}s", analysis_start.elapsed().as_secs_f64());
    println!();

    // Print summary
    analysis.print_summary();

    // ===========================================================================
    // k-NN Classification
    // ===========================================================================

    println!();
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: k-NN Classification Evaluation                               │");
    println!("└──────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("   🔧 Building k-NN classifier...");
    let classifier = KnnClassifier::new(feature_matrix.clone(), call_types.clone(), file_names.clone());

    println!("   🔍 Finding optimal k (testing k=1,3,5,7,9)...");
    let k_values = vec![1, 3, 5, 7, 9];
    let cv_results = classifier.find_optimal_k(&k_values, 5);

    println!();
    println!("   ┌────────────────────────────────────────────────────────────────┐");
    println!("   │  k-NN Cross-Validation Results                                  │");
    println!("   └────────────────────────────────────────────────────────────────┘");
    println!();
    println!("   📊 Optimal k: {}", cv_results.optimal_k);
    println!("   📊 Mean Accuracy: {:.1}%", cv_results.mean_accuracy * 100.0);
    println!();

    println!("   📊 Per-Class Accuracy:");
    let mut per_class: Vec<_> = cv_results.per_class_accuracy.iter().collect();
    per_class.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (class, &acc) in per_class {
        let bar_len = (acc * 30.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("      • {}: {:.1}% {}", class, acc * 100.0, bar);
    }
    println!();

    // Print confusion matrix summary
    println!("   📊 Confusion Matrix (showing misclassifications):");
    for entry in cv_results.confusion_matrix.iter() {
        if entry.predicted != entry.actual && entry.count > 0 {
            println!("      • Predicted {}, Actual {}: {} times", entry.predicted, entry.actual, entry.count);
        }
    }
    println!();

    // ===========================================================================
    // Similarity Search Demo
    // ===========================================================================

    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 5: Similarity Search Demo                                       │");
    println!("└──────────────────────────────────────────────────────────────────────┘");
    println!();

    let index = SimilarityIndex::new(feature_matrix, file_names.clone(), call_types.clone());

    // Find a sample from each call type
    let mut demo_files: Vec<(String, String)> = Vec::new();
    for ct in type_counts.keys() {
        if let Some(idx) = call_types.iter().position(|c| c == ct) {
            demo_files.push((file_names[idx].clone(), ct.clone()));
        }
    }

    println!("   🔍 Finding similar phrases for each call type:");
    println!();

    for (demo_file, demo_type) in demo_files.iter().take(5) {
        println!("   📂 Query: {} ({})", demo_file, demo_type);

        match index.search_by_file(demo_file, 5) {
            Some(results) => {
                println!("      Top 5 similar:");
                for (i, r) in results.iter().enumerate() {
                    let match_marker = if &r.call_type == demo_type { "✓" } else { "✗" };
                    println!("         {}. {} {:.3} ({}) {}",
                             i + 1, r.call_type, r.similarity, match_marker, r.file_name);
                }
            }
            None => {
                println!("      No results found");
            }
        }
        println!();
    }

    // ===========================================================================
    // Comparison: k-NN vs Clustering
    // ===========================================================================

    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 6: k-NN vs Clustering Comparison                                │");
    println!("└──────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("   ╔═════════════════════════════════════════════════════════════════╗");
    println!("   ║                   APPROACH COMPARISON                            ║");
    println!("   ╠═════════════════════════════════════════════════════════════════╣");
    println!("   ║                                                                 ║");
    println!("   ║   CLUSTERING (HDBSCAN)              k-NN CLASSIFICATION         ║");
    println!("   ║   ─────────────────────              ───────────────────        ║");
    println!("   ║   • Hard assignments                • Soft confidence scores   ║");
    println!("   ║   • Expects discrete islands        • Works with gradients     ║");
    println!("   ║   • Many samples → noise            • Every sample classified  ║");
    println!("   ║   • Single partition                • Query-dependent          ║");
    println!("   ║   • \"This is cluster 5\"             • \"87% similar to Phee\"    ║");
    println!("   ║                                                                 ║");
    println!("   ║   FOR CONTINUOUS ACOUSTIC MANIFOLDS:                             ║");
    println!("   ║   ══════════════════════════════════                             ║");
    println!("   ║   k-NN is preferred because:                                     ║");
    println!("   ║   • Respects acoustic gradients                                  ║");
    println!("   ║   • Provides confidence scores                                   ║");
    println!("   ║   • No samples discarded as noise                                ║");
    println!("   ║   • Interpretable similarity scores                              ║");
    println!("   ║                                                                 ║");
    println!("   ╚═════════════════════════════════════════════════════════════════╝");
    println!();

    // ===========================================================================
    // Save Results
    // ===========================================================================

    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 7: Saving Results                                               │");
    println!("└──────────────────────────────────────────────────────────────────────┘");
    println!();

    fs::create_dir_all(&output_dir)?;

    // Save analysis
    let analysis_path = output_dir.join("similarity_analysis.json");
    fs::write(&analysis_path, serde_json::to_string_pretty(&analysis)?)?;
    println!("   💾 Analysis: {}", analysis_path.display());

    // Save CV results
    let cv_path = output_dir.join("knn_cv_results.json");
    fs::write(&cv_path, serde_json::to_string_pretty(&cv_results)?)?;
    println!("   💾 k-NN CV: {}", cv_path.display());

    // Save summary report
    let summary = SummaryReport {
        n_samples,
        n_dims,
        call_type_distribution: type_counts.clone(),
        similarity_analysis: analysis.clone(),
        knn_results: cv_results.clone(),
    };

    let summary_path = output_dir.join("similarity_summary.json");
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    println!("   💾 Summary: {}", summary_path.display());

    println!();

    // ===========================================================================
    // Final Summary
    // ===========================================================================

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                      ANALYSIS COMPLETE                                ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                      ║");
    println!("║  📊 RESULTS:                                                         ║");
    println!("║      • Samples analyzed: {}                                          ║", n_samples);
    println!("║      • Feature dimensions: {}                                        ║", n_dims);
    println!("║      • Call types: {}                                                ║", type_counts.len());
    println!("║      • k-NN accuracy: {:.1}%                                         ║", cv_results.mean_accuracy * 100.0);
    println!("║      • Separation ratio: {:.2}x                                      ║", analysis.separation_ratio);
    println!("║                                                                      ║");
    println!("║  🎯 RECOMMENDATION:                                                  ║");
    if analysis.separation_ratio > 2.0 {
        println!("║      Good separation between call types.                             ║");
        println!("║      k-NN classification should work well.                           ║");
    } else if analysis.separation_ratio > 1.5 {
        println!("║      Moderate separation between call types.                         ║");
        println!("║      Consider adding more discriminative features.                   ║");
    } else {
        println!("║      Low separation between call types.                              ║");
        println!("║      Call types may overlap significantly in feature space.          ║");
        println!("║      Consider:                                                       ║");
        println!("║      • Adding temporal features (DTW on sequences)                   ║");
        println!("║      • Learning feature weights from labeled data                    ║");
        println!("║      • Using metric learning (Siamese networks)                      ║");
    }
    println!("║                                                                      ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}

// =============================================================================
// Summary Report
// =============================================================================

#[derive(Serialize)]
struct SummaryReport {
    n_samples: usize,
    n_dims: usize,
    call_type_distribution: HashMap<String, usize>,
    similarity_analysis: SimilarityAnalysis,
    knn_results: KnnCvResults,
}
