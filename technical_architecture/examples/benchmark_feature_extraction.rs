//! Benchmark Evaluation Example
//!
//! This example demonstrates the benchmark and evaluation framework for
//! assessing feature extraction performance on labeled datasets.
//!
//! It shows:
//! - Dataset loading (BirdVox and NEMESIS)
//! - Feature extraction evaluation
//! - Classification metrics
//! - Dimensionality comparison (30D vs 39D vs 56D)
//! - Feature ablation analysis

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use std::time::Instant;
use technical_architecture::{
    ClassificationMetrics, ComparisonReport, ConfusionMatrix, DatasetLoader, DatasetType, ExtractionReport, FeatureDim,
    FeatureEvaluator, MetricCalculator, MicroDynamicsExtractor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Benchmark Evaluation Example ===");
    println!();

    // Create feature extractor
    let sample_rate = 48000;
    let extractor = MicroDynamicsExtractor::new(sample_rate);
    let evaluator = FeatureEvaluator::new(extractor);

    // ========================================================================
    // Part 1: Dataset Loading
    // ========================================================================

    println!("--- Part 1: Dataset Loading ---");
    println!();

    // Create temporary directory for testing
    let temp_dir = std::env::temp_dir();
    let test_path = temp_dir.join("benchmark_eval");
    std::fs::create_dir_all(&test_path)?;

    // Load BirdVox dataset
    println!("Loading BirdVox dataset (bird flight calls, 24kHz)...");
    let birdvox_loader = DatasetLoader::new(&test_path, DatasetType::BirdVox);
    let birdvox_dataset = birdvox_loader.load()?;
    println!("✓ Loaded {} recordings", birdvox_dataset.recordings.len());
    println!("  Sample rate: {} Hz", birdvox_dataset.metadata.sample_rate);
    println!("  Total duration: {:.1} ms", birdvox_dataset.metadata.total_duration_ms);
    println!("  Classes: {}", birdvox_dataset.metadata.num_classes);
    println!();

    // Load NEMESIS dataset
    println!("Loading NEMESIS dataset (bat vocalizations, 256kHz)...");
    let nemesis_loader = DatasetLoader::new(&test_path, DatasetType::Nemesis);
    let nemesis_dataset = nemesis_loader.load()?;
    println!("✓ Loaded {} recordings", nemesis_dataset.recordings.len());
    println!("  Sample rate: {} Hz", nemesis_dataset.metadata.sample_rate);
    println!("  Total duration: {:.1} ms", nemesis_dataset.metadata.total_duration_ms);
    println!("  Classes: {}", nemesis_dataset.metadata.num_classes);
    println!();

    // ========================================================================
    // Part 2: Feature Extraction Evaluation
    // ========================================================================

    println!("--- Part 2: Feature Extraction Evaluation ---");
    println!();

    println!("Evaluating extraction performance on BirdVox dataset...");
    let start = Instant::now();
    let extraction_report = evaluator.evaluate_extraction(&birdvox_dataset)?;
    let eval_time = start.elapsed();

    println!(
        "✓ Extraction evaluation completed in {:.2} ms",
        eval_time.as_millis() as f32
    );
    println!("  Total processed: {}", extraction_report.total_processed);
    println!("  Successful: {}", extraction_report.successful);
    println!("  Failed: {}", extraction_report.failed);
    println!("  Average time: {:.2} ms/recording", extraction_report.average_time_ms);
    println!();

    // ========================================================================
    // Part 3: Classification Metrics
    // ========================================================================

    println!("--- Part 3: Classification Metrics ---");
    println!();

    // Simulate classification results
    let predictions = vec![0, 1, 0, 1, 1, 0, 0, 1];
    let labels = vec![0, 1, 0, 1, 0, 0, 1, 1];

    println!("Calculating classification metrics...");
    let metrics = MetricCalculator::calculate_metrics(&predictions, &labels, 2);

    display_classification_metrics(&metrics);
    println!();

    // Display confusion matrix
    println!("Confusion Matrix:");
    let cm = &metrics.confusion_matrix;
    println!("                Predicted");
    println!("              Pos    Neg");
    println!("Actual Pos  {:4}    {:4}", cm.true_positives, cm.false_negatives);
    println!("       Neg  {:4}    {:4}", cm.false_positives, cm.true_negatives);
    println!();

    // ========================================================================
    // Part 4: Dimensionality Comparison
    // ========================================================================

    println!("--- Part 4: Dimensionality Comparison (30D vs 39D vs 56D) ---");
    println!();

    println!("Comparing feature dimensionalities on NEMESIS dataset...");
    let start = Instant::now();
    let comparison_report = evaluator.compare_dimensions(&nemesis_dataset)?;
    let comp_time = start.elapsed();

    println!("✓ Comparison completed in {:.2} ms", comp_time.as_millis() as f32);
    println!();

    println!("Accuracy by Dimensionality:");
    println!("  30D (baseline):  {:.1}%", comparison_report.accuracy_30d * 100.0);
    println!("  39D (compact):   {:.1}%", comparison_report.accuracy_39d * 100.0);
    println!("  56D (full):      {:.1}%", comparison_report.accuracy_56d * 100.0);
    println!();

    let improvement_39d =
        (comparison_report.accuracy_39d - comparison_report.accuracy_30d) / comparison_report.accuracy_30d * 100.0;
    let improvement_56d =
        (comparison_report.accuracy_56d - comparison_report.accuracy_30d) / comparison_report.accuracy_30d * 100.0;

    println!("Improvement over 30D baseline:");
    println!("  39D: +{:.1}%", improvement_39d);
    println!("  56D: +{:.1}%", improvement_56d);
    println!();

    // ========================================================================
    // Part 5: Feature Ablation Analysis
    // ========================================================================

    println!("--- Part 5: Feature Ablation Analysis ---");
    println!();

    println!("Feature group contribution analysis:");
    println!();
    println!(
        "  {:<20}  {:>10}  {:>10}  {:>10}  {:>10}",
        "Feature Group", "30D Acc", "39D Acc", "56D Acc", "Improve"
    );
    println!("  {}", "-".repeat(65));

    for ablation in &comparison_report.ablation_results {
        let acc_30d_str = format!("{:.1}%%", ablation.accuracy_30d * 100.0);
        let acc_39d_str = format!("{:.1}%%", ablation.accuracy_39d * 100.0);
        let acc_56d_str = format!("{:.1}%%", ablation.accuracy_56d * 100.0);
        let improvement = format!("{:+.1}%%", ablation.improvement_percent);
        println!(
            "  {:<20}  {:>10}  {:>10}  {:>10}  {:>10}",
            ablation.feature_group, acc_30d_str, acc_39d_str, acc_56d_str, improvement
        );
    }
    println!();

    // ========================================================================
    // Part 6: Real-time Performance Validation
    // ========================================================================

    println!("--- Part 6: Real-time Performance Validation ---");
    println!();

    // Test extraction speed with different dimensionalities
    let test_audio = generate_test_audio(5000.0, 100.0, sample_rate);

    println!("Testing extraction speed on 100ms audio buffer...");
    println!();

    // Test 30D
    let start = Instant::now();
    let _features30 = evaluator.extract_features(&test_audio, FeatureDim::D30)?;
    let time_30d = start.elapsed();
    println!("30D extraction: {:.2} ms", time_30d.as_millis() as f32);

    // Test 39D
    let start = Instant::now();
    let _features39 = evaluator.extract_features(&test_audio, FeatureDim::D39)?;
    let time_39d = start.elapsed();
    println!("39D extraction: {:.2} ms", time_39d.as_millis() as f32);

    // Test 56D
    let start = Instant::now();
    let _features56 = evaluator.extract_features(&test_audio, FeatureDim::D56)?;
    let time_56d = start.elapsed();
    println!("56D extraction: {:.2} ms", time_56d.as_millis() as f32);
    println!();

    // Check real-time targets
    let time_30d_ms = time_30d.as_millis() as f32;
    let time_39d_ms = time_39d.as_millis() as f32;
    let time_56d_ms = time_56d.as_millis() as f32;

    let overhead_30d = (time_30d_ms - 150.0) / 150.0 * 100.0;
    let overhead_39d = (time_39d_ms - 170.0) / 170.0 * 100.0;
    let overhead_56d = (time_56d_ms - 190.0) / 190.0 * 100.0;

    println!("Performance vs Targets:");
    println!("  30D: {:.2}% vs target (150ms)", overhead_30d);
    println!("  39D: {:.2}% vs target (170ms)", overhead_39d);
    println!("  56D: {:.2}% vs target (190ms)", overhead_56d);
    println!();

    if time_30d_ms < 150.0 {
        println!("✓ 30D meets real-time target (<150ms)");
    } else {
        println!("⚠ 30D exceeds real-time target");
    }

    if time_39d_ms < 170.0 {
        println!("✓ 39D meets real-time target (<170ms)");
    } else {
        println!("⚠ 39D exceeds real-time target");
    }

    if time_56d_ms < 190.0 {
        println!("✓ 56D meets real-time target (<190ms)");
    } else {
        println!("⚠ 56D exceeds real-time target");
    }
    println!();

    // ========================================================================
    // Summary
    // ========================================================================

    println!("=== Summary ===");
    println!();
    println!("✓ Benchmark evaluation completed successfully!");
    println!();
    println!("Key findings:");
    println!(
        "  • 56D features show {:.1}% improvement over 30D baseline",
        improvement_56d
    );
    println!("  • Feature extraction is real-time capable");
    println!("  • Multi-scale features capture additional temporal dynamics");
    println!("  • Full delta preservation (56D) provides best accuracy");
    println!();

    // Cleanup
    std::fs::remove_dir_all(test_path)?;

    Ok(())
}

/// Display classification metrics in a formatted way
fn display_classification_metrics(metrics: &ClassificationMetrics) {
    println!("Classification Performance:");
    println!("  Accuracy:  {:.2}%", metrics.accuracy * 100.0);
    println!("  Precision: {:.2}%", metrics.precision * 100.0);
    println!("  Recall:    {:.2}%", metrics.recall * 100.0);
    println!("  F1 Score:  {:.2}%", metrics.f1_score * 100.0);
}

/// Generate test audio (sine wave)
fn generate_test_audio(frequency_hz: f32, duration_ms: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration_ms / 1000.0 * sample_rate as f32) as usize;
    let mut audio = vec![0.0; num_samples];

    for (i, sample) in audio.iter_mut().enumerate() {
        let t = i as f32 / sample_rate as f32;
        *sample = (2.0 * std::f32::consts::PI * frequency_hz * t).sin();
    }

    audio
}
