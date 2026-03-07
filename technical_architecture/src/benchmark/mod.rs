//! Benchmark and evaluation module
//!
//! This module provides evaluation framework for assessing feature extraction
//! on labeled datasets (BirdVox, NEMESIS).

pub mod dataset_loader;
pub mod evaluator;
pub mod metrics;

pub use dataset_loader::{BenchmarkDataset, DatasetLoader, DatasetMetadata, DatasetType, Label, Recording};
pub use evaluator::{ClassificationReport, ComparisonReport, ExtractionReport, FeatureEvaluator};
pub use metrics::{ClassificationMetrics, ConfusionMatrix, FeatureAblationResults, MetricCalculator};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_module_compatibility() {
        // Verify all benchmark components can be created
        let _type = DatasetType::BirdVox;
        let _type2 = DatasetType::Nemesis;

        // Verify metric types exist
        let metrics = ClassificationMetrics::default();
        assert_eq!(metrics.accuracy, 0.0);
        assert_eq!(metrics.precision, 0.0);
    }

    #[test]
    fn test_dataset_type_equality() {
        assert_eq!(DatasetType::BirdVox, DatasetType::BirdVox);
        assert_eq!(DatasetType::Nemesis, DatasetType::Nemesis);
        assert_ne!(DatasetType::BirdVox, DatasetType::Nemesis);
    }
}
