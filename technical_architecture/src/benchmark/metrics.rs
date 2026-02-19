//! Classification metrics for evaluation
//!
//! This module provides metrics for evaluating classification performance
//! including accuracy, precision, recall, F1, and confusion matrix.

/// Classification metrics
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ClassificationMetrics {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub confusion_matrix: ConfusionMatrix,
}

/// Confusion matrix representation
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ConfusionMatrix {
    pub true_positives: usize,
    pub false_positives: usize,
    pub true_negatives: usize,
    pub false_negatives: usize,
}

impl ConfusionMatrix {
    pub fn new(tp: usize, fp: usize, tn: usize, fn_count: usize) -> Self {
        Self {
            true_positives: tp,
            false_positives: fp,
            true_negatives: tn,
            false_negatives: fn_count,
        }
    }

    pub fn accuracy(&self) -> f32 {
        let total =
            self.true_positives + self.false_positives + self.true_negatives + self.false_negatives;
        if total == 0 {
            return 0.0;
        }
        (self.true_positives + self.true_negatives) as f32 / total as f32
    }

    pub fn precision(&self) -> f32 {
        let predicted_positive = self.true_positives + self.false_positives;
        if predicted_positive == 0 {
            return 0.0;
        }
        self.true_positives as f32 / predicted_positive as f32
    }

    pub fn recall(&self) -> f32 {
        let actual_positive = self.true_positives + self.false_negatives;
        if actual_positive == 0 {
            return 0.0;
        }
        self.true_positives as f32 / actual_positive as f32
    }

    pub fn f1_score(&self) -> f32 {
        let precision = self.precision();
        let recall = self.recall();
        if precision + recall == 0.0 {
            return 0.0;
        }
        2.0 * precision * recall / (precision + recall)
    }
}

/// Metric calculator
pub struct MetricCalculator;

impl MetricCalculator {
    /// Calculate metrics from predictions and labels
    pub fn calculate_metrics(
        predictions: &[usize],
        labels: &[usize],
        _num_classes: usize,
    ) -> ClassificationMetrics {
        if predictions.is_empty() || labels.is_empty() || predictions.len() != labels.len() {
            return ClassificationMetrics::default();
        }

        // For binary classification (simplified)
        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_count = 0;

        for (&pred, &label) in predictions.iter().zip(labels.iter()) {
            if pred == label {
                if pred > 0 {
                    tp += 1;
                } else {
                    tn += 1;
                }
            } else {
                if pred > 0 {
                    fp += 1;
                } else {
                    fn_count += 1;
                }
            }
        }

        let cm = ConfusionMatrix::new(tp, fp, tn, fn_count);

        ClassificationMetrics {
            accuracy: cm.accuracy(),
            precision: cm.precision(),
            recall: cm.recall(),
            f1_score: cm.f1_score(),
            confusion_matrix: cm,
        }
    }
}

/// Feature ablation results
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureAblationResults {
    pub feature_group: String,
    pub accuracy_30d: f32,
    pub accuracy_39d: f32,
    pub accuracy_56d: f32,
    pub improvement_percent: f32,
}

impl FeatureAblationResults {
    pub fn new(feature_group: &str, acc_30d: f32, acc_39d: f32, acc_56d: f32) -> Self {
        let improvement_percent = if acc_30d > 0.0 {
            ((acc_56d - acc_30d) / acc_30d) * 100.0
        } else {
            0.0
        };

        Self {
            feature_group: feature_group.to_string(),
            accuracy_30d: acc_30d,
            accuracy_39d: acc_39d,
            accuracy_56d: acc_56d,
            improvement_percent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Confusion Matrix Tests (4 tests)
    // =========================================================================

    #[test]
    fn test_confusion_matrix_creation() {
        let cm = ConfusionMatrix::new(10, 5, 80, 5);
        assert_eq!(cm.true_positives, 10);
        assert_eq!(cm.false_positives, 5);
        assert_eq!(cm.true_negatives, 80);
        assert_eq!(cm.false_negatives, 5);
    }

    #[test]
    fn test_confusion_matrix_accuracy() {
        let cm = ConfusionMatrix::new(10, 5, 80, 5);
        let acc = cm.accuracy();
        assert!((acc - 0.9).abs() < 0.01, "Accuracy should be ~0.9");
    }

    #[test]
    fn test_confusion_matrix_precision_recall() {
        let cm = ConfusionMatrix::new(10, 5, 80, 5);
        let precision = cm.precision();
        let recall = cm.recall();

        assert_eq!(precision, 10.0 / 15.0);
        assert_eq!(recall, 10.0 / 15.0);
    }

    #[test]
    fn test_confusion_matrix_f1_score() {
        let cm = ConfusionMatrix::new(10, 5, 80, 5);
        let f1 = cm.f1_score();
        assert!(f1 > 0.0 && f1 <= 1.0);
    }

    // =========================================================================
    // Metric Calculator Tests (4 tests)
    // =========================================================================

    #[test]
    fn test_calculate_metrics_perfect() {
        let predictions = vec![1, 1, 0, 0];
        let labels = vec![1, 1, 0, 0];

        let metrics = MetricCalculator::calculate_metrics(&predictions, &labels, 2);

        assert_eq!(metrics.accuracy, 1.0);
        assert_eq!(metrics.precision, 1.0);
        assert_eq!(metrics.recall, 1.0);
        assert_eq!(metrics.f1_score, 1.0);
    }

    #[test]
    fn test_calculate_metrics_half_correct() {
        let predictions = vec![1, 0, 1, 0];
        let labels = vec![1, 1, 0, 0];

        let metrics = MetricCalculator::calculate_metrics(&predictions, &labels, 2);

        assert_eq!(metrics.accuracy, 0.5);
        assert!(metrics.f1_score > 0.0);
    }

    #[test]
    fn test_calculate_metrics_empty() {
        let predictions: Vec<usize> = vec![];
        let labels: Vec<usize> = vec![];

        let metrics = MetricCalculator::calculate_metrics(&predictions, &labels, 2);

        assert_eq!(metrics.accuracy, 0.0);
        assert_eq!(metrics.precision, 0.0);
        assert_eq!(metrics.recall, 0.0);
        assert_eq!(metrics.f1_score, 0.0);
    }

    #[test]
    fn test_calculate_metrics_mismatched_lengths() {
        let predictions = vec![1, 0, 1];
        let labels = vec![1, 0];

        let metrics = MetricCalculator::calculate_metrics(&predictions, &labels, 2);

        assert_eq!(metrics.accuracy, 0.0);
    }

    // =========================================================================
    // Feature Ablation Tests (4 tests)
    // =========================================================================

    #[test]
    fn test_ablation_results_creation() {
        let results = FeatureAblationResults::new("MFCC", 0.8, 0.85, 0.9);

        assert_eq!(results.feature_group, "MFCC");
        assert_eq!(results.accuracy_30d, 0.8);
        assert_eq!(results.accuracy_39d, 0.85);
        assert_eq!(results.accuracy_56d, 0.9);
    }

    #[test]
    fn test_ablation_improvement_calculation() {
        let results = FeatureAblationResults::new("MFCC", 0.8, 0.85, 0.9);

        let expected_improvement = (0.9 - 0.8) / 0.8 * 100.0;
        assert!((results.improvement_percent - expected_improvement).abs() < 0.01);
    }

    #[test]
    fn test_ablation_zero_baseline() {
        let results = FeatureAblationResults::new("MFCC", 0.0, 0.0, 0.0);

        assert_eq!(results.improvement_percent, 0.0);
    }

    #[test]
    fn test_ablation_negative_improvement() {
        let results = FeatureAblationResults::new("MFCC", 0.9, 0.85, 0.8);

        assert!(results.improvement_percent < 0.0);
    }
}
