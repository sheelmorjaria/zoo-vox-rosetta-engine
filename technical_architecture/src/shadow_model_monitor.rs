// Shadow Model Monitoring
//
// Detects "concept drift" where the AI learns incorrect patterns in the field
// by comparing active model against frozen baseline.

use crate::ptp::PtpTimestamp;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ============================================================================
// Data Structures
// ============================================================================

/// Model prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPrediction {
    pub label: String,
    pub confidence: f32,
    pub category: String,
    pub raw_scores: HashMap<String, f32>,
}

/// Input features for model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputFeatures {
    pub features: Vec<f32>,
    pub metadata: HashMap<String, String>,
}

impl InputFeatures {
    pub fn new(features: Vec<f32>) -> Self {
        Self {
            features,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Drift sample point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftSample {
    pub timestamp: PtpTimestamp,
    pub divergence_ratio: f32,  // 0.0 to 1.0
    pub sample_count: usize,
    pub category_drift: HashMap<String, f32>,
}

impl DriftSample {
    pub fn new(timestamp: PtpTimestamp, divergence_ratio: f32, sample_count: usize) -> Self {
        Self {
            timestamp,
            divergence_ratio,
            sample_count,
            category_drift: HashMap::new(),
        }
    }
}

/// Model comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    pub input_features: Vec<f32>,
    pub active_prediction: String,
    pub shadow_prediction: String,
    pub confidence_difference: f32,
    pub category_match: bool,
    pub divergence_ratio: f32,
}

/// Alert level for drift
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertLevel {
    Warning,   // 10-20% divergence
    Critical,  // 20-40% divergence
    Emergency, // >40% divergence
}

impl AlertLevel {
    pub fn from_divergence(divergence: f32) -> Self {
        if divergence >= 0.4 {
            Self::Emergency
        } else if divergence >= 0.2 {
            Self::Critical
        } else {
            Self::Warning // Warning for everything below 0.2
        }
    }
}

/// Drift alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftAlert {
    pub timestamp: PtpTimestamp,
    pub alert_level: AlertLevel,
    pub current_divergence: f32,
    pub threshold: f32,
    pub recommendations: Vec<String>,
}

impl DriftAlert {
    pub fn new(timestamp: PtpTimestamp, alert_level: AlertLevel, current_divergence: f32, threshold: f32) -> Self {
        let recommendations = match alert_level {
            AlertLevel::Warning => vec![
                "Monitor drift trend closely".to_string(),
                "Consider retraining with recent data".to_string(),
            ],
            AlertLevel::Critical => vec![
                "FREEZE: Model should not be used for new decisions".to_string(),
                "Notify research team immediately".to_string(),
                "Initiate model retraining pipeline".to_string(),
            ],
            AlertLevel::Emergency => vec![
                "EMERGENCY: Automatic rollback recommended".to_string(),
                "Disable AI decision-making".to_string(),
                "Escalate to senior researchers".to_string(),
                "Use fallback rules-based system".to_string(),
            ],
        };

        Self {
            timestamp,
            alert_level,
            current_divergence,
            threshold,
            recommendations,
        }
    }
}

/// Shadow model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowModelConfig {
    pub divergence_threshold: f32,  // e.g., 0.2 (20%)
    pub window_size: usize,         // Samples for averaging
    pub alert_enabled: bool,
    pub auto_freeze_enabled: bool,
    pub rollback_enabled: bool,
}

impl Default for ShadowModelConfig {
    fn default() -> Self {
        Self {
            divergence_threshold: 0.2,
            window_size: 1000,
            alert_enabled: true,
            auto_freeze_enabled: true,
            rollback_enabled: true,
        }
    }
}

// ============================================================================
// Inference Model Trait
// ============================================================================

/// Trait for inference models (both active and shadow)
pub trait InferenceModel: Send + Sync {
    fn predict(&self, input: &InputFeatures) -> ModelPrediction;
    fn model_id(&self) -> &str;
    fn model_version(&self) -> &str;
}

/// Mock active model for testing
#[derive(Debug, Clone)]
pub struct MockActiveModel {
    pub id: String,
    pub version: String,
    pub drift_factor: f32,  // Simulates drift (0.0 = no drift, 1.0 = complete divergence)
}

impl MockActiveModel {
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            id: "active_model".to_string(),
            version: version.into(),
            drift_factor: 0.0,
        }
    }

    pub fn with_drift(mut self, drift_factor: f32) -> Self {
        self.drift_factor = drift_factor;
        self
    }
}

impl InferenceModel for MockActiveModel {
    fn predict(&self, _input: &InputFeatures) -> ModelPrediction {
        // Simulate prediction with possible drift
        let base_label = "playback";
        let base_confidence = 0.85;

        let (label, confidence) = if self.drift_factor > 0.5 {
            // High drift - different prediction
            ("recording".to_string(), base_confidence * (1.0 - self.drift_factor))
        } else {
            (base_label.to_string(), base_confidence)
        };

        let mut raw_scores = HashMap::new();
        raw_scores.insert("playback".to_string(), confidence);
        raw_scores.insert("recording".to_string(), 1.0 - confidence);

        ModelPrediction {
            label,
            confidence,
            category: "vocalization".to_string(),
            raw_scores,
        }
    }

    fn model_id(&self) -> &str {
        &self.id
    }

    fn model_version(&self) -> &str {
        &self.version
    }
}

/// Mock shadow (frozen baseline) model
#[derive(Debug, Clone)]
pub struct MockShadowModel {
    pub id: String,
    pub version: String,
}

impl MockShadowModel {
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            id: "shadow_model".to_string(),
            version: version.into(),
        }
    }
}

impl InferenceModel for MockShadowModel {
    fn predict(&self, _input: &InputFeatures) -> ModelPrediction {
        // Shadow model always returns consistent baseline prediction
        let confidence = 0.85;
        let mut raw_scores = HashMap::new();
        raw_scores.insert("playback".to_string(), confidence);
        raw_scores.insert("recording".to_string(), 1.0 - confidence);

        ModelPrediction {
            label: "playback".to_string(),
            confidence,
            category: "vocalization".to_string(),
            raw_scores,
        }
    }

    fn model_id(&self) -> &str {
        &self.id
    }

    fn model_version(&self) -> &str {
        &self.version
    }
}

// ============================================================================
// Shadow Model Monitor
// ============================================================================

/// Shadow model monitor for detecting concept drift
pub struct ShadowModelMonitor {
    active_model: Box<dyn InferenceModel>,
    shadow_model: Box<dyn InferenceModel>,
    config: ShadowModelConfig,
    drift_history: VecDeque<DriftSample>,
    is_frozen: bool,
    alerts: Vec<DriftAlert>,
}

impl ShadowModelMonitor {
    pub fn new(
        active_model: Box<dyn InferenceModel>,
        shadow_model: Box<dyn InferenceModel>,
        config: ShadowModelConfig,
    ) -> Self {
        Self {
            active_model,
            shadow_model,
            config,
            drift_history: VecDeque::with_capacity(1000),
            is_frozen: false,
            alerts: Vec::new(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults(
        active_model: Box<dyn InferenceModel>,
        shadow_model: Box<dyn InferenceModel>,
    ) -> Self {
        Self::new(active_model, shadow_model, ShadowModelConfig::default())
    }

    /// Compare predictions from both models
    pub fn compare_predictions(&self, input: &InputFeatures) -> ModelComparison {
        let active_pred = self.active_model.predict(input);
        let shadow_pred = self.shadow_model.predict(input);

        let confidence_difference = (active_pred.confidence - shadow_pred.confidence).abs();
        let category_match = active_pred.category == shadow_pred.category;

        let divergence_ratio = self.calculate_divergence(&active_pred, &shadow_pred);

        ModelComparison {
            input_features: input.features.clone(),
            active_prediction: active_pred.label,
            shadow_prediction: shadow_pred.label,
            confidence_difference,
            category_match,
            divergence_ratio,
        }
    }

    /// Calculate divergence ratio between predictions
    fn calculate_divergence(&self, active: &ModelPrediction, shadow: &ModelPrediction) -> f32 {
        // If labels differ, that's full divergence
        if active.label != shadow.label {
            return 1.0;
        }

        // Calculate confidence-based divergence
        let confidence_divergence = (active.confidence - shadow.confidence).abs() / active.confidence.max(shadow.confidence);

        // Calculate score distribution divergence (simplified KL-divergence approximation)
        let score_divergence = self.calculate_score_divergence(&active.raw_scores, &shadow.raw_scores);

        // Average of both metrics
        (confidence_divergence + score_divergence) / 2.0
    }

    /// Calculate score distribution divergence
    fn calculate_score_divergence(&self, active_scores: &HashMap<String, f32>, shadow_scores: &HashMap<String, f32>) -> f32 {
        let mut total_divergence = 0.0;
        let mut count = 0;

        for (key, active_score) in active_scores {
            let shadow_score = shadow_scores.get(key).unwrap_or(&0.0);
            let diff = (active_score - shadow_score).abs();
            total_divergence += diff;
            count += 1;
        }

        if count == 0 {
            0.0
        } else {
            total_divergence / count as f32
        }
    }

    /// Add a drift sample
    pub fn add_drift_sample(&mut self, sample: DriftSample) {
        self.drift_history.push_back(sample);

        // Keep only the most recent samples (up to window_size)
        while self.drift_history.len() > self.config.window_size {
            self.drift_history.pop_front();
        }

        // Check for alert condition
        if self.config.alert_enabled {
            if let Some(alert) = self.check_alert_threshold() {
                self.alerts.push(alert);
            }
        }
    }

    /// Check if alert threshold is exceeded
    fn check_alert_threshold(&self) -> Option<DriftAlert> {
        if self.drift_history.is_empty() {
            return None;
        }

        // Calculate average divergence over window
        let avg_divergence = self.calculate_average_divergence();

        if avg_divergence >= self.config.divergence_threshold {
            let alert_level = AlertLevel::from_divergence(avg_divergence);
            Some(DriftAlert::new(
                PtpTimestamp::from(chrono::Utc::now()),
                alert_level,
                avg_divergence,
                self.config.divergence_threshold,
            ))
        } else {
            None
        }
    }

    /// Calculate average divergence over the history window
    pub fn calculate_average_divergence(&self) -> f32 {
        if self.drift_history.is_empty() {
            return 0.0;
        }

        let sum: f32 = self.drift_history.iter().map(|s| s.divergence_ratio).sum();
        sum / self.drift_history.len() as f32
    }

    /// Get current divergence (last sample or average)
    pub fn current_divergence(&self) -> f32 {
        self.drift_history
            .back()
            .map(|s| s.divergence_ratio)
            .unwrap_or(0.0)
    }

    /// Check if model should be frozen
    pub fn should_freeze(&self) -> bool {
        if !self.config.auto_freeze_enabled {
            return false;
        }

        let avg_divergence = self.calculate_average_divergence();
        avg_divergence >= self.config.divergence_threshold * 1.5  // Freeze at 1.5x threshold
    }

    /// Freeze the active model (prevent further use)
    pub fn freeze_model(&mut self) {
        self.is_frozen = true;
        log::warn!("Active model FROZEN due to excessive drift");
    }

    /// Unfreeze the active model (after rollback or retraining)
    pub fn unfreeze_model(&mut self) {
        self.is_frozen = false;
        log::info!("Active model unfrozen");
    }

    /// Check if model is frozen
    pub fn is_frozen(&self) -> bool {
        self.is_frozen
    }

    /// Get drift history
    pub fn drift_history(&self) -> &VecDeque<DriftSample> {
        &self.drift_history
    }

    /// Get alerts
    pub fn alerts(&self) -> &[DriftAlert] {
        &self.alerts
    }

    /// Clear alerts
    pub fn clear_alerts(&mut self) {
        self.alerts.clear();
    }

    /// Get active model info
    pub fn active_model_info(&self) -> (String, String) {
        (self.active_model.model_id().to_string(), self.active_model.model_version().to_string())
    }

    /// Get shadow model info
    pub fn shadow_model_info(&self) -> (String, String) {
        (self.shadow_model.model_id().to_string(), self.shadow_model.model_version().to_string())
    }

    /// Check if rollback should be triggered
    pub fn should_rollback(&self) -> bool {
        if !self.config.rollback_enabled {
            return false;
        }

        let avg_divergence = self.calculate_average_divergence();
        avg_divergence >= self.config.divergence_threshold * 2.0  // Rollback at 2x threshold
    }

    /// Generate drift visualization data
    pub fn generate_drift_visualization(&self) -> Vec<(f64, f32)> {
        self.drift_history
            .iter()
            .map(|s| {
                let ts_secs = s.timestamp.as_nanos() as f64 / 1_000_000_000.0;
                (ts_secs, s.divergence_ratio)
            })
            .collect()
    }

    /// Get per-category drift statistics
    pub fn category_drift_stats(&self) -> HashMap<String, f32> {
        let mut category_drift: HashMap<String, f32> = HashMap::new();
        let mut category_counts: HashMap<String, usize> = HashMap::new();

        for sample in &self.drift_history {
            for (category, drift) in &sample.category_drift {
                *category_drift.entry(category.clone()).or_insert(0.0) += drift;
                *category_counts.entry(category.clone()).or_insert(0) += 1;
            }
        }

        // Calculate averages
        for (category, total_drift) in category_drift.iter_mut() {
            if let Some(count) = category_counts.get(category) {
                if *count > 0 {
                    *total_drift /= *count as f32;
                }
            }
        }

        category_drift
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_monitor() -> ShadowModelMonitor {
        let active_model = Box::new(MockActiveModel::new("1.0.0"));
        let shadow_model = Box::new(MockShadowModel::new("1.0.0"));
        ShadowModelMonitor::with_defaults(active_model, shadow_model)
    }

    // ============================================================================
    // InputFeatures Tests
    // ============================================================================

    #[test]
    fn test_input_features_creation() {
        let features = InputFeatures::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(features.features, vec![1.0, 2.0, 3.0]);
        assert!(features.metadata.is_empty());
    }

    #[test]
    fn test_input_features_with_metadata() {
        let features = InputFeatures::new(vec![1.0, 2.0])
            .with_metadata("source", "test");

        assert_eq!(features.metadata.get("source"), Some(&"test".to_string()));
    }

    // ============================================================================
    // DriftSample Tests
    // ============================================================================

    #[test]
    fn test_drift_sample_creation() {
        let timestamp = PtpTimestamp::new(0, 0);
        let sample = DriftSample::new(timestamp, 0.15, 100);

        assert_eq!(sample.divergence_ratio, 0.15);
        assert_eq!(sample.sample_count, 100);
        assert!(sample.category_drift.is_empty());
    }

    // ============================================================================
    // AlertLevel Tests
    // ============================================================================

    #[test]
    fn test_alert_level_from_divergence() {
        assert_eq!(AlertLevel::from_divergence(0.05), AlertLevel::Warning);
        assert_eq!(AlertLevel::from_divergence(0.15), AlertLevel::Warning);
        assert_eq!(AlertLevel::from_divergence(0.25), AlertLevel::Critical);
        assert_eq!(AlertLevel::from_divergence(0.45), AlertLevel::Emergency);
    }

    #[test]
    fn test_alert_level_ordering() {
        assert!(AlertLevel::Warning < AlertLevel::Critical);
        assert!(AlertLevel::Critical < AlertLevel::Emergency);
    }

    // ============================================================================
    // DriftAlert Tests
    // ============================================================================

    #[test]
    fn test_drift_alert_warning() {
        let timestamp = PtpTimestamp::new(0, 0);
        let alert = DriftAlert::new(timestamp, AlertLevel::Warning, 0.15, 0.2);

        assert_eq!(alert.alert_level, AlertLevel::Warning);
        assert_eq!(alert.current_divergence, 0.15);
        assert!(!alert.recommendations.is_empty());
    }

    #[test]
    fn test_drift_alert_emergency() {
        let timestamp = PtpTimestamp::new(0, 0);
        let alert = DriftAlert::new(timestamp, AlertLevel::Emergency, 0.5, 0.2);

        assert_eq!(alert.alert_level, AlertLevel::Emergency);
        assert!(alert.recommendations.iter().any(|r| r.contains("EMERGENCY")));
    }

    // ============================================================================
    // Mock Model Tests
    // ============================================================================

    #[test]
    fn test_mock_active_model() {
        let model = MockActiveModel::new("1.0.0");
        assert_eq!(model.model_id(), "active_model");
        assert_eq!(model.model_version(), "1.0.0");
    }

    #[test]
    fn test_mock_active_model_with_drift() {
        let model = MockActiveModel::new("1.0.0").with_drift(0.8);
        let features = InputFeatures::new(vec![1.0]);
        let prediction = model.predict(&features);

        // High drift should produce different prediction
        assert_eq!(prediction.label, "recording");
    }

    #[test]
    fn test_mock_shadow_model() {
        let model = MockShadowModel::new("baseline");
        assert_eq!(model.model_id(), "shadow_model");
        assert_eq!(model.model_version(), "baseline");
    }

    #[test]
    fn test_mock_shadow_model_consistent() {
        let model = MockShadowModel::new("1.0.0");
        let features = InputFeatures::new(vec![1.0]);
        let prediction1 = model.predict(&features);
        let prediction2 = model.predict(&features);

        // Shadow model should always return the same prediction
        assert_eq!(prediction1.label, prediction2.label);
        assert_eq!(prediction1.confidence, prediction2.confidence);
    }

    // ============================================================================
    // ShadowModelMonitor Tests
    // ============================================================================

    #[test]
    fn test_monitor_creation() {
        let monitor = create_test_monitor();
        assert!(!monitor.is_frozen());
        assert_eq!(monitor.current_divergence(), 0.0);
    }

    #[test]
    fn test_compare_predictions_no_drift() {
        let monitor = create_test_monitor();
        let features = InputFeatures::new(vec![1.0, 2.0, 3.0]);

        let comparison = monitor.compare_predictions(&features);

        // Both models should agree without drift
        assert_eq!(comparison.active_prediction, comparison.shadow_prediction);
        assert!(comparison.category_match);
        assert_eq!(comparison.divergence_ratio, 0.0);
    }

    #[test]
    fn test_compare_predictions_with_drift() {
        let active_model = Box::new(MockActiveModel::new("1.0.0").with_drift(0.8));
        let shadow_model = Box::new(MockShadowModel::new("1.0.0"));
        let monitor = ShadowModelMonitor::with_defaults(active_model, shadow_model);

        let features = InputFeatures::new(vec![1.0]);
        let comparison = monitor.compare_predictions(&features);

        // With high drift, predictions should differ
        assert_ne!(comparison.active_prediction, comparison.shadow_prediction);
        assert!(comparison.divergence_ratio > 0.5);
    }

    #[test]
    fn test_add_drift_sample() {
        let mut monitor = create_test_monitor();
        let timestamp = PtpTimestamp::new(0, 0);
        let sample = DriftSample::new(timestamp, 0.1, 100);

        monitor.add_drift_sample(sample);

        assert_eq!(monitor.drift_history().len(), 1);
        assert_eq!(monitor.current_divergence(), 0.1);
    }

    #[test]
    fn test_drift_history_window() {
        let mut monitor = create_test_monitor();

        // Add more samples than window size
        let timestamp = PtpTimestamp::new(0, 0);
        for i in 0..1500 {
            let sample = DriftSample::new(timestamp, i as f32 / 1000.0, 100);
            monitor.add_drift_sample(sample);
        }

        // Should only keep window_size samples
        assert_eq!(monitor.drift_history().len(), 1000);
    }

    #[test]
    fn test_calculate_average_divergence() {
        let mut monitor = create_test_monitor();
        let timestamp = PtpTimestamp::new(0, 0);

        monitor.add_drift_sample(DriftSample::new(timestamp, 0.1, 100));
        monitor.add_drift_sample(DriftSample::new(timestamp, 0.2, 100));
        monitor.add_drift_sample(DriftSample::new(timestamp, 0.3, 100));

        let avg = monitor.calculate_average_divergence();
        assert!((avg - 0.2).abs() < 0.001);  // Should be ~0.2
    }

    #[test]
    fn test_should_freeze_threshold() {
        let mut monitor = create_test_monitor();
        let timestamp = PtpTimestamp::new(0, 0);

        // Add samples that exceed 1.5x threshold (0.2 * 1.5 = 0.3)
        for _ in 0..10 {
            monitor.add_drift_sample(DriftSample::new(timestamp, 0.35, 100));
        }

        assert!(monitor.should_freeze());
    }

    #[test]
    fn test_freeze_model() {
        let mut monitor = create_test_monitor();
        assert!(!monitor.is_frozen());

        monitor.freeze_model();
        assert!(monitor.is_frozen());
    }

    #[test]
    fn test_unfreeze_model() {
        let mut monitor = create_test_monitor();
        monitor.freeze_model();
        assert!(monitor.is_frozen());

        monitor.unfreeze_model();
        assert!(!monitor.is_frozen());
    }

    #[test]
    fn test_should_rollback_threshold() {
        let mut monitor = create_test_monitor();
        let timestamp = PtpTimestamp::new(0, 0);

        // Add samples that exceed 2x threshold (0.2 * 2.0 = 0.4)
        for _ in 0..10 {
            monitor.add_drift_sample(DriftSample::new(timestamp, 0.45, 100));
        }

        assert!(monitor.should_rollback());
    }

    #[test]
    fn test_generate_drift_visualization() {
        let mut monitor = create_test_monitor();
        let timestamp = PtpTimestamp::new(0, 0);

        monitor.add_drift_sample(DriftSample::new(timestamp, 0.1, 100));
        monitor.add_drift_sample(DriftSample::new(timestamp, 0.2, 100));

        let viz = monitor.generate_drift_visualization();
        assert_eq!(viz.len(), 2);
    }

    #[test]
    fn test_alert_generation() {
        let mut monitor = create_test_monitor();
        let timestamp = PtpTimestamp::new(0, 0);

        // Add samples that exceed threshold
        for _ in 0..10 {
            monitor.add_drift_sample(DriftSample::new(timestamp, 0.25, 100));
        }

        // Should have generated an alert
        assert!(!monitor.alerts().is_empty());
        assert_eq!(monitor.alerts()[0].alert_level, AlertLevel::Critical);
    }

    #[test]
    fn test_clear_alerts() {
        let mut monitor = create_test_monitor();
        let timestamp = PtpTimestamp::new(0, 0);

        for _ in 0..10 {
            monitor.add_drift_sample(DriftSample::new(timestamp, 0.25, 100));
        }

        assert!(!monitor.alerts().is_empty());

        monitor.clear_alerts();
        assert!(monitor.alerts().is_empty());
    }

    #[test]
    fn test_model_info() {
        let monitor = create_test_monitor();
        let (id, version) = monitor.active_model_info();
        assert_eq!(id, "active_model");
        assert_eq!(version, "1.0.0");

        let (id, version) = monitor.shadow_model_info();
        assert_eq!(id, "shadow_model");
        assert_eq!(version, "1.0.0");
    }

    #[test]
    fn test_category_drift_stats() {
        let mut monitor = create_test_monitor();
        let timestamp = PtpTimestamp::new(0, 0);

        let mut sample = DriftSample::new(timestamp, 0.1, 100);
        sample.category_drift.insert("playback".to_string(), 0.15);
        sample.category_drift.insert("recording".to_string(), 0.05);
        monitor.add_drift_sample(sample);

        let stats = monitor.category_drift_stats();
        assert_eq!(stats.get("playback"), Some(&0.15));
        assert_eq!(stats.get("recording"), Some(&0.05));
    }
}
