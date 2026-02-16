//! Multi-scale aggregation module
//!
//! This module provides hierarchical aggregation across multiple time scales
//! for extracting multi-scale features from frame-level representations.

pub mod aggregators;
pub mod hierarchical;

pub use aggregators::{MultiScaleFeatures, StatisticalAggregator};
pub use hierarchical::{HierarchicalAggregator, HierarchicalConfig, HierarchicalFeatures};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_scale_module_compatibility() {
        // Verify both aggregators and hierarchical can be used
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Test aggregator
        let features = StatisticalAggregator::compute_all(&data);
        assert_eq!(features.mean, 3.0);

        // Test hierarchical
        let aggregator = HierarchicalAggregator::new(48000);
        // Verify it was created successfully (private fields can't be accessed directly)
        let config = HierarchicalConfig::default();
        assert_eq!(config.frame_duration_ms, 25.0);
    }

    #[test]
    fn test_multi_scale_features_default() {
        let features = MultiScaleFeatures::default();

        assert_eq!(features.mean, 0.0);
        assert_eq!(features.std_dev, 0.0);
    }

    #[test]
    fn test_hierarchical_config_default() {
        let config = HierarchicalConfig::default();

        assert_eq!(config.frame_duration_ms, 25.0);
        assert_eq!(config.hop_duration_ms, 10.0);
    }
}
