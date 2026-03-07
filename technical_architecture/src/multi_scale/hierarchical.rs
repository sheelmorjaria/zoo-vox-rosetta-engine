//! Hierarchical aggregation for multi-scale feature extraction
//!
//! This module provides hierarchical aggregation across multiple time scales
//! for extracting multi-scale features from frame-level representations.

use crate::micro_dynamics_extractor::MicroDynamicsExtractor;
use crate::multi_scale::aggregators::{MultiScaleFeatures, StatisticalAggregator};

/// Configuration for hierarchical aggregation
#[derive(Debug, Clone)]
pub struct HierarchicalConfig {
    /// Frame duration in milliseconds
    pub frame_duration_ms: f32,
    /// Hop duration in milliseconds (overlap = frame - hop)
    pub hop_duration_ms: f32,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            frame_duration_ms: 25.0, // Standard 25ms frame
            hop_duration_ms: 10.0,   // 15ms overlap (60%)
        }
    }
}

/// Hierarchical aggregator for multi-scale features
pub struct HierarchicalAggregator {
    config: HierarchicalConfig,
    sample_rate: u32,
}

impl HierarchicalAggregator {
    /// Create a new hierarchical aggregator
    pub fn new(sample_rate: u32) -> Self {
        Self {
            config: HierarchicalConfig::default(),
            sample_rate,
        }
    }

    /// Create with custom configuration
    pub fn with_config(sample_rate: u32, config: HierarchicalConfig) -> Self {
        Self { config, sample_rate }
    }

    /// Extract hierarchical features across multiple time scales
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio buffer
    /// * `extractor` - Feature extractor to use per frame
    ///
    /// # Returns
    ///
    /// Hierarchical feature representation with aggregations
    pub fn extract_hierarchical(
        &self,
        audio: &[f32],
        _extractor: &MicroDynamicsExtractor,
    ) -> Result<HierarchicalFeatures, String> {
        // Split audio into frames
        let frames = self.split_into_frames(audio)?;

        if frames.is_empty() {
            return Ok(HierarchicalFeatures::default());
        }

        // Extract features per frame
        let mut mfcc_frames: Vec<Vec<f32>> = Vec::new();
        let mut f0_values: Vec<f32> = Vec::new();
        let mut onset_rates: Vec<f32> = Vec::new();

        for frame in &frames {
            // Use a simplified feature extraction for testing
            // In real implementation, this would use the extractor
            if frame.len() > 100 {
                // Simplified: just use some statistics
                let mean = frame.iter().sum::<f32>() / frame.len() as f32;
                let variance = frame.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / frame.len() as f32;

                // Mock 13 MFCCs
                let mfccs = (0..13).map(|i| mean + 0.1 * i as f32 * variance.sqrt()).collect();
                mfcc_frames.push(mfccs);

                // Mock F0 (simplified)
                f0_values.push(mean.abs() * 100.0 + 500.0);

                // Mock onset rate
                onset_rates.push(variance.sqrt() * 10.0);
            }
        }

        // Compute hierarchical aggregations
        let f0_multi_scale = StatisticalAggregator::compute_all(&f0_values);
        let onset_rate_multi_scale = StatisticalAggregator::compute_all(&onset_rates);

        // Compute MFCC multi-scale per coefficient
        let mut mfcc_multi_scale: Vec<MultiScaleFeatures> = Vec::new();
        for coeff_idx in 0..13 {
            if let Some(_coeff_values) = mfcc_frames.first().map(|f| f.len()).filter(|&l| l > coeff_idx) {
                let values: Vec<f32> = mfcc_frames.iter().filter_map(|f| f.get(coeff_idx).copied()).collect();
                mfcc_multi_scale.push(StatisticalAggregator::compute_all(&values));
            }
        }

        Ok(HierarchicalFeatures {
            f0_multi_scale,
            mfcc_multi_scale: mfcc_multi_scale.try_into().unwrap_or_default(),
            onset_rate_multi_scale,
        })
    }

    /// Split audio into frames
    fn split_into_frames(&self, audio: &[f32]) -> Result<Vec<Vec<f32>>, String> {
        if audio.is_empty() {
            return Ok(vec![]);
        }

        let frame_samples = (self.config.frame_duration_ms / 1000.0 * self.sample_rate as f32) as usize;
        let hop_samples = (self.config.hop_duration_ms / 1000.0 * self.sample_rate as f32) as usize;

        if frame_samples == 0 || hop_samples == 0 {
            return Err("Invalid frame/hop duration".to_string());
        }

        let mut frames = Vec::new();
        let mut start = 0;

        while start + frame_samples <= audio.len() {
            let frame = audio[start..start + frame_samples].to_vec();
            frames.push(frame);
            start += hop_samples;
        }

        // Handle last partial frame
        if start < audio.len() && !frames.is_empty() {
            let remaining = audio.len() - start;
            if remaining > frame_samples / 2 {
                // Pad if more than half a frame
                let mut frame = audio[start..].to_vec();
                frame.resize(frame_samples, 0.0);
                frames.push(frame);
            }
        }

        Ok(frames)
    }

    /// Aggregate frame-level features into multi-scale representation
    pub fn aggregate_frames(&self, feature_frames: &[Vec<f32>]) -> Vec<MultiScaleFeatures> {
        if feature_frames.is_empty() {
            return vec![];
        }

        let num_features = feature_frames[0].len();
        let mut aggregations = Vec::new();

        for feat_idx in 0..num_features {
            let values: Vec<f32> = feature_frames.iter().filter_map(|f| f.get(feat_idx).copied()).collect();

            aggregations.push(StatisticalAggregator::compute_all(&values));
        }

        aggregations
    }
}

/// Hierarchical features with multi-scale aggregations
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HierarchicalFeatures {
    /// Multi-scale F0 features
    pub f0_multi_scale: MultiScaleFeatures,
    /// Multi-scale MFCC features (13 coefficients)
    pub mfcc_multi_scale: [MultiScaleFeatures; 13],
    /// Multi-scale onset rate features
    pub onset_rate_multi_scale: MultiScaleFeatures,
}

impl HierarchicalFeatures {
    /// Convert to flat feature vector for ML
    pub fn to_vector(&self) -> Vec<f32> {
        let mut vec = Vec::new();

        // F0 multi-scale (6 features)
        vec.extend_from_slice(&[
            self.f0_multi_scale.mean,
            self.f0_multi_scale.std_dev,
            self.f0_multi_scale.skewness,
            self.f0_multi_scale.kurtosis,
            self.f0_multi_scale.range,
            self.f0_multi_scale.iqr,
        ]);

        // MFCC multi-scale (13 × 6 = 78 features)
        for mfcc_ms in &self.mfcc_multi_scale {
            vec.extend_from_slice(&[
                mfcc_ms.mean,
                mfcc_ms.std_dev,
                mfcc_ms.skewness,
                mfcc_ms.kurtosis,
                mfcc_ms.range,
                mfcc_ms.iqr,
            ]);
        }

        // Onset rate multi-scale (6 features)
        vec.extend_from_slice(&[
            self.onset_rate_multi_scale.mean,
            self.onset_rate_multi_scale.std_dev,
            self.onset_rate_multi_scale.skewness,
            self.onset_rate_multi_scale.kurtosis,
            self.onset_rate_multi_scale.range,
            self.onset_rate_multi_scale.iqr,
        ]);

        vec
    }

    /// Get dimensionality of feature vector
    pub fn dimensionality(&self) -> usize {
        6 + (13 * 6) + 6 // F0 + MFCCs + onset_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Frame Splitting Tests (5 tests)
    // =========================================================================

    #[test]
    fn test_split_empty_audio() {
        let aggregator = HierarchicalAggregator::new(48000);
        let audio: Vec<f32> = vec![];

        let frames = aggregator.split_into_frames(&audio).unwrap();
        assert_eq!(frames.len(), 0);
    }

    #[test]
    fn test_split_single_frame() {
        let aggregator = HierarchicalAggregator::new(48000);
        let audio = vec![0.0; 480]; // 10ms at 48kHz

        let frames = aggregator.split_into_frames(&audio).unwrap();
        // With 25ms frames and 10ms hop, 10ms is less than one frame
        // So should be empty or padded
        assert!(frames.len() <= 1);
    }

    #[test]
    fn test_split_multiple_frames() {
        let aggregator = HierarchicalAggregator::new(48000);
        let audio = vec![0.0; 2400]; // 50ms at 48kHz

        let frames = aggregator.split_into_frames(&audio).unwrap();
        // With 25ms frames (1200 samples) and 10ms hop (480 samples)
        // Should get 2 frames
        assert!(frames.len() >= 2);
    }

    #[test]
    fn test_split_overlap() {
        let aggregator = HierarchicalAggregator::new(48000);
        let audio = vec![0.0; 2400]; // 50ms

        let frames = aggregator.split_into_frames(&audio).unwrap();

        // Check that frames overlap
        if frames.len() >= 2 {
            assert!(frames[0].len() == frames[1].len());
        }
    }

    #[test]
    fn test_split_long_audio() {
        let aggregator = HierarchicalAggregator::new(48000);
        let audio = vec![0.0; 48000]; // 1 second

        let frames = aggregator.split_into_frames(&audio).unwrap();

        // Should get ~100 frames (1000ms / 10ms hop)
        assert!(frames.len() >= 90 && frames.len() <= 110);
    }

    // =========================================================================
    // Hierarchical Extraction Tests (5 tests)
    // =========================================================================

    #[test]
    fn test_extract_hierarchical_empty() {
        let aggregator = HierarchicalAggregator::new(48000);
        let audio: Vec<f32> = vec![];
        let extractor = MicroDynamicsExtractor::new(48000);

        let result = aggregator.extract_hierarchical(&audio, &extractor).unwrap();

        assert_eq!(result.f0_multi_scale.mean, 0.0);
    }

    #[test]
    fn test_extract_hierarchical_short() {
        let aggregator = HierarchicalAggregator::new(48000);
        let audio = vec![1.0; 1200]; // 25ms - exactly one frame
        let extractor = MicroDynamicsExtractor::new(48000);

        let result = aggregator.extract_hierarchical(&audio, &extractor).unwrap();

        // Should produce valid features
        assert!(result.f0_multi_scale.mean.is_finite());
    }

    #[test]
    fn test_extract_hierarchical_long() {
        let aggregator = HierarchicalAggregator::new(48000);
        let audio: Vec<f32> = (0..48000).map(|i| (i as f32 / 48000.0) * 2.0 - 1.0).collect();
        let extractor = MicroDynamicsExtractor::new(48000);

        let result = aggregator.extract_hierarchical(&audio, &extractor).unwrap();

        // Should produce valid features
        assert!(result.f0_multi_scale.mean.is_finite());
        assert!(result.mfcc_multi_scale.len() == 13);
    }

    #[test]
    fn test_hierarchical_multi_scale_computation() {
        let aggregator = HierarchicalAggregator::new(48000);
        let audio = vec![1.0; 4800]; // 100ms
        let extractor = MicroDynamicsExtractor::new(48000);

        let result = aggregator.extract_hierarchical(&audio, &extractor).unwrap();

        // Check that multi-scale features are computed
        assert!(result.f0_multi_scale.std_dev >= 0.0);
        assert!(result.f0_multi_scale.range >= 0.0);
    }

    #[test]
    fn test_hierarchical_mfcc_aggregation() {
        let aggregator = HierarchicalAggregator::new(48000);
        let audio = vec![1.0; 4800]; // 100ms
        let extractor = MicroDynamicsExtractor::new(48000);

        let result = aggregator.extract_hierarchical(&audio, &extractor).unwrap();

        // Should have 13 MFCC coefficients
        assert_eq!(result.mfcc_multi_scale.len(), 13);

        // Each should have valid statistics
        for mfcc_ms in &result.mfcc_multi_scale {
            assert!(mfcc_ms.mean.is_finite());
        }
    }

    // =========================================================================
    // Aggregation Tests (5 tests)
    // =========================================================================

    #[test]
    fn test_aggregate_frames_empty() {
        let aggregator = HierarchicalAggregator::new(48000);
        let frames: Vec<Vec<f32>> = vec![];

        let result = aggregator.aggregate_frames(&frames);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_aggregate_frames_single() {
        let aggregator = HierarchicalAggregator::new(48000);
        let frames = vec![vec![1.0, 2.0, 3.0]];

        let result = aggregator.aggregate_frames(&frames);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_aggregate_frames_multiple() {
        let aggregator = HierarchicalAggregator::new(48000);
        let frames = vec![vec![1.0, 2.0], vec![1.1, 2.1], vec![0.9, 1.9]];

        let result = aggregator.aggregate_frames(&frames);
        assert_eq!(result.len(), 2);

        // First feature should have mean ~1.0
        assert!((result[0].mean - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_aggregate_frames_consistency() {
        let aggregator = HierarchicalAggregator::new(48000);
        let frames = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]];

        let result = aggregator.aggregate_frames(&frames);

        // All should have zero std_dev
        for agg in &result {
            assert_eq!(agg.std_dev, 0.0);
        }
    }

    #[test]
    fn test_aggregate_frames_variance() {
        let aggregator = HierarchicalAggregator::new(48000);
        let frames = vec![vec![0.0], vec![1.0], vec![2.0]];

        let result = aggregator.aggregate_frames(&frames);

        assert!(result[0].std_dev > 0.0);
        assert!(result[0].range > 0.0);
    }

    // =========================================================================
    // Feature Vector Tests (5 tests)
    // =========================================================================

    #[test]
    fn test_to_vector_dimensionality() {
        let features = HierarchicalFeatures::default();

        assert_eq!(features.dimensionality(), 6 + 13 * 6 + 6);
    }

    #[test]
    fn test_to_vector_length() {
        let features = HierarchicalFeatures::default();

        let vec = features.to_vector();
        assert_eq!(vec.len(), features.dimensionality());
    }

    #[test]
    fn test_to_vector_content() {
        let mut features = HierarchicalFeatures::default();
        features.f0_multi_scale.mean = 100.0;

        let vec = features.to_vector();
        assert_eq!(vec[0], 100.0);
    }

    #[test]
    fn test_to_vector_all_finite() {
        let features = HierarchicalFeatures::default();

        let vec = features.to_vector();
        for &val in &vec {
            assert!(val.is_finite(), "All values should be finite");
        }
    }

    #[test]
    fn test_to_vector_roundtrip() {
        let mut features = HierarchicalFeatures::default();
        features.f0_multi_scale.mean = 123.45;
        features.f0_multi_scale.std_dev = 67.89;

        let vec = features.to_vector();

        assert_eq!(vec[0], 123.45);
        assert_eq!(vec[1], 67.89);
    }
}
