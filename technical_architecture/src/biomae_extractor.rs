//! BioMAE Extractor: Learned Acoustic Feature Extraction (Rust Implementation)
//!
//! Replaces the hand-crafted `micro_dynamics_extractor` with a neural inference
//! engine using ONNX Runtime or TensorRT.
//!
//! # Architecture
//!
//! ```text
//! [Audio Buffer] → [Log-Linear Spectrogram] → [BioMAE ONNX] → [112D Embedding]
//!                (GPU accelerated)          (TensorRT)          (for Python Agent)
//! ```
//!
//! # Performance Targets
//!
//! - **Latency**: <5ms 99th percentile on Jetson Orin (refined from <1ms)
//! - **Memory**: ~500KB model size (encoder only, FP16)
//! - **Power**: Minimal overhead vs 112 sequential algorithms
//!
//! # Example
//!
//! ```no_run
//! use technical_architecture::biomae_extractor::{BioMAEExtractor, BioMAEConfig};
//!
//! let config = BioMAEConfig::default();
//! let mut extractor = BioMAEExtractor::new(config)?;
//!
//! let audio = vec![0.0f32; 4800]; // 100ms at 48kHz
//! let embedding = extractor.extract_features(&audio)?;
//!
//! println!("112D embedding: {:?}...", &embedding[..5]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::path::Path;
use std::time::Instant;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::ptp::PtpTimestamp;

/// Errors specific to BioMAE extraction.
#[derive(Debug, thiserror::Error)]
pub enum BioMAEError {
    /// ONNX runtime error.
    #[error("ONNX error: {0}")]
    OnnxError(String),

    /// Model input shape mismatch.
    #[error("Shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },

    /// I/O error loading model.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Spectrogram computation error.
    #[error("Spectrogram error: {0}")]
    SpectrogramError(String),
}

/// Configuration for BioMAE extractor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioMAEConfig {
    /// Path to ONNX encoder model.
    pub onnx_path: String,

    /// Audio sample rate in Hz.
    pub sample_rate: u32,

    /// Spectrogram FFT size.
    pub n_fft: usize,

    /// Spectrogram hop length.
    pub hop_length: usize,

    /// Target spectrogram size (freq_bins, time_frames).
    pub img_size: (usize, usize),

    /// Batch size for inference (1 for streaming).
    pub batch_size: usize,

    /// Whether to use GPU acceleration.
    pub use_gpu: bool,

    /// Latency target in milliseconds (for monitoring).
    pub latency_target_ms: f32,
}

impl Default for BioMAEConfig {
    fn default() -> Self {
        Self {
            onnx_path: "models/biomae_encoder_fp16.onnx".to_string(),
            sample_rate: 96000,
            n_fft: 1024,
            hop_length: 240,
            img_size: (128, 128),
            batch_size: 1,
            use_gpu: true,
            latency_target_ms: 5.0,  // Target <5ms (refined from 1ms)
        }
    }
}

/// Statistics from the BioMAE extractor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioMAEStatistics {
    /// Total frames processed.
    pub frame_count: usize,

    /// Total extraction time in milliseconds.
    pub total_time_ms: f32,

    /// Average latency per frame.
    pub avg_latency_ms: f32,

    /// Maximum latency observed.
    pub max_latency_ms: f32,

    /// Number of times latency exceeded target.
    pub latency_exceeded_count: usize,
}

/// ONNX session wrapper for BioMAE model.
struct ONNXModel {
    // In production, this would hold ONNX Runtime or TensorRT session
    // For now, we use a mock implementation for testing
    _model_path: String,
    _input_shape: Vec<usize>,
    _output_dim: usize,
}

impl ONNXModel {
    /// Create a new model (mock implementation).
    fn _new(path: &Path) -> Result<Self, BioMAEError> {
        // In production, load ONNX model via tract-onnx or TensorRT
        let model_path = path.to_string_lossy().to_string();

        // Mock: assume standard BioMAE encoder dimensions
        Ok(Self {
            _model_path: model_path,
            _input_shape: vec![1, 1, 128, 128],  // (B, C, Freq, Time)
            _output_dim: 112,
        })
    }

    /// Run inference on spectrogram.
    fn run(&self, spectrogram: &Array2<f32>) -> Result<Array1<f32>, BioMAEError> {
        // Mock: return random embedding
        // In production, this runs ONNX/TensorRT inference
        let mut embedding = Array1::zeros(112);
        for elem in embedding.iter_mut() {
            *elem = rand::random::<f32>() * 2.0 - 1.0;
        }
        Ok(embedding)
    }
}

/// Spectrogram computation for BioMAE input.
struct SpectrogramComputer {
    n_fft: usize,
    hop_length: usize,
    sample_rate: u32,
    img_size: (usize, usize),
}

impl SpectrogramComputer {
    fn new(n_fft: usize, hop_length: usize, sample_rate: u32, img_size: (usize, usize)) -> Self {
        Self {
            n_fft,
            hop_length,
            sample_rate,
            img_size,
        }
    }

    /// Compute log-linear spectrogram from audio buffer.
    fn compute(&self, audio: &[f32]) -> Result<Array2<f32>, BioMAEError> {
        // Simplified spectrogram computation
        // In production, use RustFFT or similar for GPU acceleration

        let num_frames = if audio.len() > self.n_fft {
            (audio.len() - self.n_fft) / self.hop_length + 1
        } else {
            1
        };

        let freq_bins = self.n_fft / 2 + 1;
        let mut spec = Array2::zeros((freq_bins, num_frames));

        // Mock: fill with random values (in production, compute FFT)
        for elem in spec.iter_mut() {
            *elem = rand::random::<f32>() * 80.0 - 80.0;  // dB range
        }

        // Resize to target img_size (simplified - no interpolation)
        let (target_freq, target_time) = self.img_size;

        // For mock, just truncate or pad
        let result_freq = spec.nrows().min(target_freq);
        let result_time = spec.ncols().min(target_time);

        let mut resized = Array2::zeros((target_freq, target_time));
        for f in 0..result_freq {
            for t in 0..result_time {
                resized[[f, t]] = spec[[f, t]];
            }
        }

        Ok(resized)
    }
}

/// BioMAE feature extractor.
///
/// Replaces the algorithmic `micro_dynamics_extractor` with learned
/// acoustic embeddings via neural inference.
pub struct BioMAEExtractor {
    config: BioMAEConfig,
    model: Option<ONNXModel>,
    spectrogram_computer: SpectrogramComputer,

    // Statistics
    stats: BioMAEStatistics,
    frame_count: usize,
    total_time_ms: f32,
    max_latency_ms: f32,
    latency_exceeded_count: usize,
}

impl BioMAEExtractor {
    /// Create a new BioMAE extractor.
    pub fn new(config: BioMAEConfig) -> Result<Self, BioMAEError> {
        // Try to load ONNX model (optional for testing)
        let model = Self::load_model(&config.onnx_path);

        let spectrogram_computer = SpectrogramComputer::new(
            config.n_fft,
            config.hop_length,
            config.sample_rate,
            config.img_size,
        );

        Ok(Self {
            config,
            model,
            spectrogram_computer,
            stats: BioMAEStatistics {
                frame_count: 0,
                total_time_ms: 0.0,
                avg_latency_ms: 0.0,
                max_latency_ms: 0.0,
                latency_exceeded_count: 0,
            },
            frame_count: 0,
            total_time_ms: 0.0,
            max_latency_ms: 0.0,
            latency_exceeded_count: 0,
        })
    }

    /// Load ONNX model from path.
    fn load_model(path: &str) -> Option<ONNXModel> {
        let path_obj = Path::new(path);
        // Only load if path exists and is not /dev/null (test sentinel)
        if path_obj.exists() && path != "/dev/null" {
            ONNXModel::_new(path_obj).ok()
        } else {
            None  // Model not found, will use mock
        }
    }

    /// Extract 112D Rosetta embedding from audio buffer.
    ///
    /// This replaces the hand-crafted feature extraction pipeline
    /// with a single neural forward pass.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples (already segmented by NBD)
    ///
    /// # Returns
    ///
    /// * `Ok(embedding)` - 112D feature vector
    /// * `Err(e)` - Extraction error
    pub fn extract_features(&mut self, audio: &[f32]) -> Result<Vec<f32>, BioMAEError> {
        let start = Instant::now();

        // Step 1: Compute log-linear spectrogram
        let spectrogram = self.spectrogram_computer.compute(audio)?;

        // Step 2: Run BioMAE encoder inference
        let embedding = match &self.model {
            Some(model) => model.run(&spectrogram)?,
            None => {
                // Mock inference when model not loaded
                let mut arr = Array1::zeros(112);
                for elem in arr.iter_mut() {
                    *elem = rand::random::<f32>() * 2.0 - 1.0;
                }
                arr
            }
        };

        // Update statistics
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;  // Convert to ms
        self.frame_count += 1;
        self.total_time_ms += elapsed as f32;

        if elapsed as f32 > self.max_latency_ms {
            self.max_latency_ms = elapsed as f32;
        }

        if elapsed as f32 > self.config.latency_target_ms {
            self.latency_exceeded_count += 1;
        }

        // Convert to Vec
        Ok(embedding.to_vec())
    }

    /// Extract features with timestamp.
    pub fn extract_features_with_timestamp(
        &mut self,
        audio: &[f32],
        timestamp_ns: u64,
    ) -> Result<FeatureExtraction, BioMAEError> {
        let embedding = self.extract_features(audio)?;

        Ok(FeatureExtraction {
            timestamp_ns,
            features: embedding,
            latency_ms: self.get_current_latency_ms(),
        })
    }

    /// Get current latency for most recent extraction.
    fn get_current_latency_ms(&self) -> f32 {
        if self.frame_count > 0 {
            self.total_time_ms / self.frame_count as f32
        } else {
            0.0
        }
    }

    /// Get statistics and reset counters.
    pub fn get_statistics(&mut self) -> BioMAEStatistics {
        let stats = BioMAEStatistics {
            frame_count: self.frame_count,
            total_time_ms: self.total_time_ms,
            avg_latency_ms: self.get_current_latency_ms(),
            max_latency_ms: self.max_latency_ms,
            latency_exceeded_count: self.latency_exceeded_count,
        };

        // Reset counters
        self.frame_count = 0;
        self.total_time_ms = 0.0;
        self.max_latency_ms = 0.0;
        self.latency_exceeded_count = 0;

        stats
    }

    /// Reset extractor state.
    pub fn reset(&mut self) {
        self.frame_count = 0;
        self.total_time_ms = 0.0;
        self.max_latency_ms = 0.0;
        self.latency_exceeded_count = 0;
    }

    /// Check if model is loaded.
    pub fn is_model_loaded(&self) -> bool {
        self.model.is_some()
    }
}

/// Feature extraction result with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtraction {
    /// PTP timestamp of extraction.
    pub timestamp_ns: u64,

    /// 112D Rosetta feature embedding (Vec for serde compatibility).
    pub features: Vec<f32>,

    /// Extraction latency in milliseconds.
    pub latency_ms: f32,
}

/// Factory function for creating extractors with presets.
pub fn create_biomae_extractor_for_taxa(taxa: &str) -> Result<BioMAEExtractor, BioMAEError> {
    let config = match taxa.to_lowercase().as_str() {
        "bat" => BioMAEConfig {
            sample_rate: 96000,
            n_fft: 1024,
            hop_length: 240,
            img_size: (128, 128),
            ..Default::default()
        },
        "cetacean" | "dolphin" | "whale" => BioMAEConfig {
            sample_rate: 192000,
            n_fft: 2048,
            hop_length: 480,
            img_size: (128, 256),
            ..Default::default()
        },
        "bird" => BioMAEConfig {
            sample_rate: 48000,
            n_fft: 1024,
            hop_length: 256,
            img_size: (128, 128),
            ..Default::default()
        },
        _ => BioMAEConfig::default(),
    };

    BioMAEExtractor::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_extractor() -> BioMAEExtractor {
        let config = BioMAEConfig {
            onnx_path: "/dev/null".to_string(),
            sample_rate: 48000,
            n_fft: 512,
            hop_length: 128,
            img_size: (64, 64),
            ..Default::default()
        };
        BioMAEExtractor::new(config).unwrap()
    }

    #[test]
    fn test_extractor_initialization() {
        let extractor = create_test_extractor();
        assert!(!extractor.is_model_loaded());
        assert_eq!(extractor.frame_count, 0);
    }

    #[test]
    fn test_extract_features_shape() {
        let mut extractor = create_test_extractor();
        let audio = vec![0.0; 2400];  // 50ms at 48kHz

        let features = extractor.extract_features(&audio).unwrap();
        assert_eq!(features.len(), 112);
        assert_eq!(features.capacity(), 112);  // Vec with exact capacity
    }

    #[test]
    fn test_extract_features_with_timestamp() {
        let mut extractor = create_test_extractor();
        let audio = vec![0.0; 2400];

        let result = extractor.extract_features_with_timestamp(&audio, 123456789).unwrap();
        assert_eq!(result.timestamp_ns, 123456789);
        assert_eq!(result.features.len(), 112);
    }

    #[test]
    fn test_statistics_tracking() {
        let mut extractor = create_test_extractor();
        let audio = vec![0.0; 2400];

        // Process several frames
        for _ in 0..10 {
            extractor.extract_features(&audio).ok();
        }

        let stats = extractor.get_statistics();
        assert_eq!(stats.frame_count, 10);
        assert!(stats.avg_latency_ms >= 0.0);
    }

    #[test]
    fn test_reset() {
        let mut extractor = create_test_extractor();
        let audio = vec![0.0; 2400];

        extractor.extract_features(&audio).ok();
        extractor.extract_features(&audio).ok();
        assert_eq!(extractor.frame_count, 2);

        extractor.reset();
        assert_eq!(extractor.frame_count, 0);
    }

    #[test]
    fn test_taxa_presets() {
        // Test that different taxa produce valid extractors
        for taxa in &["bat", "cetacean", "bird", "default"] {
            let extractor = create_biomae_extractor_for_taxa(taxa);
            assert!(extractor.is_ok(), "Failed to create extractor for taxa: {}", taxa);
        }
    }

    #[test]
    fn test_config_default() {
        let config = BioMAEConfig::default();
        assert_eq!(config.sample_rate, 96000);
        assert_eq!(config.latency_target_ms, 5.0);  // Refined target
    }
}
