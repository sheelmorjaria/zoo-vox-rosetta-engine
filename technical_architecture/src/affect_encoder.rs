//! Affective VAE Encoder (Risk A Mitigation)
//! ==========================================
//!
//! ONNX Runtime-based inference for the β-VAE encoder (Stream 1).
//! Encodes 54D affective features to 16D continuous latent space.

use anyhow::{Context, Result};
use ndarray::Array1;
use tract_onnx::prelude::*;
use std::path::Path;

// ═══════════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════════

/// ONNX Runtime-based β-VAE encoder for Stream 1 (affective features).
///
/// Encodes 54D affective features to 16D continuous latent space.
/// The model is exported from Python via `affective_export.py`.
///
/// Note: Due to tract's complex generic types, we use a load-and-run pattern
/// rather than storing the runnable. This is slightly less efficient but
/// ensures compatibility across tract versions.
pub struct AffectiveEncoder {
    /// Path to the ONNX model file
    model_path: std::path::PathBuf,
    /// Input dimension (54D)
    input_dim: usize,
    /// Output dimension (16D)
    output_dim: usize,
}

impl AffectiveEncoder {
    /// Input dimension for affective features (54D)
    pub const INPUT_DIM: usize = 54;

    /// Output dimension for affect vector (16D)
    pub const OUTPUT_DIM: usize = 16;

    /// Load an ONNX model from file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        // Verify the file exists
        if !path.exists() {
            anyhow::bail!("Model file not found: {}", path.display());
        }

        Ok(Self {
            model_path: path.to_path_buf(),
            input_dim: Self::INPUT_DIM,
            output_dim: Self::OUTPUT_DIM,
        })
    }

    /// Encode 54D affective features to 16D latent vector.
    pub fn encode(&self, features: &[f32]) -> Result<Array1<f32>> {
        // Validate input
        if features.len() != self.input_dim {
            anyhow::bail!(
                "Input shape mismatch: expected {}, got {}",
                self.input_dim,
                features.len()
            );
        }

        // Load model, create runnable, and run inference in one expression
        // (We can't store the runnable due to complex generic types)
        let result = tract_onnx::onnx()
            .model_for_path(&self.model_path)
            .context("Failed to load ONNX model")?
            .into_runnable()
            .context("Failed to create runnable")?
            .run(tvec!(Tensor::from_shape(&[1, self.input_dim], features)
                .context("Failed to create input tensor")?.into()))
            .context("ONNX inference failed")?;

        // Extract output: (1, 16) -> Vec<f32>
        let output = &result[0];

        // Convert TValue to Tensor then to array
        let output_tensor: Tensor = output
            .clone()
            .into_tensor();

        // Get raw bytes and convert to f32 slice
        let output_bytes = output_tensor.as_bytes();

        // Each f32 is 4 bytes
        let mut output_vec = Vec::with_capacity(self.output_dim);
        for i in 0..self.output_dim {
            let start = i * 4;
            let end = start + 4;
            if end <= output_bytes.len() {
                let bytes = [output_bytes[start], output_bytes[start + 1],
                             output_bytes[start + 2], output_bytes[start + 3]];
                let val = f32::from_le_bytes(bytes);
                output_vec.push(val);
            }
        }

        Ok(Array1::from(output_vec))
    }

    /// Get the input dimension.
    pub const fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get the output dimension.
    pub const fn output_dim(&self) -> usize {
        self.output_dim
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_constants() {
        assert_eq!(AffectiveEncoder::INPUT_DIM, 54);
        assert_eq!(AffectiveEncoder::OUTPUT_DIM, 16);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Integration test: Requires actual ONNX model from Python export
    #[test]
    #[ignore] // Requires ONNX model from Python export
    fn test_real_model_inference() {
        let encoder = AffectiveEncoder::load("models/dual_stream/affective_encoder.onnx")
            .expect("Failed to load model");

        let features: Vec<f32> = (0..54).map(|i| i as f32 / 100.0).collect();
        let affect = encoder.encode(&features).expect("Inference failed");

        assert_eq!(affect.len(), 16);

        for val in affect.iter() {
            assert!(val.is_finite());
        }
    }
}
