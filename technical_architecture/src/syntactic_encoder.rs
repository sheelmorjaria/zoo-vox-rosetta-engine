//! Syntactic VQ-VAE Encoder (Risk A Mitigation)
//! ============================================
//!
//! ONNX Runtime-based inference for the VQ-VAE encoder (Stream 2).
//! Encodes 44D syntactic features to discrete tokens (0-63).

use anyhow::{Context, Result};
use tract_onnx::prelude::*;
use std::path::Path;

// ═══════════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════════

/// ONNX Runtime-based VQ-VAE encoder for Stream 2 (syntactic features).
///
/// Encodes 44D syntactic features to discrete token (0-63).
/// The model is exported from Python via `syntactic_export.py`.
///
/// Note: Due to tract's complex generic types, we use a load-and-run pattern
/// rather than storing the runnable. This is slightly less efficient but
/// ensures compatibility across tract versions.
pub struct SyntacticEncoder {
    /// Path to the ONNX model file
    model_path: std::path::PathBuf,
    /// Input dimension (44D)
    input_dim: usize,
    /// Codebook size (64 tokens)
    codebook_size: usize,
}

impl SyntacticEncoder {
    /// Input dimension for syntactic features (44D)
    pub const INPUT_DIM: usize = 44;

    /// Codebook size / vocabulary size (64 tokens)
    pub const CODEBOOK_SIZE: usize = 64;

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
            codebook_size: Self::CODEBOOK_SIZE,
        })
    }

    /// Tokenize 44D syntactic features to discrete token.
    pub fn tokenize(&self, features: &[f32]) -> Result<u32> {
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

        // Extract output: (1, 1) -> u32 token
        let output = &result[0];

        // Convert TValue to Tensor then to array
        let output_tensor: Tensor = output
            .clone()
            .into_tensor();

        // Get raw bytes and interpret as i64
        let output_bytes = output_tensor.as_bytes();

        // Parse as i64 (little endian) - each i64 is 8 bytes
        if output_bytes.len() < 8 {
            anyhow::bail!("Output too short for i64: got {} bytes", output_bytes.len());
        }

        let token = i64::from_le_bytes([
            output_bytes[0], output_bytes[1], output_bytes[2], output_bytes[3],
            output_bytes[4], output_bytes[5], output_bytes[6], output_bytes[7],
        ]);

        // Validate token range
        if token < 0 || token >= self.codebook_size as i64 {
            anyhow::bail!(
                "Token out of range: expected 0-{}, got {}",
                self.codebook_size - 1,
                token
            );
        }

        Ok(token as u32)
    }

    /// Get the input dimension.
    pub const fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get the codebook size.
    pub const fn codebook_size(&self) -> usize {
        self.codebook_size
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
        assert_eq!(SyntacticEncoder::INPUT_DIM, 44);
        assert_eq!(SyntacticEncoder::CODEBOOK_SIZE, 64);
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
        let encoder = SyntacticEncoder::load("models/dual_stream/syntactic_encoder.onnx")
            .expect("Failed to load model");

        let features: Vec<f32> = (0..44).map(|i| i as f32 / 100.0).collect();
        let token = encoder.tokenize(&features).expect("Inference failed");

        assert!(token < 64);
    }
}
