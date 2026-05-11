//! Continuous Manifold Encoder (Stage 3 Replacement)
//! ===============================================
//!
//! ONNX Runtime-based inference for the UMAP+VAE manifold pipeline.
//! Replaces PCA+BGMM with a continuous probabilistic manifold.
//!
//! Architecture:
//!   112D BioMAE → [UMAP Encoder] → 30D → [VAE Encoder] → 16D Latent
//!
//! Key improvements over PCA+BGMM:
//! - UMAP preserves non-linear local gradients (arousal, affect continua)
//! - VAE provides smooth interpolable latent space
//! - Long-tail rescue: rare calls preserved, not pruned

use anyhow::{Context, Result};
use ndarray::Array1;
use tract_onnx::prelude::*;
use std::path::Path;

// ═══════════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════════

/// UMAP Encoder: 112D BioMAE → 30D non-linear embedding
///
/// Trained via parametric UMAP to preserve local neighborhood structure.
/// Exports to ONNX via `parametric_umap.py`.
pub struct UMAPEncoder {
    /// Runnable ONNX model
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    input_dim: usize,
    output_dim: usize,
}

impl UMAPEncoder {
    /// Input dimension: 112D BioMAE features
    pub const INPUT_DIM: usize = 112;

    /// Output dimension: 30D UMAP embedding
    pub const OUTPUT_DIM: usize = 30;

    /// Load UMAP encoder ONNX model
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        let model = tract_onnx::onnx()
            .model_for_path(path)
            .context("Failed to load UMAP ONNX model")?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self {
            model,
            input_dim: Self::INPUT_DIM,
            output_dim: Self::OUTPUT_DIM,
        })
    }

    /// Encode 112D BioMAE features to 30D UMAP embedding
    pub fn encode(&self, features_112d: &[f32]) -> Result<Array1<f32>> {
        if features_112d.len() != self.input_dim {
            anyhow::bail!(
                "UMAP input shape mismatch: expected {}, got {}",
                self.input_dim,
                features_112d.len()
            );
        }

        let input = Tensor::from_shape(&[1, self.input_dim], features_112d)?;
        let result = self.model.run(tvec!(input.into()))?;

        // Extract output: (1, 30) -> Vec<f32>
        self.extract_output(&result[0])
    }

    fn extract_output(&self, output: &TValue) -> Result<Array1<f32>> {
        let output_tensor: Tensor = output.clone().into_tensor();
        let output_bytes = output_tensor.as_bytes();

        let mut output_vec = Vec::with_capacity(self.output_dim);
        for i in 0..self.output_dim {
            let start = i * 4;
            let end = start + 4;
            if end <= output_bytes.len() {
                let bytes = [
                    output_bytes[start],
                    output_bytes[start + 1],
                    output_bytes[start + 2],
                    output_bytes[start + 3],
                ];
                output_vec.push(f32::from_le_bytes(bytes));
            }
        }

        Ok(Array1::from(output_vec))
    }

    pub const fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub const fn output_dim(&self) -> usize {
        self.output_dim
    }
}

/// VAE Encoder: 30D UMAP → 16D continuous latent
///
/// Standard VAE with direct mu/logvar outputs (β-VAE for disentanglement).
/// Exports to ONNX via `vocal_vae.py` (encoder wrapper).
pub struct VAEEncoder {
    /// Runnable ONNX model (encoder only, returns mu)
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    input_dim: usize,
    output_dim: usize,
}

impl VAEEncoder {
    /// Input dimension: 30D UMAP embedding
    pub const INPUT_DIM: usize = 30;

    /// Output dimension: 16D VAE latent (mu only)
    pub const OUTPUT_DIM: usize = 16;

    /// Load VAE encoder ONNX model
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        let model = tract_onnx::onnx()
            .model_for_path(path)
            .context("Failed to load VAE encoder ONNX model")?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self {
            model,
            input_dim: Self::INPUT_DIM,
            output_dim: Self::OUTPUT_DIM,
        })
    }

    /// Encode 30D UMAP embedding to 16D VAE latent (mu)
    pub fn encode(&self, umap_30d: &[f32]) -> Result<Array1<f32>> {
        if umap_30d.len() != self.input_dim {
            anyhow::bail!(
                "VAE input shape mismatch: expected {}, got {}",
                self.input_dim,
                umap_30d.len()
            );
        }

        let input = Tensor::from_shape(&[1, self.input_dim], umap_30d)?;
        let result = self.model.run(tvec!(input.into()))?;

        // Extract output: (1, 16) -> Vec<f32>
        self.extract_output(&result[0])
    }

    fn extract_output(&self, output: &TValue) -> Result<Array1<f32>> {
        let output_tensor: Tensor = output.clone().into_tensor();
        let output_bytes = output_tensor.as_bytes();

        let mut output_vec = Vec::with_capacity(self.output_dim);
        for i in 0..self.output_dim {
            let start = i * 4;
            let end = start + 4;
            if end <= output_bytes.len() {
                let bytes = [
                    output_bytes[start],
                    output_bytes[start + 1],
                    output_bytes[start + 2],
                    output_bytes[start + 3],
                ];
                output_vec.push(f32::from_le_bytes(bytes));
            }
        }

        Ok(Array1::from(output_vec))
    }

    pub const fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub const fn output_dim(&self) -> usize {
        self.output_dim
    }
}

/// Continuous Manifold Encoder: Full pipeline 112D → 16D
///
/// Combines UMAP encoder + VAE encoder for end-to-end inference.
/// This is the runtime replacement for PCA+BGMM.
pub struct ManifoldEncoder {
    /// UMAP encoder (112D → 30D)
    umap: UMAPEncoder,
    /// VAE encoder (30D → 16D)
    vae: VAEEncoder,
}

impl ManifoldEncoder {
    /// Input dimension: 112D BioMAE features
    pub const INPUT_DIM: usize = 112;

    /// Final output dimension: 16D VAE latent
    pub const OUTPUT_DIM: usize = 16;

    /// Intermediate UMAP dimension: 30D
    pub const UMAP_DIM: usize = 30;

    /// Load both encoders from ONNX model files
    pub fn load(
        umap_path: impl AsRef<Path>,
        vae_encoder_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let umap = UMAPEncoder::load(umap_path)?;
        let vae = VAEEncoder::load(vae_encoder_path)?;

        Ok(Self { umap, vae })
    }

    /// Encode 112D BioMAE features directly to 16D latent manifold
    ///
    /// Pipeline: 112D → UMAP(30D) → VAE(16D)
    pub fn encode(&self, features_112d: &[f32]) -> Result<Array1<f32>> {
        // Step 1: UMAP encoding (112D → 30D)
        let umap_30d = self.umap.encode(features_112d)?;

        // Step 2: VAE encoding (30D → 16D)
        let latent_16d = self.vae.encode(umap_30d.as_slice().unwrap())?;

        Ok(latent_16d)
    }

    /// Get the UMAP encoder for intermediate access
    pub fn umap_encoder(&self) -> &UMAPEncoder {
        &self.umap
    }

    /// Get the VAE encoder for intermediate access
    pub fn vae_encoder(&self) -> &VAEEncoder {
        &self.vae
    }

    /// Get input dimension
    pub const fn input_dim(&self) -> usize {
        Self::INPUT_DIM
    }

    /// Get output dimension
    pub const fn output_dim(&self) -> usize {
        Self::OUTPUT_DIM
    }

    /// Get intermediate UMAP dimension
    pub const fn umap_dim(&self) -> usize {
        Self::UMAP_DIM
    }
}

/// Manifold statistics for runtime validation
#[derive(Debug, Clone)]
pub struct ManifoldStats {
    /// Mean of latent coordinates (16D)
    pub latent_mean: [f32; 16],
    /// Std of latent coordinates (16D)
    pub latent_std: [f32; 16],
    /// Min of latent coordinates (16D)
    pub latent_min: [f32; 16],
    /// Max of latent coordinates (16D)
    pub latent_max: [f32; 16],
}

impl Default for ManifoldStats {
    fn default() -> Self {
        Self {
            latent_mean: [0.0; 16],
            latent_std: [1.0; 16],
            latent_min: [-1.0; 16],
            latent_max: [1.0; 16],
        }
    }
}

impl ManifoldStats {
    /// Load statistics from manifest JSON
    pub fn from_manifest(_manifest_path: impl AsRef<Path>) -> Result<Self> {
        // TODO: Load from continuous_manifold_manifest.json
        // For now, return defaults
        Ok(Self::default())
    }

    /// Validate that a latent coordinate is within expected bounds
    pub fn validate(&self, latent: &[f32]) -> Result<()> {
        if latent.len() != 16 {
            anyhow::bail!("Latent dimension mismatch: expected 16, got {}", latent.len());
        }

        // Check for NaN/Inf
        for (i, &val) in latent.iter().enumerate() {
            if !val.is_finite() {
                anyhow::bail!("Latent dimension {} is not finite: {}", i, val);
            }
        }

        // Optional: Check bounds (with some tolerance for outliers)
        for (i, &val) in latent.iter().enumerate() {
            let min_bound = self.latent_min[i] - 3.0 * self.latent_std[i];
            let max_bound = self.latent_max[i] + 3.0 * self.latent_std[i];

            if val < min_bound || val > max_bound {
                // Log warning but don't fail - rare calls can be outliers
                log::warn!(
                    "Latent dimension {} is outside expected range: {} (expected {:.2} to {:.2})",
                    i, val, min_bound, max_bound
                );
            }
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_umap_constants() {
        assert_eq!(UMAPEncoder::INPUT_DIM, 112);
        assert_eq!(UMAPEncoder::OUTPUT_DIM, 30);
    }

    #[test]
    fn test_vae_constants() {
        assert_eq!(VAEEncoder::INPUT_DIM, 30);
        assert_eq!(VAEEncoder::OUTPUT_DIM, 16);
    }

    #[test]
    fn test_manifold_constants() {
        assert_eq!(ManifoldEncoder::INPUT_DIM, 112);
        assert_eq!(ManifoldEncoder::OUTPUT_DIM, 16);
        assert_eq!(ManifoldEncoder::UMAP_DIM, 30);
    }

    #[test]
    fn test_manifold_stats_default() {
        let stats = ManifoldStats::default();
        assert_eq!(stats.latent_mean.len(), 16);
        assert_eq!(stats.latent_std.len(), 16);
    }

    #[test]
    fn test_manifold_stats_validate_finite() {
        let stats = ManifoldStats::default();

        // Valid latent
        let valid_latent: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        assert!(stats.validate(&valid_latent).is_ok());

        // Invalid: contains NaN
        let mut nan_latent = valid_latent.clone();
        nan_latent[0] = f32::NAN;
        assert!(stats.validate(&nan_latent).is_err());

        // Invalid: contains Inf
        let mut inf_latent = valid_latent;
        inf_latent[0] = f32::INFINITY;
        assert!(stats.validate(&inf_latent).is_err());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Requires ONNX models from Python export
    #[test]
    #[ignore]
    fn test_real_umap_encoder() {
        let umap = UMAPEncoder::load("models/umap_encoder.onnx")
            .expect("Failed to load UMAP model");

        let features: Vec<f32> = (0..112).map(|i| i as f32 / 100.0).collect();
        let embedding = umap.encode(&features).expect("UMAP encoding failed");

        assert_eq!(embedding.len(), 30);

        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    /// Requires ONNX models from Python export
    #[test]
    #[ignore]
    fn test_real_vae_encoder() {
        let vae = VAEEncoder::load("models/vae/vae_encoder.onnx")
            .expect("Failed to load VAE encoder");

        let umap_output: Vec<f32> = (0..30).map(|i| i as f32 / 100.0).collect();
        let latent = vae.encode(&umap_output).expect("VAE encoding failed");

        assert_eq!(latent.len(), 16);

        for val in latent.iter() {
            assert!(val.is_finite());
        }
    }

    /// Full pipeline test with real ONNX models
    #[test]
    #[ignore]
    fn test_real_manifold_encoder() {
        let manifold = ManifoldEncoder::load(
            "models/umap_encoder.onnx",
            "models/vae/vae_encoder.onnx",
        ).expect("Failed to load manifold encoder");

        let features: Vec<f32> = (0..112).map(|i| i as f32 / 100.0).collect();
        let latent = manifold.encode(&features).expect("Manifold encoding failed");

        assert_eq!(latent.len(), 16);

        for val in latent.iter() {
            assert!(val.is_finite());
        }
    }
}
