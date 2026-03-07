//! Rosetta-Net: Hybrid Deep Learning Architecture for Bioacoustics
//! ===============================================================
//!
//! A multi-task learning architecture that combines the 112D Rosetta feature
//! vector with neural network learning for improved bioacoustic classification.
//!
//! **Architecture:**
//! ```text
//! INPUT: Spectrogram (Time x Frequency)
//!          ↓
//! ┌─────────────────────────────────────────┐
//! │ ENCODER (CNN / EfficientNet)            │
//! │ Learns to compress spectrograms         │
//! └─────────────────────────────────────────┘
//!          ↓
//!     [Latent Vector (128D)]
//!          ↓
//! ┌─────────────────────────────────────────┐
//! │ HEAD A: "Rosetta Regression"            │
//! │ Predicts 112D Micro-Dynamics Vector     │
//! │ - Layer 1: Physics (46D)                │
//! │ - Layer 2: Macro Texture (30D)          │
//! │ - Layer 3: Micro Texture (36D)          │
//! Loss: MSE (Mean Squared Error)            │
//! └─────────────────────────────────────────┘
//!          ↓
//! ┌─────────────────────────────────────────┐
//! │ HEAD B: "Cascaded Classification"       │
//! │ Predicts Phrase Type / Species          │
//! Loss: Cross-Entropy                       │
//! └─────────────────────────────────────────┘
//! ```
//!
//! **Key Insight:**
//! By forcing the network to predict the 112D vector (Head A), we prevent it
//! from "cheating" with background noise. It MUST learn the physics of sound
//! (F0, FM Slope, ICI, HNR) plus texture features (harmonic, pitch, GLCM,
//! temporal, AM/FM, rhythm, psychoacoustics).
//!
//! **112D Feature Breakdown:**
//! - Layer 1 (0-45): Base Physics - F0, duration, FM, AM, HNR, ICI, energy, release
//! - Layer 2 (46-75): Macro Texture - Harmonic, pitch geometry, GLCM, temporal
//! - Layer 3 (76-111): Micro Texture - AM/FM spectrum, rhythm, psychoacoustics
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use anyhow::Result;
use ndarray::{Array1, Array2, Array3, ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Rosetta-Net
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RosettaNetConfig {
    /// Input spectrogram dimensions (time_frames, frequency_bins)
    pub spectrogram_shape: (usize, usize),
    /// Latent dimension (encoder output)
    pub latent_dim: usize,
    /// Number of Rosetta features (112D)
    pub rosetta_dim: usize,
    /// Number of classification classes
    pub num_classes: usize,
    /// Encoder type
    pub encoder_type: EncoderType,
    /// Dropout rate
    pub dropout_rate: f32,
    /// Learning rate for training
    pub learning_rate: f32,
    /// Weight for Rosetta regression loss
    pub rosetta_loss_weight: f32,
    /// Weight for classification loss
    pub classification_loss_weight: f32,
}

impl Default for RosettaNetConfig {
    fn default() -> Self {
        Self {
            spectrogram_shape: (128, 128), // 128 time frames x 128 frequency bins
            latent_dim: 128,
            rosetta_dim: 112,
            num_classes: 100, // Default: 100 species
            encoder_type: EncoderType::Cnn,
            dropout_rate: 0.3,
            learning_rate: 0.001,
            rosetta_loss_weight: 1.0,
            classification_loss_weight: 1.0,
        }
    }
}

/// Encoder architecture type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncoderType {
    /// Simple CNN encoder
    Cnn,
    /// Deeper CNN with residual connections
    ResNet,
    /// Lightweight efficient net
    EfficientNet,
    /// Transformer encoder
    Transformer,
    /// Hybrid: CNN (spectral) + TCN (temporal) - RECOMMENDED
    Hybrid,
    /// Pure Temporal Convolutional Network
    Tcn,
}

// ============================================================================
// Spectrogram Input
// ============================================================================

/// Spectrogram representation for neural network input
#[derive(Debug, Clone)]
pub struct Spectrogram {
    /// Time-frequency representation (time_frames x frequency_bins)
    pub data: Array2<f32>,
    /// Sample rate of original audio
    pub sample_rate: u32,
    /// Hop length in samples
    pub hop_length: usize,
    /// FFT size used
    pub fft_size: usize,
}

impl Spectrogram {
    /// Create a new spectrogram
    pub fn new(data: Array2<f32>, sample_rate: u32, hop_length: usize, fft_size: usize) -> Self {
        Self {
            data,
            sample_rate,
            hop_length,
            fft_size,
        }
    }

    /// Create from audio samples using STFT
    pub fn from_audio(audio: &[f32], sample_rate: u32, fft_size: usize, hop_length: usize) -> Self {
        use rustfft::num_complex::Complex;
        use rustfft::num_traits::Zero;
        use rustfft::{FftDirection, FftPlanner};

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft(fft_size, FftDirection::Forward);

        let num_frames = (audio.len() - fft_size) / hop_length + 1;
        let num_bins = fft_size / 2 + 1;

        let mut spectrogram = Array2::<f32>::zeros((num_frames, num_bins));

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_length;
            let mut buffer: Vec<Complex<f32>> = vec![Complex::zero(); fft_size];

            // Apply Hann window and copy audio
            for i in 0..fft_size {
                let audio_idx = start + i;
                if audio_idx < audio.len() {
                    let window = 0.5
                        * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos());
                    buffer[i] = Complex::new(audio[audio_idx] * window, 0.0);
                }
            }

            fft.process(&mut buffer);

            // Store magnitude
            for bin_idx in 0..num_bins {
                spectrogram[[frame_idx, bin_idx]] = buffer[bin_idx].norm();
            }
        }

        // Convert to log scale (dB)
        spectrogram.mapv_inplace(|x: f32| 20.0 * (x + 1e-10).log10());

        Self {
            data: spectrogram,
            sample_rate,
            hop_length,
            fft_size,
        }
    }

    /// Normalize to [0, 1] range
    pub fn normalize(&mut self) {
        let min = self.data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        if range > 1e-10 {
            self.data.mapv_inplace(|x| (x - min) / range);
        }
    }

    /// Resize to target shape
    pub fn resize(&self, target_shape: (usize, usize)) -> Array2<f32> {
        // Simple bilinear-like interpolation
        let (src_rows, src_cols) = self.data.dim();
        let (tgt_rows, tgt_cols) = target_shape;

        let mut result = Array2::<f32>::zeros(target_shape);

        for i in 0..tgt_rows {
            for j in 0..tgt_cols {
                let src_i = (i as f32 * src_rows as f32 / tgt_rows as f32) as usize;
                let src_j = (j as f32 * src_cols as f32 / tgt_cols as f32) as usize;
                let src_i = src_i.min(src_rows - 1);
                let src_j = src_j.min(src_cols - 1);
                result[[i, j]] = self.data[[src_i, src_j]];
            }
        }

        result
    }
}

// ============================================================================
// Neural Network Layers
// ============================================================================

/// 2D Convolution layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv2d {
    /// Filter weights (out_channels, in_channels, kernel_h, kernel_w)
    pub weights: Array4<f32>,
    /// Bias (out_channels)
    pub bias: Array1<f32>,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
}

type Array4<T> = ndarray::Array<T, ndarray::IxDyn>;

impl Conv2d {
    /// Create a new conv layer with He initialization
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // He initialization
        let std = (2.0 / (in_channels * kernel_size * kernel_size) as f32).sqrt();

        let weights_shape = IxDyn(&[out_channels, in_channels, kernel_size, kernel_size]);
        let weights = Array4::from_shape_fn(weights_shape, |_| rng.gen::<f32>() * 2.0 * std - std);

        let bias = Array1::zeros(out_channels);

        Self {
            weights,
            bias,
            stride,
            padding,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        // input: (channels, height, width)
        let (in_channels, in_h, in_w) = input.dim();
        let weights_dim = self.weights.shape();
        let out_channels = weights_dim[0];
        let kernel_size = weights_dim[2];

        let out_h = (in_h + 2 * self.padding - kernel_size) / self.stride + 1;
        let out_w = (in_w + 2 * self.padding - kernel_size) / self.stride + 1;

        let mut output = Array3::<f32>::zeros((out_channels, out_h, out_w));

        // Simple convolution (not optimized)
        for oc in 0..out_channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = self.bias[oc];

                    for ic in 0..in_channels {
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let ih = oh * self.stride + kh;
                                let iw = ow * self.stride + kw;

                                // Handle padding by checking bounds
                                if ih >= self.padding && iw >= self.padding {
                                    let ih = ih - self.padding;
                                    let iw = iw - self.padding;

                                    if ih < in_h && iw < in_w {
                                        let weight_idx = IxDyn(&[oc, ic, kh, kw]);
                                        sum += self.weights[weight_idx] * input[[ic, ih, iw]];
                                    }
                                }
                            }
                        }
                    }

                    output[[oc, oh, ow]] = sum;
                }
            }
        }

        output
    }
}

/// Batch normalization layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchNorm2d {
    pub gamma: Array1<f32>,
    pub beta: Array1<f32>,
    pub running_mean: Array1<f32>,
    pub running_var: Array1<f32>,
    pub epsilon: f32,
    pub momentum: f32,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self {
        Self {
            gamma: Array1::ones(num_features),
            beta: Array1::zeros(num_features),
            running_mean: Array1::zeros(num_features),
            running_var: Array1::ones(num_features),
            epsilon: 1e-5,
            momentum: 0.1,
        }
    }

    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let (channels, height, width) = input.dim();
        let mut output = Array3::<f32>::zeros((channels, height, width));

        for c in 0..channels {
            let mean = self.running_mean[c];
            let var = self.running_var[c];
            let scale = self.gamma[c] / (var + self.epsilon).sqrt();
            let shift = self.beta[c] - mean * scale;

            for h in 0..height {
                for w in 0..width {
                    output[[c, h, w]] = input[[c, h, w]] * scale + shift;
                }
            }
        }

        output
    }
}

/// ReLU activation
pub fn relu(input: &Array3<f32>) -> Array3<f32> {
    input.mapv(|x: f32| x.max(0.0))
}

/// Max pooling layer
pub fn max_pool2d(input: &Array3<f32>, kernel_size: usize, stride: usize) -> Array3<f32> {
    let (channels, in_h, in_w) = input.dim();
    let out_h = (in_h - kernel_size) / stride + 1;
    let out_w = (in_w - kernel_size) / stride + 1;

    let mut output = Array3::<f32>::zeros((channels, out_h, out_w));

    for c in 0..channels {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut max_val = f32::NEG_INFINITY;
                for kh in 0..kernel_size {
                    for kw in 0..kernel_size {
                        let ih = oh * stride + kh;
                        let iw = ow * stride + kw;
                        max_val = max_val.max(input[[c, ih, iw]]);
                    }
                }
                output[[c, oh, ow]] = max_val;
            }
        }
    }

    output
}

/// Fully connected (linear) layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Linear {
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Xavier initialization
        let std = (2.0 / (in_features + out_features) as f32).sqrt();

        let weights = Array2::from_shape_fn((out_features, in_features), |_| {
            rng.gen::<f32>() * 2.0 * std - std
        });

        let bias = Array1::zeros(out_features);

        Self { weights, bias }
    }

    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        self.weights.dot(input) + &self.bias
    }

    /// Backward pass for linear layer
    /// Returns gradient with respect to input
    pub fn backward(
        &self,
        grad_output: &Array1<f32>,
        input: &Array1<f32>,
        grad_weights: &mut Array2<f32>,
        grad_bias: &mut Array1<f32>,
    ) -> Array1<f32> {
        // Gradient w.r.t. weights: outer(grad_output, input)
        for i in 0..grad_output.len() {
            for j in 0..input.len() {
                grad_weights[[i, j]] += grad_output[i] * input[j];
            }
        }

        // Gradient w.r.t. bias: grad_output
        for i in 0..grad_output.len() {
            grad_bias[i] += grad_output[i];
        }

        // Gradient w.r.t. input: weights^T * grad_output
        self.weights.t().dot(grad_output)
    }

    /// Update weights using gradients
    pub fn update(
        &mut self,
        grad_weights: &Array2<f32>,
        grad_bias: &Array1<f32>,
        learning_rate: f32,
    ) {
        self.weights = &self.weights - &(grad_weights * learning_rate);
        self.bias = &self.bias - &(grad_bias * learning_rate);
    }
}

// ============================================================================
// Activation Functions with Backward
// ============================================================================

/// ReLU backward (gradient mask)
pub fn relu_backward(input: &Array3<f32>, grad_output: &Array3<f32>) -> Array3<f32> {
    let mask = input.mapv(|x| if x > 0.0 { 1.0f32 } else { 0.0f32 });
    grad_output * mask
}

/// ReLU for 1D
pub fn relu_1d(input: &Array1<f32>) -> Array1<f32> {
    input.mapv(|x: f32| x.max(0.0))
}

/// ReLU backward for 1D
pub fn relu_1d_backward(input: &Array1<f32>, grad_output: &Array1<f32>) -> Array1<f32> {
    let mask = input.mapv(|x| if x > 0.0 { 1.0f32 } else { 0.0f32 });
    grad_output * mask
}

/// Softmax
pub fn softmax(input: &Array1<f32>) -> Array1<f32> {
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals = input.mapv(|x| (x - max_val).exp());
    let sum: f32 = exp_vals.sum();
    exp_vals / sum
}

/// Softmax backward (combined with cross-entropy for efficiency)
/// Returns gradient of loss w.r.t. logits
pub fn softmax_cross_entropy_backward(probs: &Array1<f32>, target_class: usize) -> Array1<f32> {
    // d(loss)/d(logits) = probs - one_hot(target)
    let mut grad = probs.clone();
    grad[target_class] -= 1.0;
    grad
}

// ============================================================================
// Encoder Network
// ============================================================================

/// CNN Encoder for spectrograms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CnnEncoder {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    conv3: Conv2d,
    bn3: BatchNorm2d,
    fc: Linear,
    latent_dim: usize,
}

impl CnnEncoder {
    pub fn new(config: &RosettaNetConfig) -> Self {
        // Using global average pooling after conv3 -> 128 features
        // FC layer maps from 128 channels to latent_dim
        let conv1 = Conv2d::new(1, 32, 3, 1, 1);
        let bn1 = BatchNorm2d::new(32);
        let conv2 = Conv2d::new(32, 64, 3, 2, 1);
        let bn2 = BatchNorm2d::new(64);
        let conv3 = Conv2d::new(64, 128, 3, 2, 1);
        let bn3 = BatchNorm2d::new(128);

        // Global average pooling gives us 128 features
        let fc = Linear::new(128, config.latent_dim);

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            fc,
            latent_dim: config.latent_dim,
        }
    }

    /// Forward pass through encoder
    pub fn forward(&self, spectrogram: &Array2<f32>) -> Array1<f32> {
        // Add channel dimension: (1, H, W)
        let input = spectrogram.clone().insert_axis(ndarray::Axis(0));

        // Conv block 1: (1, H, W) -> (32, H, W) -> pool -> (32, H/2, W/2)
        let x = self.conv1.forward(&input);
        let x = self.bn1.forward(&x);
        let x = relu(&x);
        let x = max_pool2d(&x, 2, 2);

        // Conv block 2: (32, H/2, W/2) -> (64, H/4, W/4) -> pool -> (64, H/8, W/8)
        let x = self.conv2.forward(&x);
        let x = self.bn2.forward(&x);
        let x = relu(&x);
        let x = max_pool2d(&x, 2, 2);

        // Conv block 3: (64, H/8, W/8) -> (128, H/16, W/16)
        let x = self.conv3.forward(&x);
        let x = self.bn3.forward(&x);
        let x = relu(&x);

        // Global average pooling: (128, H', W') -> (128,)
        let shape = x.shape();
        let channels = shape[0];
        let spatial_size = shape[1] * shape[2];

        let mut pooled = Array1::<f32>::zeros(channels);
        for c in 0..channels {
            let mut sum = 0.0f32;
            for h in 0..shape[1] {
                for w in 0..shape[2] {
                    sum += x[[c, h, w]];
                }
            }
            pooled[c] = sum / spatial_size as f32;
        }

        // FC to latent: (128,) -> (latent_dim,)
        self.fc.forward(&pooled)
    }
}

// ============================================================================
// Temporal Convolutional Network (TCN)
// ============================================================================

/// 1D Causal Convolution with Dilation
///
/// Key insight from Random Forest: duration_ms is 74% important.
/// TCN uses dilated convolutions to capture long-range temporal patterns
/// that determine call duration, attack, decay, and rhythm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalConv1d {
    /// Weights (out_channels, in_channels, kernel_size)
    pub weights: Array3<f32>,
    /// Bias (out_channels)
    pub bias: Array1<f32>,
    /// Dilation factor
    pub dilation: usize,
    /// Kernel size
    pub kernel_size: usize,
}

impl CausalConv1d {
    /// Create new causal conv1d with He initialization
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // He initialization
        let std = (2.0 / (in_channels * kernel_size) as f32).sqrt();

        let weights = Array3::from_shape_fn((out_channels, in_channels, kernel_size), |_| {
            rng.gen::<f32>() * 2.0 * std - std
        });

        let bias = Array1::zeros(out_channels);

        Self {
            weights,
            bias,
            dilation,
            kernel_size,
        }
    }

    /// Forward pass with causal padding
    /// Input: (channels, time_steps)
    /// Output: (channels, time_steps) - same length due to causal padding
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let (in_channels, time_steps) = input.dim();
        let out_channels = self.weights.dim().0;

        // Causal padding: (kernel_size - 1) * dilation on the left only
        let padding = (self.kernel_size - 1) * self.dilation;

        // Create padded input
        let mut padded = Array2::<f32>::zeros((in_channels, time_steps + padding));
        for c in 0..in_channels {
            for t in 0..time_steps {
                padded[[c, t + padding]] = input[[c, t]];
            }
        }

        let mut output = Array2::<f32>::zeros((out_channels, time_steps));

        for oc in 0..out_channels {
            for t in 0..time_steps {
                let mut sum = self.bias[oc];

                for ic in 0..in_channels {
                    for k in 0..self.kernel_size {
                        let input_t = t + padding - k * self.dilation;
                        let weight = self.weights[[oc, ic, k]];
                        sum += weight * padded[[ic, input_t]];
                    }
                }

                output[[oc, t]] = sum;
            }
        }

        output
    }
}

/// Weight normalization for stability
pub fn weight_norm_1d(conv: &mut CausalConv1d) {
    let (out_ch, in_ch, kernel) = conv.weights.dim();
    for oc in 0..out_ch {
        let mut norm = 0.0f32;
        for ic in 0..in_ch {
            for k in 0..kernel {
                norm += conv.weights[[oc, ic, k]].powi(2);
            }
        }
        norm = norm.sqrt();
        if norm > 1e-10 {
            for ic in 0..in_ch {
                for k in 0..kernel {
                    conv.weights[[oc, ic, k]] /= norm;
                }
            }
        }
    }
}

/// Temporal Block with Residual Connection
///
/// Structure: DilatedConv -> ReLU -> DilatedConv -> ReLU -> Residual
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalBlock {
    conv1: CausalConv1d,
    conv2: CausalConv1d,
    /// Downsample for residual connection if channels differ
    downsample: Option<CausalConv1d>,
}

impl TemporalBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> Self {
        let conv1 = CausalConv1d::new(in_channels, out_channels, kernel_size, dilation);
        let conv2 = CausalConv1d::new(out_channels, out_channels, kernel_size, dilation);

        let downsample = if in_channels != out_channels {
            Some(CausalConv1d::new(in_channels, out_channels, 1, 1))
        } else {
            None
        };

        Self {
            conv1,
            conv2,
            downsample,
        }
    }

    /// Forward pass with residual connection
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        // First conv + ReLU
        let x = self.conv1.forward(input);
        let x = x.mapv(|v: f32| v.max(0.0));

        // Second conv + ReLU
        let x = self.conv2.forward(&x);
        let x = x.mapv(|v: f32| v.max(0.0));

        // Residual connection
        let residual = if let Some(ref ds) = self.downsample {
            ds.forward(input)
        } else {
            input.clone()
        };

        // Add residual
        &x + &residual
    }
}

/// Temporal Convolutional Network (TCN)
///
/// Stack of temporal blocks with exponentially increasing dilation.
/// This captures temporal patterns at multiple scales:
/// - Block 1: dilation=1 (fine-grained temporal features)
/// - Block 2: dilation=2 (attack/decay patterns)
/// - Block 3: dilation=4 (short duration patterns)
/// - Block 4: dilation=8 (rhythm patterns)
/// - Block 5: dilation=16 (long duration patterns)
///
/// Random Forest insight: duration_ms is 74% important.
/// The TCN learns to capture this directly from the spectrogram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConvNet {
    blocks: Vec<TemporalBlock>,
    num_channels: usize,
}

impl TemporalConvNet {
    /// Create TCN with specified channel progression
    pub fn new(num_inputs: usize, num_channels: Vec<usize>, kernel_size: usize) -> Self {
        let mut blocks = Vec::new();
        let mut in_channels = num_inputs;

        for (i, &out_channels) in num_channels.iter().enumerate() {
            let dilation = 2_usize.pow(i as u32);
            let block = TemporalBlock::new(in_channels, out_channels, kernel_size, dilation);
            blocks.push(block);
            in_channels = out_channels;
        }

        Self {
            blocks,
            num_channels: num_channels.last().copied().unwrap_or(num_inputs),
        }
    }

    /// Forward pass through all temporal blocks
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut x = input.clone();
        for block in &self.blocks {
            x = block.forward(&x);
        }
        x
    }

    /// Get output channels
    pub fn output_channels(&self) -> usize {
        self.num_channels
    }
}

/// Duration Encoder - Specialized for temporal feature extraction
///
/// This encoder focuses on capturing duration-related features that
/// the Random Forest identified as 74% important:
/// - duration_ms
/// - attack_time_ms
/// - decay_time_ms
/// - sustain_level
/// - rhythm/ICI patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationEncoder {
    /// TCN for temporal modeling
    tcn: TemporalConvNet,
    /// Final projection to latent dim
    fc: Linear,
    /// Latent dimension
    latent_dim: usize,
}

impl DurationEncoder {
    /// Create duration encoder
    /// Input: (frequency_bins, time_steps) spectrogram slice
    pub fn new(freq_bins: usize, latent_dim: usize) -> Self {
        // TCN channel progression: 32 -> 64 -> 128 -> 256
        let num_channels = vec![32, 64, 128, 256];
        let tcn = TemporalConvNet::new(freq_bins, num_channels.clone(), 3);

        let fc = Linear::new(*num_channels.last().unwrap(), latent_dim);

        Self {
            tcn,
            fc,
            latent_dim,
        }
    }

    /// Forward pass
    /// Input: (freq_bins, time_steps) - treat freq as channels, time as sequence
    /// Output: (latent_dim,) - temporal features
    pub fn forward(&self, spectrogram: &Array2<f32>) -> Array1<f32> {
        // TCN forward: (freq, time) -> (256, time)
        let temporal_features = self.tcn.forward(spectrogram);

        // Global average pooling over time: (256, time) -> (256,)
        let (channels, time_steps) = temporal_features.dim();
        let mut pooled = Array1::<f32>::zeros(channels);
        for c in 0..channels {
            let mut sum = 0.0f32;
            for t in 0..time_steps {
                sum += temporal_features[[c, t]];
            }
            pooled[c] = sum / time_steps.max(1) as f32;
        }

        // Project to latent: (256,) -> (latent_dim,)
        self.fc.forward(&pooled)
    }
}

// ============================================================================
// Hybrid Encoder: CNN (Spectral) + TCN (Temporal)
// ============================================================================

/// Hybrid Encoder combining spectral CNN and temporal TCN
///
/// Architecture:
/// ```text
/// Spectrogram (Time x Freq)
///       │
///       ├──────────────────────┐
///       │                      │
///       ▼                      ▼
/// ┌─────────────┐      ┌─────────────┐
/// │ CNN Encoder │      │ TCN Encoder │
/// │ (Spectral)  │      │ (Temporal)  │
/// └─────────────┘      └─────────────┘
///       │                      │
///       ▼                      ▼
///   [128D]                 [128D]
///       │                      │
///       └──────────┬───────────┘
///                  │
///                  ▼
///           Concat [256D]
///                  │
///                  ▼
///            FC [latent_dim]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridEncoder {
    /// Spectral encoder (CNN)
    spectral_encoder: CnnEncoder,
    /// Temporal encoder (TCN)
    temporal_encoder: DurationEncoder,
    /// Fusion layer
    fusion_fc: Linear,
    /// Latent dimension
    latent_dim: usize,
}

impl HybridEncoder {
    pub fn new(config: &RosettaNetConfig) -> Self {
        let spectral_encoder = CnnEncoder::new(config);
        let temporal_encoder = DurationEncoder::new(config.spectrogram_shape.1, config.latent_dim);

        // Fusion: concatenate spectral (128) + temporal (128) -> latent
        let fusion_fc = Linear::new(config.latent_dim * 2, config.latent_dim);

        Self {
            spectral_encoder,
            temporal_encoder,
            fusion_fc,
            latent_dim: config.latent_dim,
        }
    }

    /// Forward pass through hybrid encoder
    pub fn forward(&self, spectrogram: &Array2<f32>) -> Array1<f32> {
        // Spectral features: (H, W) -> (latent_dim,)
        let spectral_features = self.spectral_encoder.forward(spectrogram);

        // Temporal features: (H, W) -> (latent_dim,)
        // Note: We use the transpose to treat time as sequence dimension
        let transposed = spectrogram.t().to_owned();
        let temporal_features = self.temporal_encoder.forward(&transposed);

        // Concatenate
        let mut combined = Array1::<f32>::zeros(self.latent_dim * 2);
        for i in 0..self.latent_dim {
            combined[i] = spectral_features[i];
            combined[self.latent_dim + i] = temporal_features[i];
        }

        // Fusion
        self.fusion_fc.forward(&combined)
    }

    /// Get individual feature streams (for analysis)
    pub fn forward_with_features(
        &self,
        spectrogram: &Array2<f32>,
    ) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
        let spectral = self.spectral_encoder.forward(spectrogram);
        let transposed = spectrogram.t().to_owned();
        let temporal = self.temporal_encoder.forward(&transposed);

        let mut combined = Array1::<f32>::zeros(self.latent_dim * 2);
        for i in 0..self.latent_dim {
            combined[i] = spectral[i];
            combined[self.latent_dim + i] = temporal[i];
        }

        let fused = self.fusion_fc.forward(&combined);

        (spectral, temporal, fused)
    }
}

// ============================================================================
// Rosetta Regression Head (Head A)
// ============================================================================

/// Head A: Rosetta 112D Regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RosettaRegressionHead {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    dropout_rate: f32,
}

impl RosettaRegressionHead {
    pub fn new(latent_dim: usize, rosetta_dim: usize, dropout_rate: f32) -> Self {
        let fc1 = Linear::new(latent_dim, latent_dim * 2);
        let fc2 = Linear::new(latent_dim * 2, latent_dim);
        let fc3 = Linear::new(latent_dim, rosetta_dim);

        Self {
            fc1,
            fc2,
            fc3,
            dropout_rate,
        }
    }

    pub fn forward(&self, latent: &Array1<f32>) -> Array1<f32> {
        let x = self.fc1.forward(latent);
        let x = x.mapv(|v: f32| v.max(0.0)); // ReLU

        // Simple dropout approximation (scale during training)
        let x = if self.dropout_rate > 0.0 {
            x.mapv(|v| v / (1.0 - self.dropout_rate))
        } else {
            x
        };

        let x = self.fc2.forward(&x);
        let x = x.mapv(|v: f32| v.max(0.0)); // ReLU

        self.fc3.forward(&x) // Linear output for regression
    }
}

// ============================================================================
// Classification Head (Head B)
// ============================================================================

/// Head B: Species/Phrase Classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationHead {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    num_classes: usize,
    dropout_rate: f32,
}

impl ClassificationHead {
    pub fn new(latent_dim: usize, num_classes: usize, dropout_rate: f32) -> Self {
        let fc1 = Linear::new(latent_dim, latent_dim * 2);
        let fc2 = Linear::new(latent_dim * 2, latent_dim);
        let fc3 = Linear::new(latent_dim, num_classes);

        Self {
            fc1,
            fc2,
            fc3,
            num_classes,
            dropout_rate,
        }
    }

    pub fn forward(&self, latent: &Array1<f32>) -> Array1<f32> {
        let x = self.fc1.forward(latent);
        let x = x.mapv(|v: f32| v.max(0.0)); // ReLU

        let x = self.fc2.forward(&x);
        let x = x.mapv(|v: f32| v.max(0.0)); // ReLU

        self.fc3.forward(&x) // Logits (apply softmax externally)
    }
}

// ============================================================================
// Complete Rosetta-Net Model
// ============================================================================

/// Complete Rosetta-Net model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RosettaNet {
    config: RosettaNetConfig,
    encoder: CnnEncoder,
    rosetta_head: RosettaRegressionHead,
    classification_head: ClassificationHead,
}

/// Output from Rosetta-Net forward pass
#[derive(Debug, Clone)]
pub struct RosettaNetOutput {
    /// Latent representation
    pub latent: Array1<f32>,
    /// Predicted 112D Rosetta features
    pub rosetta_features: Array1<f32>,
    /// Classification logits
    pub logits: Array1<f32>,
    /// Predicted class (after argmax)
    pub predicted_class: usize,
}

impl RosettaNet {
    /// Create a new Rosetta-Net model
    pub fn new(config: RosettaNetConfig) -> Self {
        let encoder = CnnEncoder::new(&config);
        let rosetta_head =
            RosettaRegressionHead::new(config.latent_dim, config.rosetta_dim, config.dropout_rate);
        let classification_head =
            ClassificationHead::new(config.latent_dim, config.num_classes, config.dropout_rate);

        Self {
            config,
            encoder,
            rosetta_head,
            classification_head,
        }
    }

    /// Forward pass through the entire network
    pub fn forward(&self, spectrogram: &Array2<f32>) -> RosettaNetOutput {
        // Encode spectrogram to latent vector
        let latent = self.encoder.forward(spectrogram);

        // Head A: Rosetta regression
        let rosetta_features = self.rosetta_head.forward(&latent);

        // Head B: Classification
        let logits = self.classification_head.forward(&latent);

        // Predicted class
        let predicted_class = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        RosettaNetOutput {
            latent,
            rosetta_features,
            logits,
            predicted_class,
        }
    }

    /// Forward pass with concatenated features (for cascaded classification)
    pub fn forward_cascaded(&self, spectrogram: &Array2<f32>) -> RosettaNetOutput {
        // The latent vector already encodes the "understanding" from Head A
        // Classification head uses this understanding
        self.forward(spectrogram)
    }

    /// Get the configuration
    pub fn config(&self) -> &RosettaNetConfig {
        &self.config
    }
}

// ============================================================================
// Loss Functions
// ============================================================================

/// Mean Squared Error loss for regression
pub fn mse_loss(predicted: &Array1<f32>, target: &Array1<f32>) -> f32 {
    let diff = predicted - target;
    let n = predicted.len() as f32;
    diff.mapv(|x| x * x).sum() / n
}

/// Cross-entropy loss for classification
pub fn cross_entropy_loss(logits: &Array1<f32>, target_class: usize) -> f32 {
    // Softmax
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
    let log_probs: Array1<f32> = logits.mapv(|x| (x - max_logit) - exp_sum.ln());

    // Negative log likelihood
    -log_probs[target_class]
}

/// Combined multi-task loss
pub fn multi_task_loss(
    rosetta_predicted: &Array1<f32>,
    rosetta_target: &Array1<f32>,
    logits: &Array1<f32>,
    target_class: usize,
    rosetta_weight: f32,
    classification_weight: f32,
) -> f32 {
    let rosetta_loss = mse_loss(rosetta_predicted, rosetta_target);
    let class_loss = cross_entropy_loss(logits, target_class);

    rosetta_weight * rosetta_loss + classification_weight * class_loss
}

// ============================================================================
// Rosetta-Net with TCN (Temporal Convolutional Network)
// ============================================================================

/// Rosetta-Net with Hybrid CNN+TCN Encoder
///
/// This version incorporates the Random Forest insight that duration_ms
/// is 74% important. The TCN encoder specializes in capturing temporal
/// dynamics (duration, attack, decay, rhythm) while the CNN captures
/// spectral features (frequency content, harmonics).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RosettaNetWithTCN {
    config: RosettaNetConfig,
    /// Hybrid encoder: CNN (spectral) + TCN (temporal)
    encoder: HybridEncoder,
    /// Rosetta 45D regression head
    rosetta_head: RosettaRegressionHead,
    /// Classification head
    classification_head: ClassificationHead,
}

/// Output from Rosetta-Net with TCN
#[derive(Debug, Clone)]
pub struct RosettaNetWithTCNOutput {
    /// Spectral features from CNN
    pub spectral_features: Array1<f32>,
    /// Temporal features from TCN
    pub temporal_features: Array1<f32>,
    /// Fused latent representation
    pub latent: Array1<f32>,
    /// Predicted 112D Rosetta features
    pub rosetta_features: Array1<f32>,
    /// Classification logits
    pub logits: Array1<f32>,
    /// Predicted class
    pub predicted_class: usize,
}

impl RosettaNetWithTCN {
    /// Create new Rosetta-Net with TCN encoder
    pub fn new(config: RosettaNetConfig) -> Self {
        let encoder = HybridEncoder::new(&config);
        let rosetta_head =
            RosettaRegressionHead::new(config.latent_dim, config.rosetta_dim, config.dropout_rate);
        let classification_head =
            ClassificationHead::new(config.latent_dim, config.num_classes, config.dropout_rate);

        Self {
            config,
            encoder,
            rosetta_head,
            classification_head,
        }
    }

    /// Forward pass with detailed feature extraction
    pub fn forward_with_features(&self, spectrogram: &Array2<f32>) -> RosettaNetWithTCNOutput {
        // Get individual feature streams
        let (spectral, temporal, latent) = self.encoder.forward_with_features(spectrogram);

        // Head A: Rosetta regression
        let rosetta_features = self.rosetta_head.forward(&latent);

        // Head B: Classification
        let logits = self.classification_head.forward(&latent);

        // Predicted class
        let predicted_class = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        RosettaNetWithTCNOutput {
            spectral_features: spectral,
            temporal_features: temporal,
            latent,
            rosetta_features,
            logits,
            predicted_class,
        }
    }

    /// Forward pass (returns just latent, rosetta, logits)
    pub fn forward(&self, spectrogram: &Array2<f32>) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
        let latent = self.encoder.forward(spectrogram);
        let rosetta = self.rosetta_head.forward(&latent);
        let logits = self.classification_head.forward(&latent);
        (latent, rosetta, logits)
    }

    /// Get temporal feature importance (how much TCN contributes vs CNN)
    pub fn analyze_temporal_importance(&self, spectrogram: &Array2<f32>) -> TemporalImportance {
        let (spectral, temporal, _) = self.encoder.forward_with_features(spectrogram);

        // Compute energy in each feature stream
        let spectral_energy: f32 = spectral.mapv(|x| x * x).sum();
        let temporal_energy: f32 = temporal.mapv(|x| x * x).sum();
        let total_energy = spectral_energy + temporal_energy;

        TemporalImportance {
            spectral_contribution: if total_energy > 0.0 {
                spectral_energy / total_energy
            } else {
                0.5
            },
            temporal_contribution: if total_energy > 0.0 {
                temporal_energy / total_energy
            } else {
                0.5
            },
            spectral_features: spectral,
            temporal_features: temporal,
        }
    }

    /// Predict class
    pub fn predict(&self, spectrogram: &Array2<f32>) -> usize {
        let (_, _, logits) = self.forward(spectrogram);
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get configuration
    pub fn config(&self) -> &RosettaNetConfig {
        &self.config
    }
}

/// Analysis of temporal vs spectral feature contributions
#[derive(Debug, Clone)]
pub struct TemporalImportance {
    /// Fraction of features from spectral encoder (0.0 to 1.0)
    pub spectral_contribution: f32,
    /// Fraction of features from temporal encoder (0.0 to 1.0)
    pub temporal_contribution: f32,
    /// Raw spectral features
    pub spectral_features: Array1<f32>,
    /// Raw temporal features
    pub temporal_features: Array1<f32>,
}

// ============================================================================
// Siamese Network for Metric Learning
// ============================================================================

/// Siamese network for learning distance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiameseRosettaNet {
    /// Shared encoder
    encoder: CnnEncoder,
    /// Weight prediction network
    weight_predictor: Linear,
    /// Rosetta dimension
    rosetta_dim: usize,
}

impl SiameseRosettaNet {
    pub fn new(config: &RosettaNetConfig) -> Self {
        let encoder = CnnEncoder::new(config);
        // Predict 45 weights from latent vector
        let weight_predictor = Linear::new(config.latent_dim, config.rosetta_dim);

        Self {
            encoder,
            weight_predictor,
            rosetta_dim: config.rosetta_dim,
        }
    }

    /// Forward pass for one branch
    pub fn encode(&self, spectrogram: &Array2<f32>) -> Array1<f32> {
        self.encoder.forward(spectrogram)
    }

    /// Predict attention weights from latent vector
    pub fn predict_weights(&self, latent: &Array1<f32>) -> Array1<f32> {
        let weights = self.weight_predictor.forward(latent);
        // Apply sigmoid to get positive weights in [0, 1]
        weights.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    /// Calculate weighted distance between two samples
    pub fn weighted_distance(
        &self,
        features_a: &Array1<f32>,
        features_b: &Array1<f32>,
        weights: &Array1<f32>,
    ) -> f32 {
        let diff = features_a - features_b;
        let weighted_diff = &diff * weights;
        weighted_diff.mapv(|x| x * x).sum().sqrt()
    }

    /// Full siamese forward pass
    pub fn forward_pair(
        &self,
        spec_a: &Array2<f32>,
        spec_b: &Array2<f32>,
        rosetta_a: &Array1<f32>,
        rosetta_b: &Array1<f32>,
    ) -> (Array1<f32>, Array1<f32>, f32) {
        // Encode both
        let latent_a = self.encode(spec_a);
        let latent_b = self.encode(spec_b);

        // Predict weights from sample A
        let weights = self.predict_weights(&latent_a);

        // Calculate weighted distance
        let distance = self.weighted_distance(rosetta_a, rosetta_b, &weights);

        (latent_a, latent_b, distance)
    }
}

/// Contrastive loss for siamese training
pub fn contrastive_loss(distance: f32, is_same: bool, margin: f32) -> f32 {
    if is_same {
        // Same class: minimize distance
        distance * distance
    } else {
        // Different class: maximize distance (up to margin)
        let diff = margin - distance;
        diff.max(0.0) * diff.max(0.0)
    }
}

// ============================================================================
// Training Utilities
// ============================================================================

/// Training batch
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    /// Spectrograms
    pub spectrograms: Vec<Array2<f32>>,
    /// Target 112D features
    pub rosetta_targets: Vec<Array1<f32>>,
    /// Target class indices
    pub class_targets: Vec<usize>,
}

impl TrainingBatch {
    pub fn new(capacity: usize) -> Self {
        Self {
            spectrograms: Vec::with_capacity(capacity),
            rosetta_targets: Vec::with_capacity(capacity),
            class_targets: Vec::with_capacity(capacity),
        }
    }

    pub fn add(&mut self, spectrogram: Array2<f32>, rosetta: Array1<f32>, class: usize) {
        self.spectrograms.push(spectrogram);
        self.rosetta_targets.push(rosetta);
        self.class_targets.push(class);
    }

    pub fn len(&self) -> usize {
        self.spectrograms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spectrograms.is_empty()
    }
}

/// Training metrics
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub rosetta_loss: f32,
    pub classification_loss: f32,
    pub total_loss: f32,
    pub accuracy: f32,
}

/// Model trainer
pub struct RosettaNetTrainer {
    model: RosettaNet,
    config: RosettaNetConfig,
    metrics_history: Vec<TrainingMetrics>,
}

impl RosettaNetTrainer {
    pub fn new(config: RosettaNetConfig) -> Self {
        let model = RosettaNet::new(config.clone());
        Self {
            model,
            config,
            metrics_history: Vec::new(),
        }
    }

    /// Evaluate model on a batch
    pub fn evaluate_batch(&self, batch: &TrainingBatch) -> TrainingMetrics {
        let mut total_rosetta_loss = 0.0;
        let mut total_class_loss = 0.0;
        let mut correct = 0;

        for i in 0..batch.len() {
            let output = self.model.forward(&batch.spectrograms[i]);

            total_rosetta_loss += mse_loss(&output.rosetta_features, &batch.rosetta_targets[i]);
            total_class_loss += cross_entropy_loss(&output.logits, batch.class_targets[i]);

            if output.predicted_class == batch.class_targets[i] {
                correct += 1;
            }
        }

        let n = batch.len() as f32;
        TrainingMetrics {
            epoch: 0,
            rosetta_loss: total_rosetta_loss / n,
            classification_loss: total_class_loss / n,
            total_loss: self.config.rosetta_loss_weight * (total_rosetta_loss / n)
                + self.config.classification_loss_weight * (total_class_loss / n),
            accuracy: correct as f32 / n,
        }
    }

    /// Get the model
    pub fn model(&self) -> &RosettaNet {
        &self.model
    }

    /// Get mutable model
    pub fn model_mut(&mut self) -> &mut RosettaNet {
        &mut self.model
    }

    /// Get metrics history
    pub fn metrics_history(&self) -> &[TrainingMetrics] {
        &self.metrics_history
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_spectrogram_creation() {
        let audio = vec![0.5f32; 16000]; // 1 second at 16kHz
        let spec = Spectrogram::from_audio(&audio, 16000, 512, 1024);

        assert!(spec.data.nrows() > 0);
        assert!(spec.data.ncols() > 0);
        assert_eq!(spec.sample_rate, 16000);
    }

    #[test]
    fn test_spectrogram_normalize() {
        let mut spec = Spectrogram::new(
            Array2::from_shape_vec((2, 2), vec![-10.0, 0.0, 10.0, 20.0]).unwrap(),
            16000,
            512,
            1024,
        );
        spec.normalize();

        assert!((spec.data[[0, 0]] - 0.0).abs() < 0.01);
        assert!((spec.data[[1, 1]] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_spectrogram_resize() {
        let spec = Spectrogram::new(
            Array2::from_shape_vec(
                (4, 4),
                vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0,
                ],
            )
            .unwrap(),
            16000,
            512,
            1024,
        );

        let resized = spec.resize((2, 2));
        assert_eq!(resized.dim(), (2, 2));
    }

    #[test]
    fn test_conv2d_creation() {
        let conv = Conv2d::new(1, 32, 3, 1, 1);
        assert_eq!(conv.weights.shape()[0], 32); // out_channels
        assert_eq!(conv.weights.shape()[1], 1); // in_channels
        assert_eq!(conv.weights.shape()[2], 3); // kernel_h
        assert_eq!(conv.weights.shape()[3], 3); // kernel_w
    }

    #[test]
    fn test_conv2d_forward() {
        let conv = Conv2d::new(1, 16, 3, 1, 1);
        let input = Array3::<f32>::zeros((1, 8, 8));
        let output = conv.forward(&input);

        assert_eq!(output.shape()[0], 16); // out_channels
        assert_eq!(output.shape()[1], 8); // height (same due to padding)
        assert_eq!(output.shape()[2], 8); // width
    }

    #[test]
    fn test_batch_norm_forward() {
        let bn = BatchNorm2d::new(16);
        let input = Array3::<f32>::zeros((16, 8, 8));
        let output = bn.forward(&input);

        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_relu() {
        let input = Array3::from_shape_vec((1, 1, 4), vec![-1.0, 0.0, 1.0, 2.0]).unwrap();
        let output = relu(&input);

        assert_eq!(output[[0, 0, 0]], 0.0);
        assert_eq!(output[[0, 0, 1]], 0.0);
        assert_eq!(output[[0, 0, 2]], 1.0);
        assert_eq!(output[[0, 0, 3]], 2.0);
    }

    #[test]
    fn test_max_pool2d() {
        let input = Array3::from_shape_vec(
            (1, 4, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .unwrap();

        let output = max_pool2d(&input, 2, 2);

        assert_eq!(output.shape(), &[1, 2, 2]);
        assert_eq!(output[[0, 0, 0]], 6.0); // max of 1,2,5,6
        assert_eq!(output[[0, 1, 1]], 16.0); // max of 11,12,15,16
    }

    #[test]
    fn test_linear_creation() {
        let linear = Linear::new(128, 45);
        assert_eq!(linear.weights.dim(), (45, 128));
        assert_eq!(linear.bias.len(), 45);
    }

    #[test]
    fn test_linear_forward() {
        let linear = Linear::new(10, 5);
        let input = Array1::<f32>::zeros(10);
        let output = linear.forward(&input);

        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_cnn_encoder_creation() {
        let config = RosettaNetConfig::default();
        let encoder = CnnEncoder::new(&config);

        assert_eq!(encoder.latent_dim, config.latent_dim);
    }

    #[test]
    fn test_cnn_encoder_forward() {
        let config = RosettaNetConfig::default();
        let encoder = CnnEncoder::new(&config);

        let spectrogram = Array2::<f32>::zeros(config.spectrogram_shape);
        let latent = encoder.forward(&spectrogram);

        assert_eq!(latent.len(), config.latent_dim);
    }

    #[test]
    fn test_rosetta_regression_head() {
        let head = RosettaRegressionHead::new(128, 112, 0.3);
        let latent = Array1::<f32>::zeros(128);
        let output = head.forward(&latent);

        assert_eq!(output.len(), 112);
    }

    #[test]
    fn test_classification_head() {
        let head = ClassificationHead::new(128, 100, 0.3);
        let latent = Array1::<f32>::zeros(128);
        let logits = head.forward(&latent);

        assert_eq!(logits.len(), 100);
    }

    #[test]
    fn test_rosetta_net_creation() {
        let config = RosettaNetConfig::default();
        let model = RosettaNet::new(config);

        assert_eq!(model.config().latent_dim, 128);
        assert_eq!(model.config().rosetta_dim, 112);
    }

    #[test]
    fn test_rosetta_net_forward() {
        let config = RosettaNetConfig::default();
        let model = RosettaNet::new(config.clone());

        let spectrogram = Array2::<f32>::zeros(config.spectrogram_shape);
        let output = model.forward(&spectrogram);

        assert_eq!(output.latent.len(), 128);
        assert_eq!(output.rosetta_features.len(), config.rosetta_dim);
        assert_eq!(output.logits.len(), config.num_classes);
        assert!(output.predicted_class < config.num_classes);
    }

    #[test]
    fn test_mse_loss() {
        let predicted = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let target = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let loss = mse_loss(&predicted, &target);
        assert!((loss - 0.0).abs() < 1e-6);

        let predicted2 = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        let loss2 = mse_loss(&predicted2, &target);
        assert!((loss2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let logits = Array1::from_vec(vec![0.0, 1.0, 2.0]); // Class 2 should have highest prob

        let loss_class2 = cross_entropy_loss(&logits, 2);
        let loss_class0 = cross_entropy_loss(&logits, 0);

        // Loss for correct class should be lower
        assert!(loss_class2 < loss_class0);
    }

    #[test]
    fn test_multi_task_loss() {
        let rosetta_pred = Array1::zeros(112);
        let rosetta_target = Array1::zeros(112);
        let logits = Array1::zeros(10);

        let loss = multi_task_loss(&rosetta_pred, &rosetta_target, &logits, 0, 1.0, 1.0);

        assert!(loss >= 0.0);
    }

    #[test]
    fn test_siamese_network_creation() {
        let config = RosettaNetConfig::default();
        let siamese = SiameseRosettaNet::new(&config);

        assert_eq!(siamese.rosetta_dim, 112);
    }

    #[test]
    fn test_siamese_weight_prediction() {
        let config = RosettaNetConfig::default();
        let siamese = SiameseRosettaNet::new(&config);

        let spectrogram = Array2::<f32>::zeros(config.spectrogram_shape);
        let latent = siamese.encode(&spectrogram);
        let weights = siamese.predict_weights(&latent);

        assert_eq!(weights.len(), config.rosetta_dim);
        // All weights should be positive (sigmoid output)
        for &w in weights.iter() {
            assert!((0.0..=1.0).contains(&w ));
        }
    }

    #[test]
    fn test_weighted_distance() {
        let siamese = SiameseRosettaNet::new(&RosettaNetConfig::default());

        let a = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0, 1.0, 1.0]);

        let dist = siamese.weighted_distance(&a, &b, &weights);
        assert!((dist - 3.0f32.sqrt()).abs() < 0.01);
    }

    #[test]
    fn test_contrastive_loss_same() {
        let loss = contrastive_loss(0.5, true, 1.0);
        assert!((loss - 0.25).abs() < 0.01); // 0.5^2
    }

    #[test]
    fn test_contrastive_loss_different() {
        let loss = contrastive_loss(0.5, false, 1.0);
        assert!((loss - 0.25).abs() < 0.01); // (1.0 - 0.5)^2
    }

    #[test]
    fn test_contrastive_loss_different_far() {
        let loss = contrastive_loss(2.0, false, 1.0);
        assert!((loss - 0.0).abs() < 0.01); // max(0, 1 - 2)^2 = 0
    }

    #[test]
    fn test_training_batch() {
        let mut batch = TrainingBatch::new(10);

        batch.add(Array2::<f32>::zeros((64, 64)), Array1::<f32>::zeros(112), 0);

        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_trainer_creation() {
        let config = RosettaNetConfig::default();
        let trainer = RosettaNetTrainer::new(config);

        assert!(trainer.metrics_history().is_empty());
    }

    #[test]
    fn test_trainer_evaluate() {
        let config = RosettaNetConfig {
            spectrogram_shape: (32, 32),
            latent_dim: 64,
            rosetta_dim: 112,
            num_classes: 10,
            ..Default::default()
        };

        let trainer = RosettaNetTrainer::new(config.clone());
        let mut batch = TrainingBatch::new(2);

        batch.add(
            Array2::<f32>::zeros(config.spectrogram_shape),
            Array1::<f32>::zeros(config.rosetta_dim),
            0,
        );
        batch.add(
            Array2::<f32>::zeros(config.spectrogram_shape),
            Array1::<f32>::zeros(config.rosetta_dim),
            1,
        );

        let metrics = trainer.evaluate_batch(&batch);

        assert!(metrics.rosetta_loss >= 0.0);
        assert!(metrics.classification_loss >= 0.0);
        assert!(metrics.total_loss >= 0.0);
        assert!((0.0..=1.0).contains(&metrics.accuracy ));
    }

    #[test]
    fn test_encoder_type_serialization() {
        let encoder_type = EncoderType::ResNet;
        let json = serde_json::to_string(&encoder_type).unwrap();
        let decoded: EncoderType = serde_json::from_str(&json).unwrap();
        assert_eq!(encoder_type, decoded);
    }

    #[test]
    fn test_config_serialization() {
        let config = RosettaNetConfig {
            spectrogram_shape: (64, 128),
            latent_dim: 256,
            rosetta_dim: 112,
            num_classes: 50,
            encoder_type: EncoderType::EfficientNet,
            dropout_rate: 0.5,
            learning_rate: 0.0001,
            rosetta_loss_weight: 0.8,
            classification_loss_weight: 1.2,
        };

        let json = serde_json::to_string(&config).unwrap();
        let decoded: RosettaNetConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.spectrogram_shape, decoded.spectrogram_shape);
        assert_eq!(config.latent_dim, decoded.latent_dim);
        assert_eq!(config.encoder_type, decoded.encoder_type);
    }

    // ========================================================================
    // TCN (Temporal Convolutional Network) Tests
    // ========================================================================

    #[test]
    fn test_causal_conv1d_creation() {
        let conv = CausalConv1d::new(32, 64, 3, 1);
        assert_eq!(conv.weights.dim().0, 64); // out_channels
        assert_eq!(conv.weights.dim().1, 32); // in_channels
        assert_eq!(conv.weights.dim().2, 3); // kernel_size
        assert_eq!(conv.dilation, 1);
    }

    #[test]
    fn test_causal_conv1d_forward() {
        let conv = CausalConv1d::new(16, 32, 3, 1);
        let input = Array2::<f32>::zeros((16, 100)); // 16 channels, 100 time steps
        let output = conv.forward(&input);

        // Output should have same time steps (causal)
        assert_eq!(output.dim().1, 100);
        // Output channels should match conv out_channels
        assert_eq!(output.dim().0, 32);
    }

    #[test]
    fn test_causal_conv1d_dilation() {
        let conv = CausalConv1d::new(16, 32, 3, 4); // dilation=4
        let input = Array2::<f32>::zeros((16, 100));
        let output = conv.forward(&input);

        // Should still produce same length output
        assert_eq!(output.dim().1, 100);
        assert_eq!(output.dim().0, 32);
    }

    #[test]
    fn test_temporal_block() {
        let block = TemporalBlock::new(32, 64, 3, 2);
        let input = Array2::<f32>::zeros((32, 100));
        let output = block.forward(&input);

        // Output channels should match block output
        assert_eq!(output.dim().0, 64);
        // Time dimension preserved (residual connection requires it)
        assert_eq!(output.dim().1, 100);
    }

    #[test]
    fn test_temporal_block_same_channels() {
        // When in_channels == out_channels, no downsampling needed
        let block = TemporalBlock::new(64, 64, 3, 2);
        assert!(block.downsample.is_none());
    }

    #[test]
    fn test_temporal_block_different_channels() {
        // When in_channels != out_channels, downsampling is needed
        let block = TemporalBlock::new(32, 64, 3, 2);
        assert!(block.downsample.is_some());
    }

    #[test]
    fn test_temporal_conv_net() {
        let tcn = TemporalConvNet::new(64, vec![32, 64, 128], 3);
        let input = Array2::<f32>::zeros((64, 100));
        let output = tcn.forward(&input);

        // Output channels should be last in num_channels
        assert_eq!(output.dim().0, 128);
        assert_eq!(output.dim().1, 100);
    }

    #[test]
    fn test_tcn_dilation_progression() {
        // Create TCN and verify dilation increases exponentially
        // Block 0: dilation=1, Block 1: dilation=2, Block 2: dilation=4
        let tcn = TemporalConvNet::new(64, vec![32, 64, 128], 3);

        assert_eq!(tcn.blocks.len(), 3);
        assert_eq!(tcn.blocks[0].conv1.dilation, 1);
        assert_eq!(tcn.blocks[1].conv1.dilation, 2);
        assert_eq!(tcn.blocks[2].conv1.dilation, 4);
    }

    #[test]
    fn test_duration_encoder() {
        let encoder = DurationEncoder::new(128, 64); // 128 freq bins, 64 latent
        let spectrogram = Array2::<f32>::zeros((128, 100)); // freq x time

        let output = encoder.forward(&spectrogram);

        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_hybrid_encoder() {
        let config = RosettaNetConfig::default();
        let encoder = HybridEncoder::new(&config);
        let spectrogram = Array2::<f32>::zeros(config.spectrogram_shape);

        let output = encoder.forward(&spectrogram);

        assert_eq!(output.len(), config.latent_dim);
    }

    #[test]
    fn test_hybrid_encoder_with_features() {
        let config = RosettaNetConfig::default();
        let encoder = HybridEncoder::new(&config);
        let spectrogram = Array2::<f32>::zeros(config.spectrogram_shape);

        let (spectral, temporal, fused) = encoder.forward_with_features(&spectrogram);

        assert_eq!(spectral.len(), config.latent_dim);
        assert_eq!(temporal.len(), config.latent_dim);
        assert_eq!(fused.len(), config.latent_dim);
    }

    #[test]
    fn test_rosetta_net_with_tcn() {
        let config = RosettaNetConfig::default();
        let model = RosettaNetWithTCN::new(config.clone());
        let spectrogram = Array2::<f32>::zeros(config.spectrogram_shape);

        let output = model.forward_with_features(&spectrogram);

        assert_eq!(output.latent.len(), config.latent_dim);
        assert_eq!(output.rosetta_features.len(), config.rosetta_dim);
        assert_eq!(output.logits.len(), config.num_classes);
    }

    #[test]
    fn test_temporal_importance_analysis() {
        let config = RosettaNetConfig::default();
        let model = RosettaNetWithTCN::new(config.clone());
        let spectrogram = Array2::<f32>::zeros(config.spectrogram_shape);

        let importance = model.analyze_temporal_importance(&spectrogram);

        // Contributions should sum to 1.0
        let sum = importance.spectral_contribution + importance.temporal_contribution;
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rosetta_net_with_tcn_predict() {
        let config = RosettaNetConfig {
            num_classes: 10,
            ..Default::default()
        };
        let model = RosettaNetWithTCN::new(config.clone());
        let spectrogram = Array2::<f32>::zeros(config.spectrogram_shape);

        let predicted = model.predict(&spectrogram);
        assert!(predicted < config.num_classes);
    }

    #[test]
    fn test_encoder_type_hybrid() {
        let encoder_type = EncoderType::Hybrid;
        let json = serde_json::to_string(&encoder_type).unwrap();
        let decoded: EncoderType = serde_json::from_str(&json).unwrap();
        assert_eq!(encoder_type, decoded);
    }

    #[test]
    fn test_encoder_type_tcn() {
        let encoder_type = EncoderType::Tcn;
        let json = serde_json::to_string(&encoder_type).unwrap();
        let decoded: EncoderType = serde_json::from_str(&json).unwrap();
        assert_eq!(encoder_type, decoded);
    }
}
