# Implementation Plan: 112D DDSP Neural Decoder for Jetson Deployment

## Context

The Zoo Vox Rosetta Engine currently uses a **7D MicroDynamicsDelta bottleneck** between Python (cognitive layer) and Rust (execution layer). This limits synthesis to discrete grain playback rather than true generative synthesis. The goal is to replace this with a **112D-conditioned DDSP Neural Decoder** optimized for NVIDIA Jetson, enabling continuous acoustic synthesis with <50ms round-trip latency.

### Current State

**Bottleneck Identified:**
- Python sends 7D deltas via JSON over ZMQ (`realtime/action_publisher.py`)
- Rust supports 45D+ deltas but receives only 7D (`technical_architecture/src/synthesis.rs`)
- Synthesis uses grain concatenation, not generative synthesis

**Existing Foundation:**
- 112D RosettaFeatures extraction (Rust: `micro_dynamics_extractor.rs`)
- DDSP components exist (`cognitive_intelligence/ddsp_synthesis.py`) but are not PyTorch-differentiable
- 45-cluster vocabulary with centroids (`synthesis_manifest.json`)
- ZMQ IPC already configured (heartbeat and actions channels)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      NEW 112D DDSP PIPELINE                                     │
│                                                                                  │
│   Live Audio                                                                     │
│      │                                                                           │
│      ▼                                                                           │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │              NeuralBoundaryDetector (NBD)                                  │  │
│   │   Discovers semantic boundaries → variable-length phrases                  │  │
│   │   Example: [30ms opener, 180ms territorial, 45ms social]                   │  │
│   └──────────────────────────────────┬───────────────────────────────────────┘  │
│                                      │                                           │
│                                      ▼                                           │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │          MicroDynamicsExtractor (Variable-Length Input)                   │  │
│   │                                                                          │  │
│   │   Input: 180ms phrase                                                     │  │
│   │   Internal Framing: 3 frames @ 100ms/50ms hop                             │  │
│   │   Aggregation: mean/std/max across frames                                 │  │
│   │   Output: Single 112D vector representing the ENTIRE 180ms gesture        │  │
│   └──────────────────────────────────┬───────────────────────────────────────┘  │
│                                      │                                           │
│                                      ▼                                           │
│                              112D RosettaFeatures                                │
│                              (duration_ms = 180.0 ✓)                            │
│                                      │                                           │
│                                      ▼                                           │
│   Python (Cognitive)                                                          │
│      ┌──────────────┐    112D      ┌──────────────┐                              │
│      │ Interaction  │─────────────►│ DDSPDecoder  │ (PyTorch/TensorRT)          │
│      │ Agent        │   features   │ (MLP 112→65) │                              │
│      └──────────────┘              └──────┬───────┘                              │
│                                            │                                      │
│                                            ▼                                      │
│      ┌──────────────────────────────────────────────────┐                        │
│      │              DDSPSynthesizer                      │                        │
│      │  (60 harmonics + 5 noise bands → PCM audio)       │                        │
│      └──────────────────────────┬───────────────────────┘                        │
│                                 │ PCM audio                                        │
│                                 ▼                                                   │
│   Rust (Execution)                                                              │
│      ┌──────────────┐                                                              │
│      │   Audio      │                                                              │
│      │   Output     │──────────►► DAC                                              │
│      └──────────────┘                                                              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 0: NBD→112D Variable-Length Segment Compliance (PREREQUISITE) ✅ COMPLETED

### Critical Issue

The 112D extractor **MUST** process variable-length NBD segments directly. Using fixed 100ms sliding windows would:
- **Destroy duration_ms** (Index 4) - reports "100ms" instead of true phrase length
- **Break ADSR envelope** (Indices 36-45) - captures random slice instead of full lifecycle
- **Corrupt f0_contour_slope** (Index 46) - cannot compute across true gesture boundaries

### Objective

Ensure `MicroDynamicsExtractor` accepts variable-length NBD segments and produces valid 112D vectors through **internal frame aggregation** rather than external fixed windows.

### Files to Modify

1. **`technical_architecture/src/micro_dynamics_extractor.rs`**
   - Verify `extract_rosetta()` accepts variable-length audio
   - Ensure `duration_ms` is calculated from actual input length
   - Implement internal frame aggregation (mean/std/max) for statistical features

2. **`technical_architecture/src/neural_boundary.rs`**
   - Already has `segment_into_phrases()` helper
   - Verify it produces correct variable-length segments

3. **`technical_architecture/src/smart_segmenter.rs`**
   - Ensure pipeline uses NBD boundaries before 112D extraction

### Implementation Steps

**Step 0.1: Verify Duration Calculation**
```rust
// technical_architecture/src/micro_dynamics_extractor.rs
impl MicroDynamicsExtractor {
    pub fn extract_rosetta(&self, audio: &[f32]) -> Result<RosettaFeatures> {
        // CRITICAL: duration_ms MUST reflect actual input length
        let duration_ms = (audio.len() as f32 / self.sample_rate as f32) * 1000.0;

        // Internal framing for statistical aggregation
        let internal_frames = self.create_internal_frames(audio, 100, 50);

        RosettaFeatures {
            duration_ms,  // TRUE duration, not internal window size
            mean_f0_hz: self.aggregate_mean(&internal_frames, |f| f.f0),
            f0_range_hz: self.aggregate_range(&internal_frames, |f| f.f0),
            f0_contour_slope: self.fit_f0_slope(&internal_frames),
            // ... rest of 112D features
        }
    }
}
```

**Step 0.2: Implement Trajectory Fitting**
```rust
// For f0_contour_slope (Index 46) and f0_curvature (Index 47)
fn fit_f0_slope(&self, frames: &[FrameFeatures]) -> f32 {
    let f0_values: Vec<f32> = frames.iter().map(|f| f.f0).collect();
    // Linear regression: slope = covariance(x,y) / variance(x)
    let n = f0_values.len();
    let x_mean = (n - 1) as f32 / 2.0;
    let y_mean = f0_values.iter().sum::<f32>() / n as f32;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &f0) in f0_values.iter().enumerate() {
        let x = i as f32 - x_mean;
        let y = f0 - y_mean;
        numerator += x * y;
        denominator += x * x;
    }

    numerator / denominator.max(0.001)  // slope in Hz/frame
}
```

**Step 0.3: Add Zero-Padding for Short Segments**
```rust
// For segments shorter than internal FFT size
fn create_internal_frames(&self, audio: &[f32], frame_ms: usize, hop_ms: usize) -> Vec<FrameFeatures> {
    let frame_samples = (frame_ms * self.sample_rate as usize / 1000).min(audio.len());
    let hop_samples = (hop_ms * self.sample_rate as usize / 1000);

    let mut frames = Vec::new();

    for start in (0..audio.len()).step_by(hop_samples) {
        let end = (start + frame_samples).min(audio.len());
        let frame_audio = &audio[start..end];

        // Zero-pad if frame is incomplete
        let padded = if frame_audio.len() < frame_samples {
            let mut padded = vec![0.0f32; frame_samples];
            padded[..frame_audio.len()].copy_from_slice(frame_audio);
            padded
        } else {
            frame_audio.to_vec()
        };

        frames.push(self.extract_frame_features(&padded));
    }

    frames
}
```

**Step 0.4: Verify Pipeline Integration**
```rust
// technical_architecture/src/smart_segmenter.rs or pipeline controller
pub fn process_nbd_to_112d(audio: &[f32], nbd: &mut NeuralBoundaryDetector) -> Vec<(RosettaFeatures, Phrase)> {
    // 1. Detect boundaries (NBD)
    let boundaries = nbd.detect_boundaries(audio);

    // 2. Segment into variable-length phrases
    let phrases = segment_into_phrases(audio, &boundaries, nbd.sample_rate());

    // 3. Extract 112D from EACH variable-length phrase
    let mut results = Vec::new();
    for phrase_audio in phrases {
        let features = extractor.extract_rosetta(&phrase_audio)?;
        results.push((features, phrase_audio));
    }

    results
}
```

### Tests to Create

**`tests/test_nbd_to_112d_compliance.py`** (or Rust tests in `micro_dynamics_extractor.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extracts_features_from_short_nbd_segment() {
        // 30ms staccato opener (1440 samples @ 48kHz)
        let short_audio = vec![0.0f32; 1440];
        let extractor = MicroDynamicsExtractor::new(48000);
        let features = extractor.extract_rosetta(&short_audio).unwrap();

        // Must report the ACTUAL NBD segment length, not internal window
        assert_eq!(features.duration_ms, 30.0);
    }

    #[test]
    fn test_extracts_features_from_long_nbd_segment() {
        // 500ms graded closer (24000 samples @ 48kHz)
        let long_audio = vec![0.0f32; 24000];
        let extractor = MicroDynamicsExtractor::new(48000);
        let features = extractor.extract_rosetta(&long_audio).unwrap();

        assert_eq!(features.duration_ms, 500.0);
    }

    #[test]
    fn test_aggregates_internal_frames_correctly() {
        // A signal that rises in pitch from 4000Hz to 8000Hz over 200ms
        let chirp_audio = generate_chirp(4000.0, 8000.0, 200.0, 48000);
        let extractor = MicroDynamicsExtractor::new(48000);
        let features = extractor.extract_rosetta(&chirp_audio).unwrap();

        // f0_contour_slope (Index 46) must be POSITIVE (rising pitch)
        assert!(features.f0_mean_derivative > 0.0);
        // f0_range_hz (Index 2) must reflect the full 4000Hz sweep
        assert!(features.f0_range_hz > 3500.0);
    }

    #[test]
    fn test_zero_padding_for_sub_frame_segments() {
        // 5ms segment (240 samples @ 48kHz) - shorter than FFT size
        let tiny_audio = vec![0.5f32; 240];
        let extractor = MicroDynamicsExtractor::new(48000);
        let result = extractor.extract_rosetta(&tiny_audio);

        // Should succeed with zero-padding, not crash
        assert!(result.is_ok());
        assert_eq!(result.unwrap().duration_ms, 5.0);
    }

    #[test]
    fn test_rejects_empty_audio() {
        let empty_audio = vec![0.0f32; 0];
        let extractor = MicroDynamicsExtractor::new(48000);
        let result = extractor.extract_rosetta(&empty_audio);

        assert!(result.is_err());
    }
}
```

### Success Criteria
- `duration_ms` always reflects actual input length, not internal frame size
- Short segments (<100ms) are handled via zero-padding
- `f0_contour_slope` correctly captures pitch trajectories across variable-length gestures
- Pipeline integration verified: NBD → variable-length segments → 112D extraction

---

## Module 1: The 112D Delta Protocol (IPC Upgrade) ✅ COMPLETED

### Objective
Upgrade ZMQ IPC to transmit 112D feature modifications instead of 7D deltas, and add new AudioBufferEvent for PCM audio transmission.

### Files to Modify

1. **`realtime/action_publisher.py`**
   - Add `delta_112d: Optional[np.ndarray]` field to `SynthesisAction`
   - Add `AudioBufferEvent` class for PCM transmission

2. **`technical_architecture/src/synthesis.rs`**
   - Update `SynthesisAction` struct to accept `delta_112d: Option<Vec<f32>>`
   - Add `AudioBufferEvent` struct for receiving PCM from Python

3. **`realtime/feature_subscriber.py`**
   - Update to handle new audio buffer events

### Implementation Steps

**Step 1.1: Update Python SynthesisAction**
```python
# realtime/action_publisher.py
@dataclass
class SynthesisAction:
    action_type: str
    timeline: List[TimelineEvent]
    delta_112d: Optional[np.ndarray] = None  # NEW: 112D delta
    deltas: Optional[MicroDynamicsDelta] = None  # DEPRECATED
    priority: str = "normal"

@dataclass
class AudioBufferEvent:
    """PCM audio buffer generated by DDSP synthesizer."""
    audio_data: np.ndarray  # Shape: (samples,)
    sample_rate: int
    duration_ms: float
    timestamp: float
    sequence: int
```

**Step 1.2: Update Rust SynthesisAction**
```rust
// technical_architecture/src/synthesis.rs
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynthesisAction {
    pub action_type: String,
    pub timeline: Vec<TimelineEvent>,
    pub delta_112d: Option<Vec<f32>>,  // NEW: 112D delta
    #[serde(default)]
    pub deltas: Option<MicroDynamicsDelta>,  // DEPRECATED
    pub priority: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioBufferEvent {
    pub audio_data: Vec<f32>,
    pub sample_rate: usize,
    pub duration_ms: f32,
    pub timestamp: f64,
    pub sequence: u64,
}
```

**Step 1.3: Optimize Serialization (Optional Refactor)**
- Replace JSON with MessagePack for faster 112D array serialization
- Add `rmp-serde` dependency to Cargo.toml

### Tests to Create

**`tests/test_112d_ipc.py`**
- `test_synthesis_action_accepts_112d_delta()`
- `test_audio_buffer_event_serialization()`
- `test_112d_roundtrip_python_rust()`

---

## Module 2: The DDSP Decoder (Training Pipeline) ✅ COMPLETED

### Objective
Implement a PyTorch MLP that maps 112D RosettaFeatures to 65 DDSP control parameters (60 harmonic amplitudes + 5 noise coefficients).

### Files to Create

1. **`cognitive_intelligence/ddsp_decoder.py`** - New DDSP decoder module
2. **`cognitive_intelligence/ddsp_training.py`** - Training pipeline
3. **`cognitive_intelligence/multiscale_spectral_loss.py`** - Loss function

### Implementation Steps

**Step 2.1: DDSPDecoder Architecture**
```python
# cognitive_intelligence/ddsp_decoder.py
class DDSPDecoder(nn.Module):
    """
    MLP: 112D RosettaFeatures → 65 DDSP parameters
    Output: [60 harmonic amplitudes, 5 noise magnitudes]
    """
    def __init__(self, hidden_dim=256, num_harmonics=60, num_noise_bands=5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(112, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_harmonics + num_noise_bands),
        )
        self.num_harmonics = num_harmonics
        self.num_noise_bands = num_noise_bands

    def forward(self, features_112d):
        """
        Args:
            features_112d: (B, 112) RosettaFeatures
        Returns:
            harmonic_amps: (B, 60) - softmax normalized
            noise_mags: (B, 5) - relu activated
        """
        x = self.mlp(features_112d)
        harmonic_amps = F.softmax(x[:, :self.num_harmonics], dim=-1)
        noise_mags = F.relu(x[:, self.num_harmonics:])
        return harmonic_amps, noise_mags
```

**Step 2.2: MultiScale Spectral Loss**
```python
# cognitive_intelligence/multiscale_spectral_loss.py
class MultiScaleSpectralLoss(nn.Module):
    """STFT loss at multiple resolutions."""
    def __init__(self, frame_lengths=[512, 1024, 2048]):
        super().__init__()
        self.frame_lengths = frame_lengths

    def forward(self, pred_audio, target_audio):
        """
        Args:
            pred_audio: (B, 1, T)
            target_audio: (B, 1, T)
        Returns:
            loss: scalar
        """
        # Compute STFT at multiple resolutions
        # Combine L1 + L2 spectral distance
        # ...
```

**Step 2.3: Training Dataset**
```python
# cognitive_intelligence/ddsp_training.py
class VocalizationDataset(Dataset):
    """Dataset for training DDSP decoder."""
    def __init__(self, segments_json, audio_dir):
        # Load cached segments (8.9M from BEANS-Zero)
        # Each item: (features_112d, audio_waveform, f0)
        pass
```

### Tests to Create

**`tests/test_ddsp_decoder.py`**
- `test_ddsp_decoder_output_shapes()`
- `test_multiscale_spectral_loss()`
- `test_decoder_forward_pass()`
- `test_training_step()`

---

## Module 3: The DDSP Synthesizer (Differentiable Audio Engine)

### Objective
Implement a PyTorch-differentiable synthesizer that converts DDSP parameters to PCM audio using additive harmonics + filtered noise.

### Files to Modify

1. **`cognitive_intelligence/ddsp_synthesis.py`** - Convert existing to PyTorch modules

### Implementation Steps

**Step 3.1: Convert to PyTorch Modules**
```python
# cognitive_intelligence/ddsp_synthesis.py

class DifferentiableSineOscillator(nn.Module):
    """Phase-continuous sine oscillator."""
    def __init__(self, sample_rate=48000):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(self, f0, phase_acc=None):
        """
        Args:
            f0: (B, T_frames) frequency in Hz
            phase_acc: (B,) accumulated phase from previous call
        Returns:
            audio: (B, T_samples)
            phase_acc: (B,) new accumulated phase
        """
        # Cumulative phase integration
        # Sine generation
        pass

class DifferentiableNoiseFilter(nn.Module):
    """Differentiable FIR filter for noise shaping."""
    def forward(self, white_noise, filter_coefficients):
        """
        Args:
            white_noise: (B, T_samples)
            filter_coefficients: (B, 5) band magnitudes
        Returns:
            filtered_noise: (B, T_samples)
        """
        # Frequency-domain filtering
        pass

class DDSPSynthesizer(nn.Module):
    """Full DDSP synthesizer."""
    def __init__(self, sample_rate=48000, hop_size=480):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.oscillator = DifferentiableSineOscillator(sample_rate)
        self.noise_filter = DifferentiableNoiseFilter()

    def forward(self, f0, harmonic_amps, noise_mags, phase_acc=None):
        """
        Args:
            f0: (B, T_frames) fundamental frequency
            harmonic_amps: (B, T_frames, 60) harmonic amplitudes
            noise_mags: (B, T_frames, 5) noise magnitudes
            phase_acc: (B,) accumulated phase
        Returns:
            audio: (B, T_samples) output audio
            phase_acc: (B,) updated phase accumulator
        """
        # Generate harmonic component
        # Generate noise component
        # Mix and return
        pass
```

### Tests to Create

**`tests/test_ddsp_synthesizer.py`**
- `test_sine_oscillator_phase_continuity()`
- `test_harmonic_synthesis()`
- `test_noise_filtering()`
- `test_full_ddsp_synthesis()`
- `test_synthesis_output_length()`

---

## Module 4: Jetson Edge Deployment

### Objective
Deploy trained model on NVIDIA Jetson using TensorRT for FP16 inference with <2ms latency.

### Files to Create

1. **`cognitive_intelligence/jetson_export.py`** - ONNX/TensorRT export
2. **`realtime/ddsp_agent.py`** - Real-time inference agent

### Implementation Steps

**Step 4.1: ONNX Export**
```python
# cognitive_intelligence/jetson_export.py
def export_ddsp_decoder_to_onnx(checkpoint_path, output_path):
    """
    Export trained DDSPDecoder to ONNX for TensorRT.
    """
    model = DDSPDecoder.load_from_checkpoint(checkpoint_path)
    model.eval()

    dummy_input = torch.randn(1, 112)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['features_112d'],
        output_names=['harmonic_amps', 'noise_mags'],
        dynamic_axes={
            'features_112d': {0: 'batch_size'},
            'harmonic_amps': {0: 'batch_size'},
            'noise_mags': {0: 'batch_size'},
        },
        opset_version=14,
    )
```

**Step 4.2: TensorRT Optimization**
```python
# cognitive_intelligence/jetson_export.py
def build_tensorrt_engine(onnx_path, engine_path, fp16=True):
    """
    Build TensorRT engine from ONNX model.
    """
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(engine)
```

**Step 4.3: Real-time Inference Agent**
```python
# realtime/ddsp_agent.py
class DDSPAgent:
    """
    Real-time DDSP synthesis agent for Jetson deployment.
    """
    def __init__(self, model_path, synthesis_manifest_path):
        # Load TensorRT engine
        # Load cluster centroids
        # Initialize ZMQ publishers
        pass

    def handle_feature_event(self, event: FeatureEvent):
        """
        Generate response vocalization using DDSP.
        """
        # 1. Get cluster centroid
        # 2. Apply delta_112d modifications
        # 3. Run DDSP decoder (TensorRT)
        # 4. Run DDSP synthesizer
        # 5. Publish AudioBufferEvent
        pass
```

### Tests to Create

**`tests/test_jetson_deployment.py`**
- `test_onnx_export()`
- `test_tensorrt_engine_build()`
- `test_jetson_inference_latency()` (requires Jetson hardware)
- `test_audio_buffer_transmission()`

---

## Implementation Order

### Phase 0: Prerequisite (Week 0)
1. **Module 0**: NBD→112D variable-length segment compliance
2. Verify `duration_ms` is correct for all segment lengths
3. Implement trajectory fitting for f0_contour_slope

### Phase 1: Foundation (Week 1)
1. Module 1: 112D IPC upgrade
2. Module 2: DDSPDecoder architecture
3. Basic tests for both

### Phase 2: Training (Week 2)
1. Module 2: MultiScaleSpectralLoss
2. Module 2: Training pipeline
3. Train on cached segments (with correct duration_ms)

### Phase 3: Synthesis (Week 3)
1. Module 3: Convert DDSP components to PyTorch
2. Module 3: Full synthesizer integration
3. End-to-end training

### Phase 4: Deployment (Week 4)
1. Module 4: ONNX export
2. Module 4: TensorRT optimization
3. Module 4: Real-time agent

---

## Critical Files Summary

### Module 0 (New - Prerequisite)
- **Modify**: `technical_architecture/src/micro_dynamics_extractor.rs`
- **Modify**: `technical_architecture/src/neural_boundary.rs`
- **Modify**: `technical_architecture/src/smart_segmenter.rs`
- **Test**: `tests/test_nbd_to_112d_compliance.py`

### New Files to Create (DDSP)
- `cognitive_intelligence/ddsp_decoder.py` - DDSPDecoder MLP
- `cognitive_intelligence/multiscale_spectral_loss.py` - Loss function
- `cognitive_intelligence/ddsp_training.py` - Training pipeline
- `cognitive_intelligence/jetson_export.py` - ONNX/TensorRT export
- `realtime/ddsp_agent.py` - Real-time inference agent
- `tests/test_ddsp_decoder.py` - Decoder tests
- `tests/test_ddsp_synthesizer.py` - Synthesizer tests
- `tests/test_jetson_deployment.py` - Deployment tests
- `tests/test_112d_ipc.py` - IPC upgrade tests

### Files to Modify (DDSP)
- `realtime/action_publisher.py` - Add delta_112d, AudioBufferEvent
- `technical_architecture/src/synthesis.rs` - Add delta_112d, AudioBufferEvent
- `cognitive_intelligence/ddsp_synthesis.py` - Convert to PyTorch modules
- `realtime/feature_subscriber.py` - Handle audio buffer events

---

## Verification

### Module 0 Verification (Critical Prerequisite)
1. Create test audio segments: 30ms, 100ms, 500ms
2. Run through NBD → 112D pipeline
3. Verify `duration_ms` matches actual input length
4. Verify `f0_contour_slope` correctly captures rising/falling pitch

### End-to-End Test (DDSP)
1. Train DDSPDecoder on cached segments (with correct duration_ms)
2. Export to ONNX and build TensorRT engine
3. Run DDSPAgent with feature events
4. Measure latency (<50ms round-trip)
5. Validate audio quality (spectral reconstruction)

### Success Criteria
- **Module 0**: `duration_ms` accurate for all segment lengths, no truncation
- DDSPDecoder trains to convergence (spectral loss < 0.1)
- ONNX inference < 2ms on Jetson
- Full synthesis (decoder + synthesizer) < 50ms
- Generated audio matches bat vocalization characteristics
