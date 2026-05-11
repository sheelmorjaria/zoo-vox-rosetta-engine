# Level 2.5: Spatial-Social Network Integration

## Overview

Level 2.5 adds **spatial awareness** to the Zoo Vox Rosetta Engine, enabling the system to understand **where** animals are located relative to each other and incorporate this information into response decisions. This enables:

- **Receiver Inference** - Identify intended recipients based on spatial proximity and orientation
- **Call Directionality** - Determine if a call is broadcast (to all nearby) or unicast (directed at specific individuals)
- **Spatial Audio Rendering** - VBAP (Vector-Based Amplitude Panning) for realistic directional playback
- **DeepLabCut Integration** - Real-time pose estimation from RTSP camera streams

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Level 2.5 Spatial Pipeline                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │   RTSP       │     │ DeepLabCut  │     │   Spatial    │                │
│  │  Cameras     │────▶│  Pose Est.  │────▶│    Frame     │                │
│  │  (4-8 cams)  │     │  (ONNX/Rust) │     │ (Agent,Pos,H)│                │
│  └──────────────┘     └──────────────┘     └──────┬───────┘                │
│                                                       │                       │
│  ┌──────────────┐     ┌──────────────┐             │                       │
│  │   Speaker    │     │ Topology     │◀────────────┘                       │
│  │   Array      │◀────│  Engine      │                                     │
│  │  (VBAP 8ch)  │     │ (Prox,LoS,   │                                     │
│  │              │     │  Broadcast)  │                                     │
│  └──────────────┘     └──────────────┘                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. DeepLabCut RTSP Ingestor

**File:** `spatial_intelligence/deeplabcut_ingestor.py`

Real-time pose estimation from camera streams with pixel-to-world coordinate transformation.

**Key Classes:**
- `DLCCameraConfig` - Camera position, orientation, calibration
- `PoseKeypoints` - Detected keypoints with heading computation
- `DeepLabCutIngestor` - Main ingestor with RTSP support

**Features:**
- Multi-camera support (RTSP, USB, FILE, TEST_PATTERN)
- Automatic pixel-to-world coordinate transformation
- Pose heading computation (nose→tail, shoulder fallback)
- Threaded capture for real-time processing

```python
from spatial_intelligence.deeplabcut_ingestor import (
    create_test_camera_config,
    DeepLabCutIngestor,
)

# Create 4-camera circular array
configs = create_test_camera_config(num_cameras=4)
ingestor = DeepLabCutIngestor(configs, area_size=10.0)

# Generate spatial frame
frame = ingestor.generate_frame(timestamp_ns=0)
```

### 2. Topology Engine

**File:** `spatial_intelligence/topology_engine.py`

Maintains real-time spatial relationships between agents.

**Key Classes:**
- `TopologyEngine` - Main topology manager
- `AgentState` - Extended state with nearby/visible agents
- `ProximityResult` - Nearby agents sorted by distance
- `LineOfSightResult` - Field-of-view check results

**Features:**
- Proximity maps (who is near whom)
- Line-of-sight calculations (120° FoV)
- Spatial queries (find agents within radius)
- Colony state management

```python
from spatial_intelligence.topology_engine import TopologyEngine

engine = TopologyEngine(max_agents=100, proximity_radius=5.0)

# Update from spatial frame
engine.update_topology(frame)

# Query proximity
result = engine.get_proximity_result("agent_001")

# Check line of sight
los = engine.check_line_of_sight("emitter", "target")
if los.in_field_of_view:
    # Target is visible to emitter
```

### 3. Spatial Audio Rendering

**File:** `technical_architecture/src/spatial_audio.rs`

VBAP-based spatial audio for directional playback.

**Key Structures:**
- `Position3D` - 3D coordinates (x, y, z)
- `Speaker` - Speaker with position and heading
- `SpeakerArray` - Multi-speaker configuration
- `SpatialAudioRenderer` - VBAP rendering engine

**Features:**
- Vector-Based Amplitude Panning
- Support for 2-16 speaker arrays
- Per-speaker gain control (-60dB to 0dB)
- Position-based rendering

```rust
use technical_architecture::spatial_audio::{SpatialAudioRenderer, Speaker, SpeakerArray};

let array = SpeakerArray::new(vec![
    Speaker::new("spk_0", Position3D::new(2.5, 0.0, 1.2), 0.0),
    // ... more speakers
]);

let renderer = SpatialAudioRenderer::new(array, -60.0, 0.0);

// Render audio at specific position
let gains = renderer.render_at_position(1.5, 0.5, 1.2);
```

### 4. Broadcast vs Unicast Classification

Determines communication type based on spatial relationships.

**Classification Logic:**
```python
def classify_communication(emitter, receivers, probabilities):
    """
    Broadcast if:
    - No single receiver has probability > 0.65
    - Or multiple receivers with similar probabilities

    Unicast if:
    - One receiver clearly dominates (p > 0.65)
    """
    max_prob = max(probabilities.values()) if probabilities else 0

    if max_prob >= 0.65:
        return Unicast(target=highest_probability_receiver)
    else:
        return Broadcast()
```

### 5. ZMQ Integration

**Files:** `realtime/action_publisher.py`, `technical_architecture/src/peer_controller.rs`

Extended DualStreamAction with spatial metadata.

```python
@dataclass
class SpatialMetadata:
    position: Tuple[float, float, float]  # x, y, z
    target_position: Optional[Tuple[float, float, float]]
    mode: SpatialMode  # BROADCAST or UNICAST
    spread_deg: float  # VBAP spread angle

@dataclass
class DualStreamAction:
    syntactic_token: int
    affect_vector: np.ndarray  # 16D
    spatial_metadata: Optional[SpatialMetadata]  # NEW
```

## Test Results

| Test Suite | Tests | Status | File |
|------------|-------|--------|------|
| DeepLabCut Ingestor | 16 | ✅ Pass | `tests/test_deeplabcut_ingestor.py` |
| Level 2.5 Validation | 15 | ✅ Pass | `tests/test_level25_validation.py` |
| Hardware-in-the-Loop | 17 | ✅ Pass | `tests/test_hardware_in_the_loop.py` |
| **Total** | **48** | **✅ All Pass** | |

### Key Test Coverage

**Unit Tests:**
- Pose keypoint creation and heading computation
- Pixel-to-world coordinate transformation
- Camera configuration validation
- Proximity calculation accuracy
- Line-of-sight field-of-view detection

**Integration Tests:**
- Multi-camera triangulation
- "Crowd Test" (20 agents)
- "Back-Turned Test" (LoS penalty)
- Full pipeline simulation
- Multi-frame consistency

**Ethological Framework:**
- Spatial Mismatch test (Conditions A, B, C)
- Response Appropriateness Score (RAS) calculation

---

## Field Deployment Guide

### Hardware Requirements

**Speaker Array (Recommended: Octagonal 8-Speaker)**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Speakers | 6 | 8 |
| Frequency Range | 1-10 kHz (species-dependent) | 0.5-20 kHz |
| SPL at 1m | 90 dB | 100+ dB |
| Power Handling | 20W | 50W+ |
| Weather Protection | IP44 | IP65+ |

**Speaker Position Reference (8-Speaker, 2.5m radius):**

| ID | Angle | X (m) | Y (m) | Description |
|----|-------|-------|-------|-------------|
| spk_0 | 0° | +2.5 | 0 | East |
| spk_1 | 45° | +1.77 | +1.77 | Northeast |
| spk_2 | 90° | 0 | +2.5 | North |
| spk_3 | 135° | -1.77 | +1.77 | Northwest |
| spk_4 | 180° | -2.5 | 0 | West |
| spk_5 | 225° | -1.77 | -1.77 | Southwest |
| spk_6 | 270° | 0 | -2.5 | South |
| spk_7 | 315° | +1.77 | -1.77 | Southeast |

**RTSP Cameras (4-8 recommended):**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Resolution | 720p | 1080p |
| Frame Rate | 15 FPS | 30 FPS |
| Protocol | RTSP | RTSP/ONVIF |
| Weather Protection | IP44 | IP65+ |

### Calibration Procedures

**Level Calibration:**
1. Generate pink noise at -20 dBFS
2. Route to single speaker
3. Measure SPL at center position (75-80 dB target)
4. Adjust amplifier gain/trim per speaker
5. Verify all speakers within ±1 dB

**VBAP Validation:**
1. Test source at 0° (East) - spk_0 should dominate
2. Test source at 45° - spk_0 and spk_1 equal contribution
3. Test source at 90° (North) - spk_2 should dominate
4. Verify smooth panning between positions

### Coordinate System

```
        +Y (North)
         |
         |
    -----+----- +X (East)
         |
         |
        -Y (South)

Origin (0,0,0) = Arena center, subject height
Z = Upward (height above floor)
Arena: 10m × 10m (default)
```

---

## Ethological Validation: Spatial Mismatch Test

### Experimental Design

**Conditions:**

| Condition | Visual Location | Acoustic Location | Prediction |
|-----------|-----------------|-------------------|------------|
| A (Congruent) | East | East | Fast, accurate orientation |
| B (Incongruent) | East | West | Confusion, delayed response |
| C (Control) | None | East | Acoustic-only orientation |

### Response Appropriateness Score (RAS)

```
RAS = w₁·Spatial_Accuracy + w₂·Temporal_Speed + w₃·Response_Type

Where:
Spatial_Accuracy = 1 - (|orientation_error| / 180)
Temporal_Speed = 1 - min(latency / 5000, 1)
Response_Type = 1.0 for orient/approach
              = 0.5 for vocal only
              = 0.0 for avoid/none

Weights: w₁=0.5, w₂=0.3, w₃=0.2
```

### Expected Results

| Condition | Expected RAS | Interpretation |
|-----------|--------------|----------------|
| A (Congruent) | 0.7 - 0.9 | Accurate, rapid response |
| B (Incongruent) | 0.3 - 0.6 | Confused, delayed response |
| C (Acoustic-only) | 0.5 - 0.7 | Intermediate accuracy |

### Statistical Analysis

- **Test:** Repeated-measures ANOVA
- **Factor:** Condition (A, B, C)
- **Post-hoc:** Tukey HSD
- **Expected:** Condition A ≠ Condition B (p < 0.05)

### Sample Size

- **Power analysis:** N = 12 subjects (f = 0.4, α = 0.05, power = 0.8)
- **Accounting for attrition:** Target N = 15

---

## Remaining for Field Deployment

### Phase 1: Hardware Installation

**Priority: HIGH**

| Task | Estimated Time | Dependencies |
|------|----------------|--------------|
| Install speaker array (8 speakers) | 4 hours | Weatherproof mounting hardware |
| Connect amplification and audio interface | 2 hours | Multi-channel amp, USB interface |
| Install RTSP cameras (4-8) | 4 hours | PoE switch, cabling |
| Network configuration | 2 hours | Router, switch setup |
| **Total** | **12 hours** | |

**Detailed Checklist:**

- [ ] Mount speakers at 2.5m radius, 1.2m height
- [ ] Verify speaker IDs match configuration file
- [ ] Connect speaker cables (shielded, burial-rated)
- [ ] Set up amplification in weatherproof enclosure
- [ ] Connect audio interface to edge device
- [ ] Mount cameras with clear FoV to arena
- [ ] Configure RTSP streams (test connectivity)
- [ ] Power all equipment (UPS recommended)
- [ ] Label all cables for maintenance

### Phase 2: Software Configuration

**Priority: HIGH**

| Task | Estimated Time | Dependencies |
|------|----------------|--------------|
| Deploy Rust synthesis engine | 1 hour | Edge device setup |
| Configure speaker channel mapping | 30 min | Audio interface drivers |
| Calibrate VBAP rendering | 2 hours | Speaker array installed |
| Set up DeepLabCut ONNX models | 2 hours | Model training/export |
| Configure camera coordinate transforms | 2 hours | Camera positions measured |
| **Total** | **7.5 hours** | |

**Configuration Files:**

```toml
# technical_architecture/deployment/speaker_config.toml
[interface]
device_name = "USB Audio Interface"
sample_rate = 48000
channels = 8

[speakers]
spk_0 = 0  # Output channel mapping
spk_1 = 1
# ... etc

[calibration]
trim_spk_0 = 0.0  # Level trim (dB)
# ... per-speaker adjustment
```

### Phase 3: Calibration and Validation

**Priority: HIGH**

| Task | Estimated Time | Dependencies |
|------|----------------|--------------|
| Level calibration (all speakers) | 1 hour | Speakers powered |
| VBAP rendering validation | 2 hours | Level calibrated |
| Camera-to-world calibration | 3 hours | Cameras operational |
| End-to-end latency test | 1 hour | Full pipeline running |
| **Total** | **7 hours** | |

**Calibration Script:**

```bash
# Run VBAP validation
cd technical_architecture
cargo run --example vbap_validation --release

# Expected output:
# Speaker 0: 45.2° error (OK)
# Speaker 1: 3.1° error (OK)
# ...
# All speakers within 5° tolerance: PASS
```

### Phase 4: Ethological Pilot

**Priority: MEDIUM**

| Task | Estimated Time | Dependencies |
|------|----------------|--------------|
| Subject habituation (2-3 sessions) | 3 hours | Hardware installed |
| Pilot data collection (Condition A) | 2 hours | Subjects habituated |
| Pilot data collection (Condition B) | 2 hours | Condition A complete |
| Data analysis and RAS calculation | 2 hours | Data collected |
| Protocol adjustment if needed | 2 hours | Analysis complete |
| **Total** | **11 hours** | |

**Pilot Success Criteria:**

- [ ] At least 5 subjects complete all conditions
- [ ] RAS(A) > RAS(B) with p < 0.1 (trend)
- [ ] No equipment failures during sessions
- [ ] Latency budget met (<125ms)

### Phase 5: Full Data Collection

**Priority: MEDIUM**

| Task | Estimated Time | Dependencies |
|------|----------------|--------------|
| Recruit subjects (N=12-15) | 1 week | IACUC approval |
| Counterbalanced condition assignment | 1 day | Subjects recruited |
| Data collection (all subjects) | 2-3 weeks | Pilot complete |
| Data processing and analysis | 1 week | Collection complete |
| **Total** | **4-5 weeks** | |

### Phase 6: Publication and Deployment

**Priority: LOW**

| Task | Estimated Time | Dependencies |
|------|----------------|--------------|
| Manuscript preparation | 2 weeks | Analysis complete |
| Peer review and revision | 3-6 months | Manuscript submitted |
| Documentation release | 1 week | Manuscript accepted |
| **Total** | **4-7 months** | |

---

## Latency Budget

| Component | Target | Notes |
|-----------|--------|-------|
| Pose Detection | 50ms | DeepLabCut ONNX inference |
| Topology Analysis | 10ms | 5 agents typical |
| Action Generation | 10ms | Probability calculation |
| ZMQ Transit | 5ms | Localhost |
| Synthesis | 50ms | DDSP rendering |
| **Total** | **125ms** | Current budget |

**Optimization Path to 100ms:**
- Optimize DeepLabCut model (TensorRT) → 25ms
- Parallel topology updates → 5ms
- **Total: ~85ms**

---

## API Reference

### Python API

```python
from spatial_intelligence import (
    DeepLabCutIngestor,
    TopologyEngine,
    SpatialObservation,
    SpatialFrame,
)

# Create ingestor
ingestor = DeepLabCutIngestor(configs, area_size=10.0)

# Generate frame
frame = ingestor.generate_frame(timestamp_ns=0)

# Create topology engine
engine = TopologyEngine(max_agents=100, proximity_radius=5.0)

# Update topology
count = engine.update_topology(frame)

# Query
proximity = engine.get_proximity_result("agent_id")
los = engine.check_line_of_sight("emitter", "target")
```

### Rust API

```rust
use technical_architecture::spatial_audio::{
    Position3D, Speaker, SpeakerArray, SpatialAudioRenderer,
};

// Create speaker array
let speakers = vec![
    Speaker::new("spk_0", Position3D::new(2.5, 0.0, 1.2), 0.0),
    // ...
];
let array = SpeakerArray::new(speakers);

// Create renderer
let renderer = SpatialAudioRenderer::new(array, -60.0, 0.0);

// Render
let gains = renderer.render_at_position(x, y, z);
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Sound from wrong direction | Speaker ID mapping incorrect | Verify channel mapping with test tones |
| Level imbalance | Speakers not equidistant | Apply calibration trim |
| VBAP sounds "blurry" | Spread parameter too high | Reduce `spread_deg` to 10-15° |
| Low pose confidence | Poor lighting or camera position | Improve lighting, adjust camera angle |
| High latency | CPU bottleneck | Optimize models, use TensorRT |

---

## References

1. **Field Deployment Guide:** `technical_architecture/LEVEL25_FIELD_DEPLOYMENT.md`
2. **Ethological Validation:** `technical_architecture/ETHOLOGICAL_VALIDATION_PROTOCOL.md`
3. **Spatial Audio Source:** `technical_architecture/src/spatial_audio.rs`
4. **DeepLabCut Ingestor:** `spatial_intelligence/deeplabcut_ingestor.py`
5. **Topology Engine:** `spatial_intelligence/topology_engine.py`

---

**Document Version:** 1.0
**Last Updated:** 2026-05-10
**Author:** Sheel Morjaria (sheelmorjaria@gmail.com)
**License:** CC BY-ND 4.0 International
