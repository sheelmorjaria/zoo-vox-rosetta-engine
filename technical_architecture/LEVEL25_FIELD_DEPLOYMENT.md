# Level 2.5 Field Deployment Guide: Multi-Speaker Array for Spatial Audio Rendering

## Overview

This guide covers the deployment of a multi-speaker array for **Vector-Based Amplitude Panning (VBAP)** spatial audio rendering, enabling realistic directional playback of animal vocalizations in field conditions.

**Target Application:** Ethological validation experiments where spatial relationships between emitter and receiver animals affect response behavior (e.g., territorial calls, alarm choruses, mating displays).

---

## 1. Hardware Requirements

### Speaker Array Configuration

**Recommended Setup: Octagonal Array (8 speakers)**

| Component | Specification | Minimum | Recommended |
|-----------|---------------|---------|-------------|
| Speakers | Weatherproof full-range | 6 | 8 |
| Frequency Range | Coverage for target species | 1-10 kHz (marmoset) | 0.5-20 kHz |
| SPL at 1m | Maximum output | 90 dB | 100+ dB |
| Power Handling | Continuous | 20W | 50W+ |
| Impedance | Matching amplifier | 4Ω or 8Ω | 8Ω |
| Weather Protection | Rain/dust resistance | IP44 | IP65+ |

**Speaker Placement:**
- **Radius:** 2-3 meters from arena center
- **Height:** 1-1.5 meters above subject platform
- **Distribution:** Even angular spacing (45° for 8 speakers, 60° for 6 speakers)

### Amplification and Signal Chain

```
[Rust Engine] → [USB/PCIe Audio Interface] → [Multi-Channel Amp] → [Speaker Array]
```

| Component | Requirement |
|-----------|-------------|
| Audio Interface | 8+ independent output channels, 48kHz+, 24-bit |
| Amplifier | Multi-channel (4-8 channels), matched gain |
| Cabling | Shielded, weather-resistant burial-rated |

### Optional: Spatial Monitoring System

For validation and debugging:
- **Reference Microphone:** Omnidirectional, calibrated (e.g., Earthworks M50)
- **Position:** Center of arena at subject height
- **Purpose:** Verify VBAP rendering accuracy and level calibration

---

## 2. Coordinate System and Arena Setup

### World Coordinate System

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
```

### Arena Dimensions

| Parameter | Value | Notes |
|-----------|-------|-------|
| Arena Size | 10m × 10m | Default for Level 2.5 |
| Subject Platform | 0.5m × 0.5m | Centered at origin |
| Camera Height | 2-3m | Above arena floor |
| Speaker Height | 1-1.5m | At subject ear level |

### Speaker Position Reference (8-Speaker Array)

| ID | Angle (°) | X (m) | Y (m) | Description |
|----|-----------|-------|-------|-------------|
| spk_0 | 0 | +2.5 | 0 | East |
| spk_1 | 45 | +1.77 | +1.77 | Northeast |
| spk_2 | 90 | 0 | +2.5 | North |
| spk_3 | 135 | -1.77 | +1.77 | Northwest |
| spk_4 | 180 | -2.5 | 0 | West |
| spk_5 | 225 | -1.77 | -1.77 | Southwest |
| spk_6 | 270 | 0 | -2.5 | South |
| spk_7 | 315 | +1.77 | -1.77 | Southeast |

---

## 3. Software Configuration

### Rust Spatial Audio Configuration

File: `technical_architecture/src/spatial_audio.rs`

```rust
use technical_architecture::spatial_audio::{
    Position3D, Speaker, SpeakerArray, SpatialAudioRenderer,
};

// Create speaker array matching physical setup
fn create_octagonal_array() -> SpeakerArray {
    let radius = 2.5;  // meters
    let height = 1.2;  // meters

    let speakers = vec![
        Speaker::new("spk_0".into(), Position3D::new(radius, 0.0, height), 0.0),
        Speaker::new("spk_1".into(), Position3D::new(radius * 0.707, radius * 0.707, height), std::f32::consts::PI / 4.0),
        Speaker::new("spk_2".into(), Position3D::new(0.0, radius, height), std::f32::consts::PI / 2.0),
        // ... continue for all 8 speakers
    ];

    SpeakerArray::new(speakers)
}

// Create renderer
fn create_renderer() -> SpatialAudioRenderer {
    let array = create_octagonal_array();
    SpatialAudioRenderer::new(array, -60.0, 0.0)  // min: -60dB, max: 0dB
}
```

### Channel Mapping Configuration

File: `technical_architecture/deployment/speaker_config.toml`

```toml
[interface]
device_name = "USB Audio Interface"
sample_rate = 48000
buffer_size = 256
channels = 8

[speakers]
spk_0 = 0  # Output channel 0
spk_1 = 1  # Output channel 1
spk_2 = 2
spk_3 = 3
spk_4 = 4
spk_5 = 5
spk_6 = 6
spk_7 = 7

[calibration]
# Level trim per speaker (dB)
trim_spk_0 = 0.0
trim_spk_1 = -0.5
trim_spk_2 = 0.0
trim_spk_3 = +0.3
trim_spk_4 = 0.0
trim_spk_5 = -0.2
trim_spk_6 = 0.0
trim_spk_7 = +0.1
```

---

## 4. Calibration Procedures

### Level Calibration

**Goal:** Equal SPL at center position from all speakers.

**Equipment:**
- Calibrated SPL meter (A-weighted, slow)
- Pink noise generator (-20 dBFS)

**Procedure:**

1. Generate pink noise at -20 dBFS
2. Route to single speaker (e.g., spk_0)
3. Measure SPL at center position (subject height)
4. Target: 75-80 dB SPL (adjustable per experiment)
5. Adjust amplifier gain/trim to match target
6. Repeat for each speaker
7. Verify all speakers within ±1 dB

### VBAP Rendering Validation

**Goal:** Verify accurate spatial positioning.

**Test Procedure:**

```bash
# Run VBAP validation script
cd technical_architecture
cargo run --example vbap_validation --release
```

**Expected Results:**
- Source at 0° (East): Highest SPL at spk_0, symmetric roll-off to adjacent speakers
- Source at 45°: Equal contribution from spk_0 and spk_1
- Source at 90° (North): Highest SPL at spk_2

**Validation Criteria:**
| Test | Pass Criteria |
|------|---------------|
| Direct position | Target speaker > adjacent by ≥6 dB |
| Mid-position (45°) | Adjacent speakers within ±1 dB |
| Rear localization | Accurate perceptual localization by human listener |

---

## 5. Integration with Level 2.5 Pipeline

### Action Flow

```
[Python InteractionAgent]
        |
        v
[SpatialFrame: (x, y, heading)]
        |
        v
[Spatial Analysis: Proximity, LoS, Broadcast/Unicast]
        |
        v
[DualStreamAction + SpatialMetadata]
        |
        v
[ZMQ → Rust PeerController]
        |
        v
[SpatialAudioRenderer::render_with_spatial_metadata()]
        |
        v
[8-Channel Audio → Speaker Array]
```

### Python Configuration

File: `realtime/interaction_agent.py`

```python
def generate_spatial_action(
    self,
    emitter_pos: Tuple[float, float],
    target_pos: Optional[Tuple[float, float]],
    is_broadcast: bool,
) -> DualStreamAction:
    """
    Generate action with spatial metadata for VBAP rendering.
    """
    spatial_metadata = SpatialMetadata(
        position=(emitter_pos[0], emitter_pos[1], 1.2),  # Include height
        target_position=target_pos,
        mode=SpatialMode.UNICAST if target_pos else SpatialMode.BROADCAST,
        spread_deg=30.0 if is_broadcast else 15.0,
    )

    return DualStreamAction(
        syntactic_token=self.selected_token,
        affect_vector=self.affect_vector,
        spatial_metadata=spatial_metadata,
    )
```

---

## 6. Field Deployment Checklist

### Pre-Deployment

- [ ] Speaker positions measured and recorded
- [ ] Speakers securely mounted (weather-resistant)
- [ ] Cabling connected and tested (continuity check)
- [ ] Amplifier levels set (avoid clipping)
- [ ] Audio interface drivers installed on edge device
- [ ] Channel mapping verified in software
- [ ] Level calibration completed (all speakers ±1 dB)
- [ ] VBAP validation passed

### Testing

- [ ] Play test tone at each speaker position
- [ ] Verify spatial rendering at 8 compass points
- [ ] Measure SPL at center (verify target level)
- [ ] Test automated playback via Rust engine
- [ ] Verify ZMQ communication (Python → Rust)

### Monitoring During Deployment

- [ ] Speaker health (no distortion, dropouts)
- [ ] Weather protection active (rain covers if needed)
- [ ] Backup recordings saved (multi-channel for post-analysis)
- [ ] Log file monitoring for VBAP errors

---

## 7. Troubleshooting

### Issue: Sound appears from wrong direction

**Possible Causes:**
1. Speaker ID mapping incorrect
2. Physical position mismatched from configuration
3. Channel routing swapped in audio interface

**Solution:**
- Verify channel mapping with speaker-by-speaker test tones
- Update `speaker_config.toml` if channels are swapped
- Re-measure physical positions

### Issue: Level imbalance across directions

**Possible Causes:**
1. Speakers not equidistant from center
2. Room/arena acoustics causing reflections
3. Amplifier gain mismatch

**Solution:**
- Re-measure distances, adjust configuration if needed
- Apply calibration trim in `speaker_config.toml`
- Consider acoustic treatment for problematic reflections

### Issue: VBAP sounds "blurry" or unfocused

**Possible Causes:**
1. Speakers too far apart (large gap between coverage)
2. Spread parameter too high
3. Room reflections interfering

**Solution:**
- Reduce `spread_deg` in spatial metadata (try 10-15°)
- Verify speaker spacing (ideally ≤2.5m radius for 8 speakers)
- Add acoustic absorption if reflections are severe

---

## 8. Example: Spatial Mismatch Experiment Setup

### Configuration

```python
# Condition A: Congruent (audio and visual from same direction)
emitter_pos = (1.5, 0.0)  # East
speaker_angle = 0  # East

# Condition B: Incongruent (audio from opposite side)
emitter_pos = (1.5, 0.0)  # East (visual)
speaker_angle = 180  # West (audio)
```

### Response Measurement

| Metric | Description |
|--------|-------------|
| Orientation Angle | Subject's body heading relative to source |
| Response Latency | Time from stimulus onset to orienting response |
| Approach/Avoidance | Movement toward or away from apparent source |
| Vocalization | Type and timing of response calls |

### Success Criteria

- **Condition A (Congruent):** Subjects orient toward speaker, faster response latency
- **Condition B (Incongruent):** Confusion, delayed response, or orientation toward visual rather than audio

---

## 9. Safety and Ethical Considerations

### Sound Exposure Limits

- **Maximum SPL:** 90 dB at subject position (temporary threshold shift risk above)
- **Duration Limits:** 15 minutes continuous at >85 dB
- **Recovery Time:** ≥5 minutes between high-SPL sessions

### Animal Welfare

- Monitor subjects for stress indicators
- Provide escape routes/avoidance options
- Do not separate dependent individuals (e.g., marmoset pairs)
- Follow institutional animal care and use committee (IACUC) guidelines

### Environmental Protection

- Weather-resistant equipment to prevent damage
- Secure cables to prevent tripping hazards
- Minimal impact on habitat during installation

---

## 10. Reference Implementations

### Rust VBAP Renderer: `technical_architecture/src/spatial_audio.rs`
### Python Integration: `realtime/action_publisher.py`
### Test Suite: `tests/test_level25_validation.py`

---

## Appendix A: VBAP Algorithm Summary

Vector-Based Amplitude Panning calculates gains for N speakers to render a virtual source at position P.

**For 2D (horizontal) rendering with 2 speakers:**

1. Find speaker pair S₁, S₂ that brackets target angle
2. Calculate unit vectors to speakers
3. Solve: P = g₁·v₁ + g₂·v₂ (vector basis)
4. Apply gains: g₁², g₂² (energy preservation)

**Extension to 3D:** Use triplet of speakers forming a polygon around target position.

---

**Document Version:** 1.0
**Last Updated:** 2026-05-10
**Author:** Sheel Morjaria (sheelmorjaria@gmail.com)
**License:** CC BY-ND 4.0 International
