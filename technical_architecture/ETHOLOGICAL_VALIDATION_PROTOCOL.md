# Ethological Validation Protocol: Spatial Mismatch Test

## Overview

This protocol describes the **Spatial Mismatch Test**, an ethological validation experiment for Level 2.5 Spatial-Social Network Integration. The test assesses whether animals integrate spatial and acoustic information when responding to conspecific vocalizations.

**Scientific Question:** Do animals weight visual/orientation cues differently from acoustic cues when they are spatially incongruent?

**Hypothesis:** In Condition A (congruent), subjects will orient rapidly toward the source. In Condition B (incongruent), subjects will show confusion, delayed response, or preferential orientation toward visual cues.

---

## 1. Experimental Design

### Conditions

| Condition | Visual Location | Acoustic Location | Prediction |
|-----------|-----------------|-------------------|------------|
| **A (Congruent)** | East | East | Fast, accurate orientation |
| **B (Incongruent)** | East | West | Confusion, delayed response |
| **C (Control)** | None | East | Acoustic-only orientation |

### Within-Subjects Design

- Each subject experiences all conditions
- Counterbalanced order to control for learning/sequence effects
- Minimum 1 hour between conditions for same subject
- 10-15 trials per condition per subject

### Subject Species Recommendations

| Species | Social Structure | Spatial Cognition | Recommended N |
|---------|------------------|-------------------|---------------|
| Marmoset (*Callithrix*) | Pair-bonded, territorial | High (3D navigation) | 6-8 pairs |
| Zebra Finch (*Taeniopygia*) | Colonial, flock | Medium (2D navigation) | 12-15 individuals |
| Dolphin (*Tursiops*) | Fission-fusion | Very High (3D) | 4-6 individuals |
| Bat (*Rousettus*) | Colonial | High (3D cave) | 8-10 individuals |

---

## 2. Apparatus Setup

### Physical Layout

```
                    North (0, 2.5m)
                         |
                         |
       West (-2.5, 0) ----+---- East (2.5, 0)
                         |
                         |
                    South (0, -2.5m)

Subject Platform: (0, 0) - Center
Speaker Array: 8 speakers at 2.5m radius
Visual Stimuli: LED/projector arrays at cardinal points
```

### Visual Stimulus System

**Purpose:** Provide apparent visual location of "emitter" animal.

**Implementation Options:**

1. **LED Array Board:** High-brightness LEDs in animal-like shape
   - Position: 1.5m from center, at speaker height
   - Brightness: Visible in ambient light
   - Pattern: Pulse synchronized with call onset

2. **Silhouette Projector:** Backlit animal silhouette
   - More natural appearance
   - Requires projector + screen

3. **Taxidermy/Robot:** Realistic mount with servos
   - Most realistic but expensive
   - Limited deployment practicality

### Acoustic Stimulus System

- 8-speaker array (see LEVEL25_FIELD_DEPLOYMENT.md)
- VBAP rendering for spatial accuracy
- Stimuli: Species-specific vocalizations from database

### Recording System

- **Primary:** Overhead camera (tracking position/orientation)
- **Secondary:** Close-up camera (subject facial expression)
- **Audio:** Reference microphone at subject position
- **Sync:** All streams timestamped via PTP clock

---

## 3. Stimulus Selection

### Call Types

| Category | Function | Spatial Expectation |
|----------|----------|---------------------|
| Contact Call | Group cohesion | Broadcast (all directions) |
| Alarm Call | Predator warning | Directed (toward threat) |
| Territorial Call | Defense | Directed (at intruder) |
| Mating Call | Courtship | Directed (at potential mate) |

### Recommended Stimuli per Species

**Marmoset:**
- **Phee Call:** Long-distance contact (broadcast)
- **Tsik Alarm:** Predator alert (directed)
- **Twitter:** Social contact (mixed)

**Zebra Finch:**
- **Distance Call:** Contact (broadcast)
- **Tet Note:** Alert (directed)
- **Song:** Courtship (directed)

**Dolphin:**
- **Signature Whistle:** Individual ID (directed/broadcast)
-**Burst Pulse:** Excitement/aggression (directed)

### Stimulus Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Duration | 200-500 ms | Natural call length |
| SPL at Subject | 75-85 dB | Below TTS threshold |
| F0 Range | Species-specific | Natural variation |
| Background | Species-typical | Ambient jungle/barn noise |

---

## 4. Trial Procedure

### Single Trial Sequence

| Time | Event | Notes |
|------|-------|-------|
| T-30s | Subject habituation | Quiet, no stimuli |
| T-5s | Visual cue onset | LED/pseudo-emitter activated |
| T0 | Acoustic stimulus | Call played from speaker |
| T+5s | Response window | Observe orientation, vocalization |
| T+30s | Inter-trial interval | Return to baseline |

### Response Categories

1. **Orient Response:** Head/body turn toward source
2. **Approach:** Movement toward apparent source
3. **Vocal Response:** Reply call, type and timing
4. **No Response:** No detectable behavior change
5. **Avoidance:** Movement away from source

### Measured Variables

| Variable | Type | Unit |
|----------|------|------|
| Orientation Latency | Temporal | ms |
| Orientation Angle | Spatial | degrees |
| Orientation Accuracy | Spatial | degrees error |
| Response Latency | Temporal | ms |
| Response Type | Categorical | enum |
| Movement Distance | Spatial | meters |
| Vocalization Type | Categorical | enum |

---

## 5. Response Appropriateness Score (RAS)

### Definition

RAS quantifies how "appropriate" a subject's response is given the spatial information provided.

### Calculation

```
RAS = w₁·Spatial_Accuracy + w₂·Temporal_Speed + w₃·Response_Type

Where:
Spatial_Accuracy = 1 - (|orientation_error| / 180)
Temporal_Speed = 1 - min(latency / 5000, 1)  # Normalized to 5s max
Response_Type = 1.0 for orient/approach
              = 0.5 for vocal only
              = 0.0 for avoid/none

Recommended weights: w₁=0.5, w₂=0.3, w₃=0.2
```

### Expected RAS by Condition

| Condition | Expected RAS | Interpretation |
|-----------|--------------|----------------|
| A (Congruent) | 0.7 - 0.9 | Accurate, rapid response |
| B (Incongruent) | 0.3 - 0.6 | Confused, delayed response |
| C (Acoustic-only) | 0.5 - 0.7 | Intermediate accuracy |

---

## 6. Statistical Analysis Plan

### Primary Analysis

**Hypothesis:** RAS differs significantly between conditions.

**Test:** Repeated-measures ANOVA
- Factor: Condition (A, B, C)
- Dependent Variable: RAS
- Post-hoc: Tukey HSD for pairwise comparisons

**Expected Results:**
- Condition A > Condition B (p < 0.05)
- Condition C intermediate (may not differ from A or B)

### Secondary Analyses

1. **Orientation Latency:** ANOVA across conditions
2. **Orientation Accuracy:** Circular statistics (Rayleigh test)
3. **Vocal Response Rate:** Chi-square test

### Sample Size Justification

**Power Analysis:**
- Effect size: f = 0.4 (medium-large, based on pilot data)
- Alpha: 0.05
- Power: 0.8
- Result: N = 12 subjects

**Accounting for Attrition:**
- Target enrollment: N + 20%

---

## 7. Implementation Code

### Python Configuration

```python
# realtime/spatial_mismatch_experiment.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import numpy as np

class Condition(Enum):
    CONGRUENT = "A"      # Visual and acoustic from same direction
    INCONGRUENT = "B"    # Visual and acoustic from opposite directions
    ACOUSTIC_ONLY = "C"  # Acoustic only (no visual cue)

@dataclass
class TrialConfig:
    """Configuration for a single trial."""
    condition: Condition
    visual_position: Tuple[float, float]  # (x, y) in meters
    acoustic_position: Tuple[float, float]  # (x, y) in meters
    call_type: str  # e.g., "phee", "alarm", "territorial"
    duration_ms: int = 300
    spl_db: float = 80.0

@dataclass
class TrialResult:
    """Results from a single trial."""
    subject_id: str
    condition: Condition
    orientation_latency_ms: Optional[int]
    orientation_angle_deg: Optional[float]
    orientation_error_deg: Optional[float]
    response_type: str  # "orient", "approach", "vocal", "none", "avoid"
    movement_distance_m: float
    vocalization_type: Optional[str]
    ras_score: float

def calculate_ras(
    orientation_error_deg: Optional[float],
    latency_ms: Optional[int],
    response_type: str,
) -> float:
    """Calculate Response Appropriateness Score."""
    w1, w2, w3 = 0.5, 0.3, 0.2  # Weights

    # Spatial accuracy
    if orientation_error_deg is None:
        spatial_acc = 0.0
    else:
        spatial_acc = 1.0 - (abs(orientation_error_deg) / 180.0)

    # Temporal speed (normalize to 5s max)
    if latency_ms is None:
        temporal_speed = 0.0
    else:
        temporal_speed = 1.0 - min(latency_ms / 5000.0, 1.0)

    # Response type
    response_scores = {
        "orient": 1.0,
        "approach": 1.0,
        "vocal": 0.5,
        "none": 0.0,
        "avoid": 0.0,
    }
    resp_score = response_scores.get(response_type, 0.0)

    return w1 * spatial_acc + w2 * temporal_speed + w3 * resp_score

def generate_trial_configs(
    num_trials: int = 15,
    counterbalance: bool = True,
) -> list[TrialConfig]:
    """Generate trial configurations for a subject."""
    configs = []

    # Cardinal directions (meters)
    positions = {
        "north": (0.0, 2.5),
        "south": (0.0, -2.5),
        "east": (2.5, 0.0),
        "west": (-2.5, 0.0),
    }

    call_types = ["phee", "alarm", "territorial"]

    for i in range(num_trials):
        # Rotate through conditions
        condition_idx = i % 3
        condition = [Condition.CONGRUENT, Condition.INCONGRUENT, Condition.ACOUSTIC_ONLY][condition_idx]

        # Select visual position
        vis_pos_key = list(positions.keys())[i % 4]
        visual_pos = positions[vis_pos_key]

        # Select acoustic position based on condition
        if condition == Condition.CONGRUENT:
            acoustic_pos = visual_pos
        elif condition == Condition.INCONGRUENT:
            # Opposite direction
            acoustic_pos = (-visual_pos[0], -visual_pos[1])
        else:  # ACOUSTIC_ONLY
            acoustic_pos = visual_pos

        config = TrialConfig(
            condition=condition,
            visual_position=visual_pos,
            acoustic_position=acoustic_pos,
            call_type=call_types[i % len(call_types)],
        )
        configs.append(config)

    # Counterbalance by reversing order for half of subjects
    if counterbalance and np.random.random() < 0.5:
        configs.reverse()

    return configs

def execute_trial(config: TrialConfig, subject_id: str) -> TrialResult:
    """
    Execute a single trial and return results.

    In production, this would:
    1. Activate visual stimulus at config.visual_position
    2. Play acoustic stimulus at config.acoustic_position via VBAP
    3. Record subject response via video/audio
    4. Analyze response offline or in real-time
    """
    # This is a placeholder - actual implementation would interface with
    # the DeepLabCut tracking system and behavioral analysis pipeline

    # Simulated result for demonstration
    if config.condition == Condition.CONGRUENT:
        orientation_error = np.random.normal(0, 15)  # Accurate
        latency = np.random.randint(200, 500)  # Fast
        response_type = "orient"
    elif config.condition == Condition.INCONGRUENT:
        orientation_error = np.random.normal(45, 30)  # Biased toward visual
        latency = np.random.randint(800, 2000)  # Slow
        response_type = "orient" if np.random.random() > 0.3 else "none"
    else:  # ACOUSTIC_ONLY
        orientation_error = np.random.normal(15, 20)  # Moderately accurate
        latency = np.random.randint(400, 800)  # Medium
        response_type = "orient"

    ras = calculate_ras(orientation_error, latency, response_type)

    return TrialResult(
        subject_id=subject_id,
        condition=config.condition,
        orientation_latency_ms=latency,
        orientation_angle_deg=np.degrees(np.arctan2(*config.acoustic_position)),
        orientation_error_deg=orientation_error,
        response_type=response_type,
        movement_distance_m=np.random.uniform(0, 1.0),
        vocalization_type=None,
        ras_score=ras,
    )
```

### Integration with Level 2.5

```python
# realtime/spatial_publisher.py
from realtime.action_publisher import DualStreamAction, SpatialMetadata
import zmq

def publish_spatial_stimulus(
    config: TrialConfig,
    publisher: zmq.Socket,
):
    """Publish spatial stimulus to Rust synthesis engine."""
    spatial_metadata = SpatialMetadata(
        position=(*config.acoustic_position, 1.2),  # x, y, z
        mode="broadcast" if config.call_type == "phee" else "unicast",
        spread_deg=30.0,
    )

    action = DualStreamAction(
        syntactic_token=get_token_for_call_type(config.call_type),
        affect_vector=get_affect_for_call_type(config.call_type),
        spatial_metadata=spatial_metadata,
        temporal_offset_ms=0,
        priority="high",
    )

    publisher.send_json(action.to_dict())
```

---

## 8. Data Recording Format

### Trial Log Entry (JSON)

```json
{
  "trial_id": "S001_T001",
  "timestamp": "2026-05-10T14:30:00.000Z",
  "subject_id": "S001",
  "condition": "A",
  "visual_position": {"x": 2.5, "y": 0.0},
  "acoustic_position": {"x": 2.5, "y": 0.0},
  "call_type": "phee",
  "results": {
    "orientation_latency_ms": 350,
    "orientation_angle_deg": 85.0,
    "orientation_error_deg": 5.0,
    "response_type": "orient",
    "movement_distance_m": 0.5,
    "vocalization_type": null,
    "ras_score": 0.82
  }
}
```

### Video/Audio Recording

- **Filename format:** `{subject_id}_{condition}_{trial_num}_{timestamp}.mp4`
- **Metadata:** Embedded JSON with trial configuration
- **Sync:** PTP timestamp in frame metadata

---

## 9. Ethical Considerations

### Animal Welfare

1. **Sound Exposure Limits:** Max 85 dB SPL at subject position
2. **Session Duration:** Maximum 30 minutes per subject
3. **Rest Periods:** Minimum 5 minutes between trials if signs of stress
4. **Termination Criteria:** Signs of distress, escape behavior, or fatigue

### IACUC Requirements

- Protocol must be approved by institutional committee
- Veterinary oversight required
- Humane endpoints defined
- Personnel trained in species-specific handling

### Data Management

- **Privacy:** Subject IDs anonymized in publications
- **Retention:** Raw video/audio retained for 5 years
- **Sharing:** De-identified data available upon request

---

## 10. Validation Criteria

### Success Metrics

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Condition A RAS | ≥0.7 | Subjects accurately localize congruent stimuli |
| Condition B RAS | ≤0.6 | Subjects show confusion with incongruent stimuli |
| A vs B difference | p<0.05 | Statistically significant effect of spatial congruence |
| Orientation accuracy (A) | ≤30° error | Within expected biological precision |

### Failure Modes

| Issue | Symptom | Remedy |
|-------|---------|--------|
| No condition effect | Similar RAS across all conditions | Check speaker placement, visual cue visibility |
| Ceiling effect | All RAS ≈1.0 | Task too easy, increase difficulty |
| Floor effect | All RAS ≈0.0 | Task too hard, check stimulus audibility |

---

## 11. Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Setup | 2 weeks | Speaker array calibration, visual stimulus construction |
| Pilot | 1 week | Test with 2-3 subjects, refine protocol |
| Data Collection | 4-6 weeks | Run full experiment |
| Analysis | 2 weeks | Process video/audio, statistical tests |
| Write-up | 2 weeks | Manuscript preparation |

---

## 12. References

1. **Spatial Cognition in Animals:** Healy, S. (1998). Spatial Representation in Animals. Oxford University Press.

2. **Cross-Modal Integration:** Rowland, H. M., et al. (2015). "Multimodal communication and camouflage." Proceedings of the Royal Society B.

3. **Marmoset Spatial Behavior:** Miller, C. T., et al. (2015). "Marmoset vocal communication." Current Opinion in Neurobiology.

4. **VBAP for Animal Research:** Pulkki, V. (1997). "Spatial sound generation and perception." Helsinki University of Technology.

---

**Document Version:** 1.0
**Last Updated:** 2026-05-10
**Author:** Sheel Morjaria (sheelmorjaria@gmail.com)
**License:** CC BY-ND 4.0 International
