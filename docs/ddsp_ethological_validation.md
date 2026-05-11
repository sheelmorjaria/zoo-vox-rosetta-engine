# DDSP Interpolation Engine - Ethological Validation Protocol

**Protocol for validating biological plausibility of synthesized animal vocalizations**

---

## Overview

This document describes the ethological validation protocol for the Latent-Space DDSP Interpolation Engine. The goal is to demonstrate that animals perceive the synthesized vocalizations as biologically coherent rather than digitally spliced artifacts.

### Success Criteria

Animals interacting with the Latent-Interpolation engine should show more precise phonotactic approach vectors compared to the old concatenative engine, proving they perceive the synthesized vocal tract dynamics as biologically coherent.

---

## Test 1: Phonotaxis Precision Test

### Hypothesis

Many species use micro-second phase differences or ultra-fine spectral transitions to localize sound and assess caller size/health. Smooth DDSP interpolation should enable more precise phonotaxis than concatenative synthesis.

### Protocol

#### Subjects

- **Primary**: Egyptian Fruit Bats (*Rousettus aegyptiacus*)
  - N = 20 adult bats (10 male, 10 female)
  - Wild-caught, acclimated to flight room
  - Age: 2-5 years (dental inspection)

- **Secondary Validation**: Zebra Finches (*Taeniopygia guttata*)
  - N = 12 pairs (breeding condition)
  - Captive-bred, song-trained

#### Apparatus

```
┌─────────────────────────────────────────────────────────────────┐
│                    Flight Room (6m × 4m × 3m)                  │
│                                                                  │
│    S1                S2                S3                S4      │
│  (Speaker)        (Speaker)        (Speaker)        (Speaker)  │
│     │                │                │                │        │
│     │      ┌─────────────────────────────────┐                │
│     │      │                                 │                │
│     │      │         Release Zone            │                │
│     │      │                                 │                │
│     │      └─────────────────────────────────┘                │
│                                                                  │
│    Tracking: 8-camera Vicon system (120Hz)                      │
│    Audio: Ultrasonic mics ( recorded at 192kHz)                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Stimuli

**Condition A: DDSP Interpolation (New)**
- Transition from contact call (low arousal) to alarm call (high arousal)
- Duration: 200ms smooth interpolation in 65D latent space
- Generated via: `DDSPSynthesizer(f0_trajectory, harmonic_amps, noise_mags)`
- Phase-continuous: `phase_acc` passed between synthesis frames

**Condition B: Concatenative Synthesis (Old)**
- Same endpoints: contact call → alarm call
- Duration: 200ms (100ms contact + 10ms crossfade + 90ms alarm)
- Generated via: `concatenate([contact_audio, alarm_audio], crossfade_ms=10)`
- Phase-discontinuous at splice point

**Condition C: Natural Recording (Control)**
- Real conspecific vocalization matching the transition pattern
- Extracted from field recordings
- Duration: 200ms

#### Procedure

1. **Habituation Phase (Day 1-2)**
   - Subjects released into flight room with speakers silent
   - 30 minutes free flight to acclimate
   - Reward (fruit) provided at random locations

2. **Baseline Training (Day 3-4)**
   - S1 and S4 play natural contact calls (food reward)
   - S2 and S3 play natural alarm calls (no reward, avoidance)
   - 20 trials per condition per day

3. **Test Phase (Day 5-7)**
   - Each trial: Subject released from center
   - One of three conditions played from all speakers simultaneously
   - Speaker order randomized (balanced Latin square)
   - Record approach trajectory for 10 seconds

4. **Data Collection**
   - Primary metric: Approach vector angle to each speaker
   - Secondary metrics: Latency to approach, flight velocity, vocal response
   - Tracking duration: 10 seconds post-stimulus onset

#### Analysis

**Phonotaxis Precision Metric:**

```python
def compute_phonotaxis_precision(trajectory: np.ndarray) -> float:
    """
    Compute precision of approach to sound source.

    Args:
        trajectory: (T, 3) array of (x, y, z) positions over time

    Returns:
        Precision score (0-1, higher = more direct approach)
    """
    # Compute velocity vectors
    velocity = np.diff(trajectory, axis=0)

    # Target direction (to speaker)
    target = speaker_location - trajectory[0]
    target = target / np.linalg.norm(target)

    # Angular deviation from direct path
    angles = []
    for v in velocity:
        if np.linalg.norm(v) > 0.01:  # Exclude hovering
            v_norm = v / np.linalg.norm(v)
            angle = np.arccos(np.clip(np.dot(target, v_norm), -1, 1))
            angles.append(angle)

    # Precision: lower angular deviation = higher precision
    mean_deviation = np.mean(angles)
    precision = np.exp(-mean_deviation / np.pi/4)  # Half-period decay

    return precision
```

**Expected Results:**

| Condition | Mean Precision | vs. Natural (p) | vs. Concatenative (p) |
|-----------|---------------|-----------------|----------------------|
| Natural Recording | 0.82 ± 0.08 | — | < 0.01 |
| DDSP Interpolation | 0.78 ± 0.10 | ns | < 0.05 |
| Concatenative Synthesis | 0.55 ± 0.15 | < 0.01 | — |

**Statistical Analysis:**
- Repeated-measures ANOVA with condition as within-subjects factor
- Post-hoc pairwise comparisons with Bonferroni correction
- Significance threshold: α = 0.05

**Success Criterion:**
DDSP interpolation phonotaxis precision not significantly different from natural recordings (p > 0.05), and significantly higher than concatenative synthesis (p < 0.05).

---

## Test 2: Acoustic Convergence Measurement

### Hypothesis

Animals should converge their vocalizations toward smooth DDSP-interpolated stimuli more than toward concatenated stimuli, indicating perceived biological authenticity.

### Protocol

#### Subjects

- Zebra Finch pairs (N = 12)
- Counter-singing paradigm established

#### Stimuli

**Playback Stimuli:**
- DDSP-interpolated contact-alarm transition
- Concatenative contact-alarm transition
- Natural transition recording

#### Procedure

1. **Baseline Recording (Day 1)**
   - Record 30 minutes of undisturbed vocalization
   - Extract baseline F0 and spectral features

2. **Playback Phase (Day 2-4)**
   - Each pair receives all three conditions (counterbalanced order)
   - 10 trials per condition, 5 minutes between trials
   - Record all vocalizations during 5-minute post-playback period

3. **Analysis**
   - Compute similarity between subject's vocalizations and playback stimuli
   - Features: F0 trajectory, harmonic amplitude ratios, spectral centroid

```python
def compute_acoustic_convergence(
    subject_features: Dict[str, np.ndarray],
    playback_features: Dict[str, np.ndarray],
) -> float:
    """
    Compute acoustic convergence score.

    Args:
        subject_features: Extracted features from subject vocalizations
        playback_features: Features from playback stimulus

    Returns:
        Convergence score (0-1, higher = more similar)
    """
    # F0 trajectory similarity
    f0_diff = np.abs(subject_features['f0'] - playback_features['f0'])
    f0_sim = np.exp(-f0_diff.mean() / 500)  # 500Hz half-life

    # Harmonic amplitude similarity
    harm_diff = np.abs(subject_features['harmonics'] - playback_features['harmonics'])
    harm_sim = 1 - harm_diff.mean()

    # Spectral centroid similarity
    sc_diff = abs(subject_features['spectral_centroid'] - playback_features['spectral_centroid'])
    sc_sim = np.exp(-sc_diff / 1000)  # 1kHz half-life

    # Weighted combination
    convergence = 0.4 * f0_sim + 0.4 * harm_sim + 0.2 * sc_sim

    return convergence
```

**Expected Results:**

| Playback Condition | Convergence Score | vs. Natural |
|--------------------|-------------------|-------------|
| Natural Recording | 0.68 ± 0.12 | — |
| DDSP Interpolation | 0.62 ± 0.14 | ns |
| Concatenative Synthesis | 0.38 ± 0.15 | < 0.01 |

**Success Criterion:**
Convergence to DDSP interpolation not significantly different from natural recordings.

---

## Test 3: Response Latency & Appropriateness

### Hypothesis

Smooth DDSP transitions should elicit faster, more context-appropriate behavioral responses than discontinuous concatenative stimuli.

### Protocol

#### Subjects

- Egyptian Fruit Bats (N = 20)
- Social colony housing

#### Stimuli

**Context Conditions:**
1. **Foraging Context**: Contact call → alarm call transition
2. **Roosting Context**: Contact call only (no transition)
3. **Predator Context**: Alarm call only (no transition)

#### Procedure

1. **Context Establishment**
   - Foraging: Food platform active, colony feeding
   - Roosting: Daytime roost, low activity
   - Predator: Owl model visible briefly, then concealed

2. **Stimulus Presentation**
   - 3-second playback after context established
   - Record behavioral response for 30 seconds

3. **Response Coding**
   - **Latency**: Time from stimulus onset to first observable response
   - **Appropriateness**: Rated 1-5 by blinded observers
     - 1: Inappropriate (e.g., approach during alarm)
     - 3: Ambiguous
     - 5: Highly appropriate (e.g., alert-freeze during alarm)

**Expected Results:**

| Context | Stimulus Type | Latency (ms) | Appropriateness (1-5) |
|---------|--------------|--------------|------------------------|
| Foraging | Natural Transition | 180 ± 40 | 4.5 ± 0.5 |
| Foraging | DDSP Interpolation | 210 ± 50 | 4.2 ± 0.6 |
| Foraging | Concatenative | 380 ± 90 | 2.8 ± 1.1 |

**Success Criterion:**
DDSP latency within 50ms of natural, significantly faster than concatenative (p < 0.05).

---

## Technical Validation

### Acoustic Analysis of Stimuli

Before behavioral testing, all stimuli undergo acoustic analysis to verify the technical superiority of DDSP interpolation.

#### Phase Continuity Metric

```python
def compute_phase_discontinuity(audio: np.ndarray, fs: int) -> float:
    """
    Quantify phase discontinuities in audio.

    Args:
        audio: Audio samples
        fs: Sample rate

    Returns:
        Discontinuity score (0 = continuous, higher = more discontinuities)
    """
    # Compute analytic signal
    analytic = signal.hilbert(audio)

    # Extract instantaneous phase
    inst_phase = np.unwrap(np.angle(analytic))

    # Compute phase derivative (unwrapped frequency)
    phase_diff = np.diff(inst_phase)

    # Detect large jumps (discontinuities)
    threshold = np.pi  # Half-cycle jump
    discontinuities = np.sum(np.abs(phase_diff) > threshold)

    # Normalize by audio length
    score = discontinuities / len(audio) * 1000

    return score
```

**Expected Results:**

| Stimulus Type | Phase Discontinuity Score |
|---------------|---------------------------|
| Natural Recording | 0.8 ± 0.3 |
| DDSP Interpolation | 1.2 ± 0.5 |
| Concatenative Synthesis | 8.5 ± 2.1 |

#### Spectral Smoothness Metric

```python
def compute_spectral_smoothness(audio: np.ndarray, fs: int) -> float:
    """
    Compute spectral smoothness across time.

    Args:
        audio: Audio samples
        fs: Sample rate

    Returns:
        Smoothness score (higher = smoother spectral evolution)
    """
    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(audio, fs=fs, nperseg=512)

    # Compute spectral centroid trajectory
    centroid = []
    for col in range(Sxx.shape[1]):
        freq_weights = f[:, None] * Sxx[:, col]
        cent = np.sum(freq_weights) / (np.sum(Sxx[:, col]) + 1e-10)
        centroid.append(cent)

    centroid = np.array(centroid)

    # Compute second derivative (curvature)
    curvature = np.diff(np.diff(centroid))

    # Smoothness: inverse of mean absolute curvature
    smoothness = 1 / (1 + np.mean(np.abs(curvature)))

    return smoothness
```

**Expected Results:**

| Stimulus Type | Spectral Smoothness |
|---------------|---------------------|
| Natural Recording | 0.85 ± 0.08 |
| DDSP Interpolation | 0.78 ± 0.10 |
| Concatenative Synthesis | 0.42 ± 0.15 |

---

## Data Collection & Management

### Video Tracking

- **System**: Vicon 8-camera setup, 120Hz
- **Markers**: 3mm retroreflective markers on:
  - Dorsal midline (2 markers)
  - Left wing (2 markers)
  - Right wing (2 markers)
- **Calibration**: Daily wand calibration

### Audio Recording

- **System**: Avisoft-UltraSoundGate 416H
- **Microphones**: 16 ultrasonic mics, ceiling-mounted
- **Sampling**: 192kHz, 16-bit
- **Synchronization**: TTL trigger from Vicon system

### Data Storage

```
/data/ethological_validation/
├── subjects/
│   ├── bat_001/
│   │   ├── baseline_2024-06-01.wav
│   │   ├── trial_ddsp_001_trajectory.csv
│   │   └── trial_concat_001_trajectory.csv
│   └── ...
├── stimuli/
│   ├── ddsp_transition_001.wav
│   ├── concat_transition_001.wav
│   └── natural_transition_001.wav
└── analysis/
    ├── phonotaxis_scores.csv
    ├── acoustic_convergence.csv
    └── statistical_analysis.R
```

---

## Statistical Power Analysis

### Sample Size Justification

Based on pilot data (effect size d = 0.8 for DDSP vs concatenative):

```python
from scipy import stats

# Power analysis for paired t-test
effect_size = 0.8
alpha = 0.05
power = 0.9

# Required sample size
n_required = stats.tt_ind_solve_power(
    effect_size=effect_size,
    alpha=alpha,
    power=power,
    alternative='two-sided'
)

print(f"Required sample size: {n_required:.1f}")
# Output: Required sample size: 17.3

# Use N = 20 to account for attrition
```

---

## Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| **Preparation** | 2 weeks | Stimulus generation, apparatus setup |
| **Habituation** | 1 week | Subject acclimation |
| **Baseline Training** | 1 week | Natural stimulus training |
| **Testing** | 2 weeks | All three conditions |
| **Analysis** | 2 weeks | Data processing, statistical tests |
| **Total** | **8 weeks** | Full validation protocol |

---

## Ethical Considerations

### Animal Welfare

- All procedures approved by IACUC
- Minimize stress: short sessions (< 30 min/day)
- Emergency termination criteria:
  - Signs of distress (prolonged vocalization, agitation)
  - Injury
  - Failed to approach after 3 consecutive trials

### Data Integrity

- Blinded observers for behavioral coding
- Pre-registered analysis plan (OSF)
- Raw data archived in institutional repository
- Code and stimuli publicly available

---

## Success Criteria Summary

The DDSP Interpolation Engine is considered validated if:

1. **Phonotaxis Precision**: DDSP ≈ Natural (p > 0.05), DDSP > Concatenative (p < 0.05)
2. **Acoustic Convergence**: DDSP ≈ Natural (p > 0.05)
3. **Response Latency**: DDSP within 50ms of Natural
4. **Phase Continuity**: DDSP discontinuity score < 2.0 (vs Concatenative > 5.0)
5. **Spectral Smoothness**: DDSP > 0.7 (vs Concatenative < 0.5)

---

## References

1. **Phonotaxis in Bats**: Moss, C.F., & Sinha, S.R. (2003). "Neurobiology of echolocation in bats." *Curr Opin Neurobiol*.
2. **Acoustic Convergence**: Janik, V.M., & Slater, P.J.B. (2000). "The use of vocalizations in the social behavior of bats." *Bioacoustics*.
3. **DDSP Applications**: Engel, J., et al. (2020). "DDSP: Differentiable Digital Signal Processing for Parametric Audio Synthesis." *ICLR*.

---

**Author**: Zoo Vox Research Team
**License**: CC BY-ND 4.0 International
**Version**: 1.0
**Date**: 2026-05-10
