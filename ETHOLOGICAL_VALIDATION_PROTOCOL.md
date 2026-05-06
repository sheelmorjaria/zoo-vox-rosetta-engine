# Ethological Validation Protocol v1.0.0

**Field Testing Framework for Bioacoustic Closed-Loop Interaction**

This document defines the protocol for validating that the Zoo Vox Rosetta Engine achieves functional two-way communication with animal species. The protocol measures whether animals accept synthesized responses as biologically valid conspecific vocalizations.

---

## Table of Contents

1. [Overview](#overview)
2. [Scientific Hypothesis](#scientific-hypothesis)
3. [The Response Appropriateness Score (RAS)](#the-response-appropriateness-score-ras)
4. [Experimental Design](#experimental-design)
5. [Safety Protocols](#safety-protocols)
6. [Data Collection](#data-collection)
7. [Analysis Methods](#analysis-methods)
8. [Success Criteria](#success-criteria)

---

## Overview

### The Turing Test for Animal Communication

The fundamental question: **Does the species accept our system as a conspecific?**

Unlike human AI Turing tests where deception is acceptable, animal communication requires **biological validity**. We are not trying to "fool" the animals—we are trying to participate in their communication system according to their rules.

### Key Principle: Syntactic Continuity

The system must only emit responses that are:
1. **Syntactically valid**: Following the 50 valid bigrams
2. **Contextually appropriate**: Matching the behavioral context
3. **Acoustically authentic**: Using exemplar-based synthesis (not generation)

---

## Scientific Hypothesis

### Primary Hypothesis (H1)

**H1**: Bats will respond to system-generated vocalizations at rates statistically indistinguishable from conspecific responses, when the system uses valid bigram syntax and authentic acoustic exemplars.

### Alternative Hypothesis (H0)

**H0**: Bats will show differential response rates to system-generated vs. conspecific vocalizations, indicating detection of artificiality.

### Predicted Outcomes

| System Configuration | Expected RAS | Interpretation |
|---------------------|--------------|----------------|
| Full system (valid bigrams + BGMM synthesis) | R > 0.7 | Functional acceptance |
| Invalid bigrams (random transitions) | R < 0.3 | Syntactic rejection |
| Synthetic tones (no exemplars) | R < 0.2 | Acoustic rejection |
| Silent control (no response) | R ≈ 0.0 | Baseline |

---

## The Response Appropriateness Score (RAS)

### Definition

The **Response Appropriateness Score (R)** measures whether the animal continues the syntactic chain after a system response:

```
R = (Number of valid follow-up responses) / (Total system responses)
```

### Calculation

```python
def calculate_ras(interaction_sequence: List[Interaction]) -> float:
    """
    Calculate Response Appropriateness Score.

    An interaction is scored as positive if:
    1. System emits valid bigram response (e.g., 8→12)
    2. Animal responds within timeout window (e.g., 2 seconds)
    3. Animal's response forms valid bigram with system's cluster

    R = (Positive responses) / (Total system responses)
    """
    positive_responses = 0
    total_system_responses = 0

    for i, interaction in enumerate(interaction_sequence):
        if interaction.source == "system":
            total_system_responses += 1

            # Check if animal responded with valid bigram
            if i + 1 < len(interaction_sequence):
                next_interaction = interaction_sequence[i + 1]
                if (next_interaction.source == "animal" and
                    is_valid_bigram(interaction.cluster_id, next_interaction.cluster_id)):
                    positive_responses += 1

    return positive_responses / max(total_system_responses, 1)
```

### RAS Interpretation

| R Score | Interpretation |
|---------|----------------|
| **R ≥ 0.7** | **Functional acceptance** - System participates as conspecific |
| 0.5 ≤ R < 0.7 | Partial acceptance - Some responses accepted |
| 0.3 ≤ R < 0.5 | Ambiguous - Borderline acceptance |
| **R < 0.3** | **Rejection** - System detected as artificial |

---

## Experimental Design

### Setup

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         FIELD DEPLOYMENT SETUP                                │
│                                                                                  │
│   ┌─────────────┐                                                              │
│   │  Bat Colony │ ──(ambient vocalizations)──►                                │
│   └─────────────┘                                                              │
│          │                                                                     │
│          ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    OBSERVATION ZONE                                       │   │
│   │  ┌──────────────┐           ┌──────────────┐                            │   │
│   │  │ Microphone   │           │  Speaker     │                            │   │
│   │  │ Array        │◄─────────┤  (Directional)│                            │   │
│   │  └──────────────┘           └──────────────┘                            │   │
│   │         │                           │                                   │   │
│   │         ▼                           ▼                                   │   │
│   │  ┌──────────────────────────────────────────────────────────────────┐  │   │
│   │  │                Rust + Python System                             │  │   │
│   │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐                  │  │   │
│   │  │  │ NBD +      │  │ 45-State   │  │ Response   │                  │  │   │
│   │  │  │ Student    │→│ Automaton  │→│ Selection  │→ Speaker         │  │   │
│   │  │  └────────────┘  └────────────┘  └────────────┘                  │  │   │
│   │  └──────────────────────────────────────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    DATA LOGGING                                          │   │
│   │  - All vocalizations timestamped                                        │   │
│   │  - Cluster assignments logged                                            │   │
│   │  - Bigram sequences recorded                                             │   │
│   │  - RAS calculated in real-time                                           │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Conditions

1. **Baseline (Control)**: No system response, passive recording
2. **Conspecific Playback**: Pre-recorded bat vocalizations (positive control)
3. **Full System**: Active interaction with valid bigrams
4. **Invalid Syntax**: Random bigram transitions (negative control)
5. **Synthetic Tones**: Pure tone sequences (acoustic control)

### Timeline

| Phase | Duration | Purpose |
|-------|----------|---------|
| **Acclimation** | 3 days | Colony habituates to equipment |
| **Baseline** | 2 days | Record natural interaction rates |
| **Testing** | 7 days | Rotate through conditions |
| **Analysis** | Ongoing | Calculate RAS daily |

---

## Safety Protocols

### Animal Welfare

1. **Volume Limits**: Maximum SPL 85 dB at 1m (below hearing damage threshold)
2. **Session Limits**: Maximum 2 hours active interaction per day
3. **Distress Monitoring**: Abort if alarm vocalizations increase > 300%
4. **Ethics Approval**: IACUC approval required before deployment

### System Safety

1. **Emergency Mute**: Hardware mute button always accessible
2. **Watchdog Timer**: Auto-mute if no heartbeat for 100ms
3. **Fail-Safe**: Power loss = immediate mute
4. **Environmental Monitor**: Force passthrough in adverse conditions

---

## Data Collection

### Required Metrics

```python
@dataclass
class InteractionEvent:
    """Single interaction event for analysis."""
    timestamp: float
    source: str  # "animal" or "system"
    cluster_id: int
    emitter_id: Optional[int]
    response_to: Optional[int]  # If this was a response to previous event
    time_since_previous: float

@dataclass
class SessionMetrics:
    """Metrics for a single session."""
    duration_seconds: float
    total_animal_vocalizations: int
    total_system_responses: int
    positive_responses: int  # Animal followed system with valid bigram
    negative_responses: int  # No response or invalid bigram
    ras_score: float
    condition: str  # "baseline", "conspecific", "full_system", etc.
```

### Logging Format

```json
{
  "session_id": "bat_colony_2025-05-06_001",
  "condition": "full_system",
  "start_time": "2025-05-06T20:00:00Z",
  "end_time": "2025-05-06T22:00:00Z",
  "interactions": [
    {
      "timestamp": 1715011200.123,
      "source": "animal",
      "cluster_id": 8,
      "emitter_id": 3,
      "response_to": null
    },
    {
      "timestamp": 1715011200.450,
      "source": "system",
      "cluster_id": 12,
      "response_to": 8,
      "bigram_valid": true,
      "bigram_probability": 0.52
    },
    {
      "timestamp": 1715011201.234,
      "source": "animal",
      "cluster_id": 8,
      "emitter_id": 1,
      "response_to": 12,
      "forms_valid_bigram": true
    }
  ],
  "metrics": {
    "ras": 0.85,
    "total_system_responses": 20,
    "positive_responses": 17,
    "negative_responses": 3
  }
}
```

---

## Analysis Methods

### Statistical Tests

1. **RAS Comparison**: ANOVA across conditions
2. **Response Latency**: Compare animal response times
3. **Vocalization Rate**: Changes in overall activity
4. **Sequence Length**: Mean bigram chain length

### Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| **RAS (Full System)** | R ≥ 0.7 |
| **RAS vs Baseline** | p < 0.05 (significant increase) |
| **RAS vs Conspecific** | p > 0.05 (no significant difference) |
| **RAS (Invalid Syntax)** | R < 0.3 (validation of control) |
| **No distress** | Alarm rate < 150% of baseline |

---

## Implementation Status

### v1.5.0 Components

| Component | Status | File |
|-----------|--------|------|
| InteractionEvent Dataclass | ✅ Implemented | `realtime/interaction_agent.py` |
| SessionMetrics Dataclass | ✅ Implemented | `realtime/interaction_agent.py` |
| RAS Metric (calculate_ras) | ✅ Implemented | `realtime/interaction_agent.py` |
| Session Logging | ✅ Implemented | `realtime/interaction_agent.py` |
| Ethological Mode Config | ✅ Implemented | `InteractionAgentConfig` |
| Field Deployment Tests | ✅ Implemented | `tests/test_interaction_agent_v1_5_0.py` |

### Test Results

```
tests/test_interaction_agent_v1_5_0.py::TestInteractionEvent::test_interaction_event_creation PASSED
tests/test_interaction_agent_v1_5_0.py::TestInteractionEvent::test_system_event_has_no_emitter PASSED
tests/test_interaction_agent_v1_5_0.py::TestSessionMetrics::test_session_metrics_creation PASSED
tests/test_interaction_agent_v1_5_0.py::TestSessionMetrics::test_session_metrics_to_dict PASSED
tests/test_interaction_agent_v1_5_0.py::TestRASCalculation::test_perfect_ras_score PASSED
tests/test_interaction_agent_v1_5_0.py::TestRASCalculation::test_zero_ras_score PASSED
tests/test_interaction_agent_v1_5_0.py::TestRASCalculation::test_partial_ras_score PASSED
tests/test_interaction_agent_v1_5_0.py::TestRASCalculation::test_invalid_bigram_counts_as_negative PASSED
tests/test_interaction_agent_v1_5_0.py::TestRASCalculation::test_ras_with_no_valid_bigrams PASSED
tests/test_interaction_agent_v1_5_0.py::TestRASCalculation::test_ras_with_empty_sequence PASSED
tests/test_interaction_agent_v1_5_0.py::TestAgentEthologicalMode::test_agent_initializes_session_metrics PASSED
tests/test_interaction_agent_v1_5_0.py::TestAgentEthologicalMode::test_agent_generates_session_id_if_not_provided PASSED
tests/test_interaction_agent_v1_5_0.py::TestAgentEthologicalMode::test_agent_tracks_animal_events PASSED
tests/test_interaction_agent_v1_5_0.py::TestAgentEthologicalMode::test_agent_tracks_system_responses PASSED
tests/test_interaction_agent_v1_5_0.py::TestRASIntegration::test_calculate_current_ras PASSED
tests/test_interaction_agent_v1_5_0.py::TestRASIntegration::test_get_session_metrics_returns_current_state PASSED
tests/test_interaction_agent_v1_5_0.py::TestRASIntegration::test_get_stats_includes_ethological_validation PASSED
tests/test_interaction_agent_v1_5_0.py::TestRASIntegration::test_interaction_history_bounded_size PASSED
tests/test_interaction_agent_v1_5_0.py::TestRASIntegration::test_ethological_mode_disabled_skips_tracking PASSED
tests/test_interaction_agent_v1_5_0.py::TestExperimentalConditions::test_baseline_condition PASSED
tests/test_interaction_agent_v1_5_0.py::TestExperimentalConditions::test_conspecific_condition PASSED
tests/test_interaction_agent_v1_5_0.py::TestExperimentalConditions::test_full_system_condition PASSED

============================= 22 passed in 5.08s ==============================
```

### Total Test Coverage

| Version | Tests | Description |
|---------|-------|-------------|
| v1.2.0 | 24 | Cluster-based semantic grounding |
| v1.3.0 | 16 | Level 2 speaker grounding |
| v1.4.0 | 15 | Probabilistic transition weights |
| v1.5.0 | 22 | Ethological validation protocol |
| **Total** | **104** | **Full pipeline validation** |

---

**Author**: Sheel Morjaria (sheelmorjaria@gmail.com)
**Date**: 2026-05-06
**Version**: 1.0.0
