# Closed-Loop Interaction Agent: Methodology Protocol

**Version:** 1.0.0
**Last Updated:** March 7, 2026
**Author:** Sheel Morjaria
**License:** CC BY-ND 4.0 International

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Communication Protocol](#3-communication-protocol)
4. [Feature Event Specification](#4-feature-event-specification)
5. [Synthesis Action Specification](#5-synthesis-action-specification)
6. [Interaction Agent Logic](#6-interaction-agent-logic)
7. [Context Inference Engine](#7-context-inference-engine)
8. [Response Generation Pipeline](#8-response-generation-pipeline)
9. [Safety and Rate Limiting](#9-safety-and-rate-limiting)
10. [State Machine Specification](#10-state-machine-specification)
11. [Integration Testing Protocol](#11-integration-testing-protocol)
12. [Deployment Configuration](#12-deployment-configuration)
13. [Performance Metrics](#13-performance-metrics)
14. [Troubleshooting Guide](#14-troubleshooting-guide)

---

## 1. Executive Summary

### 1.1 Purpose

The Closed-Loop Interaction Agent enables real-time bidirectional communication between the Rust Execution Layer (fast, safety-critical audio processing) and the Python Logic Layer (cognitive intelligence, decision making). This allows the system to:

1. **Perceive**: Receive 112D feature vectors from Rust's Neural Boundary Detection (NBD) and feature extraction pipeline
2. **Understand**: Infer behavioral context (alarm, contact, social, territorial) from acoustic features
3. **Decide**: Determine whether and how to respond based on context, confidence, and rate limits
4. **Act**: Send synthesis timelines and micro-dynamics deltas back to Rust for audio generation

### 1.2 Design Philosophy

| Principle | Implementation |
|-----------|---------------|
| **Fail-Safe** | If Python crashes, Rust continues in Passthrough Mode |
| **Zero-Copy** | Use numpy arrays for 112D features, avoid serialization overhead |
| **Non-Blocking** | All ZeroMQ operations use `DONTWAIT` flag |
| **Rate-Limited** | Response cooldown prevents feedback loops |
| **Confidence-Gated** | Only respond when context detection confidence exceeds threshold |

### 1.3 System Requirements

| Component | Requirement |
|-----------|-------------|
| Rust | 1.70+ with `zmq` crate |
| Python | 3.10+ with `pyzmq`, `numpy` |
| ZeroMQ | 4.3+ |
| IPC | Unix domain sockets (`/tmp/*.ipc`) |

---

## 2. System Architecture

### 2.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLOSED-LOOP AGENT SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     RUST EXECUTION LAYER                              │   │
│  │                                                                       │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐   │   │
│  │  │  Audio Input    │───►│       NBD       │───►│   112D Feature  │   │   │
│  │  │  (48kHz, mono)  │    │  (Boundaries)   │    │   Extraction    │   │   │
│  │  └─────────────────┘    └─────────────────┘    └────────┬────────┘   │   │
│  │                                                          │            │   │
│  │  ┌─────────────────┐                            ┌───────▼────────┐   │   │
│  │  │   Synthesis     │◄───────────────────────────│   FeatureEvent │   │   │
│  │  │   Pipeline      │                            │   Publisher    │   │   │
│  │  └────────┬────────┘                            └────────────────┘   │   │
│  │           │                                                     │   │   │
│  │  ┌────────▼────────┐                            ┌────────────────┐   │   │
│  │  │  Audio Output   │                            │   ActionSub-   │   │   │
│  │  │  (Speaker/DAC)  │                            │   scriber      │   │   │
│  │  └─────────────────┘                            └───────▲────────┘   │   │
│  │                                                         │            │   │
│  └─────────────────────────────────────────────────────────│────────────┘   │
│                                                            │                │
│                              ┌─────────────────────────────▼──────────────┐ │
│                              │         ZeroMQ IPC Transport               │ │
│                              │  ipc:///tmp/cognitive_features.ipc (PUB)   │ │
│                              │  ipc:///tmp/cognitive_actions.ipc (SUB)    │ │
│                              └─────────────────────────────┬──────────────┘ │
│                                                          │                 │
│  ┌───────────────────────────────────────────────────────▼──────────────┐  │
│  │                     PYTHON LOGIC LAYER                                │  │
│  │                                                                       │  │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐   │  │
│  │  │   FeatureSub-   │───►│  Interaction    │───►│   Action        │   │  │
│  │  │   scriber       │    │  Agent          │    │   Publisher     │   │  │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘   │  │
│  │                                │                                      │  │
│  │                                ▼                                      │  │
│  │                    ┌─────────────────────┐                           │  │
│  │                    │  Context Inference  │                           │  │
│  │                    │  (112D → Context)   │                           │  │
│  │                    └─────────────────────┘                           │  │
│  │                                │                                      │  │
│  │                    ┌───────────▼───────────┐                         │  │
│  │                    │  Response Generator   │                         │  │
│  │                    │  (Timeline + Deltas)  │                         │  │
│  │                    └───────────────────────┘                         │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | Language | Responsibility |
|-----------|----------|---------------|
| `FeatureEventPublisher` | Rust | Publishes 112D features + cluster ID to Python |
| `FeatureSubscriber` | Python | Receives feature events, dispatches to agent |
| `InteractionAgent` | Python | Orchestrates cognitive processing and response |
| `ActionPublisher` | Python | Publishes synthesis timelines to Rust |
| `ActionSubscriber` | Rust | Receives synthesis actions, executes audio output |
| `PeerController` | Rust | Monitors Python heartbeat, manages mode switching |

### 2.3 Operation Modes

| Mode | Condition | Behavior |
|------|-----------|----------|
| **Passthrough** | Python disconnected or crashed | Audio muted, recording only, no synthesis |
| **Interactive** | Python connected, heartbeats active | Full cognitive processing, synthesis enabled |

---

## 3. Communication Protocol

### 3.1 ZeroMQ Socket Configuration

| Socket | Type | Bind/Connect | Endpoint |
|--------|------|--------------|----------|
| Feature Publisher | PUB | BIND | `ipc:///tmp/cognitive_features.ipc` |
| Feature Subscriber | SUB | CONNECT | Same as above |
| Action Publisher | PUB | CONNECT | `ipc:///tmp/cognitive_actions.ipc` |
| Action Subscriber | SUB | BIND | Same as above |
| Heartbeat (existing) | PUB/SUB | - | `ipc:///tmp/cognitive_heartbeat.ipc` |

### 3.2 Socket Options

```rust
// Rust Publisher Configuration
socket.set_sndhwm(100)?;           // High water mark: 100 messages
socket.set_linger(1000)?;          // 1s linger on close
socket.set_sndtimeo(0)?;           // Non-blocking send

// Rust Subscriber Configuration
socket.set_rcvhwm(100)?;           // High water mark: 100 messages
socket.set_rcvtimeo(100)?;         // 100ms receive timeout
socket.set_subscribe(b"")?;        // Subscribe to all messages
```

```python
# Python Publisher Configuration
socket.setsockopt(zmq.SNDHWM, 10)
socket.setsockopt(zmq.LINGER, 1000)

# Python Subscriber Configuration
socket.setsockopt(zmq.RCVHWM, 100)
socket.setsockopt(zmq.RCVTIMEO, 100)
socket.setsockopt(zmq.SUBSCRIBE, b"")
```

### 3.3 Message Flow Diagram

```
Rust                           Python
 │                                │
 │  1. Audio segment detected     │
 │     (NBD boundary)             │
 │         │                      │
 │         ▼                      │
 │  2. Extract 112D features      │
 │         │                      │
 │         ▼                      │
 │  3. Create FeatureEvent        │
 │         │                      │
 │         │  ────── PUB ──────►  │  4. FeatureSubscriber receives
 │         │                      │         │
 │         │                      │         ▼
 │         │                      │  5. InteractionAgent processes
 │         │                      │     - Infer context
 │         │                      │     - Check rate limits
 │         │                      │     - Generate timeline
 │         │                      │         │
 │         │                      │         ▼
 │         │  ◄────── PUB ──────  │  6. ActionPublisher sends
 │         │                      │
 │  7. ActionSubscriber receives  │
 │         │                      │
 │         ▼                      │
 │  8. Execute synthesis          │
 │     (Stage 4 & 5)              │
 │         │                      │
 │         ▼                      │
 │  9. Audio output               │
 │                                │
```

---

## 4. Feature Event Specification

### 4.1 Data Structure

```rust
/// Feature extraction event from Rust to Python
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEvent {
    /// Event type identifier (always "feature_extraction")
    pub event_type: String,

    /// Cluster ID from corpus analysis (k=1020)
    pub cluster_id: u32,

    /// 112D feature vector (RosettaFeatures)
    /// - Layer 1: Base Physics (46D) - indices 0-45
    /// - Layer 2: Macro Texture (30D) - indices 46-75
    /// - Layer 3: Micro Texture (36D) - indices 76-111
    pub features_112d: Vec<f32>,

    /// Unix timestamp in seconds (with sub-second precision)
    pub timestamp: f64,

    /// Sequence number for ordering and gap detection
    pub sequence: u64,
}
```

### 4.2 JSON Serialization Example

```json
{
    "event_type": "feature_extraction",
    "cluster_id": 42,
    "features_112d": [
        5500.0,  // Index 0: mean_f0_hz
        0.45,    // Index 1: rms_energy
        800.0,   // Index 2: f0_range_hz
        0.12,    // Index 3: harmonic_to_noise_ratio
        150.0,   // Index 4: duration_ms
        // ... 107 more values ...
    ],
    "timestamp": 1672531200.123456,
    "sequence": 12345
}
```

### 4.3 112D Feature Index Mapping

| Index Range | Layer | Description |
|-------------|-------|-------------|
| 0-11 | Layer 1: Prosody | F0, duration, amplitude features |
| 12-23 | Layer 1: Spectral Shape | Spectral centroid, bandwidth, rolloff |
| 24-35 | Layer 1: Voice Quality | Jitter, shimmer, HNR |
| 36-45 | Layer 1: Temporal | Attack, decay, sustain, release |
| 46-57 | Layer 2: Frequency Modulation | F0 trajectory statistics |
| 58-67 | Layer 2: Amplitude Modulation | Energy trajectory statistics |
| 68-75 | Layer 2: Spectral Dynamics | Spectral flux, variation |
| 76-95 | Layer 3: Micro-Timing | Note onset precision, timing jitter |
| 96-111 | Layer 3: Micro-Harmonics | Fine spectral detail |

### 4.4 Key Feature Indices for Context Inference

```python
# Primary context indicators
F0_MEAN_IDX = 0          # Mean fundamental frequency
RMS_ENERGY_IDX = 1       # Overall energy
F0_RANGE_IDX = 2         # Pitch variation
HNR_IDX = 3              # Harmonic-to-noise ratio
DURATION_IDX = 4         # Note duration
SPECTRAL_CENTROID_IDX = 12  # Brightness

# Secondary context indicators
F0_SLOPE_IDX = 46        # Frequency trajectory slope
F0_CURVATURE_IDX = 47    # Frequency trajectory curvature
RMS_VARIANCE_IDX = 58    # Energy variation
```

---

## 5. Synthesis Action Specification

### 5.1 Data Structure

```rust
/// Synthesis action from Python to Rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisAction {
    /// Action type (e.g., "synthesize_timeline")
    pub action_type: String,

    /// Timeline of synthesis events
    pub timeline: Vec<TimelineEvent>,

    /// Optional micro-dynamics deltas for modification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deltas: Option<MicroDynamicsDelta>,

    /// Action priority
    #[serde(default)]
    pub priority: ActionPriority,
}
```

### 5.2 Timeline Event Structure

```rust
/// Single event in synthesis timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    /// Cluster ID for synthesis (maps to exemplar audio)
    pub cluster_id: u32,

    /// Start time relative to timeline start (milliseconds)
    pub start_time_ms: f64,

    /// Duration of this event (milliseconds)
    pub duration_ms: f64,

    /// Amplitude (0.0 to 1.0)
    #[serde(default = "default_amplitude")]
    pub amplitude: f32,
}

fn default_amplitude() -> f32 { 1.0 }
```

### 5.3 Micro-Dynamics Delta Structure

```rust
/// Delta transformations for synthesis modification
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MicroDynamicsDelta {
    /// Change to mean F0 (Hz)
    #[serde(skip_serializing_if = "is_zero")]
    pub delta_mean_f0_hz: f32,

    /// Change to duration (ms)
    #[serde(skip_serializing_if = "is_zero")]
    pub delta_duration_ms: f32,

    /// Change to F0 range (Hz)
    #[serde(skip_serializing_if = "is_zero")]
    pub delta_f0_range_hz: f32,

    /// Change to harmonic-to-noise ratio
    #[serde(skip_serializing_if = "is_zero")]
    pub delta_harmonic_to_noise_ratio: f32,

    /// Change to attack time (ms)
    #[serde(skip_serializing_if = "is_zero")]
    pub delta_attack_time_ms: f32,

    /// Change to sustain level
    #[serde(skip_serializing_if = "is_zero")]
    pub delta_sustain_level: f32,

    /// Change to RMS energy
    #[serde(skip_serializing_if = "is_zero")]
    pub delta_rms_energy: f32,
}
```

### 5.4 JSON Serialization Example

```json
{
    "action_type": "synthesize_timeline",
    "timeline": [
        {
            "cluster_id": 42,
            "start_time_ms": 0.0,
            "duration_ms": 150.0,
            "amplitude": 0.9
        },
        {
            "cluster_id": 99,
            "start_time_ms": 160.0,
            "duration_ms": 200.0,
            "amplitude": 0.75
        }
    ],
    "deltas": {
        "delta_mean_f0_hz": 200.0,
        "delta_duration_ms": 20.0,
        "delta_rms_energy": 0.1
    },
    "priority": "high"
}
```

### 5.5 Priority Levels

| Priority | Description | Use Case |
|----------|-------------|----------|
| `low` | Can be delayed or dropped | Background responses |
| `normal` | Standard processing | Default contact calls |
| `high` | Prioritized processing | Territorial responses |
| `critical` | Must execute immediately | Alarm responses |

---

## 6. Interaction Agent Logic

### 6.1 Processing Pipeline

```python
def _handle_feature_event(self, event: FeatureEvent) -> None:
    """
    Main event handler - processes incoming features and generates responses.

    Pipeline:
    1. Validate event structure
    2. Extract key features from 112D vector
    3. Infer behavioral context
    4. Calculate confidence
    5. Check rate limits and thresholds
    6. Generate synthesis timeline (if appropriate)
    7. Apply micro-dynamics deltas
    8. Publish action to Rust
    """
```

### 6.2 Context States

```python
class ContextState(Enum):
    """Behavioral context types"""

    ALARM = "alarm"              # Urgent threat response
    TERRITORIAL = "territorial"   # Boundary defense
    CONTACT = "contact"           # Affiliative contact
    SOCIAL = "social"             # Social interaction
    UNKNOWN = "unknown"           # Unrecognized context
```

### 6.3 Context-to-Response Mapping

| Context | Timeline Duration | Amplitude | Delta F0 | Delta Energy |
|---------|------------------|-----------|----------|--------------|
| alarm | 100ms | 0.9 | +500 Hz | +0.2 |
| territorial | 200ms | 0.85 | +200 Hz | +0.1 |
| contact | 150ms | 0.75 | 0 Hz | 0 |
| social | 250ms | 0.7 | -100 Hz | -0.1 |

### 6.4 Response Decision Logic

```python
def _should_respond(self, result: Dict[str, Any]) -> bool:
    """
    Determine if the agent should generate a response.

    Returns True if ALL conditions are met:
    1. Context is recognized (not UNKNOWN)
    2. Confidence exceeds threshold (default: 0.5)
    3. Rate limit cooldown has passed
    4. Context requires a response (alarm, territorial, contact)
    """

    # Check confidence threshold
    if result.get("confidence", 0.0) < 0.5:
        return False

    # Check rate limiting
    time_since_last = time.time() - self._last_response_time
    if time_since_last < self.config.response_cooldown_ms / 1000.0:
        return False

    # Check context type
    context = result.get("context_state", "unknown")
    response_contexts = {"alarm", "territorial", "contact"}
    if context not in response_contexts:
        return False

    return True
```

---

## 7. Context Inference Engine

### 7.1 Feature Extraction from 112D

```python
def _extract_key_features(self, features_112d: np.ndarray) -> Dict[str, float]:
    """Extract key indicators from 112D vector"""

    return {
        "f0_mean": float(features_112d[0]),
        "rms_energy": float(features_112d[1]),
        "f0_range": float(features_112d[2]),
        "hnr": float(features_112d[3]),
        "duration_ms": float(features_112d[4]),
        "spectral_centroid": float(features_112d[12]),
        "f0_slope": float(features_112d[46]),
        "f0_curvature": float(features_112d[47]),
        "rms_variance": float(features_112d[58]),
    }
```

### 7.2 Context Inference Rules

```python
def _infer_context(self, features: np.ndarray) -> str:
    """
    Infer behavioral context from 112D features.

    Decision tree based on acoustic characteristics:
    """

    f0 = features[0]
    rms = features[1]
    f0_range = features[2]
    hnr = features[3]

    # High F0 + High Energy = Alarm (urgent threat)
    if f0 > 8000 and rms > 0.6:
        return "alarm"

    # High F0 Range = Territorial (assertive display)
    if f0_range > 2000:
        return "territorial"

    # Moderate F0 + Moderate Energy = Contact (greeting)
    if 4000 < f0 < 7000 and 0.3 < rms < 0.6:
        return "contact"

    # Low F0 = Social (affiliative)
    if f0 < 4000:
        return "social"

    # Default
    return "contact"
```

### 7.3 Confidence Calculation

```python
def _calculate_confidence(
    self,
    features: np.ndarray,
    context: str
) -> float:
    """
    Calculate confidence in context detection.

    Uses feature variance and rule strength:
    - Strong rule matches: 0.8-0.95
    - Moderate matches: 0.5-0.8
    - Weak matches: 0.3-0.5
    """

    # Base confidence on feature distinctiveness
    variance = np.var(features)
    base_confidence = min(0.95, max(0.3, 0.5 + variance * 0.1))

    # Adjust for context-specific certainty
    if context == "alarm":
        # Alarm detection is highly reliable
        return min(0.95, base_confidence + 0.1)
    elif context == "territorial":
        # Territorial is moderately reliable
        return base_confidence
    else:
        # Contact/Social have more ambiguity
        return max(0.5, base_confidence - 0.1)
```

---

## 8. Response Generation Pipeline

### 8.1 Timeline Generation

```python
def _create_response_timeline(
    self,
    cluster_id: int,
    context: str,
) -> List[TimelineEvent]:
    """
    Generate synthesis timeline based on context.

    Args:
        cluster_id: Source cluster to respond to
        context: Detected behavioral context

    Returns:
        List of TimelineEvent for Rust synthesizer
    """

    if context == "alarm":
        # Short, urgent response
        return [TimelineEvent(
            cluster_id=cluster_id,
            start_time_ms=0.0,
            duration_ms=100.0,
            amplitude=0.9,
        )]

    elif context == "territorial":
        # Strong, assertive response
        return [TimelineEvent(
            cluster_id=cluster_id,
            start_time_ms=0.0,
            duration_ms=200.0,
            amplitude=0.85,
        )]

    elif context == "social":
        # Longer, conversational response
        return [
            TimelineEvent(
                cluster_id=cluster_id,
                start_time_ms=0.0,
                duration_ms=150.0,
                amplitude=0.7,
            ),
            TimelineEvent(
                cluster_id=cluster_id,
                start_time_ms=160.0,
                duration_ms=120.0,
                amplitude=0.6,
            ),
        ]

    else:  # contact
        # Standard contact call
        return [TimelineEvent(
            cluster_id=cluster_id,
            start_time_ms=0.0,
            duration_ms=150.0,
            amplitude=0.75,
        )]
```

### 8.2 Delta Generation

```python
def _create_deltas(
    self,
    context: str,
) -> Optional[MicroDynamicsDelta]:
    """
    Generate micro-dynamics deltas based on context.

    These modify the synthesis parameters to match
    the intended behavioral response.
    """

    if context == "alarm":
        # Raise pitch and energy for urgency
        return MicroDynamicsDelta(
            delta_mean_f0_hz=500.0,
            delta_rms_energy=0.2,
        )

    elif context == "territorial":
        # Slightly raise pitch and extend duration
        return MicroDynamicsDelta(
            delta_mean_f0_hz=200.0,
            delta_duration_ms=20.0,
        )

    elif context == "social":
        # Lower pitch for affiliative tone
        return MicroDynamicsDelta(
            delta_mean_f0_hz=-100.0,
            delta_sustain_level=0.1,
        )

    else:  # contact
        # Minimal modification
        return None
```

---

## 9. Safety and Rate Limiting

### 9.1 Rate Limiting Configuration

```python
@dataclass
class InteractionAgentConfig:
    # Minimum time between responses (milliseconds)
    response_cooldown_ms: float = 100.0

    # Maximum responses per second (0 = unlimited)
    max_responses_per_second: float = 5.0

    # Minimum confidence to trigger response (0.0 to 1.0)
    min_confidence_threshold: float = 0.5
```

### 9.2 Rate Limiting Implementation

```python
def _should_respond(self, result: Dict[str, Any]) -> bool:
    # Check confidence
    if result.get("confidence", 0.0) < self.config.min_confidence_threshold:
        return False

    # Check cooldown
    time_since_last = time.time() - self._last_response_time
    if time_since_last < self.config.response_cooldown_ms / 1000.0:
        return False

    # Check rate limit
    if self.config.max_responses_per_second > 0:
        if self._responses_sent / max(1, time.time() - self._start_time) > \
           self.config.max_responses_per_second:
            return False

    return True
```

### 9.3 Safety Fallback

If Python crashes or stops sending heartbeats:

```rust
// In peer_controller.rs
fn handle_timeout(&mut self) {
    if self.python_alive {
        warn!("Heartbeat timeout - Python agent appears frozen");
        self.handle_disconnect();
    }
}

fn handle_disconnect(&mut self) {
    if self.python_alive {
        error!("❌ Cognitive Agent (Python) LOST - Muting Audio");
        self.python_alive = false;
        self.audio_mute = AudioMuteState::Muted;
        // Rust continues in Passthrough Mode
    }
}
```

---

## 10. State Machine Specification

### 10.1 Agent States

```python
class AgentState(Enum):
    """Interaction agent states"""

    IDLE = "idle"          # Not processing
    LISTENING = "listening"  # Receiving features, analyzing
    RESPONDING = "responding"  # Generating and sending synthesis
```

### 10.2 State Transition Diagram

```
                    ┌─────────┐
                    │         │
            ┌──────►│  IDLE   │◄──────┐
            │       │         │       │
            │       └────┬────┘       │
            │            │            │
            │   start()  │            │  stop()
            │            ▼            │
            │       ┌─────────┐       │
            │       │         │       │
            │       │LISTENING│───────┘
            │       │         │
            │       └────┬────┘
            │            │
            │  feature   │
            │  received  │
            │            ▼
            │       ┌─────────┐
            │       │         │
            └───────│RESPONDING│
       response  │         │
       sent      └─────────┘
```

### 10.3 State Transition Implementation

```python
def _handle_feature_event(self, event: FeatureEvent) -> None:
    self._events_processed += 1

    # Transition: IDLE/LISTENING → LISTENING
    self.state = AgentState.LISTENING

    # Process features
    result = self._process_features(event)

    # Check if should respond
    if self._should_respond(result):
        # Transition: LISTENING → RESPONDING
        self.state = AgentState.RESPONDING
        self._send_response(result, event)

        # Transition: RESPONDING → LISTENING
        self.state = AgentState.LISTENING
```

---

## 11. Integration Testing Protocol

### 11.1 Test Categories

| Category | Tests | Purpose |
|----------|-------|---------|
| Serialization | 8 | Verify JSON roundtrip compatibility |
| Context Inference | 4 | Verify context detection accuracy |
| Timeline Generation | 4 | Verify response timeline creation |
| Delta Generation | 4 | Verify micro-dynamics deltas |
| Rate Limiting | 2 | Verify cooldown enforcement |
| Confidence Threshold | 2 | Verify threshold filtering |
| Statistics | 2 | Verify metric tracking |
| Integration | 4 | Verify end-to-end flow |

### 11.2 Running Tests

```bash
# Python tests
python -m pytest tests/test_feature_subscriber.py -v
python -m pytest tests/test_action_publisher.py -v
python -m pytest tests/test_interaction_agent.py -v
python -m pytest tests/test_closed_loop_integration.py -v

# Rust tests
cargo test feature_event --no-fail-fast
cargo test action_subscriber --no-fail-fast
cargo test synthesis_action --no-fail-fast

# All tests
cargo test --lib
python -m pytest tests/ -v
```

### 11.3 Test Coverage Requirements

| Component | Minimum Coverage |
|-----------|-----------------|
| FeatureEvent serialization | 100% |
| SynthesisAction serialization | 100% |
| Context inference | 90% |
| Response generation | 90% |
| Rate limiting | 95% |
| State transitions | 100% |

---

## 12. Deployment Configuration

### 12.1 Systemd Service Files

**Rust Field Engine (`rust-field-engine.service`):**
```ini
[Unit]
Description=Rust Field Engine (5-Stage Pipeline)
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/rust-field-engine
Restart=always
RestartSec=1

[Install]
WantedBy=multi-user.target
```

**Python Cognitive Agent (`python-cognitive-agent.service`):**
```ini
[Unit]
Description=Python Cognitive Agent (Closed-Loop)
After=network.target rust-field-engine.service

[Service]
Type=simple
ExecStart=/usr/bin/python3 /opt/zoo-vox/realtime/interaction_agent.py
Restart=always
RestartSec=1

[Install]
WantedBy=multi-user.target
```

### 12.2 Environment Variables

```bash
# Feature publishing endpoint
export RUST_FEATURES_ENDPOINT="ipc:///tmp/cognitive_features.ipc"

# Action command endpoint
export RUST_ACTIONS_ENDPOINT="ipc:///tmp/cognitive_actions.ipc"

# Heartbeat endpoint (existing)
export RUST_HEARTBEAT_ENDPOINT="ipc:///tmp/cognitive_heartbeat.ipc"

# Logging level
export RUST_LOG=info
export PYTHON_LOG_LEVEL=INFO
```

### 12.3 IPC Socket Permissions

```bash
# Create socket directory with correct permissions
sudo mkdir -p /tmp/cognitive
sudo chmod 777 /tmp/cognitive

# Or use abstract namespace (Linux only)
# No filesystem permissions needed
```

---

## 13. Performance Metrics

### 13.1 Key Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Feature Event Latency | < 10ms | Timestamp delta: publish → receive |
| Context Inference Time | < 5ms | Processing duration in Python |
| Action Publishing Time | < 5ms | Send time to ZeroMQ |
| Total Round-Trip | < 50ms | Feature detection → Audio output |
| Events/Second | 50-200 | Maximum sustained throughput |

### 13.2 Statistics Tracking

```python
def get_stats(self) -> Dict[str, Any]:
    return {
        "state": self.state.value,
        "running": self._running,
        "uptime_seconds": time.time() - self._start_time,
        "events_processed": self._events_processed,
        "responses_sent": self._responses_sent,
        "current_context": self._current_context,
        "context_confidence": self._context_confidence,
        "events_per_second": self._events_processed / max(1, uptime),
        "responses_per_second": self._responses_sent / max(1, uptime),
    }
```

### 13.3 Performance Benchmarks

```bash
# Run performance benchmarks
cargo bench --features "benchmarking"

# Expected results:
# - FeatureEvent serialization: < 100μs
# - SynthesisAction serialization: < 50μs
# - ZeroMQ send/receive: < 1ms
# - Context inference: < 5ms
```

---

## 14. Troubleshooting Guide

### 14.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| No events received | Subscriber not connected | Check `connect()` was called |
| Actions not sent | Publisher not connected | Check `connect()` was called |
| High latency | Large queue backlog | Reduce HWM, check processing time |
| Context always UNKNOWN | Feature extraction failed | Check 112D vector values |
| Rate limiting too aggressive | Cooldown too long | Reduce `response_cooldown_ms` |

### 14.2 Debug Logging

```bash
# Enable debug logging
export RUST_LOG=debug
export PYTHON_LOG_LEVEL=DEBUG

# View logs
journalctl -u rust-field-engine.service -f
journalctl -u python-cognitive-agent.service -f
```

### 14.3 IPC Socket Debugging

```bash
# Check for existing sockets
ls -la /tmp/cognitive_*.ipc

# Check socket connections
lsof /tmp/cognitive_features.ipc

# Monitor ZeroMQ traffic (requires debug build)
strace -e trace=socket,connect,bind,send,recv \
    -p $(pgrep -f interaction_agent)
```

---

## Appendix A: Full JSON Schema

### FeatureEvent Schema

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["event_type", "cluster_id", "features_112d", "timestamp", "sequence"],
    "properties": {
        "event_type": {
            "type": "string",
            "const": "feature_extraction"
        },
        "cluster_id": {
            "type": "integer",
            "minimum": 0,
            "maximum": 1019
        },
        "features_112d": {
            "type": "array",
            "items": { "type": "number" },
            "minItems": 112,
            "maxItems": 112
        },
        "timestamp": {
            "type": "number",
            "description": "Unix timestamp in seconds"
        },
        "sequence": {
            "type": "integer",
            "minimum": 0
        }
    }
}
```

### SynthesisAction Schema

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["action_type", "timeline"],
    "properties": {
        "action_type": {
            "type": "string",
            "enum": ["synthesize_timeline"]
        },
        "timeline": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["cluster_id", "start_time_ms", "duration_ms"],
                "properties": {
                    "cluster_id": { "type": "integer" },
                    "start_time_ms": { "type": "number" },
                    "duration_ms": { "type": "number" },
                    "amplitude": { "type": "number", "minimum": 0, "maximum": 1 }
                }
            }
        },
        "deltas": {
            "type": "object",
            "properties": {
                "delta_mean_f0_hz": { "type": "number" },
                "delta_duration_ms": { "type": "number" },
                "delta_f0_range_hz": { "type": "number" },
                "delta_harmonic_to_noise_ratio": { "type": "number" },
                "delta_attack_time_ms": { "type": "number" },
                "delta_sustain_level": { "type": "number" },
                "delta_rms_energy": { "type": "number" }
            }
        },
        "priority": {
            "type": "string",
            "enum": ["low", "normal", "high", "critical"],
            "default": "normal"
        }
    }
}
```

---

## Appendix B: Code Examples

### B.1 Basic Usage

```python
from realtime.interaction_agent import InteractionAgent

# Create and start agent
agent = InteractionAgent()
agent.start()

# Agent will automatically process events and generate responses
# ...

# Stop when done
agent.stop()
```

### B.2 With Custom Callbacks

```python
def on_feature(event: FeatureEvent):
    print(f"Received cluster {event.cluster_id}")

def on_context(context: str, confidence: float):
    print(f"Context: {context} ({confidence:.2f})")

agent = InteractionAgent(
    on_feature_event=on_feature,
    on_context_change=on_context,
)
agent.start()
```

### B.3 Rust Integration

```rust
// In Rust, publish features
let mut publisher = FeatureEventPublisher::new(EventPublisherConfig::default())?;

// After NBD and feature extraction:
let event = FeatureEvent::from_array(cluster_id, features_112d, sequence);
publisher.publish(&event)?;

// In separate thread, receive actions
let mut subscriber = ActionSubscriber::new(ActionSubscriberConfig::default())?;

loop {
    if let Some(action) = subscriber.try_recv()? {
        // Execute synthesis based on action.timeline
        for event in action.timeline {
            synthesizer.synthesize_event(&event)?;
        }
    }
}
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-07 | Sheel Morjaria | Initial release |

---

**End of Methodology Protocol**
