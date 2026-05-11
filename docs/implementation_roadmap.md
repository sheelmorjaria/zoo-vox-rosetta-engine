# ALP Re-Architecture Implementation Roadmap

**From Digital Sampler to Living Vocal Tract: A Phased Deployment Strategy**

---

## Executive Summary

This roadmap details the deployment of the re-architected Animal Language Processing (ALP) pipeline, which represents a paradigm shift from discrete categorization-and-splicing to continuous manifold modeling and de novo generation. The 8-week phased approach ensures biological safety and computational stability before engaging in closed-loop interaction with the Egyptian fruit bat colony.

**Architecture Shift:**
```
OLD: Audio → GMM Classification → Concatenative Splicing → Speaker
NEW: Audio → BioMAE → VAE/VQ-VAE → DualStreamAgent → DDSP Synthesis
                      ↓              ↓                    ↓
                 Continuous     Continuous           Latent
                 Manifold       + Discrete         Interpolation
```

---

## Phase 1: Shadow Mode Integration (Weeks 1-4)

**Objective:** Validate the entire re-architected stack in passive listening mode before engaging in playback.

### Architecture in Shadow Mode

```
┌─────────────────────────────────────────────────────────────────┐
│                    Shadow Mode Pipeline                         │
│                                                                  │
│  [Bat Audio] → [NBD] → [BioMAE] → [VAE/VQ-VAE] → [Agent]      │
│       ↓          ↓          ↓            ↓            ↓         │
│   Segmented   112D      16D + Token   DualStream   DDSP       │
│   Features   Embedding   State       Action       Synthesized  │
│                                                 Output        │
│                                                    ↓           │
│                                              [RECORD ONLY]    │
│                                              (No Playback)    │
└─────────────────────────────────────────────────────────────────┘
```

### Week 1: Rust Layer Validation

**Goal:** Ensure NBD segmentation and BioMAE embedding run stably at edge.

**Tasks:**
1. Deploy `NBDLayer` with sub-50ms boundary detection
2. Deploy `BioMAEResNet` with ONNX/TensorRT optimization
3. Validate 112D RosettaFeatures extraction
4. Profile memory usage and thermal throttling

**Success Criteria:**
- [ ] NBD detects >95% of boundaries in validation set
- [ ] BioMAE inference <10ms per frame on TensorRT
- [ ] No memory leaks over 24-hour continuous run
- [ ] Thermal manager maintains <70°C

**Verification Script:**
```python
# tests/test_shadow_mode_rust.py
async def test_nbd_stability():
    nbd = NBDLayer(config=NBDConfig())
    for audio_chunk in bat_stream:
        boundaries = nbd.segment(audio_chunk)
        assert len(boundaries) > 0
        assert all(b.latency_ms < 50 for b in boundaries)
```

### Week 2: Python Agent Validation

**Goal:** Ensure Python cognitive layer processes DualStreamState correctly.

**Tasks:**
1. Deploy VAE/VQ-VAE encoders via ONNX
2. Deploy DualStreamInteractionAgent v2.0
3. Validate affect-syntactic fusion logic
4. Profile ZMQ IPC latency

**Success Criteria:**
- [ ] VAE encoding <5ms per frame
- [ ] VQ-VAE tokenization <5ms per frame
- [ ] Agent decision latency <20ms
- [ ] ZMQ round-trip <2ms
- [ ] Total end-to-end latency <50ms

**Verification Script:**
```python
# tests/test_shadow_mode_agent.py
async def test_dual_stream_processing():
    agent = DualStreamInteractionAgent()
    state = DualStreamState(
        syntactic_token=23,
        affect_vector=np.random.randn(16),
        raw_features=np.random.randn(112),
        confidence=0.95,
        sequence=0
    )
    action = agent.handle_dual_stream_state(state)
    assert isinstance(action, DualStreamAction)
    assert action.syntactic_token >= 0
```

### Week 3: DDSP Synthesis Validation

**Goal:** Ensure DDSP synthesis produces biologically plausible audio.

**Tasks:**
1. Deploy DualStreamDDSPDecoder with FiLM layers
2. Validate phase accumulator continuity
3. Generate shadow mode output corpus
4. Run acoustic quality metrics

**Success Criteria:**
- [ ] DDSP synthesis <50ms per utterance
- [ ] Phase discontinuity score <2.0
- [ ] Spectral smoothness >0.7
- [ ] No audible artifacts in blind listening test

**Verification Script:**
```python
# tests/test_shadow_mode_ddsp.py
def test_ddsp_quality():
    synthesizer = DualStreamSynthesizer()
    audio = synthesizer.synthesize_dual_stream(
        syntactic_token=23,
        affect_vector=np.random.randn(16)
    )
    phase_score = compute_phase_discontinuity(audio, 48000)
    smooth_score = compute_spectral_smoothness(audio, 48000)
    assert phase_score < 2.0
    assert smooth_score > 0.7
```

### Week 4: End-to-End Integration

**Goal:** Validate full pipeline from microphone to synthesized output.

**Tasks:**
1. Wire all components together
2. Run 24-hour continuous shadow mode
3. Log telemetry and latencies
4. Analyze MFAS scores of AI vs natural continuation

**Success Criteria:**
- [ ] Pipeline runs 24 hours without crash
- [ ] 99th percentile latency <100ms
- [ ] No audio artifacts in synthesized output
- [ ] MFAS analysis shows AI output within natural variation

**Telemetry Dashboard:**
```python
# monitoring/shadow_mode_dashboard.py
class ShadowModeTelemetry:
    def __init__(self):
        self.metrics = {
            'nbd_latency': [],
            'bio_mae_latency': [],
            'vae_latency': [],
            'agent_latency': [],
            'ddsp_latency': [],
            'total_latency': [],
            'memory_usage': [],
            'temperature': [],
        }

    def record_frame(self, timings):
        for key, value in timings.items():
            self.metrics[key].append(value)

    def generate_report(self):
        return {
            'p50_latency': np.median(self.metrics['total_latency']),
            'p99_latency': np.percentile(self.metrics['total_latency'], 99),
            'max_memory': max(self.metrics['memory_usage']),
            'max_temp': max(self.metrics['temperature']),
        }
```

---

## Phase 2: Spatial Calibration & Level 2.5 Mapping (Weeks 5-6)

**Objective:** Enable the system to know *where* the bats are and *who* they're talking to.

### Architecture with Spatial Awareness

```
┌─────────────────────────────────────────────────────────────────┐
│                    Level 2.5 Spatial Layer                      │
│                                                                  │
│  [Camera 1-4] → [DeepLabCut] → [3D Pose]                        │
│       ↓              ↓              ↓                            │
│  [Microphone Array] → [TDOA] → [Source Localization]            │
│                           ↓                                      │
│                    [TopologyEngine]                              │
│                           ↓                                      │
│              [ReceiverInferenceEngine]                           │
│           (Proximity × Social × Line-of-Sight)                   │
│                           ↓                                      │
│              [EmitterSelection] → [Speaker i]                    │
└─────────────────────────────────────────────────────────────────┘
```

### Week 5: Multi-Camera Setup & DeepLabCut Training

**Tasks:**
1. Install 4-camera array over flight room
2. Train DeepLabCut model on bat pose landmarks
3. Calibrate camera-to-world coordinates
4. Validate 3D pose reconstruction accuracy

**Hardware Requirements:**
- 4× USB3 cameras (120fps minimum)
- Infrared illumination for night tracking
- Synchronized capture via hardware trigger

**Success Criteria:**
- [ ] DeepLabCut training loss <0.01
- [ ] 3D pose error <5cm RMS
- [ ] Tracking identity maintained >90% over 1 minute
- [ ] Real-time processing at 30fps

**Calibration Script:**
```python
# spatial/camera_calibration.py
class CameraCalibration:
    def __init__(self, num_cameras=4):
        self.num_cameras = num_cameras
        self.camera_matrix = []
        self.dist_coeffs = []

    def calibrate_chessboard(self, images_per_camera):
        """Standard OpenCV calibration."""
        for cam_idx in range(self.num_cameras):
            # ... calibration logic ...
            pass

    def calibrate_stereo(self):
        """Stereo calibration for 3D reconstruction."""
        # ... stereo calibration logic ...
        pass

    def project_3d_to_world(self, camera_points, camera_id):
        """Convert camera coords to world coords."""
        # ... projection logic ...
        pass
```

### Week 6: TopologyEngine & Receiver Inference

**Tasks:**
1. Deploy TopologyEngine for spatial graph management
2. Train ReceiverInferenceEngine on labeled interactions
3. Validate Line-of-Sight calculation
4. Tune proximity/social/LoS weights

**TopologyEngine Data Structure:**
```python
# spatial/topology_engine.py
@dataclass
class BatNode:
    bat_id: int
    position: Tuple[float, float, float]  # x, y, z in world coords
    velocity: Tuple[float, float, float]
    last_update_ms: float
    pose_confidence: float

@dataclass
class Edge:
    from_bat: int
    to_bat: int
    distance: float
    line_of_sight: bool
    social_tie_strength: float  # From historical interaction data

class TopologyEngine:
    def __init__(self):
        self.nodes: Dict[int, BatNode] = {}
        self.edges: List[Edge] = []

    def update_node(self, bat_id: int, position: np.ndarray):
        """Update bat position and velocity."""
        # ... update logic ...

    def compute_edges(self):
        """Compute all pairwise relationships."""
        # ... edge computation ...

    def find_nearest_neighbors(self, bat_id: int, k: int) -> List[int]:
        """Find k nearest neighbors."""
        # ... nearest neighbor search ...

    def check_line_of_sight(self, from_id: int, to_id: int) -> bool:
        """Check if two bats have clear line of sight."""
        # ... LOS calculation ...
```

**Receiver Inference:**
```python
# spatial/receiver_inference.py
class ReceiverInferenceEngine:
    """
    Predict the intended receiver of a vocalization based on:
    1. Proximity (closer = more likely)
    2. Social ties (stronger ties = more likely)
    3. Line-of-sight (visible = more likely)
    """
    def __init__(self, w_proximity=0.4, w_social=0.4, w_los=0.2):
        self.w_proximity = w_proximity
        self.w_social = w_social
        self.w_los = w_los

    def infer_receiver(
        self,
        caller_id: int,
        topology: TopologyEngine,
    ) -> Tuple[int, float]:
        """
        Returns: (predicted_receiver_id, confidence)
        """
        scores = {}
        for other_id in topology.nodes:
            if other_id == caller_id:
                continue

            # Get edge data
            edge = topology.get_edge(caller_id, other_id)
            if edge is None:
                continue

            # Compute score
            proximity_score = 1.0 / (1.0 + edge.distance / 100.0)  # Normalize
            social_score = edge.social_tie_strength
            los_score = 1.0 if edge.line_of_sight else 0.1

            combined = (
                self.w_proximity * proximity_score +
                self.w_social * social_score +
                self.w_los * los_score
            )
            scores[other_id] = combined

        if not scores:
            return -1, 0.0  # No receiver detected

        # Return highest-scoring receiver
        best_receiver = max(scores, key=scores.get)
        confidence = scores[best_receiver]
        return best_receiver, confidence
```

**Calibration Validation:**
```python
# tests/test_spatial_calibration.py
def test_topology_accuracy():
    """Test spatial inference against ground truth."""
    topology = TopologyEngine()
    engine = ReceiverInferenceEngine()

    # Load labeled interactions with known receivers
    interactions = load_labeled_interactions()

    correct = 0
    total = 0
    for interaction in interactions:
        predicted, confidence = engine.infer_receiver(
            interaction.caller_id,
            interaction.topology
        )
        if predicted == interaction.actual_receiver:
            correct += 1
        total += 1

    accuracy = correct / total
    assert accuracy > 0.8  # 80% accuracy minimum
```

---

## Phase 3: Acclimation Phase (Week 7)

**Objective:** Begin broadcasting without interaction to observe colony response.

### Acclimation Protocol

```
┌─────────────────────────────────────────────────────────────────┐
│                    Acclimation Mode                             │
│                                                                  │
│  [Live Colony] → [Microphones] → [Full Pipeline]                │
│                                        ↓                         │
│                                  [DDSP Synthesis]               │
│                                        ↓                         │
│                                  [Speaker Playback]            │
│                                        ↓                         │
│                            [Colony Response Monitoring]         │
│                                        ↓                         │
│                              [MFAS Telemetry]                   │
└─────────────────────────────────────────────────────────────────┘
```

### Acclimation Tasks

1. **Volume Calibration:** Play synthesized calls at 85 dB SPL (measured at roost)
2. **Directional Testing:** Test each emitter independently
3. **MFAS Monitoring:** Track colony-wide acceptance metrics
4. **Artifact Detection:** Listen for digital artifacts causing alarm responses

### Acclimation Metrics

```python
# monitoring/acclimation_monitor.py
class AcclimationMonitor:
    """
    Monitor colony response during acclimation phase.
    """
    def __init__(self, mfas_calculator):
        self.mfas = mfas_calculator
        self.metrics = {
            'alarm_rate': [],  # % of alarm tokens post-playback
            'response_rate': [],  # % of bats responding
            'mean_mfas': [],  # Mean MFAS score
            'colony_agitation': [],  # Colony-wide arousal
        }

    def assess_colony_response(self, playback_event, responses):
        """
        Assess colony response to AI playback.

        Returns:
            dict with colony-level metrics
        """
        alarm_count = sum(1 for r in responses if r.is_alarm)
        response_count = len(responses)

        # MFAS scores
        mfas_scores = [
            self.mfas.evaluate_interaction(r.interaction)
            for r in responses if r.interaction is not None
        ]

        return {
            'alarm_rate': alarm_count / max(response_count, 1),
            'response_rate': response_count / estimated_colony_size,
            'mean_mfas': np.mean([s.mfas_score for s in mfas_scores]),
            'colony_agitation': self._compute_agitation(responses),
        }

    def _compute_agitation(self, responses):
        """Compute colony-wide agitation index."""
        # High arousal + low MFAS = agitation
        agitation = 0
        for r in responses:
            if r.interaction:
                arousal = r.interaction.affect_vector[0]  # Arousal dim
                mfas = self.mfas.evaluate_interaction(r.interaction)
                agitation += arousal * (1.0 - mfas.mfas_score)
        return agitation
```

### Success Criteria

- [ ] Alarm rate <10% (baseline is ~5%)
- [ ] No sustained agitation (>30 seconds elevated arousal)
- [ ] MFAS scores show natural variation (0.3-0.8 range)
- [ ] No observable flight-to-exits behavior

### Failure Modes & Recovery

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| 300% alarm spike | Digital artifacts in synthesis | Increase MultiScaleSpectralLoss weight |
| Colony silence | Volume too low/high | Recalibrate SPL to 85 dB |
| Sustained agitation | Affective mismatch | Adjust FiLM modulation parameters |
| Flight response | Perceived intrusion | Pause playback, extend acclimation |

---

## Phase 4: Closed-Loop Deployment (Week 8+)

**Objective:** Engage in true bidirectional interaction with individual bats.

### Closed-Loop Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Closed-Loop Interaction                        │
│                                                                  │
│  [Bat i vocalizes]                                              │
│       ↓                                                          │
│  [NBD → BioMAE → VAE/VQ-VAE]                                    │
│       ↓                                                          │
│  [TopologyEngine identifies location]                            │
│       ↓                                                          │
│  [ReceiverInference predicts receiver]                           │
│       ↓                                                          │
│  [InteractionAgent generates response]                          │
│       ↓                                                          │
│  [DDSP synthesizes with targeted affect]                        │
│       ↓                                                          │
│  [EmitterSelection directs to location]                         │
│       ↓                                                          │
│  [Speaker array broadcasts]                                     │
│       ↓                                                          │
│  [Colody responds → MFAS evaluation → Agent learning]           │
└─────────────────────────────────────────────────────────────────┘
```

### Deployment Tasks

1. **Individual Identification:** Tag bats of interest for targeted interaction
2. **Context Selection:** Start with low-arousal contact calls
3. **MFAS Feedback Loop:** Use MFAS scores to guide agent responses
4. **Progressive Escalation:** Gradually attempt more complex interactions

### InteractionAgent v3.0

```python
# realtime/interaction_agent_v3.py
class InteractionAgentV3:
    """
    Closed-loop interaction agent with spatial and MFAS awareness.
    """
    def __init__(
        self,
        topology_engine: TopologyEngine,
        receiver_inference: ReceiverInferenceEngine,
        mfas_calculator: MultiFactorAcceptanceScore,
    ):
        self.topology = topology_engine
        self.receiver_inference = receiver_inference
        self.mfas = mfas_calculator

        self.recent_mfas = deque(maxlen=100)  # Rolling MFAS history
        self.conversation_state = {}  # Per-bat conversation state

    def handle_vocalization(
        self,
        caller_id: int,
        dual_stream_state: DualStreamState,
    ) -> Optional[DualStreamAction]:
        """
        Generate response to bat vocalization.

        Returns:
            DualStreamAction if response warranted, None otherwise
        """
        # Step 1: Infer intended receiver
        receiver_id, confidence = self.receiver_inference.infer_receiver(
            caller_id,
            self.topology
        )

        # Step 2: Check if AI should respond (is receiver = AI?)
        if receiver_id != -1:  # Addressed to another bat
            return None

        # Step 3: Assess conversation context
        context = self.conversation_state.get(caller_id, {
            'turns': 0,
            'last_mfas': 0.5,
            'arousal_trajectory': [],
        })

        # Step 4: Generate response based on context
        response = self._generate_response(
            dual_stream_state,
            context
        )

        # Step 5: Update conversation state
        context['turns'] += 1
        context['arousal_trajectory'].append(
            dual_stream_state.affect_vector[0]  # Arousal
        )
        self.conversation_state[caller_id] = context

        return response

    def _generate_response(
        self,
        incoming_state: DualStreamState,
        context: dict
    ) -> DualStreamAction:
        """
        Generate response affect based on incoming state and context.
        """
        incoming_arousal = incoming_state.affect_vector[0]

        # De-escalation policy
        if incoming_arousal > 0.8:
            # High arousal: respond with calming affect
            target_arousal = incoming_arousal * 0.7
        elif incoming_arousal < 0.3:
            # Low arousal: escalate slightly for engagement
            target_arousal = incoming_arousal * 1.2
        else:
            # Match for social bonding
            target_arousal = incoming_arousal

        # Generate syntactic response (follow conversation rules)
        valid_next = self.syntax_graph.get_valid_next_tokens(
            incoming_state.syntactic_token,
            top_k=5
        )
        response_token = valid_next[0][0]

        # Generate affect vector (match arousal, adjust valence)
        target_affect = incoming_state.affect_vector.copy()
        target_affect[0] = target_arousal  # Adjust arousal
        target_affect[1] *= 0.9  # Slightly more positive valence

        return DualStreamAction(
            syntactic_token=response_token,
            affect_vector=target_affect,
            temporal_offset_ms=self._calculate_response_delay(context),
        )

    def update_mfas(self, caller_id: int, mfas_result: MFASResult):
        """Update MFAS history for adaptive behavior."""
        self.recent_mfas.append(mfas_result)

        # Update conversation state
        if caller_id in self.conversation_state:
            self.conversation_state[caller_id]['last_mfas'] = \
                mfas_result.mfas_score

        # Trigger adaptive responses if needed
        if mfas_result.mfas_score < 0.3:
            # Low acceptance: back off
            self._trigger_backoff(caller_id)
```

### Success Criteria

- [ ] Individual bats engage in >3 turn exchanges
- [ ] Mean MFAS > 0.7 across interactions
- [ ] No colony-wide panic events
- [ ] Agent shows adaptive behavior (MFAS-based learning)

### Safety Protocols

```python
# safety/interaction_safety.py
class InteractionSafetyMonitor:
    """
    Real-time safety monitoring for closed-loop interaction.
    """
    def __init__(self, thresholds):
        self.max_colony_arousal = thresholds['max_colony_arousal']  # 0.8
        self.max_alarm_rate = thresholds['max_alarm_rate']  # 0.3
        self.min_mfas = thresholds['min_mfas']  # 0.3

    def check_safety(self, colony_metrics) -> Tuple[bool, str]:
        """
        Check if interaction should continue.

        Returns:
            (safe, reason) tuple
        """
        if colony_metrics['colony_agitation'] > self.max_colony_arousal:
            return False, "Colony agitation exceeded threshold"

        if colony_metrics['alarm_rate'] > self.max_alarm_rate:
            return False, "Alarm rate too high"

        if colony_metrics['mean_mfas'] < self.min_mfas:
            return False, "Acceptance too low"

        return True, "OK"

    def trigger_emergency_stop(self):
        """Immediately halt all playback."""
        # ... emergency stop logic ...
```

---

## Telemetry & Monitoring Dashboard

### Real-Time Metrics

```python
# monitoring/deployment_dashboard.py
class DeploymentDashboard:
    """
    Real-time monitoring dashboard for closed-loop deployment.
    """
    def __init__(self):
        self.metrics = {
            # System health
            'pipeline_latency': [],
            'memory_usage': [],
            'cpu_usage': [],
            'temperature': [],

            # Interaction metrics
            'interactions_per_minute': [],
            'mean_turns_per_conversation': [],
            'active_bats': [],

            # Colony response
            'alarm_rate': [],
            'response_rate': [],
            'mean_mfas': [],
            'colony_agitation': [],

            # Spatial
            'tracked_bats': [],
            'mean_position_error': [],
        }

    def update(self, metric_name: str, value: float):
        """Update a metric."""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)

    def get_summary(self) -> dict:
        """Get summary statistics."""
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'latest': values[-1],
                }
        return summary
```

### Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Pipeline latency (p99) | >80ms | >100ms | Investigate bottleneck |
| Memory usage | >80% | >95% | Trigger garbage collection |
| Temperature | >70°C | >80°C | Throttle processing |
| Alarm rate | >20% | >30% | Pause playback |
| Mean MFAS | <0.4 | <0.3 | Adjust agent parameters |
| Colony agitation | >0.6 | >0.8 | Trigger backoff |

---

## Checklist Summary

### Phase 1: Shadow Mode (Weeks 1-4)
- [ ] NBD deployment and validation
- [ ] BioMAE deployment and validation
- [ ] VAE/VQ-VAE deployment and validation
- [ ] DualStreamAgent deployment
- [ ] DDSP synthesis validation
- [ ] 24-hour continuous run
- [ ] Latency profiling (<100ms p99)

### Phase 2: Spatial Calibration (Weeks 5-6)
- [ ] Camera array installation
- [ ] DeepLabCut training
- [ ] 3D pose reconstruction validation
- [ ] TopologyEngine deployment
- [ ] ReceiverInferenceEngine training
- [ ] Spatial accuracy validation (>80%)

### Phase 3: Acclimation (Week 7)
- [ ] Volume calibration (85 dB SPL)
- [ ] Directional emitter testing
- [ ] MFAS monitoring deployment
- [ ] Acclimation protocol execution
- [ ] Colony response validation (<10% alarm spike)

### Phase 4: Closed-Loop (Week 8+)
- [ ] Individual bat identification
- [ ] InteractionAgent v3.0 deployment
- [ ] Safety monitor deployment
- [ ] First interaction (low-arousal contact)
- [ ] Progressive escalation testing
- [ ] Long-term MFAS tracking (>0.7 target)

---

## References

1. **Shadow Mode Testing**: Standard practice for ML system deployment
2. **Spatial Calibration**: DeepLabCut (Mathis et al., 2018)
3. **Acclimation Protocols**: Standard animal behavior research ethics
4. **Closed-Loop Interaction**: Real-time animal-computer interaction best practices

---

**Author**: Zoo Vox Research Team
**License**: CC BY-ND 4.0 International
**Version**: 1.0
**Date**: 2026-05-10
