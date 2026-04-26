# How This Codebase Solves the Two-Way Interspecies Communication Problem

## Requirement 1: Non-Invasive Approach

The system is entirely non-invasive. It never implants, tags, or physically interacts with organisms. Instead:

### Passive Acquisition via Neural Boundary Detection (NBD)

The Rust Execution Layer ingests raw, continuous field recordings and autonomously segments them into "acoustic gestures" without any human annotation (`technical_architecture/src/neural_boundary.rs`). No manual segmentation required — the system discovers structure on its own.

### Feature Extraction → Symbolic Vocabulary

Continuous acoustic features (112 dimensions: prosody, spectral shape, micro-harmonics) are quantized into a discrete symbolic vocabulary using a `VocabOptimizer`. For Egyptian fruit bats, the system empirically discovered an optimal vocabulary of k=1020 symbols — preserving both individual intent specificity and shared population structures (`technical_architecture/src/graded_phrase_mining.rs`).

### Fail-Open Safety Design

The peer-to-peer supervision (`peer_controller.rs`) ensures the system never disrupts natural behavior. If the Python cognitive layer crashes, Rust immediately enters **Passthrough Mode** — audio output is muted, but passive recording continues. The system physically cannot produce sound without explicit, confidence-gated authorization from the Python layer.

### Strategy Pattern for Endogenous Signals

The architecture uses a pluggable `ParsingStrategy` (`realtime/parsing_strategy.py`):

- **CompositionalStrategy** — default for species where segments are semantic units
- **HolophrasticStrategy** — species-specific (e.g., bats) where rigid idioms are atomic meaning units

This ensures the system always communicates using the organism's **own** discovered communication patterns, not human-imposed templates.

---

## Requirement 2: Multi-Context Communication Using Endogenous Signals

### Context Inference Engine

The system maps 112D feature vectors to behavioral contexts in real-time (`realtime/interaction_agent.py`):

| Context | Acoustic Signature | Response Behavior |
|---|---|---|
| **Alarm** | High F0 (>8kHz), high RMS | +500Hz frequency shift, high priority |
| **Territorial** | Mid-high F0 (6-8kHz) | Aggressive posture matching |
| **Food/Foraging** | Medium F0 (5-6.5kHz), longer duration | Slightly lower F0, extended duration |
| **Social/Contact** | Lower F0 (4-6kHz), quieter | Softer, longer responses |

### Acoustic Framing with Endogenous Roles

For bats, the system discovered and replicates natural temporal roles:

- **Openers** — short, explosive staccato bursts to grab attention (position 0)
- **Closers** — longer, cleaner harmonic tones signaling message completion (position 1)

### Grammar Discovery, Not Grammar Imposition

The system discovered that bat communication is a **fixed-code holophrastic system** — meaning is at the pattern level, not individual "words." Only **0.02% of possible bigrams are actually used**, making it an extremely restrictive grammar. Rigid idioms like `LRN-6 [114, 464, 604, 324, 94, 714]` are unbreakable atomic units. The system respects these constraints when generating responses.

### Autonomous Closed-Loop Interaction

The interaction agent (`realtime/interaction_agent.py`) operates a state machine:

```
IDLE → LISTENING → RESPONDING
```

1. Rust detects acoustic boundary → sends 112D features via ZeroMQ
2. Python parses segments using species-specific strategy (holophrastic vs compositional)
3. Context is inferred from feature patterns
4. Confidence gate: only responds if confidence > 0.5
5. Rate limiting: max 5 responses/sec, 100ms cooldown — prevents feedback loops
6. Synthesis timeline sent back to Rust using the organism's own acoustic profile
7. Rust outputs audio using `DynamicMicroharmonicParams` species defaults

### Species-Specific Synthesis

Each species gets its own acoustic profile (`technical_architecture/src/synthesis.rs`):

- **Marmoset**: harmonic communication parameters
- **Bat**: FM sweep parameters with specific attack/decay envelopes
- **Dolphin**: whistle contour parameters
- **Chimpanzee**: mixed modulation parameters

### Semiotic Validation

The semiotic engine (`semiotics/semiotic_engine.py`) validates communication quality across six semiotic dimensions: indexical, iconic, symbolic, deceptive, emergent, and directed — ensuring responses are semantically meaningful, not just acoustic mimicry.

---

## Summary: Three-Phase Pipeline

The codebase addresses both requirements through a **three-phase pipeline**:

1. **Discovery** — Non-invasive NBD + symbolic quantization discovers the organism's communication structure from raw audio
2. **Understanding** — 112D features + context inference + holophrastic/compositional parsing maps sounds to meaning across multiple behavioral contexts
3. **Interaction** — Confidence-gated, rate-limited closed-loop agent generates responses using the organism's own restrictive grammar and acoustic profile, with fail-open safety ensuring the system never disrupts natural behavior
