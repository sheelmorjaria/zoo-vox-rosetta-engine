# TDD Implementation Plan: Legacy vs. Predictive NBD Benchmarking

**Status**: Active
**Created**: 2025-05-10
**Task**: #107

---

## Overview

This plan implements the comprehensive testing protocol required to benchmark the transition from legacy heuristic-based NBD to the self-supervised Predictive NBD system.

---

## Phase 1: Performance and Latency Benchmarking

### 1.1 Execution Speed Profiling

**Target**: 99th-percentile latency ≤ 12ms
**Budget Allocation**:
- ONNX 1D Conv Encoder: ≤ 5ms
- Autoregressive Model (Mamba/TCN): ≤ 5ms
- MSE Error Computation: ≤ 1ms
- Adaptive Boundary Logic: < 1ms

**Files to Create**:
- `tests/test_predictive_nbd_latency.py` - Python latency profiling
- `technical_architecture/tests/predictive_nbd_latency.rs` - Rust latency tests

**Implementation Tasks**:
1. Create `LatencyProfiler` class with microsecond timing
2. Profile each component separately (encoder, AR, MSE, logic)
3. Run 10,000 frame benchmark
4. Report P50, P95, P99 latencies
5. Verify 12ms budget compliance

### 1.2 Memory Footprint Evaluation

**Target**:
- TCN: ~330KB (82K parameters)
- Mamba: ~600KB (150K parameters)

**Files to Create**:
- `tests/test_predictive_nbd_memory.py` - Memory profiling

**Implementation Tasks**:
1. Track VRAM/RAM usage during inference
2. Verify models fit in L2/L3 cache
3. Check for memory leaks over 24-hour soak test

---

## Phase 2: Threshold and Debounce Logic Evaluation

### 2.1 Static vs. Dynamic Thresholding

**Test**: "Drifting Noise" Test
- Inject audio with gradually increasing noise (60 seconds)
- Legacy: Fixed thresholds trigger false positives
- New: EMA baseline adapts, maintains <5% FP rate

**Files to Create**:
- `tests/test_drifting_noise.py` - Drifting noise test

**Parameters**:
- EMA decay: 0.95
- Window: 100 frames
- Target FP rate: <5%

### 2.2 Fixed vs. Adaptive Debounce

**Test**: "Avian Trill" Test
- Rapid chirps at 20-30ms intervals
- Legacy (50ms debounce): 0% recall
- New (adaptive re-arm): >90% recall

**Files to Create**:
- `tests/test_avian_trill.py` - Rapid chirp detection

**Parameters**:
- Chirp duration: 30ms
- Gap duration: 20ms
- Rearm threshold: 1.2x baseline

---

## Phase 3: Multi-Scale Boundary Classification Tests

### 3.1 Phonetic Boundaries

**Parameters**:
- Duration: ~20ms
- Threshold: 2.5x baseline

**Files to Create**:
- `tests/test_phonetic_boundaries.py` - Vowel space morphing test

### 3.2 Syllable Boundaries

**Parameters**:
- Duration: ~100ms
- Threshold: 3.0x baseline
- Min separation: 30ms

**Files to Create**:
- `tests/test_syllable_boundaries.py` - Two-tone syllable test

### 3.3 Phrase Boundaries

**Parameters**:
- Duration: ~350ms
- Threshold: 4.0x baseline

**Files to Create**:
- `tests/test_phrase_boundaries.py` - Whistle-to-noise transition test

---

## Phase 4: Software and Edge Testing Protocol

### 4.1 Python Validation Suite (48 tests)

**Basic Tests (33)**:
- Data piping
- Tensor shapes
- EMA smoothing (0.95)
- Error multipliers (2.5x, 3.0x, 4.0x)
- Armed/Disarmed logic
- Confidence scoring

**Validation Tests (15)**:
- InfoNCE loss computation
- Mutual information maximization
- Mamba streaming states
- Sequential state updates

**Files to Create**:
- `tests/test_info_nce_loss.py` - InfoNCE validation
- `tests/test_mamba_streaming.py` - Mamba state tests
- `tests/test_ema_baseline.py` - EMA smoothing tests

### 4.2 Rust Edge Tests (8 tests)

**Files to Create**:
- `technical_architecture/tests/predictive_nbd_edge.rs`

**Tests**:
1. `test_onnx_encoder_latency_p99` - Verify ≤5ms encoder latency
2. `test_ar_model_latency_p99` - Verify ≤5ms AR latency
3. `test_total_latency_budget` - Verify ≤12ms total
4. `test_zmq_non_blocking` - Verify ZMQ DONTWAIT compliance
5. `test_state_persistence` - Verify state survives ZMQ cycles
6. `test_memory_leak_soak` - 24-hour memory leak test
7. `test_mamba_hidden_state` - Verify correct state propagation
8. `test_confidence_calibration` - Verify 0.6+ confidence on detections

---

## File Structure

```
tests/
├── test_predictive_nbd_latency.py          # Phase 1.1
├── test_predictive_nbd_memory.py           # Phase 1.2
├── test_drifting_noise.py                   # Phase 2.1
├── test_avian_trill.py                      # Phase 2.2
├── test_phonetic_boundaries.py              # Phase 3.1
├── test_syllable_boundaries.py              # Phase 3.2
├── test_phrase_boundaries.py                # Phase 3.3
├── test_info_nce_loss.py                    # Phase 4.1
├── test_mamba_streaming.py                  # Phase 4.1
├── test_ema_baseline.py                     # Phase 4.1
└── test_predictive_nbd_validation.py       # Existing (keep)

technical_architecture/tests/
└── predictive_nbd_edge.rs                   # Phase 4.2
```

---

## Success Criteria

The implementation is complete when:

1. **Latency**: P99 latency ≤ 12ms on target hardware
2. **Debounce**: Avian Trill test yields >90% recall on sub-50ms boundaries
3. **Noise Robustness**: Drifting Noise test shows <5% FP rate
4. **Classification**: Multi-scale boundaries classified with ≥0.6 confidence in >85% cases
5. **Edge Tests**: All 8 Rust tests pass with zero memory leaks over 24 hours

---

## Implementation Order

1. **Week 1**: Phase 1 (Performance) - Create latency and memory profiling tests
2. **Week 2**: Phase 2 (Threshold/Debounce) - Implement Drifting Noise and Avian Trill tests
3. **Week 3**: Phase 3 (Multi-Scale) - Implement phonetic/syllable/phrase tests
4. **Week 4**: Phase 4 (Software/Edge) - Complete validation suite and Rust edge tests

---

## Dependencies

- `pytest` for Python testing
- `torch` for CPC model validation
- `ort` (ONNX Runtime) for real ONNX inference
- `tracemalloc` for memory profiling
- `psutil` for system monitoring
