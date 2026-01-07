# Vector Delta Integration Guide

## Critical System Integration: Acoustic Algebra → Rust Synthesis

### The Problem

Acoustic Algebra generates **absolute targets** (e.g., "F0=7000Hz"), but Rust synthesis needs **relative shifts** from the source buffer.

```
Bad Command (Absolute):
  "Set pitch to 7000Hz."
  ❌ Ignores that we started at 6800Hz
  ❌ Ignores that we started at 7200Hz

Good Command (Delta):
  "Shift pitch by +200Hz relative to source."
  ✅ 6800Hz + 200Hz = 7000Hz
  ✅ 7200Hz - 200Hz = 7000Hz
```

### Why This Matters

When `find_nearest_real_phrase()` returns a phrase, it might have:
- F0 = 6800Hz (or 7200Hz, or any value)

We want to synthesize at F0=7000Hz. The delta command automatically adjusts:
- If source F0=6800Hz: Shift by **+200Hz**
- If source F0=7200Hz: Shift by **-200Hz**

**Same target, different delta!** This is why delta commands are superior.

---

## Solution: SourceMetadata + Delta Commands

### 1. SourceMetadata Structure

Tracks the acoustic features of the loaded source buffer:

```python
from technical_architecture import SourceMetadata

metadata = SourceMetadata(
    mean_f0_hz=6800.0,      # Source F0
    duration_ms=50.0,        # Source duration
    f0_range_hz=400.0        # Source F0 range
)
```

### 2. Load Source with Metadata

```python
from technical_architecture import GranularConcatenativeSynthesizer, SourceMetadata

synth = GranularConcatenativeSynthesizer(sample_rate=22050)

# Load audio buffer WITH metadata
metadata = SourceMetadata(
    mean_f0_hz=6800.0,
    duration_ms=50.0,
    f0_range_hz=400.0
)
synth.load_source_with_metadata(audio_buffer, metadata)
```

### 3. Vector Delta Commands

Now you can use **relative** delta commands:

```python
# Shift pitch by +200Hz (6800 + 200 = 7000Hz)
synth.shift_pitch_by_hz(200.0)

# Shift duration by -10ms (50 - 10 = 40ms)
synth.shift_duration_by_ms(-10.0)

# Or apply all shifts at once
synth.apply_vector_delta(
    delta_f0_hz=200.0,        # Pitch shift
    delta_duration_ms=-10.0,   # Duration shift
    delta_f0_range_hz=100.0    # F0 range shift
)
```

---

## Complete Integration Workflow

### Step-by-Step: From Acoustic Algebra to Audio

```python
from analysis.rosetta_stone.contextual_map import ContextualMap
from technical_architecture import GranularConcatenativeSynthesizer, SourceMetadata

# ===== STEP 1: Generate Virtual Phrase (Acoustic Algebra) =====
map_obj = ContextualMap()
centroids = map_obj.calculate_context_centroids(phrase_vectors, context_labels)

# Generate "30% Aggressive" virtual phrase
virtual = map_obj.generate_graded_phrase('aggression', intensity=0.3)
# Result: F0=7000Hz, Dur=40ms, Range=500Hz

# ===== STEP 2: Find Nearest Real Phrase =====
nearest_key, nearest_vec, distance = map_obj.find_nearest_real_phrase(
    virtual, phrase_vectors
)
# Result: F0=6800Hz, Dur=50ms, Range=400Hz

# ===== STEP 3: Calculate Delta =====
delta_f0 = virtual.mean_f0_hz - nearest_vec.mean_f0_hz      # +200Hz
delta_dur = virtual.duration_ms - nearest_vec.duration_ms    # -10ms
delta_range = virtual.f0_range_hz - nearest_vec.f0_range_hz # +100Hz

# ===== STEP 4: Load Source with Metadata =====
synth = GranularConcatenativeSynthesizer(sample_rate=22050)
audio_buffer = load_audio_file(nearest_key)  # Your audio loading function

metadata = SourceMetadata(
    mean_f0_hz=nearest_vec.mean_f0_hz,
    duration_ms=nearest_vec.duration_ms,
    f0_range_hz=nearest_vec.f0_range_hz
)
synth.load_source_with_metadata(audio_buffer, metadata)

# ===== STEP 5: Apply Vector Delta =====
synth.apply_vector_delta(delta_f0, delta_dur, delta_range)

# ===== STEP 6: Synthesize =====
output = synth.synthesize(duration_ms=virtual.duration_ms)
```

---

## API Reference

### SourceMetadata

**Constructor:**
```python
SourceMetadata(mean_f0_hz, duration_ms, f0_range_hz)
```

**Methods:**
- `get_mean_f0_hz()` - Get F0 (Hz)
- `set_mean_f0_hz(value)` - Set F0 (Hz)
- `get_duration_ms()` - Get duration (ms)
- `set_duration_ms(value)` - Set duration (ms)
- `get_f0_range_hz()` - Get F0 range (Hz)
- `set_f0_range_hz(value)` - Set F0 range (Hz)

### GranularConcatenativeSynthesizer

**New Methods:**

**`load_source_with_metadata(source, metadata)`**
- Load audio with acoustic metadata
- Enables delta commands

**`set_source_metadata(metadata)`**
- Set metadata after loading audio
- Useful if metadata discovered later

**`shift_pitch_by_hz(delta_hz)`**
- Shift pitch by absolute Hz amount
- Positive = higher pitch
- Negative = lower pitch
- Example: `shift_pitch_by_hz(200.0)` = shift up by 200Hz

**`shift_duration_by_ms(delta_ms)`**
- Shift duration by absolute ms amount
- Positive = longer
- Negative = shorter
- Example: `shift_duration_by_ms(-10.0)` = shorten by 10ms

**`apply_vector_delta(delta_f0_hz, delta_duration_ms, delta_f0_range_hz)`**
- Apply multiple shifts simultaneously
- Primary integration point for Acoustic Algebra

---

## Example: Delta vs Absolute Comparison

```python
# Scenario: Want to synthesize at F0=7000Hz

# ===== BAD: Absolute Command (Not Supported) =====
# synth.set_pitch_to_hz(7000.0)  # ❌ Doesn't exist!
# Problem: Doesn't know if source is 6800Hz or 7200Hz

# ===== GOOD: Delta Command =====
metadata = SourceMetadata(mean_f0_hz=6800.0, duration_ms=50.0, f0_range_hz=400.0)
synth.load_source_with_metadata(audio_buffer, metadata)
synth.shift_pitch_by_hz(200.0)  # ✅ 6800 + 200 = 7000Hz

# ===== ALSO GOOD: Different Source =====
metadata = SourceMetadata(mean_f0_hz=7200.0, duration_ms=50.0, f0_range_hz=400.0)
synth.load_source_with_metadata(audio_buffer, metadata)
synth.shift_pitch_by_hz(-200.0)  # ✅ 7200 - 200 = 7000Hz
```

Both achieve the same target (7000Hz), but with **different delta commands**!

---

## Mathematical Implementation

### Pitch Shift Ratio Calculation

```python
# Formula: ratio = (source_f0 + delta_hz) / source_f0
source_f0 = 6800.0
delta_hz = 200.0
target_f0 = source_f0 + delta_hz  # 7000.0
ratio = target_f0 / source_f0  # 1.029

# Rust implementation (in synthesis.rs)
pub fn shift_pitch_by_hz(&mut self, delta_hz: f32) {
    let source_f0 = self.source_metadata.mean_f0_hz;
    let target_f0 = source_f0 + delta_hz;
    let ratio = (target_f0 / source_f0).clamp(0.5, 2.0);
    self.pitch_shift_ratio = ratio;
}
```

### Duration Stretch Ratio Calculation

```python
# Formula: ratio = (source_duration + delta_ms) / source_duration
source_dur = 50.0
delta_ms = -10.0
target_dur = source_dur + delta_ms  # 40.0
ratio = target_dur / source_dur  # 0.8

# Rust implementation (in synthesis.rs)
pub fn shift_duration_by_ms(&mut self, delta_ms: f32) {
    let source_duration = self.source_metadata.duration_ms;
    let target_duration = source_duration + delta_ms;
    let ratio = (target_duration / source_duration).clamp(0.5, 4.0);
    self.time_stretch_ratio = ratio;
}
```

---

## Files Modified

### Rust (technical_architecture/)

**src/synthesis.rs:**
- Added `SourceMetadata` struct (lines 1454-1473)
- Added `source_metadata` field to `GranularConcatenativeSynthesizer` (line 1484)
- Added `load_source_with_metadata()` method (lines 1502-1530)
- Added `set_source_metadata()` method (lines 1537-1540)
- Added `shift_pitch_by_hz()` method (lines 1542-1563)
- Added `shift_duration_by_ms()` method (lines 1565-1586)
- Added `apply_vector_delta()` method (lines 1588-1613)

**src/lib.rs (PyO3 bindings):**
- Added `PySourceMetadata` wrapper (lines 752-808)
- Added getter/setter methods (lines 774-802)
- Added Python bindings for delta commands (lines 835-959)
- Exported `SourceMetadata` in module (line 2169)

### Python

**tests/test_vector_delta_synthesis.py:**
- 12 tests covering all delta commands
- Tests for SourceMetadata creation
- Tests for pitch/duration shifts
- Tests for complete integration workflow
- Comparison test: delta vs absolute commands

---

## Test Results

```
✅ ALL 12 TESTS PASSED

Test Coverage:
  ✓ SourceMetadata creation and accessors
  ✓ load_source_with_metadata()
  ✓ set_source_metadata()
  ✓ shift_pitch_by_hz() (positive, negative, zero)
  ✓ shift_duration_by_ms() (positive, negative)
  ✓ apply_vector_delta() (complete)
  ✓ Delta vs Absolute command comparison
  ✓ End-to-end workflow
  ✓ Legacy backward compatibility
```

---

## Key Insights

1. **Delta commands are source-aware**: They automatically adjust based on the source buffer's characteristics
2. **Same target, different deltas**: Source at 6800Hz needs +200Hz, source at 7200Hz needs -200Hz (both reach 7000Hz)
3. **Acoustic Algebra integration**: `delta = virtual - nearest` → `apply_vector_delta(delta)`
4. **Backward compatible**: Legacy `load_source()` still works (uses default metadata)

---

## Future Enhancements

- **Full 17D delta support**: Currently handles F0, duration, range. Could extend to all 17 micro-dynamics features
- **Spectral manipulation**: F0 range shift currently tracked but not applied (requires spectral processing)
- **Jitter/shimmer synthesis**: Micro-perturbations for texture variation
- **Formant shifting**: Independent of pitch shifting (vocal tract length)

---

## Summary

**Before (Absolute):**
```python
synth.set_pitch(7000)  # ❌ Doesn't work with different sources
```

**After (Delta):**
```python
# Load with metadata
metadata = SourceMetadata(mean_f0_hz=6800, duration_ms=50, f0_range_hz=400)
synth.load_source_with_metadata(audio, metadata)

# Apply delta (relative to source)
synth.shift_pitch_by_hz(200)  # ✅ Works with ANY source!
```

**Result:** Robust integration between Acoustic Algebra and Rust synthesis!
