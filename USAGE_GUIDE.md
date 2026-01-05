# Animal Vocalization Analysis Framework - Usage Guide

## Overview

This framework provides a comprehensive, six-step workflow for analyzing animal vocalizations using the Universal Rosetta Stone methodology. It combines acoustic-first analysis with cognitive intelligence for cross-species communication research.

**Architecture:** Hybrid Python/Rust with peer-to-peer supervision

---

## Complete Research Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE WORKFLOW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STEP 1: Acoustic-First Analysis (analysis/)                    │
│           ↓ Raw audio → phrases → vocabulary → grammar          │
│                                                                 │
│  STEP 2: Data Import (data_import/)                             │
│           ↓ → vocalization_database.json (2,882 phrases)         │
│                                                                 │
│  STEP 3: Query Interface (query_interface/)                      │
│           ↓ → Real-time search, indexing                        │
│                                                                 │
│  STEP 4: Cognitive Intelligence (cognitive_intelligence/,        │
│                          semiotics/)                            │
│           ↓ → Deception detection, innovation tracking           │
│                                                                 │
│  STEP 5: Python Logic Layer (realtime/)                         │
│           ↓ → Cognitive decision making                         │
│                                                                 │
│  STEP 6: Rust Execution Layer (technical_architecture/)          │
│           ↓ → Safety-critical audio processing                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## STEP 1: Acoustic-First Analysis

**Purpose:** Extract phrase, sentence, and grammar structure from raw audio

**Location:** `analysis/rosetta_stone/`

### Basic Usage

```python
from analysis.rosetta_stone import UniversalRosettaStone

# Initialize analyzer
analyzer = UniversalRosettaStone(sample_rate=48000)

# Process raw audio → phrases
phrases = analyzer.segment_phrases(audio_data)

# Build vocabulary (atomic units)
vocabulary = analyzer.build_vocabulary(phrases, f0_bin_size=200)

# Discover grammar rules
grammar = analyzer.discover_grammar(phrases)

# Detect sentences
sentences = analyzer.discover_sentences(phrases, gaps)
```

### Phrase Key Format

**Universal format:** `F0_{mean}_DUR_{duration}_RANGE_{range}`

Examples:
- `F0_6400_DUR_50_RANGE_0` - 6.4kHz, 50ms, no frequency variation
- `F0_7080_DUR_50_RANGE_100` - 7.08kHz, 50ms, ±100Hz variation
- `F0_8000_DUR_75_RANGE_200` - 8kHz, 75ms, ±200Hz variation

### Modality Detection

The analyzer automatically detects acoustic modality:

| Modality | Description | Species Examples |
|----------|-------------|------------------|
| `HARMONIC` | Flat tones, stable pitch | Marmosets, Zebra Finches |
| `FM_SWEEP` | Pitch changes over time | Egyptian Fruit Bats, Dolphins |
| `TRANSIENT` | Clicks/pulses | Sperm Whales, insects |
| `RHYTHMIC` | Temporal patterns | Crickets, frogs |

### Demo for New Species

```bash
python3 src/analysis/rosetta_stone/demo_unknown_species.py
```

---

## STEP 2: Data Import

**Purpose:** Import analyzed data into unified database

**Location:** `data_import/`

### Import Database

```bash
python3 src/data_import/import_vocalization_data.py
```

This creates `vocalization_database.json` with:
- **2,882 phrases** across 4 species
- **1,351 marmoset phrases** (harmonic, 5-12 kHz)
- **516 Egyptian fruit bat phrases** (FM sweep, 20-90 kHz)
- **387 dolphin phrases** (whistle, 2-24 kHz)
- **628 chimpanzee phrases** (mixed, 200-3000 Hz)

### Database Structure

```json
{
  "species": "marmoset",
  "phrases": {
    "F0_6400_DUR_50_RANGE_0": {
      "mean_f0_hz": 6400,
      "std_f0_hz": 25,
      "mean_duration_ms": 50,
      "total_occurrences": 127,
      "contexts": ["contact", "food"],
      "acoustic_features": {...}
    }
  },
  "grammar": {
    "transitions": [...]
  },
  "sentences": [...]
}
```

---

## STEP 3: Query Interface

**Purpose:** High-performance querying with pre-built indexes

**Location:** `query_interface/`

### Initialize Interface

```python
from src import get_query_interface

interface = get_query_interface()
```

### Query by Acoustic Features

```python
# Search by F0 range (Hz)
results = interface.search_phrases_by_f0_range(5000, 10000)
for phrase_key, phrase in results[:5]:
    print(f"{phrase_key}: {phrase.acoustic_features.mean_f0_hz:.1f} Hz")

# Search by duration (ms)
results = interface.search_phrases_by_duration(50, 150)
for phrase_key, phrase in results[:5]:
    print(f"{phrase_key}: {phrase.acoustic_features.mean_duration_ms:.1f} ms")
```

### Query by Species

```python
from src import Species

# Get all marmoset phrases
marmoset_phrases = interface.get_phrases_by_species(Species.MARMOSET)
print(f"Marmoset: {len(marmoset_phrases)} phrases")

# Get all species
for species in [Species.MARMOSET, Species.EGYPTIAN_BAT,
                Species.DOLPHIN, Species.CHIMPANZEE]:
    phrases = interface.get_phrases_by_species(species)
    print(f"{species.value}: {len(phrases)} phrases")
```

### Similarity Search

```python
# Find similar phrases based on acoustic features
similar = interface.get_similar_phrases("F0_7400_DUR_0_RANGE_300", threshold=0.8)
for similarity, phrase_key, phrase in similar[:5]:
    print(f"{similarity:.3f} - {phrase_key}")
```

### Grammar Network Analysis

```python
# Get grammar transitions from a phrase
transitions = interface.get_grammar_transitions("F0_7400_DUR_0_RANGE_300")
for to_phrase, rule in transitions.items():
    print(f"{to_phrase}: {rule.frequency} occurrences")

# Get overall network statistics
network = interface.get_grammar_network()
print(f"Grammar network: {network['nodes']} nodes, {network['edges']} edges")
```

### Statistical Analysis

```python
# Get overall database statistics
stats = interface.get_phrase_statistics()
print(f"Total phrases: {stats['total_phrases']}")
print(f"Average F0: {stats['frequency_distribution']['avg']:.1f} Hz")
print(f"Species breakdown: {stats['species_breakdown']}")

# Species-specific statistics
marmoset_stats = interface.get_phrase_statistics(Species.MARMOSET)
print(f"Marmoset phrases: {marmoset_stats['total_phrases']}")
```

---

## STEP 4: Cognitive Intelligence

**Purpose:** Deception detection, innovation tracking, cross-modal fusion

**Locations:**
- `semiotics/` - Semiotic analysis
- `cognitive_intelligence/` - ML/AI components

### Semiotic Engine

```python
from src import SemioticEngine, SemioticContext, SemioticState, Species

# Initialize engine
engine = SemioticEngine()

# Create analysis context
context = SemioticContext(
    species=Species.MARMOSET,
    acoustic_features=features,
    social_context={"no_immediate_threat": True},
    behavioral_context={"problem_solving": False},
    cross_sensory_data={
        "visual_attention": 0.8,
        "acoustic_focus": 0.9,
        "spatial_coordination": 0.7
    }
)

# Analyze semiotics
result = engine.analyze_semiotics(phrase, context)

# Check results
print(f"Semiotic State: {result.semiotic_state}")
print(f"Deception Score: {result.deception_score:.3f}")
print(f"Innovation Potential: {result.innovation_potential:.3f}")
print(f"Directed Score: {result.directed_score:.3f}")

if result.semiotic_state == SemioticState.DECEPTIVE:
    print("⚠️  Deception detected!")
elif result.semiotic_state == SemioticState.EMERGENT:
    print("💡 Innovation detected!")
```

### Cross-Modal Data Fusion

```python
from cognitive_intelligence.data_fusion import DataFusionSystem
from cognitive_intelligence.visual_fusion import VisualFusionSystem

# Initialize fusion systems
visual = VisualFusionSystem()
data_fusion = DataFusionSystem()

# Process visual frame
visual_features = visual.process_frame(frame)

# Fuse visual + audio
result = data_fusion.fuse_modalities(
    visual_features=visual_features,
    audio_features=audio_features,
    species=Species.MARMOSET
)

# Result includes 20% attention boost for contact calls
print(f"Response probability: {result.response_probability:.3f}")
print(f"Attention boost applied: {result.attention_boost_applied}")
```

### Deception Detection Indicators

The SemioticEngine tracks:
- Low occurrence frequency (< 10 occurrences)
- Context mismatch (predator call with no threat)
- Social deception (dominance + resource competition)
- Cross-species deception targets
- Acoustic anomalies (high F0 variance)

### Innovation/Emergence Indicators

- First occurrence (new phrase)
- Novel situation context
- Problem-solving context
- Social learning observation
- Compositional phrases
- High observation potential

---

## STEP 5: Python Logic Layer

**Purpose:** Cognitive decision making and phrase selection logic

**Location:** `realtime/` (active modules only)

### Core Modules

**Active Python Logic Layer Files:**
- `cognitive_layer.py` - Cognitive intelligence (55KB)
- `adaptive_context_switcher.py` - Context interpretation
- `adaptive_resonance.py` - Adaptive resonance theory
- `deep_reinforcement_learning.py` - ML training
- `context_aware_synthesis.py` - Phrase selection logic
- `probabilistic_context_machine.py` - Decision making
- `phrase_audio_library.py` - Data management (98KB)
- `unified_database.py` - Data access
- `task_management.py` - Orchestration

### Cognitive Layer

```python
from realtime.cognitive_layer import CognitiveLayer

layer = CognitiveLayer()

# Make cognitive decision
decision = layer.decide(context, state)
print(f"Decision: {decision.action}")
print(f"Confidence: {decision.confidence:.3f}")
```

### Context-Aware Synthesis

```python
from realtime.context_aware_synthesis import ContextAwareSynthesizer

synthesizer = ContextAwareSynthesizer()

# Select phrases based on context
selected = synthesizer.select_phrases(context, library)
for phrase in selected:
    print(f"Selected: {phrase.phrase_key} (weight: {phrase.weight:.3f})")
```

---

## STEP 6: Rust Execution Layer

**Purpose:** Safety-critical audio processing and field deployment

**Location:** `technical_architecture/`

### Python API (via PyO3)

```python
from technical_architecture import TechnicalArchitect

# Create technical architect
config = TechArchConfig::default()
architect = TechnicalArchitect::new(config).await?

# Process audio frame
result = architect.process_audio_frame(audio, intent).await?
```

### Rust Native API

```rust
use technical_architecture::{TechnicalArchitect, TechArchConfig};

let config = TechArchConfig::default();
let architect = TechnicalArchitect::new(config).await?;

let result = architect.process_audio_frame(audio, intent).await?;
```

### Key Modules

| Module | Tests | Description |
|--------|-------|-------------|
| `synthesis.rs` | Core | Audio synthesis engines |
| `source_separation.rs` | Core | Conv-TasNet separator |
| `safety.rs` | Core | Safety monitoring |
| `thermal.rs` | Core | Thermal management |
| `ptp.rs` | Core | IEEE 1588 timing |
| `logging.rs` | Core | Provenance logging |
| `iacuc_compliance.rs` | Production (29) | Legal protocol enforcement |
| `time_series_archive.rs` | Production (24) | High-frequency storage |
| `auto_calibration.rs` | Production (17) | Self-health checks |
| `shadow_model_monitor.rs` | Production (26) | Concept drift detection |
| `web_dashboard.rs` | Production (25) | Remote monitoring |
| `multi_node_coordination.rs` | Production (21) | PTP/TDMA coordination |
| `environmental_monitor.rs` | Field (46) | Rain/temp/light monitoring |
| `power_manager.rs` | Field (54) | Solar-aware battery management |
| `wildlife_sentry.rs` | Field (24) | Background species detection |
| `data_synchronizer.rs` | Field (20) | Offline data queuing |
| `acoustic_simulator.rs` | Field (43) | TDD test fixture |

**Total:** 415 tests passing

---

## Demos

### Query Interface Demo

```bash
python3 src/query_interface/demo_query_interface.py
```

Demonstrates:
- Basic queries
- Statistical analysis
- Semantic searches
- Cross-species comparisons
- Performance benchmarks

### Semiotic Engine Demo

```bash
python3 src/semiotics/demo_semiotic_engine.py
```

Demonstrates:
- Deception detection
- Innovation tracking
- Directed communication analysis
- Cross-modal attention fusion

---

## Performance

### Query Interface Performance

- **Large dataset loading:** 2,882 phrases in < 10ms
- **F0 range query:** < 1ms for full dataset
- **Duration query:** < 1ms for full dataset
- **Similarity search:** < 5ms for 10 nearest neighbors
- **Grammar network:** < 10ms for full analysis

### Rust Performance (Release Build)

- **Audio processing:** < 1ms per 1024-sample frame
- **Heartbeat monitoring:** < 1 μs timeout detection
- **Mode switching:** Immediate flag update
- **FFT computation (1024):** ~50 μs
- **Wildlife detection:** ~200 μs per frame

---

## Species-Specific Features

### Marmoset (Harmonic Communication)
- **F0 range:** 4-8 kHz typical
- **Modality:** Harmonic with flat tones
- **Encoding:** F0 height encoding
- **Phrases:** 1,351 with detailed occurrence data
- **Contexts:** Contact, food, social, alarm

### Egyptian Fruit Bat (FM Sweep Communication)
- **F0 range:** 20-90 kHz
- **Modality:** FM sweep patterns
- **Encoding:** Frequency slope, contour shape
- **Phrases:** 516 analyzed
- **Contexts:** Navigation, feeding, social

### Dolphin (Whistle Communication)
- **F0 range:** 2-24 kHz
- **Modality:** Whistle (harmonic with modulation)
- **Encoding:** Contour shape, duration
- **Phrases:** 387 with semantic mappings
- **Contexts:** Signature whistles, contact, feeding

### Chimpanzee (Mixed Communication)
- **F0 range:** 200-3000 Hz
- **Modality:** Mixed (harmonic + transients)
- **Encoding:** Multi-individual social context
- **Phrases:** 628 tracked
- **Contexts:** Food, aggression, grooming, play

---

## Integration Examples

### Real-Time Wildlife Monitoring

```python
import asyncio
from src import get_query_interface, Species

interface = get_query_interface()

async def monitor_wildlife():
    while True:
        # Get audio from microphone
        audio = await get_audio_from_device()

        # Extract features
        f0 = extract_f0(audio)
        duration_ms = len(audio) / sample_rate * 1000

        # Query similar phrases
        results = interface.search_phrases_by_f0_range(f0 - 200, f0 + 200)

        if results:
            phrase_key, phrase = results[0]
            print(f"Detected: {phrase_key} ({phrase.species.value})")

            # Semiotic analysis
            result = engine.analyze_semiotics(phrase, context)

            if result.semiotic_state == SemioticState.DECEPTIVE:
                print("⚠️  Possible deception detected!")

        await asyncio.sleep(0.1)  # 10 Hz processing

asyncio.run(monitor_wildlife())
```

### Cross-Species Comparative Analysis

```python
from src import Species

def compare_species_features(species_list):
    """Compare acoustic features across species"""
    comparison = {}

    for species in species_list:
        stats = interface.get_phrase_statistics(species)
        if stats['total_phrases'] > 0:
            comparison[species.value] = {
                'phrase_count': stats['total_phrases'],
                'average_f0': stats['frequency_distribution']['avg'],
                'average_duration': stats['duration_distribution']['avg'],
                'dominant_modality': max(
                    stats['modality_breakdown'].items(),
                    key=lambda x: x[1]
                )[0]
            }

    return comparison

# Usage
species_to_compare = [Species.MARMOSET, Species.DOLPHIN, Species.CHIMPANZEE]
comparison = compare_species_features(species_to_compare)

for species, data in comparison.items():
    print(f"{species}:")
    print(f"  Phrases: {data['phrase_count']}")
    print(f"  Avg F0: {data['average_f0']:.1f} Hz")
    print(f"  Avg Duration: {data['average_duration']:.1f} ms")
    print(f"  Dominant Modality: {data['dominant_modality']}")
```

---

## Running Tests

### Python Tests

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_rosetta_stone_base.py -v

# Run with coverage
python3 -m pytest tests/ --cov=. --cov-report=html
```

### Rust Tests

```bash
cd technical_architecture

# Run all tests (415 tests passing)
cargo test

# Run specific module
cargo test environmental_monitor

# Run benchmarks
cargo bench
```

---

## Error Handling

```python
try:
    interface = get_query_interface()
    results = interface.search_phrases_by_f0_range(5000, 10000)
except FileNotFoundError:
    print("Database file not found. Run import script first:")
    print("  python3 src/data_import/import_vocalization_data.py")
except Exception as e:
    print(f"Query error: {e}")
    interface.refresh_database()  # Try reloading
```

---

## Advanced Features

### Microharmonic Synthesis

```python
from technical_architecture import (
    SynthesisMode, EnhancedMicroharmonicSynthesizer
)

# Create synthesizer
synthesizer = EnhancedMicroharmonicSynthesizer::new(
    species="marmoset",
    phrase_segments=segments,
    sample_rate=48000
)

# Horizontal synthesis (sequential)
result = synthesizer.synthesize_horizontal(
    phrase_keys=["F0_6400_DUR_50_RANGE_0", "F0_7080_DUR_50_RANGE_100"],
    constraints=None
).await?
```

### Multi-Node Coordination

```rust
use technical_architecture::MultiNodeCoordinator;

let config = ClusterConfig::default();
let coordinator = MultiNodeCoordinator::new("node1".to_string(), config);

// Elect grandmaster
let is_grandmaster = coordinator.elect_grandmaster(my_info).await?;

// Schedule transmission slot
let slot = coordinator.schedule_transmission_slot(100, Priority::High).await?;
```

---

## Configuration Files

### Query Interface Config

```python
# query_interface/config.py
QUERY_CONFIG = {
    'database_path': 'vocalization_database.json',
    'cache_enabled': True,
    'cache_size': 1000,
    'index_prebuild': True,
}
```

### Rust Config

```toml
# technical_architecture/deployment/config.toml
[audio]
sample_rate = 48000
channels = 1
buffer_size = 1024

[safety]
max_rms_level = 0.8
min_duration_ms = 10
max_duration_ms = 5000
```

---

## Data Export

```python
# Export query results as JSON
results = interface.search_phrases_by_f0_range(5000, 10000)
json_output = interface.export_query_results(results, 'json')

# Export as CSV
csv_output = interface.export_query_results(results, 'csv')
```

---

## License

**CC BY-ND 4.0 International** - See LICENSE file for details.

---

## Author

Sheel Morjaria (sheelmorjaria@gmail.com)

**Animal Vocalization Analysis Framework**
*Universal Rosetta Stone methodology for cross-species communication research*
