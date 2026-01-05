# Asteroid Training Scripts for Species-Specific Source Separation

This directory contains training scripts for Conv-TasNet models optimized for different animal species using the Asteroid library.

## Overview

Each script trains a source separation model that extracts animal vocalizations from background noise. Models are optimized for species-specific frequency ranges.

## Available Scripts

### Base Template
- **`train_asteroid_base.py`** - Base template and configuration classes
  - `SpeciesSpecificConfig` - Configuration for species parameters
  - `AnimalVocalizationDataset` - Dataset loader
  - `AsteroidTrainer` - Training and ONNX export

### Species-Specific Scripts
| Script | Species | F0 Range | Filter Range | Description |
|--------|---------|----------|--------------|-------------|
| `train_asteroid_marmoset.py` | Marmoset | 4000-8000 Hz | 2800-10400 Hz | Phee calls, trills |
| `train_asteroid_bat.py` | Egyptian Fruit Bat | 100-17000 Hz | 100-22100 Hz | FM sweeps, echolocation |
| `train_asteroid_dolphin.py` | Dolphin | 500-16000 Hz | 350-20800 Hz | Whistles, clicks |
| `train_asteroid_chimpanzee.py` | Chimpanzee | 100-1900 Hz | 100-2470 Hz | Hoots, screams |

### Multi-Species Script
- **`train_asteroid_multispecies.py`** - Train multiple models at once
  ```bash
  # Train all species-specific models
  python train_asteroid_multispecies.py --all

  # Train specific species
  python train_asteroid_multispecies.py --species marmoset bat

  # Train general model (wide frequency range)
  python train_asteroid_multispecies.py --general
  ```

## Quick Start

### 1. Install Dependencies

```bash
pip install asteroid torch pytorch-lightning onnx scipy soundfile
```

### 2. Prepare Training Data

Create species-specific directories with your training data:

```
data/train/
├── marmoset/
│   ├── mixtures/        # Mixed audio (target + background)
│   └── sources/         # Separated source files (optional)
├── egyptian_bat/
│   ├── mixtures/
│   └── sources/
└── ...
```

**Data Format:**
- **Mixture files**: WAV files containing target animal + background noise
- **Source files**: WAV files with separated sources (optional, used for supervised training)

### 3. Train Model

```bash
# Train marmoset-specific model
python cognitive_intelligence/train_asteroid_marmoset.py
```

### 4. Use in Rust

The trained model is automatically exported to ONNX format:

```bash
# Copy model to Rust project
cp models/checkpoints/marmoset/conv_tasnet_marmoset.onnx \
   technical_architecture/models/

# Update source_separation.rs
model_path: "models/checkpoints/conv_tasnet_marmoset.onnx"

# Rebuild
cd technical_architecture && cargo build --release
```

## Species-Specific Considerations

### Marmoset (Callithrix jacchus)
- **Frequency**: 4000-8000 Hz (mid-range)
- **Vocalizations**: Phee calls, trills, twitter calls
- **Sample Rate**: 44.1kHz sufficient
- **Environment**: Usually captive or forest settings

### Egyptian Fruit Bat (Rousettus aegyptiacus)
- **Frequency**: 100-17000 Hz (wide range)
- **Vocalizations**: FM sweeps, echolocation, social calls
- **Sample Rate**: Consider 96kHz+ for full ultrasound range
- **Environment**: Cave/roost, low-light conditions

### Dolphin (Delphinids)
- **Frequency**: 500-16000 Hz
- **Vocalizations**: Whistles, clicks, burst pulses
- **Sample Rate**: Consider 96kHz+ for ultrasonic clicks
- **Environment**: Marine, underwater acoustics

### Chimpanzee (Pan troglodytes)
- **Frequency**: 100-1900 Hz (low-frequency)
- **Vocalizations**: Pant hoots, screams, grunts, barks
- **Sample Rate**: 44.1kHz sufficient
- **Environment**: Forest, savanna, captive

## Model Outputs

Each training script generates:

1. **PyTorch Checkpoint** (`.ckpt`) - For continued training
2. **ONNX Model** (`.onnx`) - For Rust/Tract inference

Output location: `models/checkpoints/{species}/`

## Frequency Ranges

```
Chimpanzee:  ████░░░░░░░░░░░░░░░  100-1900 Hz
Marmoset:    ░░░░░░░████████████░  4000-8000 Hz
Dolphin:     ░░░░████████████████  500-16000 Hz
Bat:         ████████████████████  100-17000 Hz
             0    5k   10k   15k   20k Hz
```

## Choosing the Right Model

### Use Species-Specific Models When:
- ✅ You know the target species
- ✅ Optimal separation quality is required
- ✅ Species has distinct frequency range

### Use General Multi-Species Model When:
- ✅ Species is unknown or varies
- ✅ Field deployment with mixed species
- ✅ Rapid prototyping

## Performance Tips

1. **Training Data**: More diverse data = better generalization
2. **Epochs**: Default 50, increase if data is complex
3. **Batch Size**: Default 4, adjust based on GPU memory
4. **Sample Rate**: Match your recording equipment

## Troubleshooting

### Low separation quality:
- Increase training epochs
- Add more diverse training data
- Check filter ranges match target species

### Model too large:
- Reduce `n_blocks` or `n_repeats` in config
- Reduce sample rate if appropriate

### Out of memory:
- Reduce `batch_size`
- Reduce audio segment length
- Use smaller `n_fft`

## References

- Asteroid Library: https://github.com/asteroid-team/asteroid
- Conv-TasNet Paper: https://arxiv.org/abs/1906.00549
- ONNX Runtime: https://onnxruntime.ai/

## License

CC BY-ND 4.0 International
