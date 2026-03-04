#!/usr/bin/env python3
"""Quick test of biodenoising on a few BEANS-Zero samples - saves as float32"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path

import numpy as np
import torch

# Load biodenoising model
print("Loading biodenoising model...")
from biodenoising import dns64

model = dns64(pretrained=True)
model.eval()
print("Model loaded!")


def denoise_audio(audio, sr):
    """Denoise audio using biodenoising"""
    # Resample to 16kHz
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Convert to tensor
    audio_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)

    # Run denoising
    with torch.no_grad():
        denoised = model(audio_tensor)

    # Convert back
    denoised = denoised.squeeze().numpy()

    # Resample back
    if sr != 16000:
        import librosa

        denoised = librosa.resample(denoised, orig_sr=16000, target_sr=sr)

    return denoised


# Test on a few samples
input_dir = Path("beans_zero_cache")
output_dir = Path("beans_zero_denoised_test")
output_dir.mkdir(parents=True, exist_ok=True)

# Get first 10 detection samples
import json

with open("beans_zero_cache/beans_audio_manifest.json") as f:
    manifest = json.load(f)

detection_samples = [s for s in manifest["samples"] if s["labels"]["task"] == "detection"][:10]

print(f"\nTesting on {len(detection_samples)} samples...")

for i, sample in enumerate(detection_samples):
    audio_file = sample["audio_file"]
    print(f"  [{i + 1}/{len(detection_samples)}] {audio_file}")

    # Load raw audio (float32 format)
    audio_path = input_dir / audio_file
    audio = np.fromfile(str(audio_path), dtype=np.float32)
    sr = 44100

    # Denoise
    denoised = denoise_audio(audio, sr)

    # Save as float32 (matching original format)
    out_path = output_dir / audio_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    denoised.astype(np.float32).tofile(str(out_path))

print(f"\nDenoised samples saved to: {output_dir}")
print("Test successful!")
