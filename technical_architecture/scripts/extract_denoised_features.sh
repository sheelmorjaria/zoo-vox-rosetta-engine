#!/bin/bash
# Extract 105D features from denoised BEANS-Zero detection data

echo "Extracting 105D features from denoised audio..."
echo "This will take ~30 minutes for 34,994 samples"

export LIBTORCH=/home/sheel/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

# Run feature extraction
./target/release/extract_features_cache beans_zero_denoised/beans_audio_manifest.json

echo "Feature extraction complete!"
ls -la beans_zero_denoised/feature_cache_eval/
