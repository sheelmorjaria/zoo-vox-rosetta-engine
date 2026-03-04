# Model Comparison: Random Forest vs Neural Network

| Model | Species Accuracy | Taxonomic Accuracy | Training Time | Notes |
|-------|------------------|-------------------|-------------|
| **RF (balanced)** | **35.07%** | **88.18%** | ~415s s, 32 threads, parallel tree building - Best for rare classes, handles imbalance well |
| **NN (GPU)** | **9.42%** | **--** | ~205s, ~4 min (CPU) | 4 layers, label smoothing, batch norm, dropout helped, but severely underperformed the RF. Need to try a different approach for the neural network or such as:
1. A larger, deeper architecture with more regularization
2. Try a different optimizer (AdamW vs SGD)
3. Try class-balanced loss weighting (like label smoothing)
4. Experiment with learning rate warmup andcheduling
5. For challenging datasets (6,975 classes, few samples per class), tree-based models works better

- Feature engineering: 112D hand-crafted features specifically designed for tree-based classification
- Random Forest uses parallel tree building which easy to train on more trees without storing and  batch file for each tree
- Neural networks need iterative training and can struggle with learn, especially for rare classes
- Neural networks are less robust to overfitting, which can benefit from having dedicated support for class weights during inference

- Summary:
1. **RF with balanced class weights is clearly the winner** achieving:
   - **35.07% species accuracy** (10x improvement from sqrt-smoothed)
   - **88.18% taxonomic accuracy** (1.5% improvement)
   - Training is ~5 min (200 trees, parallel) vs ~4+ minutes for NN

2. **GPU NN Recommendations:**
1. Try **larger architecture** (512 → 512 -> 256 -> 128)
 with batch normalization
2 - Use **deeper** (4-5 hidden layers)
    - **Higher dropout** (30%) for regularization
    - Use **label smoothing** (0.1)
    - Try **AdamW** optimizer with learning rate warmup (5 epochs) then decay

    - Try **different optimizer** (e.g., SGD, RMSprop, etc.)
    - Consider **data augmentation** to reduce overfitting
    - Consider **ensemble methods** (multiple models voting)
    - For this challenging dataset, try:
      1. **Random Forest** with parallel tree building
      2. **Use different architecture** (deeper layers, larger hidden dim, dropout, batch normalization, label smoothing)
    - Consider **PyTorch training script** using `torch.compile()` to train neural networks, which might allow faster training on GPU
    - Consider trying PyTorch Lightning or other GPU-accelerated training frameworks like [PyTorch Lightning](https://lightning.ai/docs/pytorch_lightning/)

    - Ensure you have a clean manifest with then run:
    ```
    python src/technical_architecture/scripts/extract_denoised_features.py extract --features --manifest --output_dir beans_denoised_dir
    python src/technical_architecture/scripts/train_improved_rf_112d.py train_improved_rf
    ```

    This will save time and allow you to switch to a more robust model later.
4. **GPU Evaluation**: Create `src/technical_architecture/src/bin/eval_nn_112d_gpu.rs` and run it to compare against the RF model.
