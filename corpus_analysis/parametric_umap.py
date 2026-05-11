#!/usr/bin/env python3
"""
Parametric UMAP: Non-Linear Dimensionality Reduction

Replaces PCA with UMAP to preserve non-linear local gradients in vocalizations.
This module provides both GPU-accelerated (RAPIDS cuml) and CPU (umap-learn) options.

Key improvements over PCA:
- Preserves graded continua (arousal, affect, intensity)
- Maintains local neighborhood structure
- Exportable to ONNX for TensorRT deployment

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class UMAPConfig:
    """Configuration for Parametric UMAP."""
    # Dimensions
    input_dim: int = 112  # BioMAE embedding dimension
    output_dim: int = 30  # UMAP output dimension

    # UMAP hyperparameters
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"

    # Training
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3

    # Architecture
    hidden_dims: Tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    use_batch_norm: bool = True

    # Device
    device: str = "cuda"  # "cuda" or "cpu"


class VocalManifoldReducer(nn.Module):
    """
    Parametric UMAP Encoder: 112D BioMAE → 30D Non-linear space.

    Preserves the graded continuum that PCA destroys by learning a
    neural network to approximate the UMAP embedding transformation.

    Architecture:
        Input (112D) → Hidden layers → Output (30D)

    The encoder is trained to match standard UMAP embeddings, then
    exported to ONNX for fast inference.
    """

    def __init__(
        self,
        config: Optional[UMAPConfig] = None,
    ):
        super().__init__()
        if config is None:
            config = UMAPConfig()

        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim

        # Build encoder layers
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim

        # Final projection to output dimension
        layers.append(nn.Linear(prev_dim, self.output_dim))

        self.encoder = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to UMAP space.

        Args:
            x: Input tensor (B, input_dim)

        Returns:
            Embedded tensor (B, output_dim)
        """
        return self.encoder(x)


class ParametricUMAPTrainer:
    """
    Trainer for Parametric UMAP.

    Training process:
    1. Initialize with standard UMAP (cuml or umap-learn)
    2. Train neural encoder to match UMAP targets
    3. Export to ONNX for deployment
    """

    def __init__(
        self,
        config: Optional[UMAPConfig] = None,
    ):
        if config is None:
            config = UMAPConfig()

        self.config = config
        self.device = self._get_device()

        # Initialize encoder
        self.encoder = VocalManifoldReducer(config).to(self.device)

        # Initialize UMAP for target generation
        self.umap_init = self._create_umap()

    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if self.config.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _create_umap(self):
        """
        Create UMAP initializer based on available packages.

        Priority: cuml (GPU) > umap-learn (CPU)
        """
        try:
            import cuml
            logger.info("Using RAPIDS cuml for UMAP initialization")
            return cuml.UMAP(
                n_components=self.config.output_dim,
                n_neighbors=self.config.n_neighbors,
                min_dist=self.config.min_dist,
                metric=self.config.metric,
            )
        except ImportError:
            try:
                import umap
                logger.info("Using umap-learn for UMAP initialization")
                return umap.UMAP(
                    n_components=self.config.output_dim,
                    n_neighbors=self.config.n_neighbors,
                    min_dist=self.config.min_dist,
                    metric=self.config.metric,
                )
            except ImportError:
                raise ImportError(
                    "Neither cuml nor umap-learn is installed. "
                    "Install with: pip install umap-learn"
                )

    def train(
        self,
        data_112d: np.ndarray,
        val_split: float = 0.1,
    ) -> np.ndarray:
        """
        Train parametric UMAP encoder.

        Args:
            data_112d: Input BioMAE embeddings (N, 112)
            val_split: Fraction of data for validation

        Returns:
            embedding_30d: Reduced embeddings (N, 30)
        """
        logger.info(f"Training parametric UMAP on {data_112d.shape[0]} samples")

        # Step 1: Generate UMAP targets
        logger.info("Step 1: Generating UMAP target embeddings...")
        target_30d = self.umap_init.fit_transform(data_112d)
        logger.info(f"UMAP initialization complete: {target_30d.shape}")

        # Step 2: Create PyTorch datasets
        dataset = TensorDataset(
            torch.FloatTensor(data_112d),
            torch.FloatTensor(target_30d),
        )

        # Split train/val
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        # Step 3: Train encoder
        logger.info("Step 2: Training neural encoder...")
        optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.config.learning_rate
        )
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20

        for epoch in range(self.config.epochs):
            # Training
            self.encoder.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                pred = self.encoder(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.encoder.eval()
            val_loss = 0.0
            if val_size > 0:
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        pred = self.encoder(batch_x)
                        loss = criterion(pred, batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
            else:
                val_loss = train_loss  # No validation data

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Logging
            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                logger.info(
                    f"Epoch {epoch:3d}: "
                    f"Train Loss = {train_loss:.6f}, "
                    f"Val Loss = {val_loss:.6f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Step 4: Generate final embeddings
        logger.info("Step 3: Generating final embeddings...")
        self.encoder.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data_112d).to(self.device)
            embedding_30d = self.encoder(data_tensor).cpu().numpy()

        logger.info(f"Training complete. Final embedding shape: {embedding_30d.shape}")
        return embedding_30d

    def export_to_onnx(
        self,
        output_path: str = "models/umap_encoder.onnx",
    ) -> Path:
        """
        Export trained encoder to ONNX format.

        Args:
            output_path: Path for output ONNX file

        Returns:
            Path to exported ONNX file
        """
        import torch.onnx

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.encoder.eval()

        # Create dummy input
        dummy_input = torch.randn(1, self.config.input_dim).to(self.device)

        logger.info(f"Exporting UMAP encoder to {output_path}")

        # Export to ONNX
        torch.onnx.export(
            self.encoder,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=17,
            input_names=['bio_mae_embedding'],
            output_names=['umap_coords'],
            dynamic_axes={
                'bio_mae_embedding': {0: 'batch_size'},
                'umap_coords': {0: 'batch_size'},
            },
        )

        logger.info(f"Exported successfully to {output_path}")
        return output_path


def compute_manifold_quality_score(
    original_112d: np.ndarray,
    embedding_30d: np.ndarray,
) -> dict:
    """
    Compute quality metrics for the manifold embedding.

    Args:
        original_112d: Original high-dimensional data
        embedding_30d: Reduced embedding

    Returns:
        Dictionary with quality metrics
    """
    from sklearn.metrics import pairwise_distances

    # Trustworthiness: Measures how well local structure is preserved
    try:
        from sklearn.manifold import trustworthiness
        trust = trustworthiness(
            original_112d,
            embedding_30d,
            n_neighbors=15,
            metric='euclidean',
        )
    except Exception as e:
        logger.warning(f"Could not compute trustworthiness: {e}")
        trust = None

    # Continuity: Measures preservation of nearest neighbors
    orig_dists = pairwise_distances(original_112d[:1000])  # Sample for speed
    embed_dists = pairwise_distances(embedding_30d[:1000])

    # Get top-k neighbors
    k = 15
    orig_neighbors = np.argsort(orig_dists, axis=1)[:, 1:k+1]
    embed_neighbors = np.argsort(embed_dists, axis=1)[:, 1:k+1]

    # Jaccard similarity of neighbor sets
    jaccard_scores = []
    for i in range(len(orig_neighbors)):
        set_orig = set(orig_neighbors[i])
        set_embed = set(embed_neighbors[i])
        intersection = len(set_orig & set_embed)
        union = len(set_orig | set_embed)
        jaccard_scores.append(intersection / union if union > 0 else 0)

    continuity = np.mean(jaccard_scores)

    return {
        "trustworthiness": trust,
        "neighbor_continuity": continuity,
        "shape_original": original_112d.shape,
        "shape_embedded": embedding_30d.shape,
    }


# Preset configurations for different use cases

BAT_MANIFOLD_CONFIG = UMAPConfig(
    input_dim=112,
    output_dim=30,
    n_neighbors=15,
    min_dist=0.1,
    hidden_dims=(256, 128),
    dropout=0.1,
)

COMPACT_MANIFOLD_CONFIG = UMAPConfig(
    input_dim=112,
    output_dim=16,  # More aggressive reduction
    n_neighbors=10,
    min_dist=0.0,
    hidden_dims=(128, 64),
    dropout=0.0,
)


def create_umap_trainer(
    config: Optional[UMAPConfig] = None,
    checkpoint_path: Optional[str] = None,
) -> ParametricUMAPTrainer:
    """
    Factory function to create UMAP trainer.

    Args:
        config: UMAP configuration (uses BAT_MANIFOLD_CONFIG if None)
        checkpoint_path: Path to load trained encoder from

    Returns:
        Configured ParametricUMAPTrainer
    """
    if config is None:
        config = BAT_MANIFOLD_CONFIG

    trainer = ParametricUMAPTrainer(config)

    if checkpoint_path is not None:
        logger.info(f"Loading encoder from {checkpoint_path}")
        trainer.encoder.load_state_dict(
            torch.load(checkpoint_path, map_location=trainer.device)
        )

    return trainer


def main():
    """Example training script."""
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 5000
    data_112d = np.random.randn(n_samples, 112).astype(np.float32)

    # Create and train
    config = BAT_MANIFOLD_CONFIG
    config.epochs = 20  # Quick demo

    trainer = create_umap_trainer(config)
    embedding_30d = trainer.train(data_112d)

    # Compute quality metrics
    quality = compute_manifold_quality_score(data_112d, embedding_30d)
    logger.info(f"Manifold Quality: {quality}")

    # Export to ONNX
    onnx_path = trainer.export_to_onnx("models/umap_encoder_demo.onnx")
    logger.info(f"Exported to {onnx_path}")


if __name__ == '__main__':
    main()
