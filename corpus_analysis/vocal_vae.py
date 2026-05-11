#!/usr/bin/env python3
"""
Vocal VAE: Continuous Latent Space Modeling

Replaces discrete BGMM clusters with a continuous probabilistic manifold.
Enables smooth interpolation between vocalization archetypes.

Key features:
- Standard VAE formulation (mu, logvar outputs directly)
- β-VAE support for disentangled representations
- ONNX export for encoder/decoder separately
- Interpolation and sampling utilities

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
class VAEConfig:
    """Configuration for Vocal VAE."""
    # Dimensions
    input_dim: int = 30  # UMAP output dimension
    latent_dim: int = 16  # VAE latent dimension
    hidden_dim: int = 128

    # Architecture
    encoder_layers: int = 2
    decoder_layers: int = 2
    dropout: float = 0.1
    use_layer_norm: bool = True

    # Training
    beta: float = 1.0  # β-VAE: >1 for disentanglement
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128

    # Training schedule
    epochs: int = 200
    early_stopping_patience: int = 20
    warmup_epochs: int = 10

    # KL annealing (optional)
    use_kl_annealing: bool = True
    kl_anneal_cycles: int = 5

    # Device
    device: str = "cuda"


class VocalVAE(nn.Module):
    """
    Variational Autoencoder for vocal manifold modeling.

    Architecture:
        Encoder: Input → Hidden → (mu, logvar)
        Sample: z ~ N(mu, exp(0.5 * logvar))
        Decoder: z → Hidden → Input

    The encoder and decoder can be exported separately to ONNX.
    """

    def __init__(self, config: Optional[VAEConfig] = None):
        super().__init__()
        if config is None:
            config = VAEConfig()

        self.config = config
        self.input_dim = config.input_dim
        self.latent_dim = config.latent_dim
        self.hidden_dim = config.hidden_dim
        self.beta = config.beta

        # Build encoder
        encoder_layers = []
        in_dim = self.input_dim
        for _ in range(config.encoder_layers):
            encoder_layers.append(nn.Linear(in_dim, config.hidden_dim))
            if config.use_layer_norm:
                encoder_layers.append(nn.LayerNorm(config.hidden_dim))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent distribution parameters
        self.fc_mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim, config.latent_dim)

        # Build decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for _ in range(config.decoder_layers):
            decoder_layers.append(nn.Linear(in_dim, config.hidden_dim))
            if config.use_layer_norm:
                decoder_layers.append(nn.LayerNorm(config.hidden_dim))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim

        decoder_layers.append(nn.Linear(config.hidden_dim, self.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input tensor (B, input_dim)

        Returns:
            mu: Mean of latent Gaussian (B, latent_dim)
            logvar: Log variance of latent Gaussian (B, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon
        where sigma = exp(0.5 * logvar)

        Args:
            mu: Mean (B, latent_dim)
            logvar: Log variance (B, latent_dim)

        Returns:
            z: Sampled latent vector (B, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent sample to reconstruction.

        Args:
            z: Latent vector (B, latent_dim)

        Returns:
            reconstruction: Reconstructed input (B, input_dim)
        """
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Input tensor (B, input_dim)

        Returns:
            reconstruction: Reconstructed input (B, input_dim)
            mu: Latent mean (B, latent_dim)
            logvar: Latent log variance (B, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def loss_function(
        self,
        reconstruction: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss = Reconstruction + β * KL Divergence

        Args:
            reconstruction: Reconstructed input
            x: Original input
            mu: Latent mean
            logvar: Latent log variance

        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss only
            kl_loss: KL divergence only
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x, reduction='sum')

        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # β-VAE: Weight KL by beta factor
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def sample(self, num_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Sample from the prior (standard normal) and decode.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate on

        Returns:
            samples: Generated samples (num_samples, input_dim)
        """
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples

    def interpolate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Interpolate between two inputs in latent space.

        Args:
            x1: First input (1, input_dim) or (input_dim,)
            x2: Second input (1, input_dim) or (input_dim,)
            num_steps: Number of interpolation steps

        Returns:
            interpolation: Interpolated samples (num_steps, input_dim)
        """
        self.eval()

        # Ensure batch dimension
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)

        with torch.no_grad():
            # Encode to latent space
            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)

            # Interpolate in latent space
            alphas = torch.linspace(0, 1, num_steps, device=x1.device)
            interpolation = []

            for alpha in alphas:
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                x_interp = self.decode(z_interp)
                interpolation.append(x_interp)

            return torch.cat(interpolation, dim=0)


class VocalVAETrainer:
    """
    Training loop for Vocal VAE with KL annealing.
    """

    def __init__(
        self,
        config: Optional[VAEConfig] = None,
    ):
        if config is None:
            config = VAEConfig()

        self.config = config
        self.device = self._get_device()

        self.model = VocalVAE(config).to(self.device)

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-6,
        )

    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if self.config.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _get_kl_weight(self, epoch: int) -> float:
        """
        Compute KL annealing weight.

        Cycles from 0 to 1 over the first few epochs to prevent
        posterior collapse during early training.
        """
        if not self.config.use_kl_annealing:
            return 1.0

        cycle_length = self.config.epochs // self.config.kl_anneal_cycles
        position = epoch % cycle_length
        weight = min(1.0, position / self.config.warmup_epochs)
        return weight

    def train(
        self,
        data_30d: np.ndarray,
        val_split: float = 0.1,
    ) -> dict:
        """
        Train VAE on UMAP-reduced data.

        Args:
            data_30d: Input UMAP embeddings (N, 30)
            val_split: Fraction of data for validation

        Returns:
            history: Training history dictionary
        """
        logger.info(f"Training Vocal VAE on {data_30d.shape[0]} samples")

        # Create dataset
        dataset = TensorDataset(torch.FloatTensor(data_30d))

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
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        # Training loop
        history = {
            'train_loss': [],
            'train_recon': [],
            'train_kl': [],
            'val_loss': [],
            'val_recon': [],
            'val_kl': [],
            'kl_weight': [],
        }

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_total = 0.0
            train_recon = 0.0
            train_kl = 0.0

            kl_weight = self._get_kl_weight(epoch)

            for (batch,) in train_loader:
                batch = batch.to(self.device)

                # Forward pass
                recon, mu, logvar = self.model(batch)

                # Compute loss with KL annealing
                total, recon_loss, kl_loss = self.model.loss_function(
                    recon, batch, mu, logvar
                )
                total = total / len(batch)  # Normalize by batch size
                total += (kl_weight - 1) * kl_loss / len(batch)

                # Backward pass
                self.optimizer.zero_grad()
                total.backward()
                self.optimizer.step()

                train_total += total.item() * len(batch)
                train_recon += recon_loss.item()
                train_kl += kl_loss.item()

            # Record training metrics
            n_train = len(train_dataset)
            history['train_loss'].append(train_total / n_train)
            history['train_recon'].append(train_recon / n_train)
            history['train_kl'].append(train_kl / n_train)
            history['kl_weight'].append(kl_weight)

            # Validation
            self.model.eval()
            val_total = 0.0
            val_recon = 0.0
            val_kl = 0.0

            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(self.device)
                    recon, mu, logvar = self.model(batch)
                    total, recon_loss, kl_loss = self.model.loss_function(
                        recon, batch, mu, logvar
                    )

                    val_total += total.item()
                    val_recon += recon_loss.item()
                    val_kl += kl_loss.item()

            n_val = len(val_dataset)
            if n_val > 0:
                history['val_loss'].append(val_total / n_val)
                history['val_recon'].append(val_recon / n_val)
                history['val_kl'].append(val_kl / n_val)
            else:
                # No validation data
                history['val_loss'].append(train_total / n_train)
                history['val_recon'].append(train_recon / n_train)
                history['val_kl'].append(train_kl / n_train)

            # Learning rate scheduling
            self.scheduler.step()

            # Logging
            if epoch % 20 == 0 or epoch == self.config.epochs - 1:
                if n_val > 0:
                    logger.info(
                        f"Epoch {epoch:3d}: "
                        f"Train={train_total/n_train:.4f}, "
                        f"Val={val_total/n_val:.4f}, "
                        f"KL_w={kl_weight:.2f}"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch:3d}: "
                        f"Train={train_total/n_train:.4f}, "
                        f"Val=N/A, "
                        f"KL_w={kl_weight:.2f}"
                    )

            # Early stopping (skip if no validation data)
            if n_val > 0:
                current_val_loss = val_total / n_val
            else:
                current_val_loss = train_total / n_train

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        logger.info("Training complete")
        return history

    def encode(self, data_30d: np.ndarray) -> np.ndarray:
        """
        Encode data to latent space using trained model.

        Args:
            data_30d: UMAP embeddings (N, 30)

        Returns:
            latent_16d: VAE latent coordinates (N, 16)
        """
        self.model.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data_30d).to(self.device)
            mu, _ = self.model.encode(data_tensor)
            return mu.cpu().numpy()

    def export_to_onnx(
        self,
        output_dir: str = "models/vae",
    ) -> Tuple[Path, Path]:
        """
        Export VAE encoder and decoder to ONNX format.

        Args:
            output_dir: Directory for ONNX files

        Returns:
            encoder_path: Path to encoder ONNX file
            decoder_path: Path to decoder ONNX file
        """
        import torch.onnx

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()

        # Export encoder
        logger.info("Exporting VAE encoder...")
        dummy_input_30d = torch.randn(1, self.config.input_dim).to(self.device)

        # We need to export just the encode part
        # Create a wrapper for encoder
        class EncoderWrapper(nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae

            def forward(self, x):
                mu, _ = self.vae.encode(x)
                return mu

        encoder_wrapper = EncoderWrapper(self.model)

        encoder_path = output_dir / "vae_encoder.onnx"
        torch.onnx.export(
            encoder_wrapper,
            dummy_input_30d,
            str(encoder_path),
            export_params=True,
            opset_version=17,
            input_names=['umap_coords'],
            output_names=['latent_coords'],
            dynamic_axes={
                'umap_coords': {0: 'batch_size'},
                'latent_coords': {0: 'batch_size'},
            },
        )
        logger.info(f"Exported encoder to {encoder_path}")

        # Export decoder
        logger.info("Exporting VAE decoder...")
        dummy_input_16d = torch.randn(1, self.config.latent_dim).to(self.device)

        decoder_path = output_dir / "vae_decoder.onnx"
        torch.onnx.export(
            self.model.decoder,
            dummy_input_16d,
            str(decoder_path),
            export_params=True,
            opset_version=17,
            input_names=['latent_coords'],
            output_names=['umap_coords'],
            dynamic_axes={
                'latent_coords': {0: 'batch_size'},
                'umap_coords': {0: 'batch_size'},
            },
        )
        logger.info(f"Exported decoder to {decoder_path}")

        return encoder_path, decoder_path


# Preset configurations

BETA_VAE_CONFIG = VAEConfig(
    input_dim=30,
    latent_dim=16,
    hidden_dim=128,
    beta=2.0,  # Higher β for disentanglement
    use_layer_norm=True,
    dropout=0.1,
)

STANDARD_VAE_CONFIG = VAEConfig(
    input_dim=30,
    latent_dim=16,
    hidden_dim=128,
    beta=1.0,  # Standard VAE
    use_layer_norm=True,
    dropout=0.1,
)


def create_vae_trainer(
    config: Optional[VAEConfig] = None,
    checkpoint_path: Optional[str] = None,
) -> VocalVAETrainer:
    """
    Factory function to create VAE trainer.

    Args:
        config: VAE configuration (uses BETA_VAE_CONFIG if None)
        checkpoint_path: Path to load trained model from

    Returns:
        Configured VocalVAETrainer
    """
    if config is None:
        config = BETA_VAE_CONFIG

    trainer = VocalVAETrainer(config)

    if checkpoint_path is not None:
        logger.info(f"Loading model from {checkpoint_path}")
        trainer.model.load_state_dict(
            torch.load(checkpoint_path, map_location=trainer.device)
        )

    return trainer


def compute_latent_statistics(
    latent_coords: np.ndarray,
) -> dict:
    """
    Compute statistics of the latent space.

    Args:
        latent_coords: VAE latent coordinates (N, latent_dim)

    Returns:
        Dictionary with latent space statistics
    """
    return {
        "mean": latent_coords.mean(axis=0).tolist(),
        "std": latent_coords.std(axis=0).tolist(),
        "min": latent_coords.min(axis=0).tolist(),
        "max": latent_coords.max(axis=0).tolist(),
        "shape": latent_coords.shape,
    }


def main():
    """Example training script."""
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic UMAP data for demonstration
    np.random.seed(42)
    n_samples = 5000
    data_30d = np.random.randn(n_samples, 30).astype(np.float32)

    # Add some structure (two clusters)
    data_30d[:2500] += 2.0

    # Create and train
    config = BETA_VAE_CONFIG
    config.epochs = 30  # Quick demo

    trainer = create_vae_trainer(config)
    history = trainer.train(data_30d)

    # Encode to latent space
    latent_16d = trainer.encode(data_30d)

    # Compute statistics
    stats = compute_latent_statistics(latent_16d)
    logger.info(f"Latent Statistics: {stats}")

    # Export to ONNX
    encoder_path, decoder_path = trainer.export_to_onnx("models/vae_demo")
    logger.info(f"Exported to {encoder_path} and {decoder_path}")


if __name__ == '__main__':
    main()
