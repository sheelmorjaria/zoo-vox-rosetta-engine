#!/usr/bin/env python3
"""
Offline Validation (Sprint 1-2 Milestone)

Validates that 112D features can be reconstructed by combining
VAE output (continuous affect) and VQ-VAE output (discrete syntax).

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from cognitive_intelligence.affective_feature_extractor import AffectiveFeatureExtractor
from cognitive_intelligence.affective_vae import BetaVAE
from cognitive_intelligence.syntactic_feature_extractor import SyntacticFeatureExtractor
from cognitive_intelligence.syntactic_vqvae import SyntacticVQVAE

logger = logging.getLogger(__name__)


class DualStreamReconstructor:
    """
    Reconstructs 112D features from dual-stream encodings.

    Pipeline:
        112D → AffectiveExtractor (30D) → β-VAE → 16D latent → decode → 30D
        112D → SyntacticExtractor (44D) → VQ-VAE → token → decode → 44D
        30D + 44D → 112D reconstruction

    Validation Metrics:
    - Reconstruction MSE (target: < 0.1)
    - Affective reconstruction error
    - Syntactic reconstruction error
    - Feature-wise errors
    """

    def __init__(
        self,
        affective_extractor: AffectiveFeatureExtractor,
        syntactic_extractor: SyntacticFeatureExtractor,
        beta_vae: BetaVAE,
        vqvae: SyntacticVQVAE,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.affective_extractor = affective_extractor
        self.syntactic_extractor = syntactic_extractor
        self.beta_vae = beta_vae.to(device).eval()
        self.vqvae = vqvae.to(device).eval()
        self.device = torch.device(device)

        # Mapping from extracted features back to 112D
        self.affective_indices = affective_extractor.config.feature_indices
        self.syntactic_indices = syntactic_extractor.config.feature_indices

        logger.info("DualStreamReconstructor initialized")

    def encode(
        self,
        features_112d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode 112D features to dual-stream representations.

        Returns:
            (affect_16d, token) - Affect latent and discrete token
        """
        with torch.no_grad():
            # Extract features
            affective_features = self.affective_extractor.extract(features_112d)
            syntactic_features = self.syntactic_extractor.extract(features_112d)

            # Encode with β-VAE
            affective_tensor = torch.from_numpy(affective_features).float().unsqueeze(0).to(self.device)
            mu, _ = self.beta_vae.encode(affective_tensor)  # encode returns (mu, logvar)
            affect_16d = mu.cpu().numpy()[0]

            # Encode with VQ-VAE
            syntactic_tensor = torch.from_numpy(syntactic_features).float().unsqueeze(0).to(self.device)
            z_syntactic = self.vqvae.encode(syntactic_tensor)
            _, token_ids, _ = self.vqvae.vq(z_syntactic, training=False)
            token = token_ids.cpu().numpy()[0].item()

        return affect_16d, token

    def decode(
        self,
        affect_16d: np.ndarray,
        token: int,
    ) -> np.ndarray:
        """
        Decode dual-stream representations to 112D features.

        Returns:
            Reconstructed 112D features
        """
        with torch.no_grad():
            # Decode affect
            affect_tensor = torch.from_numpy(affect_16d).float().unsqueeze(0).to(self.device)
            z_affect_decoded = self.beta_vae.decode(affect_tensor)
            affective_recon = z_affect_decoded.cpu().numpy()[0]

            # Decode syntax (get codebook vector for token)
            # token_tensor needs to be indexed properly to get 1D codebook vector
            codebook_vector = self.vqvae.vq.codebook_ema[token].unsqueeze(0)
            syntactic_recon = self.vqvae.decode(codebook_vector).cpu().numpy()[0]

            # Combine to 112D
            reconstructed = np.zeros(112, dtype=np.float32)

            # Fill affective features
            for i, idx in enumerate(self.affective_indices):
                reconstructed[idx] = affective_recon[i]

            # Fill syntactic features
            for i, idx in enumerate(self.syntactic_indices):
                reconstructed[idx] = syntactic_recon[i]

        return reconstructed

    def reconstruct(
        self,
        features_112d: np.ndarray,
    ) -> np.ndarray:
        """Full encode-decode pipeline."""
        affect_16d, token = self.encode(features_112d)
        return self.decode(affect_16d, token)

    def compute_reconstruction_error(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute reconstruction error metrics.

        Returns:
            Dictionary with error metrics
        """
        mse = np.mean((original - reconstructed) ** 2)
        mae = np.mean(np.abs(original - reconstructed))

        # Feature-wise errors
        affective_mask = np.zeros(112, dtype=bool)
        for idx in self.affective_indices:
            affective_mask[idx] = True

        syntactic_mask = np.zeros(112, dtype=bool)
        for idx in self.syntactic_indices:
            syntactic_mask[idx] = True

        # Affective error
        affective_mse = np.mean((original[affective_mask] - reconstructed[affective_mask]) ** 2)

        # Syntactic error
        syntactic_mse = np.mean((original[syntactic_mask] - reconstructed[syntactic_mask]) ** 2)

        return {
            "mse": float(mse),
            "mae": float(mae),
            "affective_mse": float(affective_mse),
            "syntactic_mse": float(syntactic_mse),
        }

    def validate_reconstruction(
        self,
        features_list: List[np.ndarray],
        target_mse: float = 0.1,
    ) -> Dict:
        """
        Validate reconstruction on a set of features.

        Returns:
            Validation results dictionary
        """
        total_mse = 0.0
        total_affective_mse = 0.0
        total_syntactic_mse = 0.0
        num_samples = len(features_list)

        for features in features_list:
            reconstructed = self.reconstruct(features)
            errors = self.compute_reconstruction_error(features, reconstructed)

            total_mse += errors["mse"]
            total_affective_mse += errors["affective_mse"]
            total_syntactic_mse += errors["syntactic_mse"]

        avg_mse = total_mse / num_samples
        avg_affective_mse = total_affective_mse / num_samples
        avg_syntactic_mse = total_syntactic_mse / num_samples

        target_met = avg_mse < target_mse

        logger.info(
            f"Validation Results: "
            f"MSE={avg_mse:.4f} (target: <{target_mse}), "
            f"Affective MSE={avg_affective_mse:.4f}, "
            f"Syntactic MSE={avg_syntactic_mse:.4f}, "
            f"Target Met: {target_met}"
        )

        return {
            "avg_mse": avg_mse,
            "avg_affective_mse": avg_affective_mse,
            "avg_syntactic_mse": avg_syntactic_mse,
            "target_mse": target_mse,
            "target_met": target_met,
            "num_samples": num_samples,
        }


def run_offline_validation(
    beta_vae_path: str,
    vqvae_path: str,
    features_112d: np.ndarray,
    affective_stats_path: Optional[str] = None,
    syntactic_stats_path: Optional[str] = None,
) -> Dict:
    """
    Run offline validation of dual-stream reconstruction.

    Args:
        beta_vae_path: Path to trained β-VAE checkpoint
        vqvae_path: Path to trained VQ-VAE checkpoint
        features_112d: Array of shape (N, 112) for validation
        affective_stats_path: Path to affective normalization stats
        syntactic_stats_path: Path to syntactic normalization stats

    Returns:
        Validation results dictionary
    """
    logger.info("Starting offline validation...")

    # Create feature extractors
    affective_extractor = AffectiveFeatureExtractor()
    syntactic_extractor = SyntacticFeatureExtractor()

    # Get actual dimensions from extractors
    affective_dim = affective_extractor.output_dim
    syntactic_dim = syntactic_extractor.output_dim

    # Load normalization stats
    if affective_stats_path:
        affective_extractor.load_normalization_stats(affective_stats_path)
    if syntactic_stats_path:
        syntactic_extractor.load_normalization_stats(syntactic_stats_path)

    # Load models with correct dimensions
    device = "cuda" if torch.cuda.is_available() else "cpu"

    beta_vae_checkpoint = torch.load(beta_vae_path, map_location=device)
    # Create model with correct dimensions
    beta_vae = BetaVAE(input_dim=affective_dim, latent_dim=16, hidden_dim=32)
    beta_vae.load_state_dict(beta_vae_checkpoint["model_state_dict"])
    beta_vae.eval()

    vqvae_checkpoint = torch.load(vqvae_path, map_location=device)
    # Create model with correct dimensions
    vqvae = SyntacticVQVAE(input_dim=syntactic_dim, codebook_size=16, codebook_dim=16, hidden_dim=32)
    vqvae.load_state_dict(vqvae_checkpoint["model_state_dict"])
    vqvae.eval()

    # Create reconstructor
    reconstructor = DualStreamReconstructor(
        affective_extractor,
        syntactic_extractor,
        beta_vae,
        vqvae,
        device=device,
    )

    # Run validation
    features_list = [features_112d[i] for i in range(len(features_112d))]
    results = reconstructor.validate_reconstruction(features_list)

    logger.info(f"Offline validation complete: {results}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create dummy models for testing
    from cognitive_intelligence.affective_vae import create_beta_vae
    from cognitive_intelligence.syntactic_vqvae import create_syntactic_vqvae

    beta_vae = create_beta_vae()
    vqvae = create_syntactic_vqvae()

    # Create feature extractors
    affective_extractor = AffectiveFeatureExtractor()
    syntactic_extractor = SyntacticFeatureExtractor()

    # Create reconstructor
    reconstructor = DualStreamReconstructor(
        affective_extractor,
        syntactic_extractor,
        beta_vae,
        vqvae,
    )

    # Test with dummy data
    dummy_features = np.random.randn(112).astype(np.float32)
    reconstructed = reconstructor.reconstruct(dummy_features)
    errors = reconstructor.compute_reconstruction_error(dummy_features, reconstructed)

    print(f"Reconstruction MSE: {errors['mse']:.4f}")
    print(f"Affective MSE: {errors['affective_mse']:.4f}")
    print(f"Syntactic MSE: {errors['syntactic_mse']:.4f}")
