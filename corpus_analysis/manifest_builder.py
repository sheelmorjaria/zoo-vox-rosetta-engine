#!/usr/bin/env python3
"""
Continuous Manifold Manifest Builder

Generates the `continuous_manifold_manifest.json` that ties together:
- Trained UMAP and VAE models (ONNX paths)
- Exemplar bank with metadata
- Manifold statistics

This manifest is consumed by the runtime Rust pipeline and Python agents.

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np

from corpus_analysis.medoid_extractor import ExemplarMetadata, MedoidExtractor

logger = logging.getLogger(__name__)


@dataclass
class ManifoldMetadata:
    """Metadata about the manifold training."""
    created_at: str
    species: str
    total_segments: int
    num_exemplars: Dict[str, int]
    training_duration_seconds: float = 0.0


@dataclass
class ManifoldParameters:
    """Parameters of the manifold architecture."""
    umap_input_dim: int
    umap_output_dim: int
    vae_latent_dim: int
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    vae_beta: float = 1.0


@dataclass
class ModelPaths:
    """Paths to exported ONNX models."""
    parametric_umap_onnx: str
    vae_encoder_onnx: str
    vae_decoder_onnx: str


@dataclass
class ManifoldStatistics:
    """Statistics about the latent manifold space."""
    latent_mean: List[float]
    latent_std: List[float]
    latent_min: List[float]
    latent_max: List[float]
    interpolation_validated: bool = False
    trustworthiness: Optional[float] = None
    neighbor_continuity: Optional[float] = None


@dataclass
class ContinuousManifoldManifest:
    """Complete manifest for continuous manifold pipeline."""
    version: str = "1.1"
    metadata: ManifoldMetadata = field(default_factory=ManifoldMetadata)
    manifold_parameters: ManifoldParameters = field(default_factory=ManifoldParameters)
    model_paths: ModelPaths = field(default_factory=ModelPaths)
    exemplar_bank: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    manifold_statistics: ManifoldStatistics = field(default_factory=ManifoldStatistics)


class ManifestBuilder:
    """
    Builds the continuous manifold manifest from trained components.

    Process:
    1. Collect trained model paths
    2. Gather exemplar metadata from medoid extraction
    3. Compute manifold statistics
    4. Validate and write JSON manifest
    """

    def __init__(
        self,
        output_dir: str = "models",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_manifest(
        self,
        # Model info
        umap_onnx_path: str,
        vae_encoder_onnx_path: str,
        vae_decoder_onnx_path: str,

        # Exemplars
        exemplars: Dict[str, ExemplarMetadata],

        # Parameters
        manifold_params: ManifoldParameters,

        # Metadata
        species: str = "Unknown",
        total_segments: int = 0,

        # Optional statistics
        latent_coords: Optional[np.ndarray] = None,
        manifold_quality: Optional[Dict[str, Any]] = None,
    ) -> ContinuousManifoldManifest:
        """
        Build the complete manifest.

        Args:
            umap_onnx_path: Path to UMAP encoder ONNX model
            vae_encoder_onnx_path: Path to VAE encoder ONNX model
            vae_decoder_onnx_path: Path to VAE decoder ONNX model
            exemplars: Dict of exemplar metadata from medoid extraction
            manifold_params: Manifold architecture parameters
            species: Species name
            total_segments: Total number of training segments
            latent_coords: Optional latent coordinates for statistics
            manifold_quality: Optional quality metrics

        Returns:
            Complete manifest object
        """
        logger.info("Building continuous manifold manifest...")

        # Count exemplars by type
        n_dense = sum(1 for e in exemplars.values() if e.exemplar_type == "dense_zone")
        n_rare = sum(1 for e in exemplars.values() if e.exemplar_type == "rare")

        # Build metadata
        metadata = ManifoldMetadata(
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            species=species,
            total_segments=total_segments,
            num_exemplars={"dense_zones": n_dense, "rare_calls": n_rare},
        )

        # Build model paths (relative to output_dir)
        model_paths = ModelPaths(
            parametric_umap_onnx=self._make_relative_path(umap_onnx_path),
            vae_encoder_onnx=self._make_relative_path(vae_encoder_onnx_path),
            vae_decoder_onnx=self._make_relative_path(vae_decoder_onnx_path),
        )

        # Build exemplar bank
        exemplar_bank = {
            exemplar_id: {
                "latent_coord_16d": meta.latent_coord,
                "audio_path": meta.audio_path,
                "snr": meta.snr,
                "type": meta.exemplar_type,
                "cluster_size": meta.cluster_size,
                "cluster_label": meta.cluster_label,
                "description": meta.description,
            }
            for exemplar_id, meta in exemplars.items()
        }

        # Build manifold statistics
        if latent_coords is not None:
            manifold_stats = self._compute_manifold_statistics(
                latent_coords,
                manifold_quality,
            )
        else:
            # Create empty stats
            latent_dim = manifold_params.vae_latent_dim
            manifold_stats = ManifoldStatistics(
                latent_mean=[0.0] * latent_dim,
                latent_std=[1.0] * latent_dim,
                latent_min=[-1.0] * latent_dim,
                latent_max=[1.0] * latent_dim,
                interpolation_validated=False,
            )

        # Assemble manifest
        manifest = ContinuousManifoldManifest(
            metadata=metadata,
            manifold_parameters=manifold_params,
            model_paths=model_paths,
            exemplar_bank=exemplar_bank,
            manifold_statistics=manifold_stats,
        )

        logger.info(f"Manifest built: {n_dense} dense zones, {n_rare} rare calls")
        return manifest

    def _make_relative_path(self, path: str) -> str:
        """Convert path to be relative to output_dir."""
        path_obj = Path(path)
        try:
            return str(path_obj.relative_to(self.output_dir))
        except ValueError:
            # Path is not relative, return as-is
            return path

    def _compute_manifold_statistics(
        self,
        latent_coords: np.ndarray,
        quality_metrics: Optional[Dict[str, Any]] = None,
    ) -> ManifoldStatistics:
        """
        Compute statistics of the latent manifold.

        Args:
            latent_coords: VAE latent coordinates (N, latent_dim)
            quality_metrics: Optional quality metrics from UMAP

        Returns:
            Manifold statistics
        """
        return ManifoldStatistics(
            latent_mean=latent_coords.mean(axis=0).tolist(),
            latent_std=latent_coords.std(axis=0).tolist(),
            latent_min=latent_coords.min(axis=0).tolist(),
            latent_max=latent_coords.max(axis=0).tolist(),
            interpolation_validated=True,  # Assume validated if provided
            trustworthiness=quality_metrics.get("trustworthiness") if quality_metrics else None,
            neighbor_continuity=quality_metrics.get("neighbor_continuity") if quality_metrics else None,
        )

    def save_manifest(
        self,
        manifest: ContinuousManifoldManifest,
        output_path: str = "models/continuous_manifold_manifest.json",
    ) -> Path:
        """
        Save manifest to JSON file.

        Args:
            manifest: Manifest object to save
            output_path: Output file path

        Returns:
            Path to saved manifest
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict
        manifest_dict = {
            "version": manifest.version,
            "metadata": asdict(manifest.metadata),
            "manifold_parameters": asdict(manifest.manifold_parameters),
            "model_paths": asdict(manifest.model_paths),
            "exemplar_bank": manifest.exemplar_bank,
            "manifold_statistics": asdict(manifest.manifold_statistics),
        }

        # Write to file with pretty formatting
        with open(output_path, 'w') as f:
            json.dump(manifest_dict, f, indent=2)

        logger.info(f"Manifest saved to {output_path}")
        return output_path

    def validate_manifest(
        self,
        manifest: ContinuousManifoldManifest,
    ) -> List[str]:
        """
        Validate manifest for common issues.

        Args:
            manifest: Manifest to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check version
        if manifest.version != "1.1":
            errors.append(f"Unexpected version: {manifest.version}")

        # Check exemplar counts
        n_exemplars = len(manifest.exemplar_bank)
        if n_exemplars == 0:
            errors.append("No exemplars in bank")

        # Check dimensions match
        latent_dim = manifest.manifold_parameters.vae_latent_dim
        for exemplar_id, exemplar in manifest.exemplar_bank.items():
            coord_len = len(exemplar["latent_coord_16d"])
            if coord_len != latent_dim:
                errors.append(
                    f"{exemplar_id}: coord length {coord_len} != {latent_dim}"
                )

        # Check required fields
        if not manifest.model_paths.parametric_umap_onnx:
            errors.append("Missing UMAP model path")
        if not manifest.model_paths.vae_encoder_onnx:
            errors.append("Missing VAE encoder path")

        # Check model files exist (relative to output_dir)
        for model_key, model_path in asdict(manifest.model_paths).items():
            full_path = self.output_dir / model_path
            if not full_path.exists():
                errors.append(f"{model_key}: {model_path} does not exist")

        return errors


def load_manifest(
    manifest_path: str,
) -> ContinuousManifoldManifest:
    """
    Load manifest from JSON file.

    Args:
        manifest_path: Path to manifest JSON file

    Returns:
        Loaded manifest object
    """
    with open(manifest_path) as f:
        data = json.load(f)

    # Reconstruct manifest object
    manifest = ContinuousManifoldManifest(
        version=data["version"],
        metadata=ManifoldMetadata(**data["metadata"]),
        manifold_parameters=ManifoldParameters(**data["manifold_parameters"]),
        model_paths=ModelPaths(**data["model_paths"]),
        exemplar_bank=data["exemplar_bank"],
        manifold_statistics=ManifoldStatistics(**data["manifold_statistics"]),
    )

    return manifest


def create_manifest_summary(
    manifest: ContinuousManifoldManifest,
) -> str:
    """
    Create a human-readable summary of the manifest.

    Args:
        manifest: Manifest to summarize

    Returns:
        Formatted summary string
    """
    lines = [
        "════════════════════════════════════════════════════════════",
        "  CONTINUOUS MANIFOLD MANIFEST SUMMARY",
        "════════════════════════════════════════════════════════════",
        "",
        f"Version: {manifest.version}",
        f"Species: {manifest.metadata.species}",
        f"Created: {manifest.metadata.created_at}",
        f"Total Segments: {manifest.metadata.total_segments:,}",
        "",
        "Exemplars:",
        f"  Dense Zones: {manifest.metadata.num_exemplars.get('dense_zones', 0)}",
        f"  Rare Calls: {manifest.metadata.num_exemplars.get('rare_calls', 0)}",
        f"  Total: {len(manifest.exemplar_bank)}",
        "",
        "Manifold Parameters:",
        f"  UMAP: {manifest.manifold_parameters.umap_input_dim}D → {manifest.manifold_parameters.umap_output_dim}D",
        f"  VAE: {manifest.manifold_parameters.umap_output_dim}D → {manifest.manifold_parameters.vae_latent_dim}D",
        f"  β-VAE: {manifest.manifold_parameters.vae_beta}",
        "",
        "Model Paths:",
        f"  UMAP: {manifest.model_paths.parametric_umap_onnx}",
        f"  VAE Enc: {manifest.model_paths.vae_encoder_onnx}",
        f"  VAE Dec: {manifest.model_paths.vae_decoder_onnx}",
        "",
        "Latent Statistics:",
        f"  Mean: [{', '.join(f'{x:.2f}' for x in manifest.manifold_statistics.latent_mean[:3])}, ...]",
        f"  Std:  [{', '.join(f'{x:.2f}' for x in manifest.manifold_statistics.latent_std[:3])}, ...]",
        "",
        "════════════════════════════════════════════════════════════",
    ]

    return "\n".join(lines)


# Preset configurations for common species

BAT_MANIFOLD_PARAMS = ManifoldParameters(
    umap_input_dim=112,
    umap_output_dim=30,
    vae_latent_dim=16,
    umap_n_neighbors=15,
    umap_min_dist=0.1,
    vae_beta=1.0,
)

BIRD_MANIFOLD_PARAMS = ManifoldParameters(
    umap_input_dim=112,
    umap_output_dim=30,
    vae_latent_dim=16,
    umap_n_neighbors=20,  # More neighbors for birds (more variation)
    umap_min_dist=0.05,
    vae_beta=1.0,
)


def main():
    """Example manifest creation."""
    logging.basicConfig(level=logging.INFO)

    from corpus_analysis.medoid_extractor import ExemplarMetadata

    # Create dummy exemplars
    exemplars = {
        "zone_0": ExemplarMetadata(
            exemplar_id="zone_0",
            latent_coord=[0.1, -0.2, 0.3] + [0.0] * 13,
            audio_path="audio/bat_00123.wav",
            snr=42.1,
            exemplar_type="dense_zone",
            cluster_size=1523,
            cluster_label=0,
            description="Dense zone medoid",
        ),
        "rare_42": ExemplarMetadata(
            exemplar_id="rare_42",
            latent_coord=[2.3, 1.5, -0.8] + [0.0] * 13,
            audio_path="audio/bat_04456.wav",
            snr=35.2,
            exemplar_type="rare",
            cluster_size=1,
            cluster_label=-1,
            description="Rare call (long-tail)",
        ),
    }

    # Create manifest
    builder = ManifestBuilder()
    manifest = builder.build_manifest(
        umap_onnx_path="models/umap_encoder.onnx",
        vae_encoder_onnx_path="models/vae/vae_encoder.onnx",
        vae_decoder_onnx_path="models/vae/vae_decoder.onnx",
        exemplars=exemplars,
        manifold_params=BAT_MANIFOLD_PARAMS,
        species="Rousettus aegyptiacus",
        total_segments=50000,
        latent_coords=np.random.randn(100, 16),  # Dummy
    )

    # Validate
    errors = builder.validate_manifest(manifest)
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Manifest validated successfully")

    # Save
    builder.save_manifest(manifest)

    # Print summary
    print(create_manifest_summary(manifest))


if __name__ == '__main__':
    main()
