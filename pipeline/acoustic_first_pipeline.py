#!/usr/bin/env python3
"""
Acoustic-First Pipeline: Foundational Paradigms Implementation

This pipeline implements the two non-negotiable pillars:
1. Acoustic-First Paradigm: Raw acoustic physics as primary substrate of meaning
2. Intra-Call Paradigm: Micro-modulations within single vocalization boundaries

Pipeline Stages:
    Stage 1: Self-Supervised Predictive Boundary Detection (CPC)
        → Segments continuous audio into semantic units
        → Sub-50ms latency via Mamba/TCN autoregressive modeling

    Stage 2: BioMAE Feature Extraction & Reconstruction
        → Log-linear spectrograms (NO mel-warping for ultrasonic preservation)
        → 112D Rosetta features via masked autoencoder

    Stage 3: Dual-Stream Encoding
        → Stream 1 (Affective): pUMAP + β-VAE → 16D disentangled latent
        → Stream 2 (Syntactic): VQ-VAE → 64 discrete tokens

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn

# Stage 1: CPC
from boundary_detection.cpc_encoder import CPCEncoder, EncoderConfig
from boundary_detection.cpc_autoregressive import AutoregressiveMamba, TCNAutoregressive
from boundary_detection.predictive_boundary import (
    PredictiveBoundaryDetector,
    BoundaryDetectorConfig,
    BoundaryEvent,
)

# Stage 2: BioMAE
from feature_extraction.bio_spectrogram import UltrasonicSpectrogram, SpectrogramConfig
from feature_extraction.biomae import BioMAEModel, EncoderConfig as BioMAEEncoderConfig

# Stage 3: Dual-Stream
from cognitive_intelligence.affective_encoder import AffectiveFeatureExtractor
from cognitive_intelligence.affective_pumap_vae import AffectiveStream, AffectiveConfig
from cognitive_intelligence.syntactic_vqvae import SyntacticVQVAE, VQVAEConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the complete Acoustic-First Pipeline."""

    # ===== Stage 1: CPC Configuration =====
    cpc_sample_rate: int = 96000
    cpc_frame_size_ms: int = 10
    cpc_hidden_dim: int = 128
    cpc_channels: Tuple[int, ...] = (64, 128, 256)
    cpc_kernels: Tuple[int, ...] = (5, 5, 3)
    cpc_strides: Tuple[int, ...] = (2, 2, 1)

    # Autoregressive model (Mamba or TCN)
    use_mamba: bool = True  # Fall back to TCN if unavailable
    ar_d_model: int = 128
    ar_d_state: int = 16
    ar_d_conv: int = 4

    # Boundary detection
    boundary_threshold: float = 2.5
    boundary_threshold_lower: float = 1.5
    slow_decay: float = 0.99
    fast_decay: float = 0.9
    min_segment_duration_ms: float = 10.0

    # ===== Stage 2: BioMAE Configuration =====
    biomae_sample_rate: int = 96000
    biomae_n_fft: int = 1024
    biomae_hop_length: int = 240
    biomae_img_size: Tuple[int, int] = (128, 128)
    biomae_patch_size: Tuple[int, int] = (16, 16)
    biomae_embed_dim: int = 256
    biomae_depth: int = 4
    biomae_num_heads: int = 4
    biomae_mask_ratio: float = 0.75

    # ===== Stage 3: Dual-Stream Configuration =====
    # Affective Stream
    affective_input_dim: int = 54  # From AffectiveFeatureExtractor
    affective_pumap_hidden: Tuple[int, ...] = (256, 128)
    affective_pumap_output: int = 30
    affective_vae_latent: int = 16
    affective_beta: float = 2.0

    # Syntactic Stream
    syntactic_input_dim: int = 44  # Complement to affective (112 - 54 - 14 structural)
    syntactic_codebook_size: int = 64
    syntactic_codebook_dim: int = 32
    syntactic_commitment_cost: float = 0.25

    # ===== Energy-Based Segmentation (Fallback) =====
    enable_energy_segmentation: bool = True  # Use energy-based detection when CPC misses boundaries
    energy_drop_threshold_db: float = 5.0  # Energy drop > this = potential boundary
    energy_min_phrase_duration_ms: float = 50.0  # Minimum phrase duration
    energy_silence_threshold_db: float = -40.0  # Energy below this = silence

    # ===== General Configuration =====
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4
    num_workers: int = 0


@dataclass
class PipelineOutput:
    """Output from the Acoustic-First Pipeline."""

    # Stage 1 outputs
    boundaries: List[Tuple[float, float]] = field(default_factory=list)  # List of (start_ms, end_ms)
    cpc_latents: Optional[np.ndarray] = None  # CPC latent representations

    # Stage 2 outputs
    features_112d: Optional[np.ndarray] = None  # BioMAE 112D features per segment
    recon_loss: float = 0.0

    # Stage 3 outputs
    affective_latent_16d: Optional[np.ndarray] = None  # Affective stream
    syntactic_tokens: Optional[List[int]] = None  # Syntactic stream discrete tokens
    perplexity: float = 0.0

    # Metadata
    processing_time_ms: float = 0.0
    segment_count: int = 0


class AcousticFirstPipeline(nn.Module):
    """
    Complete Acoustic-First Pipeline implementing foundational paradigms.

    Pipeline Flow:
        Raw Audio (continuous stream)
            ↓
        Stage 1: CPC Boundary Detection
            ├── 1D Conv Encoder (64→128→256 channels)
            ├── Autoregressive Model (Mamba/TCN)
            └── Boundary Detection via MSE spikes
            ↓
        Segmented Audio Clips
            ↓
        Stage 2: BioMAE Feature Extraction
            ├── Log-Linear Spectrogram (NO mel-warping)
            ├── ViT Patching (16×16)
            ├── 75% Masking (training only)
            └── 112D Rosetta Features
            ↓
        Stage 3: Dual-Stream Encoding
            ├── Stream 1: 54D Affective → pUMAP → β-VAE → 16D
            └── Stream 2: 44D Syntactic → VQ-VAE → 64 tokens

    Example:
        >>> pipeline = AcousticFirstPipeline()
        >>> audio = np.load('bat_vocalization.wav')
        >>> output = pipeline.process_audio(audio)
        >>> print(f"Detected {output.segment_count} segments")
        >>> print(f"Affective latent: {output.affective_latent_16d.shape}")
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__()

        if config is None:
            config = PipelineConfig()

        self.config = config
        self.device = torch.device(config.device)

        # ===== Stage 1: CPC Components =====
        self.cpc_encoder = CPCEncoder(
            sample_rate=config.cpc_sample_rate,
            frame_size_ms=config.cpc_frame_size_ms,
            hidden_dim=config.cpc_hidden_dim,
            num_channels=config.cpc_channels,
            kernel_sizes=config.cpc_kernels,
            strides=config.cpc_strides,
        ).to(self.device)

        # Autoregressive model
        if config.use_mamba:
            try:
                import mamba_ssm
                self.ar_model = AutoregressiveMamba(
                    d_model=config.ar_d_model,
                    d_state=config.ar_d_state,
                    d_conv=config.ar_d_conv,
                    use_mamba=True,
                ).to(self.device)
                logger.info("Using Mamba for autoregressive modeling")
            except ImportError:
                logger.warning("Mamba unavailable, falling back to TCN")
                self.ar_model = TCNAutoregressive(d_model=config.ar_d_model).to(self.device)
        else:
            self.ar_model = TCNAutoregressive(d_model=config.ar_d_model).to(self.device)

        # Predictive boundary detector (Green Phase)
        nbd_config = BoundaryDetectorConfig(
            # Detection thresholds (Duration-Gated)
            boundary_threshold=config.boundary_threshold,
            boundary_threshold_lower=config.boundary_threshold_lower,
            syllable_threshold=3.0,
            phrase_threshold=4.0,

            # Duration requirements
            phonetic_duration_ms=10.0,
            syllable_duration_ms=30.0,
            phrase_duration_ms=80.0,

            # Derivative-based spike detection
            derivative_threshold=0.5,
            derivative_window_ms=20.0,

            # Dual-EMA baseline tracking
            baseline_window=100,
            slow_decay=config.slow_decay,
            fast_decay=config.fast_decay,

            # Armed logic
            rearm_threshold=1.2,
            rearm_trigger_threshold=1.5,

            # Confidence scoring
            min_confidence=0.6,

            # Timing
            frame_size_ms=float(config.cpc_frame_size_ms),
            sample_rate=config.cpc_sample_rate,
        )
        self.boundary_detector = PredictiveBoundaryDetector(
            config=nbd_config,
            cpc_model=None,  # Will compute error directly
        )

        # ===== Stage 2: BioMAE Components =====
        self.spectrogram = UltrasonicSpectrogram(SpectrogramConfig(
            sample_rate=config.biomae_sample_rate,
            n_fft=config.biomae_n_fft,
            hop_length=config.biomae_hop_length,
        )).to(self.device)

        biomae_encoder_config = BioMAEEncoderConfig(
            img_size=config.biomae_img_size,
            patch_size=config.biomae_patch_size,
            embed_dim=config.biomae_embed_dim,
            depth=config.biomae_depth,
            num_heads=config.biomae_num_heads,
            output_dim=112,  # Rosetta dimension
        )

        from feature_extraction.biomae import DecoderConfig
        biomae_decoder_config = DecoderConfig(
            embed_dim=config.biomae_embed_dim,
            patch_size=config.biomae_patch_size,
            img_size=config.biomae_img_size,
        )

        self.biomae = BioMAEModel(biomae_encoder_config, biomae_decoder_config).to(self.device)

        # ===== Stage 3: Dual-Stream Components =====
        # Affective stream
        self.affective_stream = AffectiveStream(AffectiveConfig(
            input_dim=config.affective_input_dim,
            pumap_hidden=config.affective_pumap_hidden,
            pumap_output=config.affective_pumap_output,
            vae_latent=config.affective_vae_latent,
            beta=config.affective_beta,
        )).to(self.device).eval()  # Set to eval mode for inference

        # Syntactic stream
        self.syntactic_vqvae = SyntacticVQVAE(
            input_dim=config.syntactic_input_dim,
            codebook_size=config.syntactic_codebook_size,
            codebook_dim=config.syntactic_codebook_dim,
            commitment_cost=config.syntactic_commitment_cost,
        ).to(self.device).eval()  # Set to eval mode for inference

        logger.info(
            f"AcousticFirstPipeline initialized on {self.device}:\n"
            f"  Stage 1: CPC ({config.cpc_hidden_dim}D latent, {config.cpc_sample_rate}Hz)\n"
            f"  Stage 2: BioMAE (112D output)\n"
            f"  Stage 3: Dual-Stream (Affective: {config.affective_vae_latent}D, "
            f"Syntactic: {config.syntactic_codebook_size} tokens)"
        )

    def detect_boundaries(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[Tuple[float, float]]:
        """
        Stage 1: Detect vocalization boundaries using Predictive NBD (Green Phase).

        Uses CPC encoder + autoregressive model to compute prediction errors,
        then applies PredictiveBoundaryDetector with dual-EMA baseline and
        duration-gated confidence for sub-50ms boundary detection.

        Args:
            audio: Raw audio samples (num_samples,)
            sample_rate: Sample rate in Hz

        Returns:
            List of (start_ms, end_ms) boundary pairs
        """
        import time
        start_time = time.time()

        # Convert to tensor
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio).float()

        audio = audio.to(self.device)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Process through CPC encoder frame by frame
        frame_size = int(sample_rate * self.config.cpc_frame_size_ms / 1000)
        num_frames = (audio.shape[1] - frame_size) // frame_size

        # Store boundary events
        boundary_events = []
        segment_start_ms = 0.0  # Track where current segment started

        # Process each frame through PredictiveBoundaryDetector
        for i in range(num_frames):
            start = i * frame_size
            end = start + frame_size
            frame = audio[:, start:end]

            # Ensure frame is the right size
            if frame.shape[1] < frame_size:
                break

            current_time_ms = i * self.config.cpc_frame_size_ms
            timestamp_ns = int(current_time_ms * 1_000_000)

            with torch.no_grad():
                # Add channel dimension for CPC encoder: (B, T) -> (B, 1, T)
                frame_channel = frame.unsqueeze(1)

                # Encode current frame
                z = self.cpc_encoder(frame_channel)  # (B, T', hidden_dim)

                # Compute prediction error
                if i < num_frames - 1:
                    next_frame = audio[:, end:end+frame_size]
                    if next_frame.shape[1] == frame_size:
                        # Add channel dimension for next frame
                        next_frame_channel = next_frame.unsqueeze(1)
                        z_next = self.cpc_encoder(next_frame_channel)

                        # Use autoregressive model to predict next latent
                        # z is already (B, T', hidden_dim) from CPC encoder
                        z_pred = self.ar_model(z)  # (B, T', hidden_dim)

                        # Compute prediction error (MSE between predicted and actual)
                        prediction_error = torch.mean((z_pred - z_next) ** 2).item()

                        # Process through PredictiveBoundaryDetector using pre-computed error
                        result = self.boundary_detector.process_frame_with_error(
                            error=prediction_error,
                            timestamp_ns=timestamp_ns,
                        )

                        # Check if boundary detected
                        if result.is_boundary and result.boundary_type is not None:
                            # Calculate segment end
                            segment_end_ms = current_time_ms + self.config.cpc_frame_size_ms

                            # Only add if segment is long enough
                            segment_duration = segment_end_ms - segment_start_ms
                            if segment_duration >= self.config.min_segment_duration_ms:
                                boundary_events.append((segment_start_ms, segment_end_ms))

                            # Start new segment
                            segment_start_ms = segment_end_ms

        # Handle final segment
        if segment_start_ms < num_frames * self.config.cpc_frame_size_ms:
            final_end_ms = num_frames * self.config.cpc_frame_size_ms
            if final_end_ms - segment_start_ms >= self.config.min_segment_duration_ms:
                boundary_events.append((segment_start_ms, final_end_ms))

        processing_time = (time.time() - start_time) * 1000
        logger.debug(
            f"Predictive NBD: {len(boundary_events)} segments in {processing_time:.1f}ms "
            f"(boundaries: {self.boundary_detector.boundary_count}, "
            f"types: {self.boundary_detector.boundary_types})"
        )

        # Fallback to energy-based segmentation if:
        # 1. Only 1 segment detected (possibly missed internal boundaries)
        # 2. Energy-based segmentation is enabled
        # 3. Audio is long enough to contain multiple phrases (>500ms)
        audio_duration_ms = len(audio.squeeze().cpu().numpy()) / sample_rate * 1000
        if (len(boundary_events) <= 1 and
            self.config.enable_energy_segmentation and
            audio_duration_ms > 500):
            logger.info("Only 1 segment detected, trying energy-based phrase segmentation...")
            energy_boundaries = self._detect_energy_boundaries(audio, sample_rate)
            if len(energy_boundaries) > 1:
                logger.info(f"Energy-based segmentation found {len(energy_boundaries)} phrases")
                return energy_boundaries

        return boundary_events

    def _detect_energy_boundaries(
        self,
        audio: torch.Tensor,
        sample_rate: int
    ) -> List[Tuple[float, float]]:
        """
        Fallback energy-based phrase segmentation.

        Detects phrase boundaries based on energy drops in the spectrogram.
        Useful when CPC-based NBD misses boundaries in continuous vocalizations.

        Args:
            audio: Audio tensor (B, T) or (T,)
            sample_rate: Sample rate in Hz

        Returns:
            List of (start_ms, end_ms) phrase boundary pairs
        """
        import numpy as np

        # Ensure audio is 1D numpy for spectrogram
        audio_np = audio.squeeze().cpu().numpy()

        # Compute spectrogram
        with torch.no_grad():
            spec = self.spectrogram(torch.from_numpy(audio_np).float().to(self.device))
            spec_db = spec.squeeze().cpu().numpy()  # (Freq, Time)

        # Compute energy over time (mean across frequency)
        energy_over_time = spec_db.mean(axis=0)

        # Find energy drops (potential boundaries)
        energy_diff = np.diff(energy_over_time)

        # Smooth the derivative to reduce noise
        window_size = 5
        energy_diff_smooth = np.convolve(energy_diff, np.ones(window_size)/window_size, mode='same')

        # Find significant drops
        drop_threshold = -self.config.energy_drop_threshold_db
        drop_indices = np.where(energy_diff_smooth < drop_threshold)[0]

        # Convert frame indices to time
        n_fft = self.config.biomae_n_fft
        hop_length = self.config.biomae_hop_length
        time_per_frame_ms = (hop_length / sample_rate) * 1000

        # Build boundary list
        boundaries = []
        start_ms = 0.0

        for idx in drop_indices:
            end_ms = idx * time_per_frame_ms

            # Check minimum phrase duration
            phrase_duration = end_ms - start_ms
            if phrase_duration >= self.config.energy_min_phrase_duration_ms:
                boundaries.append((start_ms, end_ms))
                start_ms = end_ms

        # Add final phrase
        total_duration_ms = len(audio_np) / sample_rate * 1000
        if total_duration_ms - start_ms >= self.config.energy_min_phrase_duration_ms:
            boundaries.append((start_ms, total_duration_ms))

        return boundaries

    def extract_features_112d(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        Stage 2: Extract 112D BioMAE features from audio segment.

        Args:
            audio_segment: Audio samples for one segment

        Returns:
            112D feature vector
        """
        # Convert to tensor
        if not isinstance(audio_segment, torch.Tensor):
            audio_segment = torch.from_numpy(audio_segment).float()

        audio_segment = audio_segment.to(self.device)
        if audio_segment.dim() == 1:
            audio_segment = audio_segment.unsqueeze(0)

        # Compute log-linear spectrogram
        with torch.no_grad():
            spec = self.spectrogram(audio_segment)

            # BioMAE expects (Batch, Channels, Freq, Time)
            # Add channel dimension if not present
            if spec.dim() == 3:
                spec = spec.unsqueeze(1)  # (Batch, 1, Freq, Time)

            # Resize to expected input size if needed
            if spec.shape[-2:] != self.config.biomae_img_size:
                spec = torch.nn.functional.interpolate(
                    spec,
                    size=self.config.biomae_img_size,
                    mode='bilinear',
                    align_corners=False,
                )

            # Extract 112D features
            features_112d = self.biomae.encode(spec)

        return features_112d.cpu().numpy()

    def encode_dual_stream(self, features_112d: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Stage 3: Dual-stream encoding of 112D features.

        Args:
            features_112d: 112D Rosetta features

        Returns:
            (affective_16d, syntactic_tokens)
        """
        # Convert to tensor
        if not isinstance(features_112d, torch.Tensor):
            features_112d = torch.from_numpy(features_112d).float()

        features_112d = features_112d.to(self.device)
        if features_112d.dim() == 1:
            features_112d = features_112d.unsqueeze(0)

        with torch.no_grad():
            # Extract affective features (54D from 112D)
            affective_features = AffectiveFeatureExtractor.extract_affective_features(
                features_112d.cpu().numpy()
            )
            affective_tensor = torch.from_numpy(affective_features).float().to(self.device)

            # Stream 1: Affective encoding → 16D
            affective_16d = self.affective_stream.encode(affective_tensor)

            # Stream 2: Syntactic encoding → 64 tokens
            # For syntactic, we use a different subset of the 112D features
            # This would typically be the structural/categorical features
            syntactic_features = features_112d[:, :44]  # Simplified: use first 44 dims
            syntactic_tokens = self.syntactic_vqvae.tokenize(syntactic_features)

            perplexity = self.syntactic_vqvae.codebook_utilization()

        return (
            affective_16d.cpu().numpy(),
            syntactic_tokens.cpu().numpy().tolist(),
            perplexity,
        )

    def process_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        return_intermediates: bool = False,
    ) -> PipelineOutput:
        """
        Process audio through the complete pipeline.

        Args:
            audio: Raw audio samples
            sample_rate: Sample rate in Hz
            return_intermediates: If True, return intermediate representations

        Returns:
            PipelineOutput with all stage results
        """
        import time
        start_time = time.time()

        output = PipelineOutput()

        # Stage 1: Boundary detection
        output.boundaries = self.detect_boundaries(audio, sample_rate)
        output.segment_count = len(output.boundaries)

        if output.segment_count == 0:
            logger.warning("No boundaries detected, returning early")
            return output

        # Stage 2 & 3: Process each segment
        affective_latents = []
        all_syntactic_tokens = []
        features_list = []

        for start_ms, end_ms in output.boundaries:
            start_sample = int(start_ms * sample_rate / 1000)
            end_sample = int(end_ms * sample_rate / 1000)
            segment = audio[start_sample:end_sample]

            # Skip if too short
            if len(segment) < self.config.biomae_n_fft:
                continue

            # Stage 2: Extract 112D features
            features_112d = self.extract_features_112d(segment)
            features_list.append(features_112d)

            # Stage 3: Dual-stream encoding
            affective_16d, syntactic_tokens, perplexity = self.encode_dual_stream(features_112d)
            affective_latents.append(affective_16d)
            all_syntactic_tokens.extend(syntactic_tokens)
            output.perplexity = max(output.perplexity, perplexity)

        # Aggregate outputs
        if features_list:
            output.features_112d = np.array(features_list)
            output.affective_latent_16d = np.array(affective_latents)
            output.syntactic_tokens = all_syntactic_tokens

        output.processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Pipeline complete: {output.segment_count} segments, "
            f"{output.processing_time_ms:.1f}ms processing time"
        )

        return output

    def train_step(
        self,
        audio_batch: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Training step for the pipeline (simplified).

        Args:
            audio_batch: Batch of audio samples (B, T)
            targets: Optional target values for supervised losses

        Returns:
            Dictionary of loss values
        """
        self.train()

        losses = {}

        # Stage 1: CPC training (InfoNCE loss would go here)
        # This is a placeholder - actual CPC training requires future samples
        z = self.cpc_encoder(audio_batch)
        losses['cpc_loss'] = 0.0  # Placeholder

        # Stage 2: BioMAE training (reconstruction loss)
        # Compute spectrogram
        spec = self.spectrogram(audio_batch)
        if spec.shape[-2:] != self.config.biomae_img_size:
            spec = torch.nn.functional.interpolate(
                spec.unsqueeze(1),
                size=self.config.biomae_img_size,
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)

        # Generate mask and train
        mask = self.biomae.generate_random_mask(spec.shape[0], spec.device)
        spec_recon, embedding = self.biomae(spec, mask)
        biomae_loss = F.mse_loss(spec_recon, spec)
        losses['biomae_recon_loss'] = biomae_loss.item()

        # Stage 3: Dual-stream training
        # Affective stream
        affective_features = AffectiveFeatureExtractor.extract_affective_features(
            embedding.cpu().numpy()
        )
        affective_tensor = torch.from_numpy(affective_features).float().to(self.device)

        x_recon, mu, logvar, z_pumap = self.affective_stream(affective_tensor)
        affective_loss, affective_losses = self.affective_stream.loss_function(
            affective_tensor, x_recon, mu, logvar, z_pumap, z_pumap
        )
        losses['affective_loss'] = affective_losses['total_loss']

        # Syntactic stream (VQ-VAE)
        syntactic_features = embedding[:, :44]
        x_recon_syn, z_syn, z_q_syn, token_ids, perplexity = self.syntactic_vqvae(syntactic_features)
        syntactic_losses = self.syntactic_vqvae.loss_function(syntactic_features, x_recon_syn, z_syn, z_q_syn)
        losses['syntactic_loss'] = syntactic_losses['total_loss'].item()
        losses['codebook_utilization'] = self.syntactic_vqvae.codebook_utilization()

        # Total loss
        total_loss = (
            biomae_loss +
            affective_loss +
            syntactic_losses['total_loss']
        )
        losses['total_loss'] = total_loss.item()

        return losses

    def save_checkpoint(self, path: Path) -> None:
        """Save pipeline checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'config': self.config,
            'cpc_encoder': self.cpc_encoder.state_dict(),
            'ar_model': self.ar_model.state_dict(),
            'biomae': self.biomae.state_dict(),
            'affective_stream': self.affective_stream.state_dict(),
            'syntactic_vqvae': self.syntactic_vqvae.state_dict(),
        }, path)

        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load pipeline checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)

        self.cpc_encoder.load_state_dict(checkpoint['cpc_encoder'])
        self.ar_model.load_state_dict(checkpoint['ar_model'])
        self.biomae.load_state_dict(checkpoint['biomae'])
        self.affective_stream.load_state_dict(checkpoint['affective_stream'])
        self.syntactic_vqvae.load_state_dict(checkpoint['syntactic_vqvae'])

        logger.info(f"Loaded checkpoint from {path}")


def create_pipeline(config: Optional[PipelineConfig] = None) -> AcousticFirstPipeline:
    """Factory function to create Acoustic-First Pipeline."""
    if config is None:
        config = PipelineConfig()
    return AcousticFirstPipeline(config)


# Preset configurations

BAT_PIPELINE = PipelineConfig(
    cpc_sample_rate=250000,  # Egyptian fruit bat recordings at 250kHz
    biomae_sample_rate=250000,  # Preserve ultrasonic content (Nyquist = 125kHz)
    device="cuda",
)

BIRD_PIPELINE = PipelineConfig(
    cpc_sample_rate=48000,
    biomae_sample_rate=48000,
    device="cuda",
)

MINIMAL_PIPELINE = PipelineConfig(
    cpc_sample_rate=48000,
    biomae_sample_rate=48000,
    device="cpu",
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test pipeline
    pipeline = create_pipeline(MINIMAL_PIPELINE)

    # Generate test audio
    sample_rate = 48000
    duration = 1.0  # 1 second
    audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

    # Process
    output = pipeline.process_audio(audio, sample_rate)

    print(f"\n{'='*60}")
    print(f"Acoustic-First Pipeline Test Results")
    print(f"{'='*60}")
    print(f"Segments detected: {output.segment_count}")
    print(f"Processing time: {output.processing_time_ms:.1f}ms")
    if output.affective_latent_16d is not None:
        print(f"Affective latent shape: {output.affective_latent_16d.shape}")
    if output.syntactic_tokens is not None:
        print(f"Syntactic tokens: {len(output.syntactic_tokens)} tokens")
    print(f"Codebook utilization: {output.perplexity:.1f}%")
    print(f"{'='*60}")
