#!/usr/bin/env python3
"""
Real-time DDSP Agent for Jetson Deployment

Module 4 (v1.6.0): Real-time inference agent that runs DDSP synthesis
on NVIDIA Jetson devices with TensorRT optimization.

This agent:
- Receives 112D feature events from the cognitive layer
- Runs DDSP decoder to generate control parameters
- Runs DDSP synthesizer to generate PCM audio
- Publishes AudioBufferEvent via ZMQ for playback by Rust layer
- Auto-detects Jetson device type and uses appropriate configuration

Target latency: <50ms round-trip (features → audio)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Check for dependencies
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. DDSP Agent disabled.")

try:
    import zmq

    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    logger.warning("ZMQ not available. IPC disabled.")

# Import Jetson device types from jetson_export module
try:
    from cognitive_intelligence.jetson_export import (
        JetsonDevice,
    )
    from cognitive_intelligence.jetson_export import (
        detect_jetson_device as _export_detect_jetson_device,
    )

    # Use the imported function
    detect_jetson_device = _export_detect_jetson_device
except ImportError:
    # Fallback definitions if jetson_export is not available
    class JetsonDevice(Enum):  # type: ignore
        """Detected Jetson device types."""

        NANO = "nano"
        XAVIER = "xavier"
        ORIN = "orin"
        UNKNOWN = "unknown"

    def detect_jetson_device() -> JetsonDevice:  # type: ignore
        """Fallback device detection."""
        tegra_release = Path("/etc/nv_tegra_release")
        if not tegra_release.exists():
            return JetsonDevice.UNKNOWN
        return JetsonDevice.UNKNOWN


def get_config_for_device(device: Optional[JetsonDevice] = None) -> "DDSPAgentConfig":
    """
    Get the appropriate DDSPAgentConfig for a specific device.

    Args:
        device: Device type (auto-detect if None)

    Returns:
        DDSPAgentConfig configured for the device
    """
    if device is None:
        device = detect_jetson_device()

    base_dir = "exports/ddsp_jetson"

    configs = {
        JetsonDevice.NANO: DDSPAgentConfig(
            decoder_path=f"{base_dir}/nano_fp32/ddsp_decoder.onnx",
            synthesizer_path=f"{base_dir}/nano_fp32/ddsp_synthesizer.onnx",
            use_tensorrt=False,
            fp16=False,
            num_harmonics=40,
            num_noise_bands=3,
            target_latency_ms=30.0,
        ),
        JetsonDevice.XAVIER: DDSPAgentConfig(
            decoder_path=f"{base_dir}/xavier_fp16/ddsp_decoder.trt",
            synthesizer_path=f"{base_dir}/xavier_fp16/ddsp_synthesizer.trt",
            use_tensorrt=True,
            fp16=True,
            num_harmonics=60,
            num_noise_bands=5,
            target_latency_ms=12.0,
        ),
        JetsonDevice.ORIN: DDSPAgentConfig(
            decoder_path=f"{base_dir}/orin_fp16_postfilter/ddsp_decoder.trt",
            synthesizer_path=f"{base_dir}/orin_fp16_postfilter/ddsp_synthesizer.trt",
            use_tensorrt=True,
            fp16=True,
            num_harmonics=60,
            num_noise_bands=5,
            enable_post_filter=True,
            target_latency_ms=15.0,
        ),
        JetsonDevice.UNKNOWN: DDSPAgentConfig(
            decoder_path=f"{base_dir}/universal_fp32/ddsp_decoder.onnx",
            synthesizer_path=f"{base_dir}/universal_fp32/ddsp_synthesizer.onnx",
            use_tensorrt=False,
            fp16=False,
            num_harmonics=40,
            num_noise_bands=3,
            target_latency_ms=50.0,
        ),
    }

    return configs.get(device, configs[JetsonDevice.UNKNOWN])


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DDSPAgentConfig:
    """Configuration for real-time DDSP agent."""

    # Model paths
    decoder_path: str = "exports/ddsp_jetson/ddsp_decoder.onnx"
    synthesizer_path: str = "exports/ddsp_jetson/ddsp_synthesizer.onnx"
    post_filter_path: str = ""  # Optional neural post-filter (Orin tier)
    synthesis_manifest_path: str = "technical_architecture/data/synthesis_manifest.json"

    # Inference settings
    use_tensorrt: bool = False
    fp16: bool = True
    device: str = "cuda"  # cuda or cpu

    # Audio settings
    sample_rate: int = 48000
    target_latency_ms: float = 50.0
    max_audio_duration_ms: float = 500.0

    # Synthesis settings
    num_harmonics: int = 60
    num_noise_bands: int = 5
    hop_size: int = 480

    # Post-filter (Orin tier only)
    enable_post_filter: bool = False

    # ZMQ settings
    feature_sub_port: int = 5556
    audio_pub_port: int = 5557
    heartbeat_pub_port: int = 5555

    # Cluster vocabulary
    cluster_centroids_112d: Optional[np.ndarray] = None

    # Performance monitoring
    enable_profiling: bool = True
    log_every_n_frames: int = 100


# =============================================================================
# Neural Post-Filter (Orin tier only)
# =============================================================================

if TORCH_AVAILABLE:

    class NeuralPostFilter(nn.Module):
        """
        Lightweight neural post-filter for DDSP audio refinement.

        This small network takes DDSP-generated audio and the DDSP parameters,
        then outputs refined audio with improved quality. Designed for Orin
        devices where we have GPU headroom for additional processing.

        Architecture: 1D convolutions with skip connections
        """

        def __init__(self, num_harmonics: int = 60, num_noise_bands: int = 5):
            super().__init__()
            self.num_harmonics = num_harmonics
            self.num_noise_bands = num_noise_bands

            # Input channels: audio (1) + param embedding (16)
            in_channels = 1 + 16

            # Light convolutional network
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(32, 32, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(32, 16, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(16, 1, kernel_size=7, padding=3),
                nn.Tanh(),
            )

            # Parameter embedding network
            self.param_embed = nn.Sequential(
                nn.Linear(num_harmonics + num_noise_bands, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
            )

        def forward(
            self, audio: torch.Tensor, harmonic_amps: torch.Tensor, noise_mags: torch.Tensor
        ) -> torch.Tensor:
            """
            Refine DDSP audio with neural post-filter.

            Args:
                audio: DDSP-generated audio (B, T)
                harmonic_amps: Harmonic amplitudes (B, 60)
                noise_mags: Noise magnitudes (B, 5)

            Returns:
                Refined audio (B, T)
            """
            audio_length = audio.shape[-1]

            # Embed parameters
            params = torch.cat([harmonic_amps, noise_mags], dim=-1)
            param_emb = self.param_embed(params)  # (B, 16)

            # Expand param embedding to match audio length
            param_emb_expanded = param_emb.unsqueeze(-1).expand(-1, -1, audio_length)  # (B, 16, T)

            # Stack audio with param embedding
            audio_input = audio.unsqueeze(1)  # (B, 1, T)
            combined = torch.cat([audio_input, param_emb_expanded], dim=1)  # (B, 17, T)

            # Run through network
            refinement = self.net(combined)  # (B, 1, T)

            # Add refinement to original audio (residual connection)
            refined = audio + refinement.squeeze(1)  # (B, T)

            return refined


# =============================================================================
# Real-time DDSP Agent
# =============================================================================

if TORCH_AVAILABLE:

    class RealtimeDDSPAgent:
        """
        Real-time DDSP synthesis agent for Jetson deployment.

        This agent runs an inference loop that:
        1. Receives 112D feature events via ZMQ
        2. Applies feature deltas to cluster centroids
        3. Runs DDSP decoder → synthesizer pipeline
        4. Publishes PCM audio via AudioBufferEvent

        Target: <50ms latency from feature to audio
        """

        def __init__(self, config: DDSPAgentConfig):
            """
            Initialize real-time DDSP agent.

            Args:
                config: Agent configuration
            """
            self.config = config
            self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

            # Load models
            self.decoder = self._load_decoder()
            self.synthesizer = self._load_synthesizer()

            # Load post-filter if enabled (Orin tier)
            self.post_filter = None
            if config.enable_post_filter:
                self.post_filter = self._load_post_filter()

            # Load cluster centroids
            self.centroids = self._load_centroids()

            # Setup ZMQ
            if ZMQ_AVAILABLE:
                self._setup_zmq()

            # State
            self.running = False
            self.frame_count = 0
            self.latency_samples: List[float] = []

            logger.info(f"RealtimeDDSPAgent initialized on {self.device}")
            logger.info(f"Target latency: {config.target_latency_ms}ms")
            if self.post_filter is not None:
                logger.info("Neural post-filter enabled (Orin tier)")

        def _load_decoder(self) -> nn.Module:
            """Load DDSP decoder model."""
            from cognitive_intelligence.ddsp_decoder import DDSPDecoder

            if os.path.exists(self.config.decoder_path):
                # Try to load exported model
                try:
                    import onnxruntime  # noqa: F401

                    logger.info(f"Loading ONNX decoder: {self.config.decoder_path}")
                    # For now, load PyTorch model
                    # ONNX/TensorRT loading would go here
                except ImportError:
                    logger.info("ONNX Runtime not available, using PyTorch")

            # Load PyTorch model
            decoder = DDSPDecoder().to(self.device)
            decoder.eval()

            logger.info("DDSP Decoder loaded (PyTorch)")
            return decoder

        def _load_synthesizer(self) -> nn.Module:
            """Load DDSP synthesizer model."""
            from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

            synthesizer = DDSPSynthesizer(
                sample_rate=self.config.sample_rate,
                num_harmonics=self.config.num_harmonics,
                num_noise_bands=self.config.num_noise_bands,
                hop_size=self.config.hop_size,
            ).to(self.device)
            synthesizer.eval()

            logger.info("DDSP Synthesizer loaded (PyTorch)")
            return synthesizer

        def _load_post_filter(self) -> Optional[nn.Module]:
            """Load neural post-filter model (Orin tier only)."""
            post_filter_path = self.config.post_filter_path

            # Try to load from specified path
            if post_filter_path and os.path.exists(post_filter_path):
                try:
                    post_filter = torch.load(post_filter_path, map_location=self.device)
                    post_filter.eval()
                    logger.info(f"Neural post-filter loaded: {post_filter_path}")
                    return post_filter
                except Exception as e:
                    logger.warning(f"Failed to load post-filter: {e}")

            # Create new post-filter if none exists
            logger.info("Creating new neural post-filter")
            post_filter = NeuralPostFilter(
                num_harmonics=self.config.num_harmonics,
                num_noise_bands=self.config.num_noise_bands,
            ).to(self.device)
            post_filter.eval()

            return post_filter

        def _load_centroids(self) -> np.ndarray:
            """Load cluster centroids for synthesis."""
            manifest_path = self.config.synthesis_manifest_path

            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)

                centroids = []
                for cluster in manifest.get("clusters", []):
                    centroid_112d = cluster.get("centroid_112d")
                    if centroid_112d:
                        centroids.append(centroid_112d)

                if centroids:
                    centroids_array = np.array(centroids, dtype=np.float32)
                    logger.info(f"Loaded {len(centroids)} cluster centroids")
                    return centroids_array

            # Fallback: create default centroids
            logger.warning("No centroids found, using default")
            return np.eye(112, dtype=np.float32)[:45]  # 45 default clusters

        def _setup_zmq(self):
            """Setup ZMQ sockets for IPC."""
            context = zmq.Context()

            # Subscribe to feature events
            self.feature_socket = context.socket(zmq.SUB)
            if self.config.feature_sub_port == 0:
                # For testing: bind to random port instead of connecting
                self.feature_port = self.feature_socket.bind_to_random_port(
                    "tcp://*", min_port=49152, max_port=65536
                )
            else:
                self.feature_socket.connect(f"tcp://localhost:{self.config.feature_sub_port}")
                self.feature_port = self.config.feature_sub_port
            self.feature_socket.setsockopt_string(zmq.SUBSCRIBE, "")

            # Publish audio buffers - use random port if 0 specified
            self.audio_socket = context.socket(zmq.PUB)
            if self.config.audio_pub_port == 0:
                self.audio_port = self.audio_socket.bind_to_random_port(
                    "tcp://*", min_port=49152, max_port=65536
                )
            else:
                self.audio_socket.bind(f"tcp://*:{self.config.audio_pub_port}")
                self.audio_port = self.config.audio_pub_port

            # Heartbeat publisher - use random port if 0 specified
            self.heartbeat_socket = context.socket(zmq.PUB)
            if self.config.heartbeat_pub_port == 0:
                self.heartbeat_port = self.heartbeat_socket.bind_to_random_port(
                    "tcp://*", min_port=49152, max_port=65536
                )
            else:
                self.heartbeat_socket.bind(f"tcp://*:{self.config.heartbeat_pub_port}")
                self.heartbeat_port = self.config.heartbeat_pub_port

            logger.info(  # noqa: E501
                f"ZMQ sockets configured (audio: {self.audio_port}, "
                f"heartbeat: {self.heartbeat_port})"
            )

        def synthesize_from_features(
            self,
            features_112d: np.ndarray,
            duration_ms: float = 200.0,
            base_f0: float = 6000.0,
        ) -> Tuple[np.ndarray, float]:
            """
            Synthesize audio from 112D features.

            Args:
                features_112d: Input feature vector (112,)
                duration_ms: Output duration in milliseconds
                base_f0: Base fundamental frequency in Hz

            Returns:
                audio: PCM audio samples
                latency_ms: Actual inference latency
            """
            start_time = time.perf_counter()

            # Convert to tensor
            features_tensor = torch.from_numpy(features_112d).float().unsqueeze(0).to(self.device)

            # Run decoder
            with torch.no_grad():
                harmonic_amps, noise_mags = self.decoder(features_tensor)

                # Expand to time dimension
                n_frames = int(duration_ms / 10)  # 10ms frames
                harmonic_amps = harmonic_amps.unsqueeze(1).expand(1, n_frames, -1)
                noise_mags = noise_mags.unsqueeze(1).expand(1, n_frames, -1)

                # Create F0 trajectory
                f0 = torch.ones(1, n_frames, device=self.device) * base_f0

                # Run synthesizer
                audio, _ = self.synthesizer(f0, harmonic_amps, noise_mags)

                # Apply neural post-filter if available (Orin tier)
                if self.post_filter is not None:
                    # Get mean parameters for the sequence
                    mean_harmonic = harmonic_amps.mean(dim=1)  # (B, 60)
                    mean_noise = noise_mags.mean(dim=1)  # (B, 5)

                    # Apply post-filter
                    audio = self.post_filter(audio.squeeze(1), mean_harmonic, mean_noise)

            # Convert to numpy
            audio_np = audio.squeeze(0).cpu().numpy()

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Track statistics
            self.frame_count += 1
            self.latency_samples.append(latency_ms)
            if len(self.latency_samples) > 1000:
                self.latency_samples.pop(0)

            return audio_np, latency_ms

        def synthesize_from_cluster(
            self,
            cluster_id: int,
            delta_112d: Optional[np.ndarray] = None,
            duration_ms: float = 200.0,
        ) -> Tuple[np.ndarray, float]:
            """
            Synthesize audio from cluster ID with optional delta.

            Args:
                cluster_id: Cluster ID from vocabulary
                delta_112d: Optional feature modification (112,)
                duration_ms: Output duration in milliseconds

            Returns:
                audio: PCM audio samples
                latency_ms: Actual inference latency
            """
            # Get centroid
            if cluster_id < len(self.centroids):
                features = self.centroids[cluster_id].copy()
            else:
                # Use first centroid if ID out of range
                features = self.centroids[0].copy()

            # Apply delta
            if delta_112d is not None:
                delta_112d = np.asarray(delta_112d, dtype=np.float32)
                if delta_112d.shape == (112,):
                    features = features + delta_112d

            # Derive F0 from features (use first few features)
            # In practice, this would be learned from the data
            base_f0 = 6000 + features[0] * 2000
            base_f0 = np.clip(base_f0, 3000, 15000)

            return self.synthesize_from_features(features, duration_ms, base_f0)

        def publish_audio_buffer(
            self,
            audio: np.ndarray,
            sequence: int,
        ):
            """Publish audio buffer via ZMQ."""
            if not ZMQ_AVAILABLE:
                return

            from realtime.action_publisher import AudioBufferEvent

            event = AudioBufferEvent(
                audio_data=audio,
                sample_rate=self.config.sample_rate,
                duration_ms=len(audio) / self.config.sample_rate * 1000,
                timestamp=time.time(),
                sequence=sequence,
            )

            # Publish JSON
            self.audio_socket.send_string(event.to_json())

        def send_heartbeat(self):
            """Send heartbeat signal."""
            if not ZMQ_AVAILABLE:
                return

            heartbeat = {
                "timestamp": time.time(),
                "status": "running",
                "latency_ms": np.mean(self.latency_samples) if self.latency_samples else 0,
            }

            self.heartbeat_socket.send_json(heartbeat)

        def run(self, num_iterations: int = -1):
            """
            Run the real-time inference loop.

            Args:
                num_iterations: Number of iterations (-1 for infinite)
            """
            self.running = True
            self.frame_count = 0

            logger.info("Starting real-time DDSP synthesis loop")

            try:
                while self.running and (num_iterations < 0 or self.frame_count < num_iterations):
                    # Receive feature event (with timeout for heartbeat)
                    try:
                        if ZMQ_AVAILABLE:
                            # Poll with timeout
                            if self.feature_socket.poll(timeout=100):
                                message = self.feature_socket.recv_json()
                                audio, latency = self._handle_feature_event(message)
                            else:
                                # Timeout - send heartbeat
                                self.send_heartbeat()
                                continue
                        else:
                            # Simulated mode
                            time.sleep(0.1)
                            continue

                    except zmq.ZMQError:
                        continue

                    # Update statistics
                    self.latency_samples.append(latency)
                    if len(self.latency_samples) > 1000:
                        self.latency_samples.pop(0)

                    self.frame_count += 1

                    # Log progress
                    if self.frame_count % self.config.log_every_n_frames == 0:
                        avg_latency = np.mean(self.latency_samples)
                        logger.info(f"Frame {self.frame_count} | Latency: {avg_latency:.2f}ms")

            except KeyboardInterrupt:
                logger.info("Interrupted by user")

            finally:
                self.running = False
                logger.info("Real-time DDSP synthesis loop stopped")

        def _handle_feature_event(self, message: Dict) -> Tuple[np.ndarray, float]:
            """Handle incoming feature event."""
            # Extract parameters
            cluster_id = message.get("cluster_id", 0)
            delta_112d = message.get("delta_112d")
            duration_ms = message.get("duration_ms", 200.0)

            # Synthesize
            audio, latency = self.synthesize_from_cluster(
                cluster_id=cluster_id,
                delta_112d=delta_112d,
                duration_ms=duration_ms,
            )

            # Publish audio
            self.publish_audio_buffer(audio, self.frame_count)

            return audio, latency

        def stop(self):
            """Stop the inference loop."""
            self.running = False

        def get_statistics(self) -> Dict:
            """Get performance statistics."""
            return {
                "frame_count": self.frame_count,
                "avg_latency_ms": np.mean(self.latency_samples) if self.latency_samples else 0,
                "min_latency_ms": np.min(self.latency_samples) if self.latency_samples else 0,
                "max_latency_ms": np.max(self.latency_samples) if self.latency_samples else 0,
                "target_latency_ms": self.config.target_latency_ms,
                "meets_target": (
                    np.mean(self.latency_samples) < self.config.target_latency_ms
                    if self.latency_samples
                    else False
                ),
            }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_ddsp_agent(
    decoder_path: Optional[str] = None,
    synthesizer_path: Optional[str] = None,
    use_tensorrt: Optional[bool] = None,
    device: Optional[str] = None,
    auto_detect: bool = True,
) -> RealtimeDDSPAgent:
    """
    Create a real-time DDSP agent with auto-detected or manual settings.

    Args:
        decoder_path: Path to decoder model (auto-detect if None)
        synthesizer_path: Path to synthesizer model (auto-detect if None)
        use_tensorrt: Use TensorRT for inference (auto-detect if None)
        device: Device to run on (cuda or cpu, auto-detect if None)
        auto_detect: Auto-detect Jetson device and use appropriate config

    Returns:
        Configured RealtimeDDSPAgent
    """
    if auto_detect:
        # Auto-detect Jetson device and use appropriate config
        config = get_config_for_device()

        # Override with manual settings if provided
        if decoder_path is not None:
            config.decoder_path = decoder_path
        if synthesizer_path is not None:
            config.synthesizer_path = synthesizer_path
        if use_tensorrt is not None:
            config.use_tensorrt = use_tensorrt
        if device is not None:
            config.device = device
    else:
        # Use manual configuration
        config = DDSPAgentConfig(
            decoder_path=decoder_path or "exports/ddsp_jetson/ddsp_decoder.onnx",
            synthesizer_path=synthesizer_path or "exports/ddsp_jetson/ddsp_synthesizer.onnx",
            use_tensorrt=use_tensorrt or False,
            device=device or "cuda",
        )

    return RealtimeDDSPAgent(config)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if not TORCH_AVAILABLE:
        print("PyTorch not available. Install with: pip install torch")
        exit(1)

    print("Real-time DDSP Agent for Jetson Deployment")
    print("=" * 60)

    # Create agent
    agent = create_ddsp_agent()

    # Run a single synthesis test
    print("\nTesting synthesis...")

    # Test with cluster 0
    audio, latency = agent.synthesize_from_cluster(
        cluster_id=0,
        duration_ms=100.0,
    )

    print(f"Generated audio: {len(audio)} samples ({len(audio) / 48000 * 1000:.1f}ms)")
    print(f"Latency: {latency:.2f}ms")

    # Show statistics
    stats = agent.get_statistics()
    print(f"\nTarget latency: {stats['target_latency_ms']}ms")
    print(f"Actual latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"Target met: {stats['meets_target']}")

    print("\nFor real-time operation, run with ZMQ connected.")
