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

Target latency: <50ms round-trip (features → audio)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import logging
import os
import time
from dataclasses import dataclass
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


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DDSPAgentConfig:
    """Configuration for real-time DDSP agent."""

    # Model paths
    decoder_path: str = "exports/ddsp_jetson/ddsp_decoder.onnx"
    synthesizer_path: str = "exports/ddsp_jetson/ddsp_synthesizer.onnx"
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
    decoder_path: str = "exports/ddsp_jetson/ddsp_decoder.onnx",
    synthesizer_path: str = "exports/ddsp_jetson/ddsp_synthesizer.onnx",
    use_tensorrt: bool = False,
    device: str = "cuda",
) -> RealtimeDDSPAgent:
    """
    Create a real-time DDSP agent with default settings.

    Args:
        decoder_path: Path to decoder model
        synthesizer_path: Path to synthesizer model
        use_tensorrt: Use TensorRT for inference
        device: Device to run on (cuda or cpu)

    Returns:
        Configured RealtimeDDSPAgent
    """
    config = DDSPAgentConfig(
        decoder_path=decoder_path,
        synthesizer_path=synthesizer_path,
        use_tensorrt=use_tensorrt,
        device=device,
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
