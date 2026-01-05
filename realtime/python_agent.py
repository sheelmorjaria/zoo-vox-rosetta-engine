#!/usr/bin/env python3
"""
Python Agent Logic for Hybrid Architecture.
Implements the "soft real-time" layer with cognitive intelligence.
"""

import asyncio
import zmq
import msgpack
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import threading
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PythonAgent")

class AgentState(Enum):
    """Probabilistic contextual agent states"""
    SILENCE = "silence"
    CONTACT = "contact"
    ALARM = "alarm"
    FOOD = "food"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"

@dataclass
class AudioFeatures:
    """Audio features from Rust engine"""
    rms: float
    zcr: float
    spectral_centroid: float
    timestamp: int
    mel_spectrogram: List[List[float]]
    mfcc: List[float]
    f0_estimate: Optional[float]

@dataclass
class SynthesisCommand:
    """Command sent to Rust engine for synthesis"""
    phrase_id: int
    pitch_shift: float
    time_stretch: float
    gain: float
    emotional_state: Optional[Dict[str, float]]

class ContextualAgent:
    """Probabilistic contextual agent with 6 states"""

    def __init__(self):
        # Context probabilities for each state
        self.state_probabilities = {
            AgentState.SILENCE: 0.1,
            AgentState.CONTACT: 0.1,
            AgentState.ALARM: 0.1,
            AgentState.FOOD: 0.1,
            AgentState.NEUTRAL: 0.5,
            AgentState.UNCERTAIN: 0.1
        }

        # Context history for learning
        self.context_history = deque(maxlen=1000)
        self.phrase_context_associations = {}
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.3

        # Behavioral patterns
        self.response_patterns = {
            AgentState.CONTACT: {"urgency": 0.7, "aggression": 0.2, "playfulness": 0.8},
            AgentState.ALARM: {"urgency": 0.9, "aggression": 0.8, "playfulness": 0.1},
            AgentState.FOOD: {"urgency": 0.4, "aggression": 0.1, "playfulness": 0.6},
            AgentState.NEUTRAL: {"urgency": 0.3, "aggression": 0.2, "playfulness": 0.5},
            AgentState.SILENCE: {"urgency": 0.1, "aggression": 0.0, "playfulness": 0.2},
            AgentState.UNCERTAIN: {"urgency": 0.5, "aggression": 0.3, "playfulness": 0.4}
        }

    def process_audio_features(self, features: AudioFeatures, context: str = "unknown") -> Dict[str, Any]:
        """Process audio features and update context state"""
        # Extract meaningful features
        feature_vector = np.array([
            features.rms,
            features.zcr,
            features.spectral_centroid / 10000.0,  # Normalize
            features.f0_estimate / 10000.0 if features.f0_estimate else 0.0
        ])

        # Update context history
        self.context_history.append({
            "timestamp": features.timestamp,
            "context": context,
            "features": feature_vector.tolist(),
            "state_probs": self.state_probabilities.copy()
        })

        # Perform context inference (simplified)
        self._update_context_inference(feature_vector, context)

        # Determine most likely state
        most_likely_state = max(self.state_probabilities.items(), key=lambda x: x[1])[0]

        logger.info(f"Context inferred: {most_likely_state.value} (prob: {self.state_probabilities[most_likely_state]:.3f})")

        return {
            "most_likely_state": most_likely_state.value,
            "state_probabilities": {s.value: p for s, p in self.state_probabilities.items()},
            "confidence": self.state_probabilities[most_likely_state],
            "context": context
        }

    def _update_context_inference(self, feature_vector: np.ndarray, context: str):
        """Update context inference based on features"""
        # Simple rule-based inference (in production, use ML model)
        rms, zcr, centroid, f0 = feature_vector

        # Update probabilities based on features
        if rms > 0.5:  # Loud sound
            self.state_probabilities[AgentState.ALARM] += 0.2
            self.state_probabilities[AgentState.CONTACT] += 0.1

        if zcr > 0.3:  # Noisy sound
            self.state_probabilities[AgentState.ALARM] += 0.1
            self.state_probabilities[AgentState.UNCERTAIN] += 0.1

        if centroid > 0.5:  # High frequency
            self.state_probabilities[AgentState.ALARM] += 0.1

        if f0 > 0.3:  # Clear pitch
            self.state_probabilities[AgentState.CONTACT] += 0.2
            self.state_probabilities[AgentState.FOOD] += 0.1

        # Apply context-specific updates
        if context == "contact_call":
            self.state_probabilities[AgentState.CONTACT] += 0.4
        elif context == "alarm_call":
            self.state_probabilities[AgentState.ALARM] += 0.5
        elif context == "food_call":
            self.state_probabilities[AgentState.FOOD] += 0.4
        elif context == "neutral":
            self.state_probabilities[AgentState.NEUTRAL] += 0.3

        # Normalize probabilities
        total = sum(self.state_probabilities.values())
        for state in self.state_probabilities:
            self.state_probabilities[state] /= total

        # Ensure minimum probabilities
        for state in self.state_probabilities:
            self.state_probabilities[state] = max(0.01, self.state_probabilities[state])

    def get_emotional_parameters(self) -> Dict[str, float]:
        """Get emotional parameters based on current state"""
        # Weighted average of all states
        emotional_params = {"urgency": 0.0, "aggression": 0.0, "playfulness": 0.0, "fear": 0.0}

        for state, probability in self.state_probabilities.items():
            if state in self.response_patterns:
                for param, value in self.response_patterns[state].items():
                    emotional_params[param] += probability * value

        # Add fear component (inverse of playfulness for simplicity)
        emotional_params["fear"] = 1.0 - emotional_params["playfulness"]

        return emotional_params

    def update_phrase_context_association(self, phrase_id: int, context: str, response_positive: bool = True):
        """Update phrase-to-context associations for learning"""
        if phrase_id not in self.phrase_context_associations:
            self.phrase_context_associations[phrase_id] = {}

        if context not in self.phrase_context_associations[phrase_id]:
            self.phrase_context_associations[phrase_id][context] = {
                "count": 0,
                "success_rate": 0.0,
                "total_responses": 0,
                "successful_responses": 0
            }

        association = self.phrase_context_associations[phrase_id][context]
        association["count"] += 1
        association["total_responses"] += 1

        if response_positive:
            association["successful_responses"] += 1

        association["success_rate"] = association["successful_responses"] / association["total_responses"]

        logger.info(f"Updated phrase {phrase_id} context association for {context}: "
                   f"success_rate = {association['success_rate']:.3f}")

class CognitiveProcessor:
    """Advanced cognitive processing layer"""

    def __init__(self):
        self.agent = ContextualAgent()
        self.long_term_memory = {}
        self.attention_focus = 0.5  # Attention level (0.0 to 1.0)
        self.working_memory = deque(maxlen=50)
        self.adaptation_threshold = 0.7

    def process_audio_cognitively(self, features: AudioFeatures,
                                 additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process audio through cognitive framework"""
        start_time = time.time()

        # Extract context from additional data
        context = additional_context.get("context", "unknown") if additional_context else "unknown"

        # Process through contextual agent
        context_result = self.agent.process_audio_features(features, context)

        # Update working memory
        self.working_memory.append({
            "timestamp": features.timestamp,
            "context": context,
            "context_result": context_result,
            "features": asdict(features)
        })

        # Determine synthesis parameters
        emotional_params = self.agent.get_emotional_parameters()

        # Generate synthesis command
        synthesis_cmd = self._generate_synthesis_command(context_result, emotional_params)

        # Update long-term memory if this is a meaningful event
        if context_result["confidence"] > self.adaptation_threshold:
            self._update_long_term_memory(features, context_result, context)

        processing_time = (time.time() - start_time) * 1000  # ms

        return {
            "processing_complete": True,
            "processing_time_ms": processing_time,
            "context_inference": context_result,
            "emotional_parameters": emotional_params,
            "synthesis_command": synthesis_cmd,
            "attention_level": self.attention_focus,
            "working_memory_size": len(self.working_memory)
        }

    def _generate_synthesis_command(self, context_result: Dict,
                                  emotional_params: Dict[str, float]) -> SynthesisCommand:
        """Generate synthesis command based on context and emotional state"""
        # Map context to phrase IDs (simplified mapping)
        context_to_phrase = {
            "contact_call": 1,
            "alarm_call": 2,
            "food_call": 3,
            "neutral": 4,
            "unknown": 4
        }

        most_likely_state = context_result["most_likely_state"]
        phrase_id = context_to_phrase.get(most_likely_state, 4)

        # Adjust parameters based on emotional state
        urgency = emotional_params["urgency"]
        aggression = emotional_params["aggression"]
        playfulness = emotional_params["playfulness"]

        # Pitch shift based on context and emotion
        if urgency > 0.7:
            pitch_shift = 1.2  # Higher pitch for urgency
        elif playfulness > 0.7:
            pitch_shift = 0.8  # Lower pitch for playfulness
        else:
            pitch_shift = 1.0

        # Time stretch based on context
        if aggression > 0.5:
            time_stretch = 1.3  # Slower for aggression
        else:
            time_stretch = 1.0

        # Gain based on context confidence
        gain = 0.5 + context_result["confidence"] * 0.5

        return SynthesisCommand(
            phrase_id=phrase_id,
            pitch_shift=pitch_shift,
            time_stretch=time_stretch,
            gain=gain,
            emotional_state=emotional_params
        )

    def _update_long_term_memory(self, features: AudioFeatures,
                               context_result: Dict, context: str):
        """Update long-term memory with significant events"""
        memory_key = f"{context}_{int(features.timestamp)}"

        self.long_term_memory[memory_key] = {
            "timestamp": features.timestamp,
            "context": context,
            "context_state": context_result["most_likely_state"],
            "features": asdict(features),
            "emotional_state": self.agent.get_emotional_parameters(),
            "significance": context_result["confidence"]
        }

        # Keep only recent memories (last 1000)
        if len(self.long_term_memory) > 1000:
            oldest_key = min(self.long_term_memory.keys())
            del self.long_term_memory[oldest_key]

        logger.debug(f"Updated long-term memory: {memory_key}")

class PythonAgent:
    """Main Python agent class"""

    def __init__(self, ipc_endpoint: str = "ipc:///tmp/animal_comm.ipc"):
        self.ipc_endpoint = ipc_endpoint
        self.context_processor = CognitiveProcessor()
        self.zmq_context = None
        self.sub_socket = None
        self.pub_socket = None
        self.running = False
        self.stats = {
            "messages_processed": 0,
            "synthesis_commands_sent": 0,
            "processing_errors": 0,
            "avg_processing_time": 0.0
        }

    async def initialize(self):
        """Initialize ZeroMQ connections"""
        self.zmq_context = zmq.Context()

        # Subscribe to audio features from Rust engine
        self.sub_socket = self.zmq_context.socket(zmq.SUB)
        self.sub_socket.connect(self.ipc_endpoint)
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages

        # Publish synthesis commands to Rust engine
        self.pub_socket = self.zmq_context.socket(zmq.PUB)
        self.pub_socket.connect(f"{self.ipc_endpoint.replace('ipc://', 'tcp://')}1")

        logger.info(f"Python Agent initialized with IPC endpoint: {self.ipc_endpoint}")

    async def process_message(self, raw_message: bytes) -> Optional[Dict[str, Any]]:
        """Process incoming message from Rust engine"""
        try:
            # Deserialize using MessagePack for efficiency
            message = msgpack.unpackb(raw_message, raw=False)

            if message.get("type") == "audio_features":
                # Convert to AudioFeatures object
                features = AudioFeatures(**message["data"])

                # Process cognitively
                result = self.context_processor.process_audio_cognitively(
                    features,
                    message.get("context", {})
                )

                # Update statistics
                self.stats["messages_processed"] += 1
                processing_time = result.get("processing_time_ms", 0)
                self._update_avg_processing_time(processing_time)

                # Generate synthesis command
                synthesis_cmd = result["synthesis_command"]

                # Send synthesis command back to Rust engine
                await self.send_synthesis_command(synthesis_cmd)

                return result

            elif message.get("type") == "heartbeat":
                logger.debug("Received heartbeat from Rust engine")
                return {"type": "heartbeat_ack"}

            else:
                logger.warning(f"Unknown message type: {message.get('type')}")
                return None

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.stats["processing_errors"] += 1
            return None

    async def send_synthesis_command(self, cmd: SynthesisCommand):
        """Send synthesis command to Rust engine"""
        try:
            # Serialize with MessagePack
            message = {
                "type": "synthesis_command",
                "data": asdict(cmd),
                "timestamp": int(time.time() * 1000)
            }

            packed = msgpack.packb(message, use_bin_type=True)
            self.pub_socket.send(packed)

            self.stats["synthesis_commands_sent"] += 1
            logger.debug(f"Sent synthesis command: phrase_id={cmd.phrase_id}")

        except Exception as e:
            logger.error(f"Error sending synthesis command: {e}")
            self.stats["processing_errors"] += 1

    def _update_avg_processing_time(self, new_time: float):
        """Update average processing time"""
        alpha = 0.1  # Smoothing factor
        self.stats["avg_processing_time"] = (
            alpha * new_time + (1 - alpha) * self.stats["avg_processing_time"]
        )

    async def run(self):
        """Main agent loop"""
        await self.initialize()
        self.running = True

        logger.info("Python Agent started")

        poller = zmq.Poller()
        poller.register(self.sub_socket, zmq.POLLIN)

        while self.running:
            try:
                # Poll for messages with timeout
                socks = dict(poller.poll(timeout=100))  # 100ms timeout

                if self.sub_socket in socks and socks[self.sub_socket] == zmq.POLLIN:
                    # Receive message
                    raw_message = self.sub_socket.recv()

                    # Process message
                    result = await self.process_message(raw_message)

                    if result:
                        logger.debug(f"Processed message: {result.get('type', 'unknown')}")

                # Send periodic heartbeat
                await self.send_heartbeat()

                # Small delay to prevent CPU overuse
                await asyncio.sleep(0.01)

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                self.running = False
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.stats["processing_errors"] += 1
                await asyncio.sleep(0.1)  # Brief pause on error

        await self.cleanup()

    async def send_heartbeat(self):
        """Send periodic heartbeat to Rust engine"""
        try:
            message = {
                "type": "heartbeat",
                "timestamp": int(time.time() * 1000),
                "agent_stats": self.stats
            }

            packed = msgpack.packb(message, use_bin_type=True)
            self.pub_socket.send(packed)

        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Python Agent...")

        self.running = False

        if self.sub_socket:
            self.sub_socket.close()
        if self.pub_socket:
            self.pub_socket.close()
        if self.zmq_context:
            self.zmq_context.term()

        logger.info(f"Agent stats: {self.stats}")
        logger.info("Python Agent shutdown complete")

async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Python Agent for Hybrid Audio Architecture")
    parser.add_argument("--ipc-endpoint", default="ipc:///tmp/animal_comm.ipc",
                       help="IPC endpoint for communication")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    agent = PythonAgent(args.ipc_endpoint)

    try:
        await agent.run()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")

if __name__ == "__main__":
    asyncio.run(main())