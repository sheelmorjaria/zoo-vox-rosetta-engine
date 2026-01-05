"""
Siamese Network for Few-Shot Learning
=====================================

Implements a Siamese Neural Network for few-shot learning in animal communication.
The network learns to recognize similar audio patterns and can adapt to new
contexts with minimal examples.

Architecture:
- Twin encoders with shared weights
- Distance metric (contrastive loss)
- Memory-augmented adaptation
- Real-time inference capabilities

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
import threading
from collections import deque
import logging

@dataclass
class SiameseConfig:
    """Configuration for Siamese Network"""
    input_dim: int = 64  # Feature dimension
    hidden_dim: int = 128  # Hidden layer dimension
    embedding_dim: int = 64  # Embedding dimension
    dropout: float = 0.2  # Dropout rate
    learning_rate: float = 0.001  # Learning rate
    memory_size: int = 1000  # Memory buffer size
    adaptation_rate: float = 0.1  # Learning rate for adaptation
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class Encoder(nn.Module):
    """Twin encoder for Siamese network"""

    def __init__(self, config: SiameseConfig):
        super().__init__()
        self.config = config

        # Shared encoder architecture
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fc3 = nn.Linear(config.hidden_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder"""
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.normalize(x, p=2, dim=0)  # L2 normalize along feature dimension


class DistanceMetric(nn.Module):
    """Distance metric for similarity computation"""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim * 2, 1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        """Compute distance between embeddings"""
        # Ensure inputs are 2D
        if anchor.dim() == 1:
            anchor = anchor.unsqueeze(0)
        if positive.dim() == 1:
            positive = positive.unsqueeze(0)

        # Concatenate embeddings
        concat = torch.cat([anchor, positive], dim=1)
        # Compute distance (smaller = more similar)
        distance = self.linear(concat)
        return distance


class MemoryBuffer:
    """Memory buffer for storing successful patterns"""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.keys = deque(maxlen=max_size)

    def add_pattern(self, key: str, embedding: torch.Tensor):
        """Add pattern to memory"""
        self.keys.append(key)
        self.buffer.append(embedding.clone().detach())

    def find_similar(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[Tuple[str, torch.Tensor, float]]:
        """Find most similar patterns in memory"""
        if not self.buffer:
            return []

        # Compute similarities
        similarities = []
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)

        for i, (key, memory_embedding) in enumerate(zip(self.keys, self.buffer)):
            if memory_embedding.dim() == 1:
                memory_embedding = memory_embedding.unsqueeze(0)
            similarity = F.cosine_similarity(query_embedding, memory_embedding, dim=1).mean().item()
            similarities.append((key, memory_embedding, similarity))

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]

    def clear(self):
        """Clear memory buffer"""
        self.buffer.clear()
        self.keys.clear()


class SiameseNetwork:
    """Siamese Network for few-shot learning in animal communication"""

    def __init__(self, config: SiameseConfig):
        self.config = config
        self.encoder = Encoder(config).to(config.device)
        self.distance_metric = DistanceMetric(config.embedding_dim).to(config.device)
        self.memory_buffer = MemoryBuffer(config.memory_size)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=config.learning_rate)
        self.logger = logging.getLogger(__name__)

        # Training state
        self.training = False
        self.best_similarity_threshold = 0.5  # Lower threshold for adaptive learning

    def extract_features(self, audio_features: np.ndarray) -> torch.Tensor:
        """Extract features from audio input"""
        if isinstance(audio_features, np.ndarray):
            audio_features = torch.FloatTensor(audio_features).to(self.config.device)

        with torch.no_grad():
            embedding = self.encoder(audio_features)
        return embedding

    def compute_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> float:
        """Compute similarity between two feature vectors"""
        if features1.dim() == 1:
            features1 = features1.unsqueeze(0)
        if features2.dim() == 1:
            features2 = features2.unsqueeze(0)

        distance = self.distance_metric(features1, features2)
        similarity = -distance.item()  # Convert distance to similarity
        return similarity

    def learn_from_success(self, audio_features: np.ndarray, context: str, success_weight: float = 1.0):
        """Learn from successful interaction"""
        # Extract features
        embedding = self.extract_features(audio_features)

        # Store in memory with adaptation
        memory_key = f"{context}_{int(time.time() * 1000)}"
        self.memory_buffer.add_pattern(memory_key, embedding)

        # Update similarity threshold based on success
        if success_weight > 0.5:
            self.best_similarity_threshold = max(0.5, self.best_similarity_threshold - 0.01)

        self.logger.info(f"Learned from success: {context}, similarity threshold: {self.best_similarity_threshold:.3f}")

    def predict_response(self, audio_features: np.ndarray, context: str = None) -> Dict[str, Any]:
        """Predict response based on similarity to learned patterns"""
        # Extract current features
        current_embedding = self.extract_features(audio_features)

        # Find similar patterns in memory
        similar_patterns = self.memory_buffer.find_similar(current_embedding)

        # Compute similarity scores
        similarities = []
        for key, memory_embedding, similarity in similar_patterns:
            similarities.append(similarity)

        # Determine if this is a known context
        max_similarity = max(similarities) if similarities else 0.0

        # Predict response based on similarity
        if max_similarity >= self.best_similarity_threshold:
            response_confidence = (max_similarity - self.best_similarity_threshold) / (1.0 - self.best_similarity_threshold)
            response_type = "adaptive"
        else:
            response_confidence = 0.0
            response_type = "novel"

        # Adapt parameters based on similarity
        if context and max_similarity > 0.8:
            self.adapt_parameters(current_embedding, response_confidence)

        return {
            "response_type": response_type,
            "confidence": response_confidence,
            "max_similarity": max_similarity,
            "similar_patterns": len(similar_patterns),
            "context": context,
            "adaptation_count": len(self.memory_buffer.buffer)
        }

    def adapt_parameters(self, new_embedding: torch.Tensor, adaptation_strength: float):
        """Adapt network parameters based on new successful pattern"""
        if adaptation_strength > 0.5:
            # Fine-tune encoder towards successful pattern
            with torch.no_grad():
                for param in self.encoder.parameters():
                    # Small update towards the new pattern
                    param.add_(-self.config.adaptation_rate * 0.01 * param)

    def compute_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute similarity between two feature vectors"""
        emb1 = self.extract_features(features1)
        emb2 = self.extract_features(features2)
        return self.compute_similarity(emb1, emb2)

    def train_on_batch(self, anchor_features: np.ndarray, positive_features: np.ndarray,
                      negative_features: np.ndarray = None) -> float:
        """Train network with contrastive loss"""
        if not self.training:
            return 0.0

        # Convert to tensors
        anchor = torch.FloatTensor(anchor_features).to(self.config.device)
        positive = torch.FloatTensor(positive_features).to(self.config.device)

        if negative_features is not None:
            negative = torch.FloatTensor(negative_features).to(self.config.device)
        else:
            # Generate negative from random noise
            negative = torch.randn_like(anchor)

        # Forward pass
        anchor_emb = self.encoder(anchor)
        positive_emb = self.encoder(positive)
        negative_emb = self.encoder(negative)

        # Compute losses
        pos_distance = self.distance_metric(anchor_emb, positive_emb)
        neg_distance = self.distance_metric(anchor_emb, negative_emb)

        # Contrastive loss: minimize positive distance, maximize negative distance
        pos_loss = F.relu(pos_distance + 1.0)  # Push positive apart
        neg_loss = F.relu(-neg_distance + 1.0)  # Pull negative together
        total_loss = (pos_loss + neg_loss).mean()

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def start_training(self):
        """Start training mode"""
        self.training = True
        self.encoder.train()
        self.logger.info("Siamese Network training mode enabled")

    def stop_training(self):
        """Stop training mode"""
        self.training = False
        self.encoder.eval()
        self.logger.info("Siamese Network training mode disabled")

    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'distance_state_dict': self.distance_metric.state_dict(),
            'config': self.config,
            'similarity_threshold': self.best_similarity_threshold,
            'memory_size': len(self.memory_buffer.buffer)
        }, path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.distance_metric.load_state_dict(checkpoint['distance_state_dict'])
        self.best_similarity_threshold = checkpoint['similarity_threshold']
        self.logger.info(f"Model loaded from {path}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory buffer statistics"""
        return {
            "memory_size": len(self.memory_buffer.buffer),
            "max_size": self.memory_buffer.max_size,
            "similarity_threshold": self.best_similarity_threshold,
            "training_mode": self.training,
            "device": self.config.device
        }


# Test utility function
def create_test_siamese_network() -> SiameseNetwork:
    """Create a SiameseNetwork for testing"""
    config = SiameseConfig(
        input_dim=64,
        hidden_dim=128,
        embedding_dim=64,
        device='cpu'
    )
    return SiameseNetwork(config)