#!/usr/bin/env python3
"""
Syntax Transformer: Lightweight Autoregressive Model

Replaces the rigid bigram automaton with a probabilistic Transformer
that models P(token_t | token_{t-1}, ..., token_0).

This enables:
- Novel but statistically plausible combinatorial syntax
- Temperature-controlled sampling (conservative vs. creative)
- Probability-based validation instead of binary valid/invalid

Architecture: Minimal GPT-style model
- 2-4 layers, 64-128 hidden dimensions
- Causal masking for autoregressive generation
- Fast CPU inference (<1ms per token)

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for Syntax Transformer."""
    # Vocabulary
    num_tokens: int = 64  # VQ-VAE codebook size
    max_len: int = 32  # Maximum sequence length (context window)

    # Architecture
    d_model: int = 64  # Embedding dimension
    n_heads: int = 4  # Number of attention heads
    n_layers: int = 2  # Number of transformer layers
    dim_feedforward: int = 128  # FFN hidden dimension
    dropout: float = 0.1

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 50

    # Device
    device: str = "cpu"


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model) with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class SyntaxTransformer(nn.Module):
    """
    Lightweight autoregressive model for VQ-VAE token sequences.

    Architecture:
        Token/Position Embedding → Transformer Decoder → Output Head

    Uses causal masking to ensure autoregressive property.
    """

    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()
        if config is None:
            config = TransformerConfig()

        self.config = config
        self.num_tokens = config.num_tokens
        self.d_model = config.d_model
        self.max_len = config.max_len
        self.n_heads = config.n_heads

        # Token embedding
        self.token_emb = nn.Embedding(config.num_tokens, config.d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_len)

        # Transformer decoder (uses causal masking)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.n_layers,
        )

        # Output head
        self.output_head = nn.Linear(config.d_model, config.num_tokens)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, T) token IDs
            mask: Optional causal mask

        Returns:
            logits: (B, T, num_tokens)
        """
        B, T = x.shape

        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len {self.max_len}")

        # Token embeddings
        x = self.token_emb(x)  # (B, T, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Generate causal mask if not provided
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)

        # Transformer decoder (use input as both tgt and memory)
        # This creates a decoder-only architecture like GPT
        out = self.transformer(
            tgt=x,
            memory=x,
            tgt_mask=mask,
        )

        # Output projection
        logits = self.output_head(out)  # (B, T, num_tokens)

        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: (B, T, num_tokens)
            targets: (B, T) token IDs

        Returns:
            loss: Scalar
        """
        B, T, V = logits.shape
        logits = logits.view(-1, V)
        targets = targets.view(-1)

        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        return loss

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            prefix: (B, T_prefix) starting tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more conservative)
            top_k: If set, only sample from top-k tokens
            top_p: If set, use nucleus (top-p) sampling
            eos_token: Stop generation if this token is generated

        Returns:
            generated: (B, T_prefix + T_generated) token IDs
        """
        self.eval()
        B, T = prefix.shape

        # Start with prefix
        generated = prefix.clone()

        for _ in range(max_new_tokens):
            # Truncate to max_len if necessary
            if generated.size(1) > self.max_len:
                generated = generated[:, -self.max_len:]

            # Forward pass
            logits = self.forward(generated)[:, -1, :]  # Get last token logits

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if eos_token is not None and (next_token == eos_token).all():
                break

        return generated


class SyntaxTransformerTrainer:
    """Trainer for Syntax Transformer."""

    def __init__(
        self,
        config: Optional[TransformerConfig] = None,
    ):
        if config is None:
            config = TransformerConfig()

        self.config = config
        self.device = self._get_device()

        self.model = SyntaxTransformer(config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def _get_device(self) -> torch.device:
        """Determine device."""
        if self.config.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def train(
        self,
        sequences: List[List[int]],
        val_split: float = 0.1,
    ) -> dict:
        """
        Train Syntax Transformer on token sequences.

        Args:
            sequences: List of token ID sequences (variable length)
            val_split: Fraction of data for validation

        Returns:
            Training history
        """
        # Prepare dataset
        max_len = min(self.config.max_len, max(len(s) for s in sequences))

        # Pad sequences and create inputs/targets
        # Input: tokens[0:t], Target: tokens[1:t+1]
        inputs = []
        targets = []

        for seq in sequences:
            if len(seq) < 2:
                continue  # Skip sequences that are too short

            # Truncate if necessary
            seq = seq[:max_len]

            # Create input-target pairs
            for i in range(len(seq) - 1):
                inputs.append(seq[:i+1])
                targets.append(seq[i+1])

        logger.info(f"Created {len(inputs)} training examples from {len(sequences)} sequences")

        # Pad inputs to same length
        padded_inputs = []
        for inp in inputs:
            padded = inp.copy()
            padded.extend([0] * (max_len - len(padded)))
            padded_inputs.append(padded[:max_len])

        inputs_tensor = torch.LongTensor(padded_inputs)
        targets_tensor = torch.LongTensor(targets)

        dataset = TensorDataset(inputs_tensor, targets_tensor)

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
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        # Training loop
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for inputs_batch, targets_batch in train_loader:
                inputs_batch = inputs_batch.to(self.device)
                targets_batch = targets_batch.to(self.device)

                # Forward pass
                logits = self.model(inputs_batch)

                # For next-token prediction, use only the last position
                # logits[:, -1, :] gives the prediction for what comes next
                next_token_logits = logits[:, -1, :]  # (B, num_tokens)

                # Compute loss
                loss = self.model.compute_loss(
                    next_token_logits.unsqueeze(1),  # (B, 1, num_tokens)
                    targets_batch.unsqueeze(1)       # (B, 1)
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs_batch, targets_batch in val_loader:
                    inputs_batch = inputs_batch.to(self.device)
                    targets_batch = targets_batch.to(self.device)

                    logits = self.model(inputs_batch)
                    next_token_logits = logits[:, -1, :]
                    loss = self.model.compute_loss(
                        next_token_logits.unsqueeze(1),
                        targets_batch.unsqueeze(1)
                    )
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)

            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                logger.info(
                    f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, "
                    f"Val Loss = {val_loss:.4f}"
                )

        logger.info("Training complete")
        return history

    def save_checkpoint(
        self,
        path: str = "models/syntax_transformer.pt",
    ) -> Path:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, path)

        logger.info(f"Saved checkpoint to {path}")
        return path

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
    ) -> "SyntaxTransformerTrainer":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')

        config = checkpoint['config']
        trainer = cls(config)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Loaded checkpoint from {path}")
        return trainer

    def export_to_onnx(
        self,
        path: str = "models/syntax_transformer.onnx",
    ) -> Path:
        """Export model to ONNX for potential Rust migration."""
        import torch.onnx

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval()

        # Dummy input: batch_size=1, seq_len=1 (for single token generation)
        dummy_input = torch.randint(0, self.config.num_tokens, (1, 1))

        torch.onnx.export(
            self.model,
            dummy_input,
            str(path),
            export_params=True,
            opset_version=17,
            input_names=['input_tokens'],
            output_names=['output_logits'],
            dynamic_axes={
                'input_tokens': {0: 'batch_size', 1: 'seq_len'},
                'output_logits': {0: 'batch_size', 1: 'seq_len'},
            },
        )

        logger.info(f"Exported ONNX model to {path}")
        return path


# Preset configurations

MINIMAL_TRANSFORMER_CONFIG = TransformerConfig(
    num_tokens=64,
    max_len=16,
    d_model=64,
    n_heads=4,
    n_layers=2,
    dim_feedforward=128,
    dropout=0.1,
)

STANDARD_TRANSFORMER_CONFIG = TransformerConfig(
    num_tokens=64,
    max_len=32,
    d_model=128,
    n_heads=8,
    n_layers=3,
    dim_feedforward=256,
    dropout=0.1,
)


def create_transformer_trainer(
    config: Optional[TransformerConfig] = None,
    checkpoint_path: Optional[str] = None,
) -> SyntaxTransformerTrainer:
    """
    Factory function to create transformer trainer.

    Args:
        config: Transformer configuration
        checkpoint_path: Optional path to load checkpoint from

    Returns:
        Configured SyntaxTransformerTrainer
    """
    if config is None:
        config = STANDARD_TRANSFORMER_CONFIG

    trainer = SyntaxTransformerTrainer(config)

    if checkpoint_path is not None:
        trainer = SyntaxTransformerTrainer.load_checkpoint(checkpoint_path)

    return trainer


def main():
    """Example training."""
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic training data
    np.random.seed(42)
    n_sequences = 500
    seq_len = 10
    num_tokens = 64

    sequences = []
    for _ in range(n_sequences):
        seq = np.random.randint(0, num_tokens, seq_len).tolist()
        sequences.append(seq)

    # Train
    config = MINIMAL_TRANSFORMER_CONFIG
    config.epochs = 20  # Quick demo

    trainer = create_transformer_trainer(config)
    history = trainer.train(sequences)

    # Generate
    prefix = torch.tensor([[sequences[0][0]]])
    generated = trainer.model.generate(
        prefix,
        max_new_tokens=5,
        temperature=0.8,
        top_p=0.9,
    )

    print(f"Generated sequence: {generated[0].tolist()}")

    # Export
    trainer.export_to_onnx("models/syntax_transformer_demo.onnx")


if __name__ == '__main__':
    main()
