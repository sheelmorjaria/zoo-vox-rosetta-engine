#!/usr/bin/env python3
"""
Syntax Graph Builder - Module 3 (Dual-Stream)

Builds a probabilistic syntax graph from tokenized vocalization corpus.

The syntax graph captures valid bigram transitions between syntactic tokens
using Laplace smoothing to prevent zero-probability transitions from corpus
sparsity.

Process:
    1. Load trained VQ-VAE model
    2. Tokenize corpus (segments JSON or cached features)
    3. Build transition matrix with Laplace smoothing (α=0.01)
    4. Save syntax graph for use by DualStreamInteractionAgent

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .syntactic_encoder import SyntacticFeatureExtractor
from .syntactic_vqvae import SyntacticVQVAE, VQVAECheckpoint
from .syntax_graph import SyntaxGraph

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SyntaxGraphBuilderConfig:
    """Configuration for syntax graph building."""

    # Vocabulary size
    num_tokens: int = 64

    # Laplace smoothing
    alpha: float = 0.01  # Pseudocount for smoothing

    # Filtering
    min_bigram_count: int = 1  # Minimum count to include transition

    # Token labels (for interpretability)
    token_labels: Optional[List[str]] = None

    # Model path
    vqvae_path: str = "models/dual_stream/syntactic_vqvae.pt"

    # Output
    output_path: str = "models/dual_stream/syntax_graph.json"


# =============================================================================
# Tokenization
# =============================================================================


class CorpusTokenizer:
    """Tokenize a corpus using trained VQ-VAE."""

    def __init__(self, vqvae: SyntacticVQVAE, device: torch.device):
        self.vqvae = vqvae
        self.device = device
        self.vqvae.eval()
        self.extractor = SyntacticFeatureExtractor()

    @torch.no_grad()
    def tokenize_features_112d(self, features_112d: np.ndarray) -> int:
        """Tokenize a single 112D feature vector."""
        # Extract syntactic features
        syntactic = self.extractor.extract_syntactic_features(features_112d)

        # Convert to tensor
        x = torch.from_numpy(syntactic).float().unsqueeze(0).to(self.device)

        # Tokenize
        _, _, token_ids, _ = self.vqvae(x)

        return token_ids.item()

    @torch.no_grad()
    def tokenize_batch(self, features_112d_batch: np.ndarray) -> List[int]:
        """Tokenize a batch of 112D feature vectors."""
        # Extract syntactic features
        syntactic = self.extractor.extract_syntactic_features_batch(features_112d_batch)

        # Convert to tensor
        x = torch.from_numpy(syntactic).float().to(self.device)

        # Tokenize
        _, _, token_ids, _ = self.vqvae(x)

        return token_ids.cpu().tolist()

    def tokenize_segments(self, segments_json: str) -> List[List[int]]:
        """
        Tokenize segments from JSON file.

        Returns a list of token sequences (one per segment).
        For single vocalizations, each segment becomes a sequence of length 1.
        For phrases with multiple features, tokens are extracted in order.
        """
        with open(segments_json, "r") as f:
            data = json.load(f)

        sequences = []

        for item in data.get("segments", []):
            if "features_112d" in item:
                features = np.array(item["features_112d"], dtype=np.float32)
                if len(features) == 112:
                    token_id = self.tokenize_features_112d(features)
                    sequences.append([token_id])

        logger.info(f"Tokenized {len(sequences)} segments")
        return sequences

    def tokenize_features_npy(self, features_npy: str) -> List[List[int]]:
        """
        Tokenize cached features from .npy file.

        Each feature vector becomes a single-token sequence.
        """
        features_112d = np.load(features_npy)
        logger.info(f"Loaded {len(features_112d)} feature vectors")

        sequences = []
        for features in features_112d:
            token_id = self.tokenize_features_112d(features)
            sequences.append([token_id])

        logger.info(f"Tokenized {len(sequences)} feature vectors")
        return sequences


# =============================================================================
# Syntax Graph Builder
# =============================================================================


class SyntaxGraphBuilder:
    """Build syntax graph from tokenized corpus."""

    def __init__(self, config: SyntaxGraphBuilderConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_from_sequences(self, sequences: List[List[int]], num_tokens: int) -> SyntaxGraph:
        """
        Build syntax graph from token sequences.

        Args:
            sequences: List of token sequences
            num_tokens: Size of vocabulary

        Returns:
            SyntaxGraph with learned transition probabilities
        """
        logger.info(f"Building syntax graph from {len(sequences)} sequences")

        # Initialize transition matrix with Laplace smoothing
        # Start with uniform probabilities + pseudocounts
        alpha = self.config.alpha
        transitions = np.full((num_tokens, num_tokens), alpha)

        # Count bigrams
        bigram_counts = np.zeros((num_tokens, num_tokens), dtype=int)

        for seq in sequences:
            for i in range(len(seq) - 1):
                bigram_counts[seq[i], seq[i + 1]] += 1

        # Apply Laplace smoothing
        for i in range(num_tokens):
            total = bigram_counts[i].sum() + alpha * num_tokens
            for j in range(num_tokens):
                transitions[i, j] = (bigram_counts[i, j] + alpha) / total

        # Create syntax graph
        graph = SyntaxGraph(num_tokens=num_tokens, alpha=alpha)

        # Set the pre-computed transitions
        graph.transitions = transitions

        # Compute statistics
        total_bigrams = bigram_counts.sum()
        unique_bigrams = (bigram_counts > 0).sum()
        coverage = unique_bigrams / (num_tokens * num_tokens)

        # Print statistics
        logger.info("Syntax Graph Statistics:")
        logger.info(f"  Total bigrams observed: {int(total_bigrams)}")
        logger.info(f"  Unique bigrams: {unique_bigrams}")
        logger.info(f"  Coverage: {coverage:.2%}")

        return graph

    def build_and_save(
        self,
        sequences: List[List[int]],
        num_tokens: int,
        output_path: Optional[str] = None,
    ) -> SyntaxGraph:
        """
        Build and save syntax graph.

        Args:
            sequences: Token sequences
            num_tokens: Vocabulary size
            output_path: Output path (overrides config)

        Returns:
            SyntaxGraph
        """
        # Build graph
        graph = self.build_from_sequences(sequences, num_tokens)

        # Save
        save_path = output_path or self.config.output_path
        graph.save_json(save_path)
        logger.info(f"Saved syntax graph to {save_path}")

        return graph


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build syntax graph from tokenized corpus"
    )

    # Input
    parser.add_argument(
        "--vqvae",
        type=str,
        default="models/dual_stream/syntactic_vqvae.pt",
        help="Path to trained VQ-VAE model",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to corpus (segments JSON or cached features .npy)",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["json", "npy"],
        default="json",
        help="Data format (default: json)",
    )

    # Syntax graph
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=64,
        help="Vocabulary size (default: 64)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Laplace smoothing parameter (default: 0.01)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="models/dual_stream/syntax_graph.json",
        help="Output path for syntax graph (default: models/dual_stream/syntax_graph.json)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()

    # Create config
    config = SyntaxGraphBuilderConfig(
        alpha=args.alpha,
        vqvae_path=args.vqvae,
        output_path=args.output,
    )

    # Load VQ-VAE model
    logger.info(f"Loading VQ-VAE from {args.vqvae}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = VQVAECheckpoint.load(args.vqvae)
    vqvae = SyntacticVQVAE(
        input_dim=checkpoint.config["input_dim"],
        codebook_size=checkpoint.config["codebook_size"],
        codebook_dim=checkpoint.config["codebook_dim"],
        hidden_dim=checkpoint.config["hidden_dim"],
        decay=checkpoint.config.get("decay", 0.99),
        commitment_cost=checkpoint.config.get("commitment_cost", 0.25),
    )
    vqvae.load_state_dict(checkpoint.model_state_dict)
    vqvae.to(device)
    vqvae.eval()

    logger.info(f"Loaded VQ-VAE on {device}")

    # Create tokenizer
    tokenizer = CorpusTokenizer(vqvae, device)

    # Tokenize corpus
    if args.data_type == "json":
        sequences = tokenizer.tokenize_segments(args.data)
    else:
        sequences = tokenizer.tokenize_features_npy(args.data)

    # Build and save syntax graph
    builder = SyntaxGraphBuilder(config)
    graph = builder.build_and_save(sequences, args.num_tokens, args.output)

    # Print example transitions
    logger.info("\nExample transitions from token 0:")
    for token, prob in graph.get_valid_next_tokens(0, top_k=5):
        logger.info(f"  0 -> {token}: {prob:.4f}")


if __name__ == "__main__":
    main()
