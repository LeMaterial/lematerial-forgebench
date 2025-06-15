"""MACE model implementation."""

try:
    from .calculator import (
        AVAILABLE_MACE_MODELS,
        MACECalculator,
        create_mace_calculator,
    )
    from .embeddings import MACEEmbeddingExtractor

    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False
    MACECalculator = None
    MACEEmbeddingExtractor = None
    create_mace_calculator = None
    AVAILABLE_MACE_MODELS = []

__all__ = [
    "MACECalculator",
    "MACEEmbeddingExtractor",
    "create_mace_calculator",
    "AVAILABLE_MACE_MODELS",
    "MACE_AVAILABLE",
]
