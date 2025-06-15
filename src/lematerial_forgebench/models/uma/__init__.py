"""UMA model implementation."""

try:
    from .calculator import (
        AVAILABLE_UMA_MODELS,
        AVAILABLE_UMA_TASKS,
        UMACalculator,
        create_uma_calculator,
    )
    from .embeddings import UMAASECalculator, UMAEmbeddingExtractor

    UMA_AVAILABLE = True
except ImportError:
    UMA_AVAILABLE = False
    UMACalculator = None
    UMAEmbeddingExtractor = None
    UMAASECalculator = None
    create_uma_calculator = None
    AVAILABLE_UMA_MODELS = []
    AVAILABLE_UMA_TASKS = []

__all__ = [
    "UMACalculator",
    "UMAEmbeddingExtractor",
    "UMAASECalculator",
    "create_uma_calculator",
    "AVAILABLE_UMA_MODELS",
    "AVAILABLE_UMA_TASKS",
    "UMA_AVAILABLE",
]
