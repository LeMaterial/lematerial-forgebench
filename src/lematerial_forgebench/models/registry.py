"""Model registry for easy access to different MLIPs."""

from typing import Dict, Type, Union

from lematerial_forgebench.models.base import BaseMLIPCalculator
from lematerial_forgebench.utils.logging import logger

# Import all calculator implementations
try:
    from lematerial_forgebench.models.orb.calculator import (
        AVAILABLE_ORB_MODELS,
        ORBCalculator,
        create_orb_calculator,
    )

    ORB_AVAILABLE = True
except ImportError:
    ORB_AVAILABLE = False
    logger.warning("ORB not available")

try:
    from lematerial_forgebench.models.mace.calculator import (
        AVAILABLE_MACE_MODELS,
        MACECalculator,
        create_mace_calculator,
    )

    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False
    logger.warning("MACE not available")

try:
    from lematerial_forgebench.models.equiformer.calculator import (
        EquiformerCalculator,
        create_equiformer_calculator,
    )

    EQUIFORMER_AVAILABLE = True
except ImportError:
    EQUIFORMER_AVAILABLE = False
    logger.warning("Equiformer v2 not available")

try:
    from lematerial_forgebench.models.uma.calculator import (
        AVAILABLE_UMA_MODELS,
        AVAILABLE_UMA_TASKS,
        UMACalculator,
        create_uma_calculator,
    )

    UMA_AVAILABLE = True
except ImportError:
    UMA_AVAILABLE = False
    logger.warning("UMA not available")


class ModelRegistry:
    """Registry for managing available MLIP calculators."""

    def __init__(self):
        self._calculators: Dict[str, Type[BaseMLIPCalculator]] = {}
        self._factory_functions: Dict[str, callable] = {}
        self._register_available_models()

    def _register_available_models(self):
        """Register all available models."""
        if ORB_AVAILABLE:
            self._calculators["orb"] = ORBCalculator
            self._factory_functions["orb"] = create_orb_calculator

        if MACE_AVAILABLE:
            self._calculators["mace"] = MACECalculator
            self._factory_functions["mace"] = create_mace_calculator

        if EQUIFORMER_AVAILABLE:
            self._calculators["equiformer"] = EquiformerCalculator
            self._factory_functions["equiformer"] = create_equiformer_calculator

        if UMA_AVAILABLE:
            self._calculators["uma"] = UMACalculator
            self._factory_functions["uma"] = create_uma_calculator

    def get_available_models(self) -> list[str]:
        """Get list of available model types.

        Returns
        -------
        list[str]
            List of available model names
        """
        return list(self._calculators.keys())

    def create_calculator(self, model_type: str, **kwargs) -> BaseMLIPCalculator:
        """Create a calculator for the specified model type.

        Parameters
        ----------
        model_type : str
            Type of model ("orb", "mace", "equiformer", "uma")
        **kwargs
            Model-specific arguments

        Returns
        -------
        BaseMLIPCalculator
            Configured calculator instance

        Raises
        ------
        ValueError
            If model_type is not available
        """
        if model_type not in self._factory_functions:
            available = self.get_available_models()
            raise ValueError(
                f"Model type '{model_type}' not available. "
                f"Available models: {available}"
            )

        factory_func = self._factory_functions[model_type]
        return factory_func(**kwargs)

    def get_model_info(self) -> Dict[str, Dict]:
        """Get information about available models.

        Returns
        -------
        Dict[str, Dict]
            Information about each available model
        """
        info = {}

        if ORB_AVAILABLE:
            info["orb"] = {
                "class": "ORBCalculator",
                "available_models": AVAILABLE_ORB_MODELS,
                "description": "Orbital Materials' ORB force fields",
                "supports_embeddings": True,
                "supports_relaxation": True,
            }

        if MACE_AVAILABLE:
            info["mace"] = {
                "class": "MACECalculator",
                "available_models": AVAILABLE_MACE_MODELS,
                "description": "MACE: Materials Accelerated by Chemical Embedding",
                "supports_embeddings": True,
                "supports_relaxation": True,
            }

        if EQUIFORMER_AVAILABLE:
            info["equiformer"] = {
                "class": "EquiformerCalculator",
                "available_models": ["custom"],
                "description": "Equiformer v2: Transformer for molecules and materials",
                "supports_embeddings": True,
                "supports_relaxation": True,
            }

        if UMA_AVAILABLE:
            info["uma"] = {
                "class": "UMACalculator",
                "available_models": AVAILABLE_UMA_MODELS,
                "available_tasks": AVAILABLE_UMA_TASKS,
                "description": "Universal Materials Accelerator by Meta",
                "supports_embeddings": True,
                "supports_relaxation": True,
            }

        return info


# Global registry instance
_registry = ModelRegistry()


def get_calculator(model_type: str, **kwargs) -> BaseMLIPCalculator:
    """Get a calculator for the specified model type.

    This is the main entry point for creating calculators.

    Parameters
    ----------
    model_type : str
        Type of model to create
    **kwargs
        Model-specific configuration

    Returns
    -------
    BaseMLIPCalculator
        Configured calculator

    Examples
    --------
    >>> calc = get_calculator("orb", model_type="orb_v3_conservative_inf_omat")
    >>> calc = get_calculator("mace", model_type="mp")
    >>> calc = get_calculator("uma", model_name="uma-s-1", task="omat")
    """
    return _registry.create_calculator(model_type, **kwargs)


def list_available_models() -> list[str]:
    """List all available model types.

    Returns
    -------
    list[str]
        Available model types
    """
    return _registry.get_available_models()


def get_model_info() -> Dict[str, Dict]:
    """Get information about all available models.

    Returns
    -------
    Dict[str, Dict]
        Model information
    """
    return _registry.get_model_info()


def print_model_info():
    """Print information about available models."""
    info = get_model_info()

    print("Available MLIP Models:")
    print("=" * 50)

    for model_type, details in info.items():
        print(f"\n{model_type.upper()}:")
        print(f"  Class: {details['class']}")
        print(f"  Description: {details['description']}")
        print(f"  Supports embeddings: {details['supports_embeddings']}")
        print(f"  Supports relaxation: {details['supports_relaxation']}")

        if "available_models" in details:
            print(f"  Available models: {details['available_models']}")

        if "available_tasks" in details:
            print(f"  Available tasks: {details['available_tasks']}")


# Backward compatibility functions
def get_orb_calculator(**kwargs):
    """Get ORB calculator (backward compatibility)."""
    if not ORB_AVAILABLE:
        raise ImportError("ORB not available")
    return get_calculator("orb", **kwargs)


def get_mace_calculator(**kwargs):
    """Get MACE calculator (backward compatibility)."""
    if not MACE_AVAILABLE:
        raise ImportError("MACE not available")
    return get_calculator("mace", **kwargs)


def get_equiformer_calculator(**kwargs):
    """Get Equiformer calculator (backward compatibility)."""
    if not EQUIFORMER_AVAILABLE:
        raise ImportError("Equiformer v2 not available")
    return get_calculator("equiformer", **kwargs)


def get_uma_calculator(**kwargs):
    """Get UMA calculator (backward compatibility)."""
    if not UMA_AVAILABLE:
        raise ImportError("UMA not available")
    return get_calculator("uma", **kwargs)
