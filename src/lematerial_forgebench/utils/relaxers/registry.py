"""Registry for relaxer implementations and base classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Type

from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry


@dataclass
class RelaxationResult:
    """Result of a structure relaxation.

    Parameters
    ----------
    success : bool
        Whether the relaxation was successful.
    energy : float | None
        Final energy of the relaxed structure.
    structure : Structure | None
        Final relaxed structure.
    message : str | None
        Error message if relaxation failed.
    """

    success: bool
    energy: float | None = None
    structure: Structure | None = None
    message: str | None = None


class BaseRelaxer(ABC):
    """Base class for structure relaxation implementations.

    All relaxer implementations should inherit from this class and implement
    the required methods.
    """

    @abstractmethod
    def relax(self, structure: Structure, relax: bool = False) -> RelaxationResult:
        """Relax a structure and return the result.

        Parameters
        ----------
        structure : Structure
            Structure to relax.

        Returns
        -------
        RelaxationResult
            Result of the relaxation.
        """
        pass

    @abstractmethod
    def get_computed_entry(
        self, structure: Structure, energy: float
    ) -> ComputedStructureEntry:
        """Create a ComputedStructureEntry from a relaxed structure.

        Parameters
        ----------
        structure : Structure
            The relaxed structure.
        energy : float
            The energy of the relaxed structure.

        Returns
        -------
        ComputedStructureEntry
            The computed structure entry with appropriate corrections applied.
        """
        pass


_RELAXER_REGISTRY: Dict[str, Type[BaseRelaxer]] = {}


def register_relaxer(name: str):
    """Register a relaxer implementation.

    Parameters
    ----------
    name : str
        Name of the relaxer.

    Returns
    -------
    callable
        Decorator function.
    """

    def decorator(cls: Type[BaseRelaxer]) -> Type[BaseRelaxer]:
        if not issubclass(cls, BaseRelaxer):
            raise TypeError(f"{cls.__name__} must inherit from BaseRelaxer")
        _RELAXER_REGISTRY[name] = cls
        return cls

    return decorator


def get_relaxer(relaxer_type: str, **kwargs) -> BaseRelaxer:
    """Get a relaxer implementation by name.

    Parameters
    ----------
    relaxer_type : str
        Name of the relaxer.
    **kwargs
        Additional arguments to pass to the relaxer constructor.

    Returns
    -------
    BaseRelaxer
        Relaxer instance.

    Raises
    ------
    ValueError
        If relaxer_type is not registered.
    """
    if relaxer_type not in _RELAXER_REGISTRY:
        raise ValueError(
            f"Unknown relaxer type: {relaxer_type}. "
            f"Available types: {list(_RELAXER_REGISTRY.keys())}"
        )
    return _RELAXER_REGISTRY[relaxer_type](**kwargs)
