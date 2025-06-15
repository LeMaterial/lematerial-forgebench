"""UMA model calculator implementation."""

import numpy as np
import torch
from pymatgen.core.structure import Structure

try:
    from fairchem.core import pretrained_mlip

    UMA_AVAILABLE = True
except ImportError:
    UMA_AVAILABLE = False

from lematerial_forgebench.models.base import (
    BaseMLIPCalculator,
    CalculationResult,
    EmbeddingResult,
    get_energy_above_hull_from_total_energy,
    get_formation_energy_from_total_energy,
)
from lematerial_forgebench.models.uma.embeddings import (
    UMAASECalculator,
    UMAEmbeddingExtractor,
)
from lematerial_forgebench.utils.logging import logger


class UMACalculator(BaseMLIPCalculator):
    """UMA calculator for energy/force calculations and embedding extraction."""

    def __init__(
        self,
        model_name: str = "uma-s-1",
        task: str = "omat",  # "oc20", "omat", "omol", "odac", "omc"
        device: str = "cpu",
        precision: str = "float32",
        **kwargs,
    ):
        if not UMA_AVAILABLE:
            raise ImportError(
                "UMA/FAIRChem is not available. Please install it with: "
                "pip install fairchem-core>=2.1.0"
            )

        self.model_name = model_name
        self.task = task
        super().__init__(device=device, precision=precision, **kwargs)

    def _setup_model(self, **kwargs):
        """Initialize the UMA model."""
        try:
            # Load the UMA predictor
            self.predictor = pretrained_mlip.get_predict_unit(
                self.model_name,
                device=self.device,
                precision=self.precision,
            )

            # Get the underlying model for embeddings
            self.model = self.predictor.model
            self.model.eval()

            # Create ASE calculator wrapper
            self.ase_calc = UMAASECalculator(self.predictor, self.device)

            # Create embedding extractor
            self.embedding_extractor = UMAEmbeddingExtractor(self.model, self.device)

            logger.info(f"Successfully loaded UMA model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load UMA model: {str(e)}")
            raise

    def calculate_energy_forces(self, structure: Structure) -> CalculationResult:
        """Calculate energy and forces using UMA.

        Parameters
        ----------
        structure : Structure
            Input structure

        Returns
        -------
        CalculationResult
            Energy, forces, and metadata
        """
        atoms = self._structure_to_atoms(structure)
        atoms.calc = self.ase_calc

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        return CalculationResult(
            energy=energy,
            forces=forces,
            metadata={"model_type": f"UMA-{self.model_name}", "task": self.task},
        )

    def extract_embeddings(self, structure: Structure) -> EmbeddingResult:
        """Extract embeddings using UMA.

        Parameters
        ----------
        structure : Structure
            Input structure

        Returns
        -------
        EmbeddingResult
            Node and graph embeddings
        """
        return self.embedding_extractor.extract_embeddings(structure)

    def _get_ase_calculator(self):
        """Get ASE calculator for UMA."""
        return self.ase_calc

    def calculate_formation_energy(self, structure: Structure) -> float:
        """Calculate formation energy using UMA.

        Parameters
        ----------
        structure : Structure
            Input structure

        Returns
        -------
        float
            Formation energy in eV/atom
        """
        result = self.calculate_energy_forces(structure)
        total_energy = result.energy

        return get_formation_energy_from_total_energy(
            total_energy, structure.composition
        )

    def calculate_energy_above_hull(self, structure: Structure) -> float:
        """Calculate energy above hull using UMA.

        Parameters
        ----------
        structure : Structure
            Input structure

        Returns
        -------
        float
            Energy above hull in eV/atom
        """
        result = self.calculate_energy_forces(structure)
        total_energy = result.energy

        return get_energy_above_hull_from_total_energy(
            total_energy, structure.composition
        )


def create_uma_calculator(
    model_name: str = "uma-s-1", task: str = "omat", device: str = "cpu", **kwargs
) -> UMACalculator:
    """Factory function to create UMA calculator.

    Parameters
    ----------
    model_name : str
        UMA model name (e.g., "uma-s-1")
    task : str
        Task domain for UMA
    device : str
        Device for computation
    **kwargs
        Additional arguments for the calculator

    Returns
    -------
    UMACalculator
        Configured UMA calculator
    """
    return UMACalculator(model_name=model_name, task=task, device=device, **kwargs)


# Available UMA models and tasks
AVAILABLE_UMA_MODELS = ["uma-s-1"]
AVAILABLE_UMA_TASKS = ["oc20", "omat", "omol", "odac", "omc"]
