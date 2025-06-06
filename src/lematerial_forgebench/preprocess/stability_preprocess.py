"""Relaxation metrics for evaluating material structures.

This module implements metrics for evaluating the relaxation of
material structures using various relaxation models and calculating
energy above hull.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import load_dataset
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from lematerial_forgebench.preprocess.base import BasePreprocessor, PreprocessorConfig
from lematerial_forgebench.preprocess.reference_energies import (
    get_energy_above_hull,
    get_formation_energy_from_composition_energy,
)
from lematerial_forgebench.utils.e_above_hull import (
    generate_CSE,
    get_patched_phase_diagram_mp,
)
from lematerial_forgebench.utils.logging import logger
from lematerial_forgebench.utils.relaxers import (
    BaseRelaxer,
    get_relaxer,
    relaxers,
)
import numpy as np 

class OrbFormationEnergy:
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.orbff = pretrained.orb_v3_conservative_inf_omat(
            compile=False,
            device="cpu",
            precision="float32-high",
        )
        self.calc = ORBCalculator(self.orbff, device="cpu")

    def __call__(self, structure) -> float:
        crystal_ase = AseAtomsAdaptor().get_atoms(structure)
        crystal_ase.calc = self.calc

        total_energy = crystal_ase.get_potential_energy()

        formation_energy = get_formation_energy_from_composition_energy(
            total_energy, structure.composition
        )
        if formation_energy is None:
            return 0.0
        # print("formation energy ", str(formation_energy))
        return formation_energy / self.temperature

    def move_to_shared_memory(self):
        """Move ORB model parameters to shared memory."""
        for param in self.orbff.parameters():
            param.share_memory_()


class EnergyAboveHull:
    def __init__(self, temperature: float = 1, functional: str = "pbe"):
        super().__init__()
        self.temperature = temperature
        self.orbff = pretrained.orb_v3_conservative_inf_omat(
            compile=False,
            device="cpu",
            precision="float32-high",
        )
        self.calc = ORBCalculator(self.orbff, device="cpu")

        # self.ds = load_dataset("LeMaterial/LeMat-Bulk", f"compatible_{functional}")

    def __call__(self, structure) -> float:
        crystal_ase = AseAtomsAdaptor().get_atoms(structure)
        crystal_ase.calc = self.calc

        total_energy = crystal_ase.get_potential_energy()
        energy_above_hull = get_energy_above_hull(total_energy, structure.composition)
        # print("energy_above_hull :", energy_above_hull)
        if energy_above_hull is None:
            return None
        if energy_above_hull < 0:
            return 0.0

        # print("energy_above_hull :", energy_above_hull)

        return energy_above_hull

    def move_to_shared_memory(self):
        """Move ORB model parameters to shared memory."""
        for param in self.orbff.parameters():
            param.share_memory_()


@dataclass
class StabilityPreprocessorConfig(PreprocessorConfig):
    """Configuration for the StabilityPreprocessor.

    Parameters
    ----------
    relaxer_type : str
        Type of relaxer to use (e.g., "chgnet", "eqv2", "esen").
    relaxer_config : dict
        Configuration for the specific relaxer.
    mp_entries_file : str, optional
        Path to the Materials Project entries file for e_above_hull calculation.
    """

    relaxer_type: str = "orb"
    relaxer_config: Dict[str, Any] = field(default_factory=dict)
    mp_entries_file: Optional[str] = None


class StabilityPreprocessor(BasePreprocessor):
    """Evaluate structure relaxation and energy above hull.

    This metric handles both the relaxation of structures using various models
    and the calculation of energy above hull using the Materials Project database.

    Parameters
    ----------
    relaxer_type : str
        Type of relaxer to use.
    relaxer_config : dict
        Configuration for the specific relaxer.
    mp_entries_file : str, optional
        Path to the Materials Project entries file.
    name : str, optional
        Custom name for the metric.
    description : str, optional
        Description of what the metric measures.
    lower_is_better : bool, default=True
        Lower energies indicate better stability.
    n_jobs : int, default=1
        Number of parallel jobs to run.
    """

    def __init__(
        self,
        relaxer_type: str = "orb",
        relaxer_config: Dict[str, Any] = {"fmax": 0.02, "steps": 500},
        name: str | None = None,
        description: str | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "StabilityPreprocessor",
            description=description or "Preprocesses structures for stability analysis",
            n_jobs=n_jobs,
        )
        self.config = StabilityPreprocessorConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
            relaxer_type=relaxer_type,
            relaxer_config=relaxer_config,
        )

        # Initialize the relaxer
        self.relaxer = get_relaxer(relaxer_type, **relaxer_config)

    def _get_process_attributes(self) -> dict[str, Any]:
        """Get the attributes for the process_structure method."""
        return {
            "relaxer": self.relaxer,
        }

    @staticmethod
    def process_structure(
        structure: Structure,
        relaxer: BaseRelaxer,
    ) -> Structure:
        """Process a single structure by relaxing it and computing e_above_hull.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to process.
        relaxer : BaseRelaxer
            Relaxer object to use.

        Returns
        -------
        Structure
            The processed Structure with relaxed geometry and e_above_hull in properties.

        Raises
        ------
        Exception
            If relaxation fails or other processing errors occur.
        """
        # Relax structure
        relaxation_result = relaxer.relax(structure, relax=True)
        if not relaxation_result.success:
            raise RuntimeError(f"Relaxation failed: {relaxation_result.message}")

        processed_structure = relaxation_result.structure
        structure.properties["relaxed_structure"] = processed_structure

        e_above_hull_calc = EnergyAboveHull()
        form_energy_calc = OrbFormationEnergy() # currently using orb for formation energy calculation
        # Calculate e_above_hull using LeMatBulk
        try:
            e_above_hull = e_above_hull_calc(structure)
            structure.properties["e_above_hull"] = e_above_hull
            structure.properties["formation_energy"] = form_energy_calc(structure)
            logger.debug(
                f"Computed e_above_hull: {e_above_hull:.3f} eV/atom for unrelaxed {structure.formula}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to compute e_above_hull for unrelaxed {structure.formula}: {str(e)}"
            )
            structure.properties["e_above_hull"] = 0.0
            structure.properties["formation_energy"] = 0.0
            # Still return the relaxed structure even if e_above_hull calculation fails
        try:
            e_above_hull_relaxed = e_above_hull_calc(processed_structure)
            print("energy_above_hull relaxed :", e_above_hull_relaxed)

            structure.properties["relaxed_e_above_hull"] = e_above_hull_relaxed
            structure.properties["relaxed_formation_energy"] = form_energy_calc(
                processed_structure
            )
            logger.debug(
                f"Computed e_above_hull: {e_above_hull_relaxed:.3f} eV/atom for relaxed {processed_structure.formula}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to compute e_above_hull for relaxed {processed_structure.formula}: {str(e)}"
            )
            structure.properties["relaxed_e_above_hull"] = 0.0
            structure.properties["relaxed_formation_energy"] = 0.0
        # Store additional processing metadata
        structure.properties["MLIP"] = "Orb"

        return structure
