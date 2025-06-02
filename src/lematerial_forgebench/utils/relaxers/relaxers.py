"""Implementations for structure relaxation."""

import gc
import tempfile
from abc import abstractmethod

import torch
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

# from fairchem.core import OCPCalculator
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.sets import MPRelaxSet

from lematerial_forgebench.utils.logging import logger
from lematerial_forgebench.utils.relaxers.registry import (
    BaseRelaxer,
    RelaxationResult,
    register_relaxer,
)


class BaseVASPRelaxer(BaseRelaxer):
    """Base class for relaxers that use VASP-like parameters."""

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
            The computed structure entry with MP2020 corrections applied.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            b = MPRelaxSet(structure)
            b.write_input(f"{tmpdirname}/", potcar_spec=True)
            poscar = Poscar.from_file(f"{tmpdirname}/POSCAR")
            incar = Incar.from_file(f"{tmpdirname}/INCAR")
            clean_structure = Structure.from_file(f"{tmpdirname}/POSCAR")

        # Get the U values and figure out if we should have run a GGA+U calc
        param = {"hubbards": {}}
        if "LDAUU" in incar:
            param["hubbards"] = dict(zip(poscar.site_symbols, incar["LDAUU"]))
        param["is_hubbard"] = (
            incar.get("LDAU", True) and sum(param["hubbards"].values()) > 0
        )
        if param["is_hubbard"]:
            param["run_type"] = "GGA+U"

        # Make a ComputedStructureEntry without the correction
        cse_d = {
            "structure": clean_structure,
            "energy": energy,
            "correction": 0.0,
            "parameters": param,
        }

        # Apply the MP 2020 correction scheme (anion/+U/etc)
        cse = ComputedStructureEntry.from_dict(cse_d)
        _ = MaterialsProject2020Compatibility(check_potcar=False).process_entries(
            cse,
            clean=True,
        )

        return cse


class ASERelaxerBase(BaseVASPRelaxer):
    """Base class for ASE-based relaxers."""

    def __init__(
        self,
        fmax: float = 0.02,
        steps: int = 500,
        cpu: bool = True,
        **params,
    ):
        """Initialize ASE relaxer.

        Parameters
        ----------
        fmax : float, default=0.02
            Maximum force convergence criterion.
        steps : int, default=500
            Maximum number of optimization steps.
        cpu : bool, default=False
            Whether to use CPU instead of GPU.
        **params
            Additional parameters specific to each relaxer type.
        """
        self.fmax = fmax
        self.steps = steps
        self.cpu = cpu
        self.params = params  # Store additional parameters for subclasses
        # Abstract calculator setup - must be implemented by subclasses
        self.calc = self._setup_calculator()

    @abstractmethod
    def _setup_calculator(self):
        """Set up the calculator for this relaxer.

        Returns
        -------
        Calculator
            The initialized calculator object.
        """
        pass

    def relax(self, structure: Structure, relax: bool = False) -> RelaxationResult:
        """Relax a structure using calculator.

        Parameters
        ----------
        structure : Structure
            Structure to relax.
        relax : bool, default=False
            Only relaxes the structure if True.

        Returns
        -------
        RelaxationResult
            Result of the relaxation.
        """

        try:
            if structure is None or not structure.is_valid():
                print("Skipping structure: Invalid crystal")
                return RelaxationResult(
                    success=False,
                    message="Invalid crystal",
                )

            # Convert to ASE atoms
            atoms = structure.to_ase_atoms()

            atoms.calc = self.calc

            # Relax structure

            if relax:
                dyn = FIRE(FrechetCellFilter(atoms), logfile=None)
                dyn.run(fmax=self.fmax, steps=self.steps)

            # Get results
            final_energy = atoms.get_potential_energy()

            final_structure = AseAtomsAdaptor.get_structure(atoms)

            # Clean up
            # gc.collect()
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

            return RelaxationResult(
                success=True,
                energy=final_energy,
                structure=final_structure,
            )
        except Exception as e:
            logger.error(f"ASE relaxation failed: {str(e)}")
            return RelaxationResult(
                success=False,
                message=str(e),
            )


@register_relaxer("orb")
class OrbRelaxerImpl(ASERelaxerBase):
    """Orb relaxer implementation."""

    def _setup_calculator(self):
        """Set up the Orb calculator."""
        device = "cpu" if self.cpu else "cuda"
        print(device)
        if self.params.get("direct", False):
            orbff = pretrained.orb_v3_direct_inf_mpa(
                device=device,
                compile=False,
                precision="float32-high",  # or "float32-highest" / "float64
            )
        else:
            orbff = pretrained.orb_v3_conservative_inf_mpa(
                device=device,
                compile=False,
                precision="float32-high",  # or "float32-highest" / "float64
            )
        return ORBCalculator(orbff, device=device)


# TODO: Fix Meta Fairchem Relaxers
# @register_relaxer("ocp")
# class OCPRelaxerImpl(ASERelaxerBase):
#     """OCP relaxer implementation."""

#     def _setup_calculator(self):
#         """Set up the OCP calculator."""
#         return OCPCalculator(checkpoint_path=self.checkpoint_path, cpu=self.cpu)
