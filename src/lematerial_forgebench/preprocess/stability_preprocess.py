"""Relaxation metrics for evaluating material structures.

This module implements metrics for evaluating the relaxation of
material structures using various relaxation models and calculating
energy above hull.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram
from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

from lematerial_forgebench.preprocess.base import BasePreprocessor, PreprocessorConfig
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
        mp_entries_file: Optional[
            str
        ] = "src/lematerial_forgebench/utils/relaxers/2023-02-07-ppd-mp.pkl.gz",
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
            mp_entries_file=mp_entries_file,
        )

        # Initialize the relaxer
        self.relaxer = get_relaxer(relaxer_type, **relaxer_config)

        # Load MP entries if file provided
        self.mp_entries = None
        # check if path exists
        if mp_entries_file and Path(mp_entries_file).exists():
            self.load_mp_entries(mp_entries_file)

    def load_mp_entries(self, mp_entries_file: str) -> None:
        """Load Materials Project entries for e_above_hull calculation.

        Parameters
        ----------
        mp_entries_file : str
            Path to the MP entries file.
        """
        self.mp_entries = get_patched_phase_diagram_mp(Path(mp_entries_file))
        # import pandas as pd

        # try:
        #     df = pd.read_json(mp_entries_file)
        #     df = df[df["entry"].apply(lambda x: "GGA" in x["entry_id"])]

        #     mp_computed_entries = [
        #         ComputedEntry.from_dict(x)
        #         for x in df.entry
        #         if "GGA" in x["parameters"]["run_type"]
        #     ]
        #     self.mp_entries = [
        #         entry
        #         for entry in mp_computed_entries
        #         if not np.any(["R2SCAN" in a.name for a in entry.energy_adjustments])
        #     ]
        #     logger.info(f"Loaded {len(self.mp_entries)} MP entries")
        # except Exception as e:
        #     logger.error(f"Failed to load MP entries: {str(e)}")
        #     self.mp_entries = None

    def _get_process_attributes(self) -> dict[str, Any]:
        """Get the attributes for the process_structure method."""
        return {
            "relaxer": self.relaxer,
            "mp_entries": self.mp_entries,
        }

    @staticmethod
    def process_structure(
        structure: Structure,
        relaxer: BaseRelaxer,
        mp_entries: Optional[PatchedPhaseDiagram] = None,
    ) -> Structure:
        """Process a single structure by relaxing it and computing e_above_hull.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to process.
        relaxer : BaseRelaxer
            Relaxer object to use.
        mp_entries : PatchedPhaseDiagram, optional
            MP entries for e_above_hull calculation.

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
        relaxation_result = relaxer.relax(structure, relax=False)
        if not relaxation_result.success:
            raise RuntimeError(f"Relaxation failed: {relaxation_result.message}")

        processed_structure = relaxation_result.structure

        # Calculate e_above_hull if MP entries are available
        if mp_entries is not None:
            try:
                cse = generate_CSE(processed_structure, relaxation_result.energy)
                e_above_hull = mp_entries.get_e_above_hull(cse, allow_negative=True)
                processed_structure.properties["e_above_hull"] = e_above_hull
                logger.debug(
                    f"Computed e_above_hull: {e_above_hull:.3f} eV/atom for {processed_structure.formula}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to compute e_above_hull for {processed_structure.formula}: {str(e)}"
                )
                # Still return the relaxed structure even if e_above_hull calculation fails

        # Store additional processing metadata
        processed_structure.properties["relaxed_energy"] = relaxation_result.energy

        return processed_structure
