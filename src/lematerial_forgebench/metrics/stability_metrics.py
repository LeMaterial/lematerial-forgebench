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

from lematerial_forgebench.metrics.base import BaseMetric, MetricConfig
from lematerial_forgebench.utils.e_above_hull import (
    generate_CSE,
    get_patched_phase_diagram_mp,
)
from lematerial_forgebench.utils.logging import logger
from lematerial_forgebench.utils.relaxers import BaseRelaxer, get_relaxer


@dataclass
class StabilityMetricConfig(MetricConfig):
    """Configuration for the StabilityMetric.

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


class StabilityMetric(BaseMetric):
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
        relaxer_type: str,
        relaxer_config: Dict[str, Any],
        mp_entries_file: Optional[str] = None,
        name: str | None = None,
        description: str | None = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "StabilityMetric",
            description=description or "Evaluates structure stability",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )
        self.config = StabilityMetricConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            relaxer_type=relaxer_type,
            relaxer_config=relaxer_config,
            mp_entries_file=mp_entries_file,
        )

        # Initialize the relaxer
        self.relaxer = get_relaxer(relaxer_type, **relaxer_config)

        # Initialize MP compatibility
        self.compatibility = MaterialsProject2020Compatibility(check_potcar=False)

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

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {
            "relaxer": self.relaxer,
            "compatibility": self.compatibility,
            "mp_entries": self.mp_entries,
        }

    @staticmethod
    def compute_structure(
        structure: Structure,
        relaxer: BaseRelaxer,
        compatibility: MaterialsProject2020Compatibility,
        mp_entries: Optional[PatchedPhaseDiagram] = None,
    ) -> Dict[str, Any]:
        """Compute relaxation and energy above hull for a structure.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to evaluate.
        relaxer : BaseRelaxer
            Relaxer object to use.
        compatibility : MaterialsProject2020Compatibility
            MP compatibility scheme.
        mp_entries : list[ComputedEntry], optional
            MP entries for e_above_hull calculation.

        Returns
        -------
        dict
            Dictionary containing relaxation results and e_above_hull.
        """
        try:
            # Relax structure
            relaxation_result = relaxer.relax(structure)
            if not relaxation_result.success:
                logger.error(f"Relaxation failed: {relaxation_result.message}")
                return np.nan

            result = {
                "success": True,
                "relaxed_energy": relaxation_result.energy,
                "relaxed_structure": relaxation_result.structure,
                "e_above_hull": None,
            }

            # Calculate e_above_hull if MP entries are available
            if mp_entries is not None:
                # Calculate e_above_hull
                cse = generate_CSE(
                    relaxation_result.structure, relaxation_result.energy
                )

                e_above_hull = mp_entries.get_e_above_hull(cse, allow_negative=True)
                result["e_above_hull"] = e_above_hull
            return result["e_above_hull"]
        except Exception as e:
            logger.error(f"Computation failed: {str(e)}")
            return np.nan

    def aggregate_results(self, values: list[float]) -> dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            List of computation results for each structure.

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        # Convert to numpy array for efficient operations
        values_array = np.array(values)

        # Filter out NaN values
        valid_mask = ~np.isnan(values_array)
        e_above_hull_values = values_array[valid_mask]

        if len(e_above_hull_values) > 0:
            mean_e_above_hull = np.mean(e_above_hull_values)
            e_above_hull_std = (
                np.std(e_above_hull_values) if len(e_above_hull_values) > 1 else 0.0
            )
            # Calculate ratio of stable structures (e_above_hull <= 0) using numpy
            stable_ratio = np.sum(e_above_hull_values <= 0) / len(values)
            # Calculate ratio of metastable structures (e_above_hull <= 0.1) using numpy
            metastable_ratio = np.sum(e_above_hull_values <= 0.1) / len(values)
        else:
            mean_e_above_hull = 0.0
            e_above_hull_std = 0.0
            stable_ratio = 0.0
            metastable_ratio = 0.0

        return {
            "metrics": {
                "mean_e_above_hull": mean_e_above_hull,
                "stable_ratio": stable_ratio,
                "metastable_ratio": metastable_ratio,
            },
            "primary_metric": "stable_ratio",
            "uncertainties": {
                "e_above_hull_std": {"std": e_above_hull_std},
            },
        }
