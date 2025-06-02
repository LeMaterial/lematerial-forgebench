"""Conditional generation metrics for evaluating material structures.

This module implements metrics that determine the degree to which a target metric
was produced by a set of generated structures.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List
from matgl import load_model
import torch

import numpy as np
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure

from lematerial_forgebench.metrics.base import BaseMetric, MetricConfig, MetricResult
from lematerial_forgebench.utils.logging import logger


@dataclass
class BandgapPropertyTargetConfig(MetricConfig):
    """Configuration for the ChargeNeutrality metric.

    Parameters
    ----------
    tolerance : float, default=0.1
        Tolerance for deviations from charge neutrality.
    strict : bool, default=False
        If True, oxidation states must be determinable for all atoms.
        If False, will attempt to calculate oxidation states but pass
        the structure if calculation fails.
    """

    target_theory: str = "PBE"
    target_bandgap: float = 1
    model: str = "MEGNet-MP-2019.4.1-BandGap-mfi"
    tolerance: float = 0.1

class BandgapPropertyTargetMetric(BaseMetric):
    """
    """


    def __init__(
        self,
        target_theory: str = "PBE",
        target_bandgap: float = 1,
        model: str = "MEGNet-MP-2019.4.1-BandGap-mfi",
        tolerance: float = 0.1, 
        lower_is_better: bool = True,
        name: str | None = None,
        description: str | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "Bandgap",
            description=description
            or "Computes bandgap with selected model and compares to target bandgap " + model,
            tolerance = 0.1, 
            lower_is_better = lower_is_better,
            n_jobs=n_jobs
        )
        self.config = BandgapPropertyTargetConfig(
            name=self.config.name,
            description=self.config.description,
            tolerance=self.config.tolerance, 
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            target_theory=target_theory,
            target_bandgap=target_bandgap,
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {
        "target_theory" : self.config.target_theory,
        "target_bandgap": self.config.target_bandgap,
        "model": self.config.model,
        "tolerance": self.config.tolerance
        }

    @staticmethod
    def compute_structure(
        structure: Structure, target_theory: str, target_bandgap: float, model: str, 
    ) -> float:
        """
        # density metric - sliding window??

        Parameters
        ----------

        """
        print(model)
        band_gap_model = load_model(model)
        if target_theory == "PBE":
            graph_attrs = torch.tensor([0])
        elif target_theory == "HSE":
            graph_attrs = torch.tensor([2])
        bandgap = band_gap_model.predict_structure(
                structure=structure, state_attr=graph_attrs
            )
       
        return np.abs(bandgap - target_bandgap) 

    def aggregate_results(self, values: list[float]) -> Dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            Absolute deviations from charge neutrality for each structure.

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        # Filter out NaN values
        valid_values = [v for v in values if not np.isnan(v)]
        print(valid_values)

        if not valid_values:
            raise ValueError 

        # Count how many structures are within tolerance
        within_tolerance = sum(1 for v in valid_values if v <= self.config.tolerance)
        success_rate = within_tolerance / len(valid_values)

        # Calculate mean absolute deviation
        average_property_proximity = np.mean(valid_values)

        return {
            "metrics": {
                "average_property_proximity": average_property_proximity,
                "success_rate": success_rate,
            },
            "primary_metric": "average_property_proximity",
            "uncertainties": {
                "average_property_proximity": {
                    "std": np.std(valid_values) if len(valid_values) > 1 else 0.0
                }
            },
        }