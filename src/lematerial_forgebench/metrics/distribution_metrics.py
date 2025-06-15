"""Distribution metrics for evaluating material structures.

This module implements distribution metrics that quantify the degree of similarity
between a set of structures sampled from a generative model and a database of materials.
"""

import time
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure

from lematerial_forgebench.metrics.base import BaseMetric, MetricConfig, MetricResult
from lematerial_forgebench.utils.logging import logger
from lematerial_forgebench.utils.distribution_utils import compute_mmd, generate_probabilities, compute_shannon_entropy, compute_jensen_shannon_distance


@dataclass
class DistributionMetricConfig(MetricConfig):
    """Configuration for the DistributionMetric metric.

    Parameters
    ----------
    xy : 
    xz : , 
    """

    xy: float = 0.1
    xz: bool = False

class JSDistance(BaseMetric): 

    """Calculate shannon entropy for a target distribution.

    This metric compares a set of properties for a target structure (crystal system, 
    space group, elemental composition, lattice constants, and wykoff positions) to 
    a database of structures and determines the similarity of that crystal to the 
    distribution in that database

    Parameters
    ----------
    a : float, 
    b : bool, 
    c : str, optional
    d : str,
    e : bool, 
    f : int, default=x
    """

    def __init__(
        self,
        tolerance: float = 0.1,
        strict: bool = False,
        name: str | None = None,
        description: str | None = None,
        lower_is_better: bool = True,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "Distribution",
            description=description
            or "Measures how close a structure is the target distribution",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )
        self.config = DistributionMetricConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            tolerance=tolerance,
            strict=strict,
        )
    
    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {
            "tolerance": self.config.tolerance,
            "strict": self.config.strict,
            "bv_analyzer": self.bv_analyzer,
        }

    @staticmethod
    def compute_structure(
        structure: pd.DataFrame,
        reference_df: str 
    ) -> float:
        """Compute the similarity of the structure to a target distribution.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to evaluate. TODO list of structures? may already 
            be what this is primed to deal with? 


        Returns
        -------
        float
            Jensen-Shannon Distance
        """

        quantities = structure.columns
        for quant in quantities:
            if quant in reference_df.columns:
                js = compute_jensen_shannon_distance(reference_df, structure, quant)
                dist_metrics = {quant:js}
        
        return dist_metrics 

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

        if not valid_values:
            return {
                "metrics": {
                    "charge_neutrality_error": float("nan"),
                    "charge_neutral_ratio": 0.0,
                },
                "primary_metric": "charge_neutrality_error",
                "uncertainties": {},
            }

        # Count how many structures are within tolerance
        within_tolerance = sum(1 for v in valid_values if v <= self.config.tolerance)
        charge_neutral_ratio = within_tolerance / len(valid_values)

        # Calculate mean absolute deviation
        mean_abs_deviation = np.mean(valid_values)

        return {
            "metrics": {
                "charge_neutrality_error": mean_abs_deviation,
                "charge_neutral_ratio": charge_neutral_ratio,
            },
            "primary_metric": "charge_neutrality_error",
            "uncertainties": {
                "charge_neutrality_error": {
                    "std": np.std(valid_values) if len(valid_values) > 1 else 0.0
                }
            },
        }



class MMD(BaseMetric): 

    """Calculate shannon entropy for a target distribution.

    This metric compares a set of properties for a target structure (crystal system, 
    space group, elemental composition, lattice constants, and wykoff positions) to 
    a database of structures and determines the similarity of that crystal to the 
    distribution in that database

    Parameters
    ----------
    a : float, 
    b : bool, 
    c : str, optional
    d : str,
    e : bool, 
    f : int, default=x
    """

    def __init__(
        self,
        tolerance: float = 0.1,
        strict: bool = False,
        name: str | None = None,
        description: str | None = None,
        lower_is_better: bool = True,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "Distribution",
            description=description
            or "Measures how close a structure is the target distribution",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )
        self.config = DistributionMetricConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            tolerance=tolerance,
            strict=strict,
        )
    
    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {
            "tolerance": self.config.tolerance,
            "strict": self.config.strict,
            "bv_analyzer": self.bv_analyzer,
        }

    @staticmethod
    def compute_structure(
        structure: pd.DataFrame,
        reference_df: str 
    ) -> float:
        """Compute the similarity of the structure to a target distribution.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to evaluate. TODO list of structures? may already 
            be what this is primed to deal with? 


        Returns
        -------
        float
            Jensen-Shannon Distance
        """

        quantities = structure.columns
        for quant in quantities:
            if quant in reference_df.columns:
                mmd = compute_mmd(reference_df, structure, quant)
                dist_metrics = {quant:mmd}
        
        return dist_metrics 

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

        if not valid_values:
            return {
                "metrics": {
                    "charge_neutrality_error": float("nan"),
                    "charge_neutral_ratio": 0.0,
                },
                "primary_metric": "charge_neutrality_error",
                "uncertainties": {},
            }

        # Count how many structures are within tolerance
        within_tolerance = sum(1 for v in valid_values if v <= self.config.tolerance)
        charge_neutral_ratio = within_tolerance / len(valid_values)

        # Calculate mean absolute deviation
        mean_abs_deviation = np.mean(valid_values)

        return {
            "metrics": {
                "charge_neutrality_error": mean_abs_deviation,
                "charge_neutral_ratio": charge_neutral_ratio,
            },
            "primary_metric": "charge_neutrality_error",
            "uncertainties": {
                "charge_neutrality_error": {
                    "std": np.std(valid_values) if len(valid_values) > 1 else 0.0
                }
            },
        }



