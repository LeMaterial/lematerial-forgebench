"""Distribution metrics for evaluating material structures.

This module implements distribution metrics that quantify the degree of similarity
between a set of structures sampled from a generative model and a database of materials.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure

from lematerial_forgebench.metrics.base import BaseMetric, MetricConfig, MetricResult
from lematerial_forgebench.utils.distribution_utils import (
    compute_frechetdist,
    compute_jensen_shannon_distance,
    compute_mmd,
    compute_shannon_entropy,
)
from lematerial_forgebench.utils.logging import logger


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

    """Calculate Jensen-Shannon distance between two distributions.

    This metric compares a set of distribution wide properties (crystal system, 
    space group, elemental composition, lattice constants, and wykoff positions) 
    between two samples of crystal structures and determines the degree of similarity 
    between those two distributions for the particular structural property. 

    Parameters
    ----------

    """

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "Distribution",
            description=description
            or "Measures distance between two reference distributions",
            n_jobs=n_jobs,
        )
        self.config = DistributionMetricConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
        )
    
    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {
        }

    @staticmethod
    def compute_structure(
        structure: pd.DataFrame,
        reference_df: str 
    ) -> dict:
        """Compute the similarity of the structure to a target distribution.

        Parameters
        ----------
        structure : pandas DataFrame 
            Contains the values of the structural properties of interest for 
            each of the structures in the distribution. This dataframe is 
            calculated by "src/lematerial_forgebench/preprocess/distribution_preprocess.py"
            which specifies the format, column names etc used here for compatibility with
            the reference datasets. When changing the reference dataset, ensure the 
            column names etc correspond to those found in the above script. 
        
        Returns
        -------
        dict 
            Jensen-Shannon Distances, where the keys are the structural property 
            and the values are the JS Distances. 
        """

        quantities = structure.columns
        dist_metrics = {}
        for quant in quantities:
            if quant in reference_df.columns:
                # print(quant)
                # print(type(quant))
                if isinstance(reference_df[quant].iloc[0], np.float64):
                    pass
                else:
                    js = compute_jensen_shannon_distance(reference_df, structure, quant,
                                                         metric_type=type(reference_df[quant].iloc[0]))
                    dist_metrics[quant] = js
                    print(dist_metrics)
        
        for quant in ["CompositionCounts", "Composition"]:
            print(quant)
            print(type(quant))
            js = compute_jensen_shannon_distance(reference_df, structure, quant,
                                                metric_type=type(structure[quant].iloc[0]))
            dist_metrics[quant] = js
            print(dist_metrics)
        return dist_metrics 

    def aggregate_results(self, values: dict[str, float]) -> Dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : dict[str, float]
            Jensen-Shannon Distance values for each structural property. 

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        # Filter out NaN values
        valid_values = [v for v in values.values() if not np.isnan(v)]
        if not valid_values:
            return {
                "metrics": {
                    "Jensen_Shannon_Distance": float("nan"),
                },
                "primary_metric": "Jensen_Shannon_Distance",
                "uncertainties": {},
            }

        return {
            "metrics": {
                "Jensen_Shannon_Distance": values,
            },
            "primary_metric": "Jensen_Shannon_Distance",
            "uncertainties": {}
        }


class MMD(BaseMetric): 

    """Calculate MMD between two distributions.

    This metric compares a set of distribution wide properties (crystal system, 
    space group, elemental composition, lattice constants, and wykoff positions) 
    between two samples of crystal structures and determines the degree of similarity 
    between those two distributions for the particular structural property. 

    """

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "Distribution",
            description=description
            or "Measures distance between two reference distributions",
            n_jobs=n_jobs,
        )
        self.config = DistributionMetricConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
        )
    
    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {
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
            MMD 
        """
        print("starting MMD")
        np.random.seed(32)
        if len(reference_df) > 10000: 
            ref_ints = np.random.randint(0,len(reference_df), 10000)
            ref_sample_df = reference_df.iloc[ref_ints]
        else:
            ref_sample_df = reference_df
        if len(structure) > 10000: 
            strut_ints = np.random.randint(0,len(structure), 10000)
            strut_sample_df = structure.iloc[strut_ints]
        else:
            strut_sample_df = structure
        dist_metrics = {}
        quantities = strut_sample_df.columns
        for quant in quantities:
            if quant in ref_sample_df.columns:
                print(quant)
                print(type(quant))
                if isinstance(ref_sample_df[quant].iloc[0], np.int64):
                    pass
                else:
                    try:
                        mmd = compute_mmd(ref_sample_df, strut_sample_df, quant)
                        dist_metrics[quant] = mmd
                        print(dist_metrics)

                        dist_metrics[quant] = mmd
                    except ValueError:
                        pass
        
        return dist_metrics 

    def aggregate_results(self, values: dict[str, float]) -> Dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : dict[str, float]
            Jensen-Shannon Distance values for each structural property. 

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        # Filter out NaN values
        valid_values = [v for v in values.values() if not np.isnan(v)]
        if not valid_values:
            return {
                "metrics": {
                    "MMD": float("nan"),
                },
                "primary_metric": "MMD",
                "uncertainties": {},
            }

        return {
            "metrics": {
                "MMD": values,
            },
            "primary_metric": "MMD",
            "uncertainties": {}
        }


class FrechetDistance(BaseMetric): 

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
                mmd = compute_frechetdist(reference_df, structure, quant)
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
        print("filtering")
        print(values)
        valid_values = [v for v in values if not np.isnan(v)]
        print("made it here")

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