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
from lematerial_forgebench.utils.logging import logger


class StabilityMetric(BaseMetric):
    """Evaluate structure metastability using precomputed e_above_hull values.

    This metric assumes that e_above_hull values have already been computed
    and stored in structure.properties['e_above_hull']. It calculates stability
    statistics without performing any relaxation or recomputation.
    """

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "StabilityMetric",
            description=description
            or "Evaluates structure stability from precomputed e_above_hull",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {}

    @staticmethod
    def compute_structure(structure: Structure) -> float:
        """Extract precomputed e_above_hull from structure properties.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object with e_above_hull in properties.

        Returns
        -------
        float
            The precomputed e_above_hull value, or NaN if not available.
        """
        try:
            # Extract e_above_hull from structure properties
            e_above_hull = structure.properties.get("e_above_hull", None)

            if e_above_hull is None:
                logger.warning(
                    "Structure missing e_above_hull in properties, please compute it first using StabilityMetric"
                )
                return np.nan

            return float(e_above_hull)

        except Exception as e:
            logger.error(f"Failed to extract e_above_hull: {str(e)}")
            return np.nan

    def aggregate_results(self, values: list[float]) -> dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            List of e_above_hull values for each structure.

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
            # Calculate ratio of stable structures (e_above_hull <= 0) using numpy
            stable_ratio = np.sum(e_above_hull_values <= 0) / len(values)
            e_above_hull_std = np.std(e_above_hull_values)
            mean_e_above_hull = np.mean(e_above_hull_values)
        else:
            stable_ratio = 0.0
            e_above_hull_std = 0.0
            mean_e_above_hull = 0.0

        return {
            "metrics": {
                "stable_ratio": stable_ratio,
                "mean_e_above_hull": mean_e_above_hull,
            },
            "primary_metric": "stable_ratio",
            "uncertainties": {
                "e_above_hull_std": {"std": e_above_hull_std},
            },
        }


class MetastabilityMetric(BaseMetric):
    """Evaluate structure metastability using precomputed e_above_hull values.

    This metric assumes that e_above_hull values have already been computed
    and stored in structure.properties['e_above_hull']. It calculates stability
    statistics without performing any relaxation or recomputation.
    """

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        lower_is_better: bool = True,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "MetastabilityMetric",
            description=description
            or "Evaluates structure metastability from precomputed e_above_hull",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {}

    @staticmethod
    def compute_structure(structure: Structure) -> float:
        """Extract precomputed e_above_hull from structure properties.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object with e_above_hull in properties.

        Returns
        -------
        float
            The precomputed e_above_hull value, or NaN if not available.
        """
        try:
            # Extract e_above_hull from structure properties
            e_above_hull = structure.properties.get("e_above_hull", None)

            if e_above_hull is None:
                logger.warning(
                    f"Structure `{structure.formula}` missing e_above_hull in properties, please compute it first using StabilityPreprocessor"
                )
                return np.nan

            return float(e_above_hull)

        except Exception as e:
            logger.error(f"Failed to extract e_above_hull: {str(e)}")
            return np.nan

    def aggregate_results(self, values: list[float]) -> dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            List of e_above_hull values for each structure.

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
            # Calculate ratio of metastable structures (e_above_hull <= 0.1) using numpy
            metastable_ratio = np.sum(e_above_hull_values <= 0.1) / len(values)
        else:
            metastable_ratio = 0.0

        return {
            "metrics": {
                "metastable_ratio": metastable_ratio,
            },
            "primary_metric": "metastable_ratio",
            "uncertainties": {},
        }
