"""
Herfindahl-Hirschman Index (HHI) metrics for material supply risk assessment.

This module implements metrics for evaluating the concentration of element
production and reserves, which can indicate supply risk for materials
generation.
"""

import sys
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Composition, Structure

from lematerial_forgebench.metrics.base import (
    BaseMetric,
    MetricConfig,
    MetricResult,
)


def _load_hhi_data():
    """Load HHI data from data_props.py file."""
    try:
        # Get the root directory (go up from metrics folder to
        # lematerial_forgebench to src to root)
        current_file = Path(__file__).resolve()
        root_dir = current_file.parent.parent.parent.parent  # Go up to root
        data_props_path = root_dir / "data" / "data_props.py"

        if not data_props_path.exists():
            raise ImportError(f"data_props.py not found at {data_props_path}")

        # Add root directory to path and import
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))

        from data.data_props import hhi_production, hhi_reserve

        return hhi_production, hhi_reserve
    except ImportError as e:
        raise ImportError(
            f"Could not import HHI data from data.data_props: {e}. "
            "Make sure the data_props.py file is in the root/data directory."
        )


@dataclass
class HHIMetricConfig(MetricConfig):
    """Configuration for HHI metrics.

    Parameters
    ----------
    scale_to_0_10 : bool, default=True
        If True, divide the classical 0 to 10,000 HHI by 1000 to get the 0 to 10
        convenience scale used in the MatterGen paper.
    """

    scale_to_0_10: bool = True


class BaseHHIMetric(BaseMetric, ABC):
    """Base class for HHI-based metrics.

    This class provides common functionality for both HHI production
    and HHI reserve metrics.
    """

    def __init__(
        self,
        hhi_table: dict[str, int],
        name: str | None = None,
        description: str | None = None,
        scale_to_0_10: bool = True,
        n_jobs: int = 1,
    ):
        """Initialize HHI metric with element lookup table.

        Parameters
        ----------
        hhi_table : dict[str, int]
            Per-element HHI values (either production or reserve).
        name : str, optional
            Custom name for the metric.
        description : str, optional
            Description of what the metric measures.
        scale_to_0_10 : bool, default=True
            If True, divide by 1000 to get 0-10 scale.
        n_jobs : int, default=1
            Number of parallel jobs to run.
        """
        self.hhi_table = hhi_table
        self.config = HHIMetricConfig(
            name=name or self.__class__.__name__,
            description=description,
            lower_is_better=True,  # Lower HHI indicates less concentration risk
            n_jobs=n_jobs,
            scale_to_0_10=scale_to_0_10,
        )

    def get_low_risk_structures(
        self,
        structures: list[Structure],
        result: "MetricResult" = None,
        threshold: float = 3.0,
    ) -> tuple[list[int], list[float]]:
        """Identify structures with low HHI values (low supply risk).

        Parameters
        ----------
        structures : list[Structure]
            List of structures that were evaluated.
        result : MetricResult, optional
            Previously computed result. If None, will compute fresh.
        threshold : float, default=3.0
            HHI threshold below which structures are considered low-risk.
            Assumes scaled (0-10) values if scale_to_0_10=True.

        Returns
        -------
        tuple[list[int], list[float]]
            - List of indices of low-risk structures
            - List of corresponding HHI values for those structures

        Examples
        --------
        >>> metric = HHIProductionMetric()
        >>> structures = [...]  # Your structures
        >>> result = metric.compute(structures)
        >>> low_risk_indices, low_risk_values = metric.get_low_risk_structures(
        ...     structures, result, threshold=2.5
        ... )
        >>> print(f"Found {len(low_risk_indices)} low-risk structures")
        >>> # Get the actual low-risk structures
        >>> low_risk_structures = [structures[i] for i in low_risk_indices]
        """
        if result is None:
            result = self.compute(structures)

        low_risk_indices = []
        low_risk_values = []

        for i, value in enumerate(result.individual_values):
            if not np.isnan(value) and value <= threshold:
                low_risk_indices.append(i)
                low_risk_values.append(value)

        return low_risk_indices, low_risk_values

    def get_individual_values_with_structures(
        self,
        structures: list[Structure],
        result: "MetricResult" = None,
        include_failed: bool = False,
    ) -> list[tuple[int, Structure, float]]:
        """Get individual HHI values paired with their corresponding structures.

        Parameters
        ----------
        structures : list[Structure]
            List of structures that were evaluated.
        result : MetricResult, optional
            Previously computed result. If None, will compute fresh.
        include_failed : bool, default=False
            Whether to include structures that failed computation (with NaN
            values).

        Returns
        -------
        list[tuple[int, Structure, float]]
            List of (index, structure, hhi_value) tuples.

        Examples
        --------
        >>> metric = HHIProductionMetric()
        >>> structures = [...]  # Your structures
        >>> structure_values = metric.get_individual_values_with_structures
        >>> # Sort by HHI value to find lowest risk
        >>> structure_values.sort(key=lambda x: x[2])
        >>> print(f"Lowest risk structure has HHI = {structure_values[0][2]}")
        """
        if result is None:
            result = self.compute(structures)

        structure_values = []
        for i, (structure, value) in enumerate(
            zip(structures, result.individual_values)
        ):
            if include_failed or not np.isnan(value):
                structure_values.append((i, structure, value))

        return structure_values

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get attributes needed for computing the metric.

        Returns
        -------
        dict[str, Any]
            Dictionary containing hhi_table and scale_to_0_10.
        """
        return {
            "hhi_table": self.hhi_table,
            "scale_to_0_10": self.config.scale_to_0_10,
        }

    @staticmethod
    def compute_structure(structure: Structure, **compute_args: Any) -> float:
        """Compute HHI for a single structure.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to evaluate.
        **compute_args : Any
            Additional keyword arguments containing hhi_table and scale_to_0_10.

        Returns
        -------
        float
            The computed HHI value for this structure.
        """
        try:
            hhi_table = compute_args["hhi_table"]
            scale_to_0_10 = compute_args["scale_to_0_10"]

            # Get fractional composition
            comp = Composition(structure.composition).fractional_composition

            # Calculate weighted HHI
            # For missing elements, assign maximum HHI value
            # (10000 unscaled / 10 scaled)
            max_hhi_value = 10.0 if scale_to_0_10 else 10000.0

            hhi = 0.0
            for el in comp:
                element_symbol = el.symbol
                if element_symbol in hhi_table:
                    element_hhi = hhi_table[element_symbol]
                    # Apply scaling if requested
                    if scale_to_0_10:
                        element_hhi = element_hhi / 1000.0
                else:
                    # Element not found in HHI table - assign maximum risk value
                    element_hhi = max_hhi_value

                hhi += comp[el] * element_hhi

            return hhi

        except Exception as e:
            # Other computation errors
            raise ValueError(f"Failed to compute HHI: {str(e)}")

    def aggregate_results(self, values: list[float]) -> dict[str, Any]:
        """Aggregate individual HHI results into final metrics.

        Parameters
        ----------
        values : list[float]
            Individual HHI values for each structure.

        Returns
        -------
        dict[str, Any]
            Dictionary containing aggregated metrics and uncertainties.
            Individual values are preserved and accessible through
            MetricResult.individual_values and also included in the metrics
            dictionary as 'individual_hhi_values'.
        """
        valid_values = [v for v in values if not np.isnan(v)]

        if not valid_values:
            primary_metric_name = f"{self.name.lower()}_mean"
            return {
                "metrics": {
                    primary_metric_name: float("nan"),
                    "individual_hhi_values": values,  # Include even if all NaN
                },
                "primary_metric": primary_metric_name,
                "uncertainties": {},
            }

        # Compute statistics
        mean_hhi = np.mean(valid_values)
        std_hhi = np.std(valid_values) if len(valid_values) > 1 else 0.0
        min_hhi = np.min(valid_values)
        max_hhi = np.max(valid_values)
        median_hhi = np.median(valid_values)

        # Calculate percentiles for risk assessment
        percentile_25 = np.percentile(valid_values, 25)
        percentile_75 = np.percentile(valid_values, 75)

        # Count low-risk structures (below different thresholds)
        # These thresholds assume scaled (0-10) values
        low_risk_count_2 = sum(1 for v in valid_values if v <= 2.0)
        low_risk_count_3 = sum(1 for v in valid_values if v <= 3.0)
        low_risk_count_5 = sum(1 for v in valid_values if v <= 5.0)

        primary_metric_name = f"{self.name.lower()}_mean"
        prefix = self.name.lower()
        count_valid = len(valid_values)

        metrics = {
            # Individual values - in same order as input structures
            "individual_hhi_values": values,
            # Basic statistics
            primary_metric_name: mean_hhi,
            f"{prefix}_std": std_hhi,
            f"{prefix}_min": min_hhi,
            f"{prefix}_max": max_hhi,
            f"{prefix}_median": median_hhi,
            f"{prefix}_25th_percentile": percentile_25,
            f"{prefix}_75th_percentile": percentile_75,
            # Risk assessment metrics
            f"{prefix}_low_risk_count_2": low_risk_count_2,
            f"{prefix}_low_risk_count_3": low_risk_count_3,
            f"{prefix}_low_risk_count_5": low_risk_count_5,
            f"{prefix}_low_risk_fraction_2": (low_risk_count_2 / count_valid),
            f"{prefix}_low_risk_fraction_3": (low_risk_count_3 / count_valid),
            f"{prefix}_low_risk_fraction_5": (low_risk_count_5 / count_valid),
            # Count metrics
            "total_structures_evaluated": count_valid,
            "failed_structures_count": len(values) - count_valid,
        }

        uncertainties = {
            primary_metric_name: {
                "std": std_hhi,
                "std_error": (std_hhi / np.sqrt(count_valid) if valid_values else 0),
            }
        }

        return {
            "metrics": metrics,
            "primary_metric": primary_metric_name,
            "uncertainties": uncertainties,
        }


class HHIProductionMetric(BaseHHIMetric):
    """Herfindahl-Hirschman Index for production concentration.

    This metric evaluates the concentration of element production sources,
    indicating supply risk based on market concentration.
    Higher values indicate more concentrated supply chains and higher risk.

    Elements not found in the HHI production data are automatically assigned
    the maximum HHI value (10000 unscaled / 10 scaled), representing maximum
    supply risk for rare or untracked elements.
    """

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        scale_to_0_10: bool = True,
        n_jobs: int = 1,
    ):
        """Initialize HHI Production metric.

        Parameters
        ----------
        name : str, optional
            Custom name for the metric.
        description : str, optional
            Description of the metric.
        scale_to_0_10 : bool, default=True
            If True, divide by 1000 to get 0-10 scale.
        n_jobs : int, default=1
            Number of parallel jobs to run.
        """
        # Load HHI data
        hhi_production, _ = _load_hhi_data()

        super().__init__(
            hhi_table=hhi_production,
            name=name or "HHIProduction",
            description=description
            or (
                "Herfindahl-Hirschman Index for element production "
                "concentration. Higher values indicate more concentrated "
                "supply chains and higher supply risk."
            ),
            scale_to_0_10=scale_to_0_10,
            n_jobs=n_jobs,
        )


class HHIReserveMetric(BaseHHIMetric):
    """Herfindahl-Hirschman Index for reserve concentration.

    This metric evaluates the concentration of element reserves,
    indicating long-term supply risk based on reserve distribution.
    Higher values indicate more concentrated reserves and higher long-term risk.

    Elements not found in the HHI reserve data are automatically assigned
    the maximum HHI value (10000 unscaled / 10 scaled), representing maximum
    supply risk for rare or untracked elements.
    """

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        scale_to_0_10: bool = True,
        n_jobs: int = 1,
    ):
        """Initialize HHI Reserve metric.

        Parameters
        ----------
        name : str, optional
            Custom name for the metric.
        description : str, optional
            Description of the metric.
        scale_to_0_10 : bool, default=True
            If True, divide by 1000 to get 0-10 scale.
        n_jobs : int, default=1
            Number of parallel jobs to run.
        """
        # Load HHI data
        _, hhi_reserve = _load_hhi_data()

        super().__init__(
            hhi_table=hhi_reserve,
            name=name or "HHIReserve",
            description=description
            or (
                "Herfindahl-Hirschman Index for element reserve concentration. "
                "Higher values indicate more concentrated reserves and higher "
                "long-term supply risk."
            ),
            scale_to_0_10=scale_to_0_10,
            n_jobs=n_jobs,
        )


# Convenience function for easy usage
def compound_hhi(formula: str, hhi_table: dict, scale_to_0_10: bool = True) -> float:
    """
    Return the Herfindahl-Hirschman Index of a compound.

    Parameters
    ----------
    formula : str
        Chemical formula, e.g. "Nd2Fe14B".
    hhi_table : dict[str, int]
        Per-element HHI values (either production or reserve).
    scale_to_0_10 : bool, optional
        If True, divide the classical 0 to 10,000 HHI by 1000 to get the 0 to 10
        convenience scale used in the MatterGen paper.

    Returns
    -------
    float
        The weighted HHI value for the compound.
        Elements not found in hhi_table are assigned maximum HHI value
        (10000/10).

    Examples
    --------
    >>> hhi_production, hhi_reserve = _load_hhi_data()
    >>> compound_hhi("Nd2Fe14B", hhi_production)
    5.234
    >>> compound_hhi("LiFePO4", hhi_reserve, scale_to_0_10=False)
    3456.7
    >>> compound_hhi("UPt3", hhi_production)  # U might not be in table
    8.5  # High value due to rare U
    """
    comp = Composition(formula).fractional_composition

    # For missing elements, assign maximum HHI value
    max_hhi_value = 10.0 if scale_to_0_10 else 10000.0

    hhi = 0.0
    for el in comp:
        element_symbol = el.symbol
        if element_symbol in hhi_table:
            element_hhi = hhi_table[element_symbol]
            # Apply scaling if requested
            if scale_to_0_10:
                element_hhi = element_hhi / 1000.0
        else:
            # Element not found in HHI table - assign maximum risk value
            element_hhi = max_hhi_value

        hhi += comp[el] * element_hhi

    return hhi
