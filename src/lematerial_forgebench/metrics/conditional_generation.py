"""Conditional generation metrics for evaluating material structures.

This module implements metrics that determine the degree to which a target metric
was produced by a set of generated structures.
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
from matgl import load_model
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from lematerial_forgebench.metrics.base import BaseMetric, MetricConfig
from lematerial_forgebench.utils.logging import logger


@dataclass(kw_only=True)
class DiscreteTargetConfig(MetricConfig):
    """Configuration for the DiscreteTarget metric.

    Parameters
    ----------
    target_value : Any
        The target discrete value to match.
    """

    target_value: Any


class DiscreteTargetMetric(BaseMetric):
    """Generic metric for evaluating if structures match a target discrete value.

    Parameters
    ----------
    target_value : Any
        The target discrete value to match.
    lower_is_better : bool, default=True
        Whether a lower value is better.
    name : str, default=None
        The name of the metric.
    description : str, default=None
        The description of the metric.
    n_jobs : int, default=1
        The number of jobs to run in parallel.
    """

    def __init__(
        self,
        target_value: Any,
        lower_is_better: bool = True,
        name: str | None = None,
        description: str | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or f"DiscreteTarget_{str(target_value)}",
            description=description
            or f"Measures fraction of structures matching target value {str(target_value)}",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )
        self.config = DiscreteTargetConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            target_value=target_value,
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {
            "cls": self.__class__,
        }

    @staticmethod
    def value_extractor(structure: Structure) -> Any:
        """Extract the relevant discrete value from a structure.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement value_extractor")

    @staticmethod  # TODO: Should this be a classmethod?
    def compute_structure(structure: Structure, cls, **kwargs) -> float | None:
        """Computes and returns the extracted value from the structure.
        Returns None if value extraction fails.
        """
        try:
            value = cls.value_extractor(structure, **kwargs)
            return float(value) if value is not None else np.nan
        except Exception as e:
            logger.warning(
                f"Value extraction failed for structure {structure.formula}: {e}"
            )
            return None

    def aggregate_results(self, values: list[float | None]) -> Dict[str, Any]:
        """Aggregate results into final metric values."""
        valid_values = [v for v in values if v is not None]

        if not valid_values:
            raise ValueError("No valid structures for discrete target metric.")

        # Convert values to success/failure based on exact equality
        success_values = [
            1.0 if v == self.config.target_value else 0.0 for v in valid_values
        ]

        success_rate = np.mean(success_values)

        return {
            "metrics": {
                "success_rate": success_rate,
                "mean_value": np.mean(valid_values),
            },
            "primary_metric": "success_rate",
            "uncertainties": {
                "success_rate": {
                    "std": np.std(success_values) if len(success_values) > 1 else 0.0
                },
                "mean_value": {
                    "std": np.std(valid_values) if len(valid_values) > 1 else 0.0
                },
            },
        }


@dataclass(kw_only=True)
class ContinuousTargetConfig(MetricConfig):
    """Configuration for the ContinuousTarget metric.

    Parameters
    ----------
    target_value : float
        The target continuous value to match.
    tolerance : Optional[float], default=0.1
        Tolerance for deviations from target value.
        If None, the metric returns the distance metric between the target value and the structure value.
    top_k : int | None = None
        The number of best structures to consider for the metric.
        If None, all structures are considered.
    distance_metric : str, default="absolute"
        The metric to use for measuring distance to target.
        Options:
        - "absolute": Absolute difference |x - target|
        - "relative": Relative difference |x - target|/|target|
        - "squared": Squared difference (x - target)**2
    """

    target_value: float
    tolerance: float | None = None
    top_k: int | None = None
    distance_metric: str = "absolute"


class ContinuousTargetMetric(BaseMetric):
    """Generic metric for evaluating how close structures are to a target continuous value."""

    DISTANCE_METRICS = {
        "absolute": lambda x, target: abs(x - target),
        "relative": lambda x, target: abs(x - target)
        / (abs(target) if target != 0 else 1.0),
        "squared": lambda x, target: (x - target) ** 2,
    }

    def __init__(
        self,
        target_value: float,
        tolerance: float | None = None,
        top_k: int | None = None,
        distance_metric: str = "absolute",
        name: str | None = None,
        description: str | None = None,
        n_jobs: int = 1,
        lower_is_better: bool | None = None,  # Added parameter
    ):
        # If tolerance is None, the metric uses distance so lower is better
        if lower_is_better is None:
            lower_is_better = tolerance is None

        if distance_metric not in self.DISTANCE_METRICS:
            raise ValueError(
                f"Invalid distance metric '{distance_metric}'. "
                f"Must be one of: {list(self.DISTANCE_METRICS.keys())}"
            )

        super().__init__(
            name=name or f"ContinuousTarget_{str(target_value)}",
            description=description
            or f"Measures {distance_metric} distance to target value {str(target_value)}",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )
        self.config = ContinuousTargetConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            target_value=target_value,
            tolerance=tolerance,
            top_k=top_k,
            distance_metric=distance_metric,
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {
            "target_value": self.config.target_value,
            "distance_metric": self.config.distance_metric,
            "cls": self.__class__,
        }

    @staticmethod
    def value_extractor(structure: Structure) -> float:
        """Extract the relevant continuous value from a structure.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement value_extractor")

    @staticmethod
    def compute_structure(
        structure: Structure, target_value: float, distance_metric: str, cls, **kwargs
    ) -> float:
        """Computes the distance from the target value using the specified metric."""
        try:
            value = cls.value_extractor(structure, **kwargs)
            return cls.DISTANCE_METRICS[distance_metric](value, target_value)
        except Exception as e:
            logger.warning(
                f"Value extraction failed for structure {structure.formula}: {e}"
            )
            return float("nan")

    def aggregate_results(self, values: list[float]) -> Dict[str, Any]:
        """Aggregate results into final metric values."""
        valid_values = [v for v in values if not np.isnan(v)]

        if not valid_values:
            raise ValueError("No valid structures for continuous target metric.")

        if self.config.top_k is not None:
            valid_values = np.sort(valid_values)[: self.config.top_k]

        # Count how many structures are within tolerance
        if self.config.tolerance is not None:
            within_tolerance = sum(
                1 for v in valid_values if v <= self.config.tolerance
            )
            success_rate = within_tolerance / len(valid_values)
        else:
            success_rate = None

        # Calculate mean distance
        average_distance = np.mean(valid_values)

        metric_name = f"average_{self.config.distance_metric}_distance"
        return {
            "metrics": {
                metric_name: average_distance,
                "success_rate": success_rate,
            },
            "primary_metric": metric_name
            if self.config.tolerance is not None
            else "success_rate",
            "uncertainties": {
                metric_name: {
                    "std": np.std(valid_values) if len(valid_values) > 1 else 0.0
                }
            },
        }


@dataclass(kw_only=True)
class BandgapPropertyTargetConfig(ContinuousTargetConfig):
    """Configuration for the BandgapPropertyTarget metric.

    Parameters
    ----------
    target_theory : str, default="PBE"
        The theory level for bandgap prediction ("PBE" or "HSE").
    target_bandgap : float, default=1
        Target bandgap value in eV.
    model : str, default="MEGNet-MP-2019.4.1-BandGap-mfi"
        The model to use for bandgap prediction.
    tolerance : float, default=0.1
        Tolerance for deviations from target bandgap.
    """

    target_theory: str = "PBE"
    model: str = "MEGNet-MP-2019.4.1-BandGap-mfi"


class BandgapPropertyTargetMetric(ContinuousTargetMetric):
    """Metric for evaluating if structures match a target bandgap value."""

    def __init__(
        self,
        target_theory: str = "PBE",
        target_bandgap: float = 1,
        model: str = "MEGNet-MP-2019.4.1-BandGap-mfi",
        tolerance: float = 0.1,
        distance_metric: str = "absolute",
        name: str | None = None,
        description: str | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            target_value=target_bandgap,
            tolerance=tolerance,
            distance_metric=distance_metric,
            name=name or "Bandgap",
            description=description
            or f"Computes bandgap with {model} and compares to target bandgap {target_bandgap} eV",
            n_jobs=n_jobs,
        )

        self.config = BandgapPropertyTargetConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            target_value=target_bandgap,
            tolerance=tolerance,
            distance_metric=distance_metric,
            target_theory=target_theory,
            model=model,
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        super_attrs = super()._get_compute_attributes()
        return {
            **super_attrs,
            "target_theory": self.config.target_theory,
            "model": self.config.model,
        }

    @staticmethod
    def value_extractor(structure: Structure, model: str, target_theory: str) -> float:
        """Extract bandgap value from structure using the configured model."""
        band_gap_model = load_model(model)
        graph_attrs = torch.tensor([0 if target_theory == "PBE" else 2])
        return band_gap_model.predict_structure(
            structure=structure, state_attr=graph_attrs
        )


@dataclass(kw_only=True)
class SpacegroupTargetConfig(DiscreteTargetConfig):
    """Configuration for the SpacegroupTarget metric.

    Parameters
    ----------
    target_sg : int
        Target space group number.
    symprec : float, default=0.01
        Symmetry precision for SpacegroupAnalyzer.
    tolerance : float, default=0.1
        Tolerance for considering a match successful (not used for space groups).
    """

    symprec: float = 0.01


class SpacegroupTargetMetric(DiscreteTargetMetric):
    """Checks if a structure has a target space group."""

    def __init__(
        self,
        target_sg: int,
        symprec: float = 0.01,
        lower_is_better: bool = False,
        name: str | None = None,
        description: str | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            target_value=target_sg,
            lower_is_better=lower_is_better,
            name=name or f"SpacegroupMatch_{target_sg}",
            description=description
            or f"Measures fraction of structures with space group {target_sg}",
            n_jobs=n_jobs,
        )

        self.config = SpacegroupTargetConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            target_value=target_sg,
            symprec=symprec,
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        super_attrs = super()._get_compute_attributes()
        return {
            **super_attrs,
            "symprec": self.config.symprec,
        }

    @staticmethod
    def value_extractor(structure: Structure, symprec: float) -> int:
        """Extract space group number from structure."""
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        return sga.get_space_group_number()
