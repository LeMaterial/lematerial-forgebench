"""Stability benchmark for material structures.

This module implements a benchmark that evaluates the stability of
generated material structures using various relaxation methods.
"""

from typing import Any, Dict

from lematerial_forgebench.benchmarks.base import BaseBenchmark
from lematerial_forgebench.evaluator import EvaluationResult, EvaluatorConfig
from lematerial_forgebench.metrics.stability_metrics import (
    MetastabilityMetric,
    StabilityMetric,
    FormationEnergyMetric,
    RelaxationStabilityMetric,
)


class StabilityBenchmark(BaseBenchmark):
    """Benchmark for evaluating the stability of generated material structures."""

    def __init__(
        self,
        name: str = "StabilityBenchmark",
        description: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ):
        """Initialize the stability benchmark.

        Parameters
        ----------
        relaxer_type : str, default="orb"
            Type of relaxer to use (e.g., "orb", "ocp").
        relaxer_config : dict, optional
            Configuration for the relaxer. If None, uses default config.
        mp_entries_file : str
            Path to the Materials Project entries file.
        name : str
            Name of the benchmark.
        description : str, optional
            Description of the benchmark.
        metadata : dict, optional
            Additional metadata for the benchmark.
        """
        if description is None:
            description = (
                "Evaluates the stability and metastability of crystal structures"
            )

        # Initialize the stability metric
        stability_metric = StabilityMetric()

        # Set up evaluator configs
        evaluator_configs = {
            "stability": EvaluatorConfig(
                name="Stability",
                description="Evaluates structure stability",
                metrics={"stability": stability_metric},
                weights={"stability": 1.0},
                aggregation_method="weighted_mean",
            ),
        }

        # Add metastability evaluator if requested
        metastability_metric = MetastabilityMetric()
        evaluator_configs["metastability"] = EvaluatorConfig(
            name="Metastability Analysis",
            description="Evaluates metastability from precomputed e_above_hull values",
            metrics={"metastability": metastability_metric},
            weights={"metastability": 1.0},
            aggregation_method="weighted_mean",
        )

        # Add formation energy evaluator if requested
        formation_energy_metric = FormationEnergyMetric()
        evaluator_configs["formation_energy"] = EvaluatorConfig(
            name="Formation Energy Analysis",
            description="Evaluates formation energy from precomputed values",
            metrics={"formation_energy": formation_energy_metric},
            weights={"formation_energy": 1.0},
            aggregation_method="weighted_mean",
        )

        # Add relxation stability evaluator if requested
        relaxation_stability_metric = RelaxationStabilityMetric()
        evaluator_configs["relaxation_stability"] = EvaluatorConfig(
            name="Relaxation Stability Analysis",
            description="Evaluates relaxation stability from precomputed relaxed structure",
            metrics={"relaxation_stability": relaxation_stability_metric},
            weights={"relaxation_stability": 1.0},
            aggregation_method="weighted_mean",
        )

        # Create benchmark metadata
        benchmark_metadata = {
            "version": "0.1.0",
            "category": "stability",
            **(metadata or {}),
        }

        super().__init__(
            name=name,
            description=description,
            evaluator_configs=evaluator_configs,
            metadata=benchmark_metadata,
        )

    def aggregate_evaluator_results(
        self, evaluator_results: Dict[str, EvaluationResult]
    ) -> Dict[str, float]:
        """Aggregate results from multiple evaluators into final scores.

        Parameters
        ----------
        evaluator_results : dict[str, EvaluationResult]
            Results from each evaluator.

        Returns
        -------
        dict[str, float]
            Final aggregated scores.
        """
        import math

        def safe_float(value, default=0.0):
            """Safely convert value to float, handling None and NaN."""
            if value is None:
                return default
            try:
                float_val = float(value)
                if math.isnan(float_val):
                    return default
                return float_val
            except (TypeError, ValueError):
                return default

        final_scores = {
            "stable_ratio": 0.0,
            "metastable_ratio": 0.0,
            "mean_e_above_hull": 0.0,
            "mean_formation_energy": 0.0,
            "mean_relaxation_RMSE": 0.0,
        }

        # Extract stability results
        stability_results = evaluator_results.get("stability")

        if stability_results:
            # Main stability ratio
            final_scores["stable_ratio"] = safe_float(
                stability_results.get("combined_value")
            )
            # Extract individual metrics from stability metric
            final_scores["mean_e_above_hull"] = safe_float(
                stability_results["metric_results"]["stability"].metrics[
                    "mean_e_above_hull"
                ]
            )

        # Extract metastability results if available
        metastability_results = evaluator_results.get("metastability")
        if metastability_results:
            # Main metastability score
            final_scores["metastable_ratio"] = safe_float(
                metastability_results.get("combined_value")
            )

        # Extract formation energy results if available
        formation_energy_results = evaluator_results.get("formation_energy")
        if formation_energy_results:
            # Main metastability score
            final_scores["mean_formation_energy"] = safe_float(
                formation_energy_results.get("combined_value")
            )

        # Extract relxation stability results if available
        relaxation_stability_results = evaluator_results.get("relaxation_stability")
        if relaxation_stability_results:
            # Main metastability score
            final_scores["mean_relaxation_RMSE"] = safe_float(
                relaxation_stability_results.get("combined_value")
            )

        print(final_scores)

        return final_scores
