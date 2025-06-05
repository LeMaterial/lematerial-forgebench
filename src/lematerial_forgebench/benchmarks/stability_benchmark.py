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
            stability_metric_results = stability_results.get("metric_results", {}).get(
                "stability", {}
            )
            stability_metrics = stability_metric_results.get("metrics", {})

            final_scores["mean_e_above_hull"] = safe_float(
                stability_metrics.get("mean_e_above_hull")
            )

        # Extract metastability results if available
        metastability_results = evaluator_results.get("metastability")
        if metastability_results:
            # Main metastability score
            final_scores["metastable_ratio"] = safe_float(
                metastability_results.get("combined_value")
            )

        return final_scores
