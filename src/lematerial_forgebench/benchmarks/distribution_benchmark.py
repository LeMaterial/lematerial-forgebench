"""Stability benchmark for material structures.

This module implements a benchmark that evaluates the stability of
generated material structures using various relaxation methods.
"""

from typing import Any, Dict

import numpy as np

from lematerial_forgebench.benchmarks.base import BaseBenchmark
from lematerial_forgebench.evaluator import EvaluationResult, EvaluatorConfig
from lematerial_forgebench.metrics.distribution_metrics import JSDistance, MMD, FrechetDistance



class DistributionBenchmark(BaseBenchmark):
    """Benchmark for evaluating quantitative similarity of two distributions of 
    materials structures."""

    def __init__(
        self,
        name: str = "DistributionBenchmark",
        description: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ):
        """Initialize the distribution benchmark.

        Parameters
        ----------
        name : str
            Name of the benchmark.
        distribution_functions : list[str], optional
            A list of strings containing the distribution functions to compare. If none, 
            defaults to all currently encoded. 
        description : str, optional
            Description of the benchmark.
        metadata : dict, optional
            Additional metadata for the benchmark.
        """
        if description is None:
            description = (
                "Evaluates the stability and metastability of crystal structures"
            )

        # Initialize the JSDistance metric
        JSDistance_metric = JSDistance()

        # Set up evaluator configs
        evaluator_configs = {
            "JSDistance": EvaluatorConfig(
                name="JSDistance",
                description="Calculates the JS Distance between two distributions",
                metrics={"JSDistance": JSDistance_metric},
                weights={"JSDistance": 1.0},
                aggregation_method="weighted_mean",
            ),
        }

        # Initialize the MMD metric
        MMD_metric = MMD()
        evaluator_configs["MMD"] = EvaluatorConfig(
            name="MMD Analysis",
            description="Calculates the MMD between two distributions",
            metrics={"MMD": MMD_metric},
            weights={"MMD": 1.0},
            aggregation_method="weighted_mean",
        )

        # FrechetDistance_metric = FrechetDistance()
        # evaluator_configs["FrechetDistance"] = EvaluatorConfig(
        #     name="FrechetDistance Analysis",
        #     description="Calculates the Frechet Distance between two distributions",
        #     metrics={"FrechetDistance": FrechetDistance_metric},
        #     weights={"FrechetDistance": 1.0},
        #     aggregation_method="weighted_mean",
        # )


        # Create benchmark metadata
        benchmark_metadata = {
            "version": "0.1.0",
            "category": "distribution",
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

        def safe_float(value):
            """Safely convert value to float, handling None and NaN."""
            return value
            # if value is None:
            #     raise ValueError

            # float_val = float(value)
            # if math.isnan(float_val):
            #     raise ValueError
            # return float_val

        final_scores = {
            "JSDistance": np.nan,
            "MMD": np.nan,
            "FrechetDistance": np.nan,
        }

        # Extract stability results
        JSDistance_results = evaluator_results.get("JSDistance")
        if JSDistance_results:
            # Main stability ratio
            final_scores["JSDistance"] = safe_float(
                JSDistance_results.get("combined_value")
            )

        # Extract metastability results if available
        MMD_results = evaluator_results.get("MMD")
        print("MMD_results")
        if MMD_results:
            # Main metastability score
            final_scores["MMD"] = safe_float(
                MMD_results.get("combined_value")
            )

        # FrechetDistance_results = evaluator_results.get("FrechetDistance")
        # if FrechetDistance_results:
        #     # E_hull score
        #     final_scores["FrechetDistance"] = safe_float(
        #         FrechetDistance_results.get("combined_value")
        #     )

        return final_scores
