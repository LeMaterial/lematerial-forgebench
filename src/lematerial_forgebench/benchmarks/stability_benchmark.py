"""Stability benchmark for material structures.

This module implements a benchmark that evaluates the stability of
generated material structures using various relaxation methods.
"""

from typing import Any, Dict

from lematerial_forgebench.benchmarks.base import BaseBenchmark
from lematerial_forgebench.evaluator import EvaluatorConfig
from lematerial_forgebench.metrics.stability_metrics import StabilityMetric


class StabilityBenchmark(BaseBenchmark):
    """Benchmark for evaluating the stability of generated material structures."""

    def __init__(
        self,
        relaxer_type: str = "orb",
        relaxer_config: Dict[str, Any] | None = None,
        mp_entries_file: str = "src/lematerial_forgebench/utils/relaxers/2023-02-07-ppd-mp.pkl.gz",
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
                f"Evaluates the stability of crystal structures using {relaxer_type.upper()} "
                "relaxation and energy above hull calculations."
            )

        # Set default relaxer config if not provided
        if relaxer_config is None:
            relaxer_config = {"steps": 500, "fmax": 0.02}

        # Initialize the stability metric
        stability_metric = StabilityMetric(
            relaxer_type=relaxer_type,
            relaxer_config=relaxer_config,
            mp_entries_file=mp_entries_file,
        )

        # Set up evaluator config
        evaluator_configs = {
            "stability": EvaluatorConfig(
                name=f"{relaxer_type.upper()} Stability",
                description=f"Evaluates structure stability using {relaxer_type.upper()}",
                metrics={"stability": stability_metric},
                weights={"stability": 1.0},
                aggregation_method="weighted_mean",
            ),
        }

        # Create benchmark metadata
        benchmark_metadata = {
            "version": "0.1.0",
            "category": "stability",
            "relaxer_type": relaxer_type,
            "relaxer_config": relaxer_config,
            "mp_entries_file": mp_entries_file,
            **(metadata or {}),
        }

        super().__init__(
            name=name,
            description=description,
            evaluator_configs=evaluator_configs,
            metadata=benchmark_metadata,
        )

    def aggregate_evaluator_results(
        self, evaluator_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate results from multiple evaluators into final scores.

        Parameters
        ----------
        evaluator_results : dict[str, dict[str, Any]]
            Results from each evaluator, as structured by BaseBenchmark.evaluate.
            Example: {"evaluator_name": {"combined_value": 0.X, "metric_name_value": 0.Y}}

        Returns
        -------
        dict[str, float]
            Final aggregated scores.
        """
        final_scores = {
            "stability_score": 0.0,
            "stable_ratio": 0.0,
            "mean_e_above_hull": 0.0,
            "metastable_ratio": 0.0,
        }

        stability_eval_data: Dict[str, Any] | None = evaluator_results.get("stability")
        print("stability_eval_data", stability_eval_data)
        if stability_eval_data:
            if stability_eval_data.get("combined_value") is not None:
                final_scores["stability_score"] = stability_eval_data["combined_value"]

            if stability_eval_data.get("stability_value") is not None:
                final_scores["stable_ratio"] = stability_eval_data["stability_value"]

            if stability_eval_data.get("mean_e_above_hull") is not None:
                final_scores["mean_e_above_hull"] = stability_eval_data[
                    "mean_e_above_hull"
                ]

            if stability_eval_data.get("metastable_ratio") is not None:
                final_scores["metastable_ratio"] = stability_eval_data[
                    "metastable_ratio"
                ]

        return final_scores
