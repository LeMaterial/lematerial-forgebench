"""Distribution benchmark for material structures.

This module implements a benchmark that compares two distributions of crystal structures. 
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from lematerial_forgebench.benchmarks.base import BaseBenchmark
from lematerial_forgebench.evaluator import EvaluationResult, EvaluatorConfig
from lematerial_forgebench.metrics.distribution_metrics import (
    MMD,
    JSDistance,
)
from lematerial_forgebench.utils.distribution_utils import safe_float


class DistributionBenchmark(BaseBenchmark):
    """Benchmark for evaluating quantitative similarity of two distributions of
    materials structures."""

    def __init__(
        self,
        reference_df: pd.DataFrame, 
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
                "Compares the distribution of structural parameters from a sample of " \
                "crystals to a reference distribution."
            )
                
        # Initialize the JSDistance metric
        JSDistance_metric = JSDistance(reference_df=reference_df)
        # Set up evaluator configs
        evaluator_configs = {"JSDistance": EvaluatorConfig(
                name="JSDistance",
                description="Calculates the JS Distance between two distributions",
                metrics={"JSDistance": JSDistance_metric},
                weights={"JSDistance": 1.0},
                aggregation_method="weighted_mean",
                )
            }

        # Initialize the MMD metric
        MMD_metric = MMD(reference_df=reference_df)

        # add to evaluator config
        evaluator_configs["MMD"] = EvaluatorConfig(
            name="MMD Analysis",
            description="Calculates the MMD between two distributions",
            metrics={"MMD": MMD_metric},
            weights={"MMD": 1.0},
            aggregation_method="weighted_mean",
        )

        # Initialize the MFrechetDistanceMD metric
        # FrechetDistance_metric = FrechetDistance(reference_df=reference_df)

        # # add to evaluator config
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

        final_scores = {
            "JSDistance": np.nan,
            "MMD": np.nan,
            "FrechetDistance": np.nan,
        }

        # Extract stability results
        JSDistance_results = evaluator_results.get("JSDistance")
        if JSDistance_results:
            # Main stability ratio
            final_scores["JSDistance"] = JSDistance_results.get("combined_value")

        # Extract metastability results if available
        MMD_results = evaluator_results.get("MMD")
        print("MMD_results")
        if MMD_results:
            # Main metastability score
            final_scores["MMD"] = safe_float(MMD_results.get("combined_value"))

        FrechetDistance_results = evaluator_results.get("FrechetDistance")
        if FrechetDistance_results:
            # E_hull score
            final_scores["FrechetDistance"] = safe_float(
                FrechetDistance_results.get("combined_value")
            )

        return final_scores


if __name__ == '__main__':
    import pickle

    from pymatgen.util.testing import PymatgenTest

    from lematerial_forgebench.preprocess.distribution_preprocess import (
        DistributionPreprocessor,
    )
    
    with open("data/small_lematbulk.pkl", "rb") as f:
        test_lemat = pickle.load(f)
    test = PymatgenTest()

    structures = [
        test.get_structure("Si"),
        test.get_structure("LiFePO4"),
    ]

    distribution_preprocessor = DistributionPreprocessor()
    preprocessor_result = distribution_preprocessor(structures)

    test_df = pd.DataFrame(preprocessor_result.processed_structures, columns = ["Volume", "Density(g/cm^3)", "Density(atoms/A^3)", 
                                                                            "SpaceGroup", "CrystalSystem", "CompositionCounts",
                                                                            "Composition"])
    
    benchmark = DistributionBenchmark(reference_df=test_lemat)
    benchmark_result = benchmark.evaluate([test_df])
    print(benchmark_result.evaluator_results["JSDistance"]["JSDistance_value"])
    print(benchmark_result.evaluator_results["MMD"]["MMD_value"])