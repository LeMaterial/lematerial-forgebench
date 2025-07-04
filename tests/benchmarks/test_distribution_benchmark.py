"""Tests for validity benchmark."""

import pickle

import pytest
from pymatgen.util.testing import PymatgenTest

from lematerial_forgebench.benchmarks.distribution_benchmark import (
    DistributionBenchmark,
)
from lematerial_forgebench.preprocess.distribution_preprocess import (
    DistributionPreprocessor,
)

from lematerial_forgebench.preprocess.universal_stability_preprocess import UniversalStabilityPreprocessor
from lematerial_forgebench.preprocess.base import PreprocessorResult

@pytest.fixture
def valid_structures():
    """Create valid test structures."""
    test = PymatgenTest()
    structures = [
        test.get_structure("Si"),  # Silicon
        test.get_structure("LiFePO4"),  # Lithium iron phosphate
        test.get_structure("CsCl"),  # Cesium chloride
    ]
    return structures


@pytest.fixture
def reference_data():
    "create reference dataset"
    with open("data/small_lematbulk.pkl", "rb") as f:
        reference_df = pickle.load(f)

    return reference_df


class TestDistributionBenchmark:
    """Test suite for DistributionBenchmark class."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        benchmark = DistributionBenchmark(reference_df=reference_data)

        # Check name and properties
        assert benchmark.config.name == "DistributionBenchmark"
        assert "version" in benchmark.config.metadata

        # Check correct evaluators
        assert len(benchmark.evaluators) == 3
        assert "JSDistance" in benchmark.evaluators
        assert "MMD" in benchmark.evaluators
        assert "FrechetDistance" in benchmark.evaluators

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        benchmark = DistributionBenchmark(
            reference_df=reference_data,
            name="Custom Benchmark",
            description="Custom description",
            metadata={"test_key": "test_value"},
        )

        # Check custom values
        assert benchmark.config.name == "Custom Benchmark"
        assert benchmark.config.description == "Custom description"
        assert benchmark.config.metadata["test_key"] == "test_value"

    def test_evaluate(self):
        """Test benchmark evaluation on structures."""

        test = PymatgenTest()

        structures = [
            test.get_structure("Si"),
            test.get_structure("LiFePO4"),
        ]

        distribution_preprocessor = DistributionPreprocessor()
        dist_preprocessor_result = distribution_preprocessor(structures)

        stability_preprocessor = UniversalStabilityPreprocessor(model_name="orb")
        stability_preprocessor_result = stability_preprocessor(structures)

        final_processed_structures = []

        for ind in range(0, len(dist_preprocessor_result.processed_structures)): 
            combined_structure = dist_preprocessor_result.processed_structures[ind]
            for entry in stability_preprocessor_result.processed_structures[ind].properties.keys():
                combined_structure.properties[entry] = stability_preprocessor_result.processed_structures[ind].properties[entry]
            final_processed_structures.append(combined_structure)

        preprocessor_result = PreprocessorResult(processed_structures=final_processed_structures,
                config={
                    "stability_preprocessor_config":stability_preprocessor_result.config,
                    "distribution_preprocessor_config": dist_preprocessor_result.config,
                },
                computation_time={
                    "stability_preprocessor_computation_time": stability_preprocessor_result.computation_time,
                    "distribution_preprocessor_computation_time": dist_preprocessor_result.computation_time,
                },
                n_input_structures=stability_preprocessor_result.n_input_structures,
                failed_indices={
                    "stability_preprocessor_failed_indices": stability_preprocessor_result.failed_indices,
                    "distribution_preprocessor_failed_indices": dist_preprocessor_result.failed_indices,
                },
                warnings={
                    "stability_preprocessor_warnings": stability_preprocessor_result.warnings,
                    "distribution_preprocessor_warnings": dist_preprocessor_result.warnings,
                },
            )

        benchmark = DistributionBenchmark(reference_df=reference_data)
        result = benchmark.evaluate(preprocessor_result.processed_structures)

        # Check result format
        assert len(result.evaluator_results) == 3
        assert "JSDistance" in result.final_scores
        assert "MMD" in result.final_scores
        assert "FrechetDistance" in result.final_scores

        # Check score ranges
        for name, score in result.final_scores.items():
            if "score" in name or "ratio" in name:
                assert 0 <= score <= 1.0, f"{name} should be between 0 and 1"
