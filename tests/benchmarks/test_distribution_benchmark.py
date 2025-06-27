"""Tests for validity benchmark."""

import pickle

import pandas as pd
import pytest
from pymatgen.util.testing import PymatgenTest

from lematerial_forgebench.benchmarks.distribution_benchmark import (
    DistributionBenchmark,
)
from lematerial_forgebench.preprocess.distribution_preprocess import (
    DistributionPreprocessor,
)


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
        assert len(benchmark.evaluators) == 2
        assert "JSDistance" in benchmark.evaluators
        assert "MMD" in benchmark.evaluators
        # assert "FrechetDistance" in benchmark.evaluators


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
        preprocessor_result = distribution_preprocessor(structures)

        test_df = pd.DataFrame(preprocessor_result.processed_structures, columns = ["Volume", "Density(g/cm^3)", "Density(atoms/A^3)", 
                                                                                "SpaceGroup", "CrystalSystem", "CompositionCounts",
                                                                                "Composition"])
        
        benchmark = DistributionBenchmark(reference_df=reference_data)
        result = benchmark.evaluate([test_df])

        # Check result format
        assert len(result.evaluator_results) == 2
        assert "JSDistance" in result.final_scores
        assert "MMD" in result.final_scores
        # assert "FrechetDistance" in result.final_scores


        # Check score ranges
        for name, score in result.final_scores.items():
            if "score" in name or "ratio" in name:
                assert 0 <= score <= 1.0, f"{name} should be between 0 and 1"


