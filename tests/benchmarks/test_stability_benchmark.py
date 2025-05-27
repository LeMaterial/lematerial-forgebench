"""Tests for stability benchmark."""

from pymatgen.util.testing import PymatgenTest

from lematerial_forgebench.benchmarks.stability_benchmark import StabilityBenchmark


class TestStabilityBenchmark:
    """Test suite for StabilityBenchmark class."""

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        benchmark = StabilityBenchmark(mp_entries_file=None)

        # Check name and properties
        assert benchmark.config.name == "StabilityBenchmark"
        assert "version" in benchmark.config.metadata
        assert benchmark.config.metadata["category"] == "stability"

        # Check default relaxer type
        assert benchmark.config.metadata["relaxer_type"] == "orb"

        # Check correct evaluator
        assert len(benchmark.evaluators) == 1
        assert "stability" in benchmark.evaluators

    def test_initialization_custom(self):
        """Test initialization with custom relaxer configuration."""
        relaxer_config = {"steps": 2000, "fmax": 0.01, "direct": True}

        benchmark = StabilityBenchmark(
            relaxer_type="orb",
            relaxer_config=relaxer_config,
            mp_entries_file="custom/path/mp_entries.json",
            name="Custom Stability Benchmark",
            description="Custom description",
            metadata={"test_key": "test_value"},
        )

        # Check custom values
        assert benchmark.config.name == "Custom Stability Benchmark"
        assert benchmark.config.description == "Custom description"
        assert benchmark.config.metadata["test_key"] == "test_value"
        assert benchmark.config.metadata["relaxer_type"] == "orb"
        assert benchmark.config.metadata["relaxer_config"] == relaxer_config
        assert (
            benchmark.config.metadata["mp_entries_file"]
            == "custom/path/mp_entries.json"
        )

    def test_evaluate_with_mp_entries(self):
        """Test benchmark evaluation on structures"""
        benchmark = StabilityBenchmark()

        # Create test structures
        test = PymatgenTest()
        structures = [test.get_structure("Si"), test.get_structure("LiFePO4")]

        # Run benchmark
        result = benchmark.evaluate(structures)

        # Check result format
        assert len(result.evaluator_results) == 1
        assert "stability" in result.evaluator_results
        assert "stability_score" in result.final_scores
        assert "stable_ratio" in result.final_scores
        print(result.final_scores)

        # Check score types
        assert isinstance(result.final_scores["stability_score"], (int, float))
        assert isinstance(result.final_scores["stable_ratio"], (int, float))
        assert isinstance(result.final_scores["metastable_ratio"], (int, float))
        assert isinstance(result.final_scores["mean_e_above_hull"], (int, float))

    def test_empty_structures(self):
        """Test behavior with empty structure list."""
        benchmark = StabilityBenchmark()

        # Test behavior with no structures - should not raise error
        result = benchmark.evaluate([])

        # Should get default values
        assert result.final_scores["stability_score"] == 0.0
        assert result.final_scores["stable_ratio"] == 0.0
        assert result.final_scores["metastable_ratio"] == 0.0
        assert result.final_scores["mean_e_above_hull"] == 0.0

    def test_aggregate_evaluator_results(self):
        """Test result aggregation logic."""
        benchmark = StabilityBenchmark()

        # Mock evaluator_results as passed by BaseBenchmark.evaluate
        # It contains the evaluator's combined_value and the primary metric value
        # of each metric configured for that evaluator (e.g., "metric_name_value").
        mock_evaluator_results_from_base = {
            "stability": {  # Name of the evaluator
                "combined_value": 0.75,  # Evaluator's combined score
                # The StabilityMetric instance within the 'stability' evaluator was named "stability".
                # Its primary metric (stable_ratio) is flattened by BaseBenchmark to "stability_value".
                "stability_value": 0.6,
            }
        }

        # Aggregate results
        scores = benchmark.aggregate_evaluator_results(mock_evaluator_results_from_base)

        # Check scores
        # aggregate_evaluator_results should pick up combined_value as stability_score
        # and stability_value as stable_ratio.
        # mean_e_above_hull and metastable_ratio will be defaults (nan, 0.0) because
        # they are not present in the input dict.
        assert scores["stability_score"] == 0.75
        assert scores["stable_ratio"] == 0.6
        assert (
            "mean_e_above_hull" in scores
            and scores["mean_e_above_hull"] != scores["mean_e_above_hull"]
        )  # Check for NaN
        assert scores["metastable_ratio"] == 0.0

    def test_benchmark_metadata(self):
        """Test benchmark metadata structure."""
        benchmark = StabilityBenchmark(
            relaxer_type="orb",
            relaxer_config={"direct": True},
            mp_entries_file="test.json",
        )

        metadata = benchmark.config.metadata

        # Check required metadata fields
        assert metadata["version"] == "0.1.0"
        assert metadata["category"] == "stability"
        assert metadata["relaxer_type"] == "orb"
        assert metadata["relaxer_config"]["direct"] is True
        assert metadata["mp_entries_file"] == "test.json"


def test_benchmark_description_generation():
    """Test automatic description generation."""
    # Test with ORB relaxer
    orb_benchmark = StabilityBenchmark(relaxer_type="orb")
    assert "ORB" in orb_benchmark.config.description
    assert "relaxation and energy above hull" in orb_benchmark.config.description

    # Test with custom description
    custom_benchmark = StabilityBenchmark(
        relaxer_type="orb", description="Custom description"
    )
    assert custom_benchmark.config.description == "Custom description"


def test_evaluator_configuration():
    """Test that evaluator is properly configured."""
    benchmark = StabilityBenchmark(relaxer_type="orb")

    # Check evaluator configuration
    stability_evaluator = benchmark.evaluators["stability"]
    print(stability_evaluator)
    assert stability_evaluator.config.name == "stability"
    assert "ORB" in stability_evaluator.config.description
    assert stability_evaluator.config.weights == {"stability": 1.0}
    assert stability_evaluator.config.aggregation_method == "weighted_mean"


def test_orb_relaxer_type():
    """Test benchmark with ORB relaxer type."""
    benchmark = StabilityBenchmark(relaxer_type="orb")

    # Check that the relaxer type is properly set
    assert benchmark.config.metadata["relaxer_type"] == "orb"

    # Check that evaluator name reflects the relaxer type
    expected_name = "stability"
    assert benchmark.evaluators["stability"].config.name == expected_name
