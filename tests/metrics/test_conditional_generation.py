"""Tests for conditional generation metrics."""

import numpy as np
import pytest
from pymatgen.core.structure import Lattice, Structure

from lematerial_forgebench.metrics.conditional_generation import (
    BandgapPropertyTargetMetric,
    ContinuousTargetMetric,
    DiscreteTargetMetric,
    MaxDensityTargetMetric,
    SpacegroupTargetMetric,
    StableMagnets,
)


def create_test_structures():
    """Create a list of test structures."""
    # Create a simple cubic structure
    lattice = Lattice.cubic(4.0)
    coords = [[0, 0, 0]]

    structures = [
        Structure(lattice, ["Si"], coords),  # Simple Si structure
        Structure(lattice, ["Fe"], coords),  # Simple Fe structure
        Structure(lattice, ["Cu"], coords),  # Simple Cu structure
    ]
    return structures


class SimpleDiscreteMetric(DiscreteTargetMetric):
    """A simple discrete metric that returns the number of atoms."""

    @staticmethod
    def value_extractor(structure: Structure) -> int:
        return len(structure)


class SimpleContinuousMetric(ContinuousTargetMetric):
    """A simple continuous metric that returns the volume."""

    @staticmethod
    def value_extractor(structure: Structure) -> float:
        return structure.volume


def test_discrete_target_metric():
    """Test the discrete target metric."""
    structures = create_test_structures()

    # Test with target value of 1 (all structures have 1 atom)
    metric = SimpleDiscreteMetric(target_value=1)
    result = metric(structures)

    assert result.metrics["success_rate"] == 1.0  # All structures match
    assert result.metrics["mean_value"] == 1.0

    # Test with target value of 2 (no structures have 2 atoms)
    metric = SimpleDiscreteMetric(target_value=2)
    result = metric(structures)

    assert result.metrics["success_rate"] == 0.0  # No structures match
    assert result.metrics["mean_value"] == 1.0


def test_continuous_target_metric():
    """Test the continuous target metric."""
    structures = create_test_structures()
    volume = structures[0].volume  # All structures have same volume

    # Test with exact volume match
    metric = SimpleContinuousMetric(
        target_value=volume, tolerance=0.1, distance_metric="absolute"
    )
    result = metric(structures)

    assert result.metrics["success_rate"] == 1.0  # All structures within tolerance
    assert result.metrics["average_absolute_distance"] == 0.0

    # Test with no matches
    metric = SimpleContinuousMetric(
        target_value=volume * 2, tolerance=0.1, distance_metric="absolute"
    )
    result = metric(structures)

    assert result.metrics["success_rate"] == 0.0  # No structures within tolerance
    assert result.metrics["average_absolute_distance"] > 0.0


def test_spacegroup_target_metric():
    """Test the spacegroup target metric."""
    structures = create_test_structures()

    # Simple cubic structures should have space group 221
    metric = SpacegroupTargetMetric(target_sg=221)
    result = metric(structures)

    assert result.metrics["success_rate"] == 1.0
    assert result.metrics["mean_value"] == 221.0


def test_max_density_target_metric():
    """Test the max density target metric."""
    structures = create_test_structures()

    # Test basic functionality
    metric = MaxDensityTargetMetric()
    result = metric(structures)

    # Check that metrics are present and have expected properties
    assert "average_identity_distance" in result.metrics
    assert (
        result.metrics["average_identity_distance"] > 0.0
    )  # Should be positive since it's density
    assert (
        result.metrics["success_rate"] is None
    )  # No tolerance set, so no success rate

    # Test with top_k parameter
    metric_top_k = MaxDensityTargetMetric(top_k=2)
    result_top_k = metric_top_k(structures)

    # Should only consider top 2 densest structures
    assert (
        result_top_k.metrics["average_identity_distance"]
        >= result.metrics["average_identity_distance"]
    )

    # Verify that individual values are densities
    for value in result.individual_values:
        assert value > 0.0  # Density should be positive
        assert not np.isnan(value)  # No NaN values expected


def test_bandgap_property_target():
    """Test the bandgap property target metric."""
    structures = create_test_structures()

    metric = BandgapPropertyTargetMetric(
        target_bandgap=1.0, tolerance=0.5, distance_metric="absolute"
    )
    result = metric(structures)

    # Basic checks - actual values will depend on the model
    assert "success_rate" in result.metrics
    assert "average_absolute_distance" in result.metrics
    assert 0 <= result.metrics["success_rate"] <= 1.0
    assert result.metrics["average_absolute_distance"] >= 0.0


def test_continuous_target_different_metrics():
    """Test continuous target metric with different distance metrics."""
    structures = create_test_structures()
    volume = structures[0].volume

    for metric_type in ["absolute", "relative", "squared"]:
        metric = SimpleContinuousMetric(
            target_value=volume * 2, distance_metric=metric_type
        )
        result = metric(structures)

        metric_name = f"average_{metric_type}_distance"
        assert metric_name in result.metrics
        assert result.metrics[metric_name] > 0.0


def test_continuous_target_top_k():
    """Test continuous target metric with top_k parameter."""
    structures = (
        create_test_structures() * 2
    )  # Duplicate structures to have more samples
    volume = structures[0].volume

    metric = SimpleContinuousMetric(
        target_value=volume, top_k=3, distance_metric="absolute"
    )
    result = metric(structures)

    # Should only consider top 3 closest structures
    assert result.metrics["average_absolute_distance"] == 0.0


def test_invalid_distance_metric():
    """Test that invalid distance metric raises error."""
    with pytest.raises(ValueError, match="Invalid distance metric"):
        SimpleContinuousMetric(target_value=1.0, distance_metric="invalid")


def test_empty_structures():
    """Test behavior with empty structure list."""
    metric = SimpleDiscreteMetric(target_value=1)
    result = metric.compute([])

    # Empty list should return a MetricResult with NaN values
    metric_name = metric.name  # Use the metric's name
    assert np.isnan(result.metrics[metric_name])
    assert result.n_structures == 0
    assert len(result.individual_values) == 0
    assert len(result.failed_indices) == 0


def test_all_failed_structures():
    """Test behavior when all structures fail value extraction."""

    class FailingMetric(DiscreteTargetMetric):
        @staticmethod
        def value_extractor(structure: Structure) -> int:
            raise ValueError("Always fails")

    metric = FailingMetric(target_value=1)
    structures = create_test_structures()
    result = metric.compute(structures)

    # All failed structures should return a MetricResult with NaN values
    metric_name = metric.name  # Use the metric's name
    assert np.isnan(result.metrics[metric_name])
    assert result.n_structures == len(structures)
    assert len(result.individual_values) == len(structures)
    assert all(np.isnan(v) for v in result.individual_values)
    assert len(result.failed_indices) == len(structures)
    assert all(i in result.failed_indices for i in range(len(structures)))


# Mock the external model predictions to avoid actual model loading
@pytest.fixture(autouse=True)
def mock_model_predictions(monkeypatch):
    def mock_bandgap_predict(*args, **kwargs):
        # Return a fixed bandgap value for testing
        return 2.0

    def mock_hhi_compute(*args, **kwargs):
        # Return fixed HHI values for testing
        return 0.3

    monkeypatch.setattr(
        "lematerial_forgebench.metrics.conditional_generation.BandgapPropertyTargetMetric.value_extractor",
        mock_bandgap_predict,
    )
    monkeypatch.setattr(
        "lematerial_forgebench.metrics.conditional_generation.HHIProductionMetric.compute_structure",
        mock_hhi_compute,
    )
    monkeypatch.setattr(
        "lematerial_forgebench.metrics.conditional_generation.HHIReserveMetric.compute_structure",
        mock_hhi_compute,
    )


def test_stable_magnets_initialization():
    """Test that StableMagnets initializes correctly with various parameters."""
    metric = StableMagnets(min_bandgap=1.0, max_bandgap=3.0, hhi_value=0.35, top_k=10)

    assert metric.config.min_bandgap == 1.0
    assert metric.config.max_bandgap == 3.0
    assert metric.config.hhi_value == 0.35
    assert metric.config.top_k == 10
    assert metric.config.target_theory == "PBE"
    assert "MEGNet" in metric.config.model


def test_stable_magnets_compute_structure():
    """Test that compute_structure returns expected dictionary format."""
    metric = StableMagnets(min_bandgap=1.0, max_bandgap=3.0, hhi_value=0.35)

    # Create a simple test structure
    lattice = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    species = ["Si"]
    coords = [[0, 0, 0]]
    structure = Structure(lattice, species, coords)

    result = metric.compute_structure(structure, **metric._get_compute_attributes())

    assert isinstance(result, dict)
    assert "bandgap" in result
    assert "hhi_production" in result
    assert "hhi_reserve" in result
    assert result["bandgap"] == 2.0  # From our mock
    assert result["hhi_production"] == 0.3  # From our mock
    assert result["hhi_reserve"] == 0.3  # From our mock


def test_stable_magnets_aggregate_results():
    """Test that aggregate_results correctly evaluates conditions."""
    metric = StableMagnets(min_bandgap=1.0, max_bandgap=3.0, hhi_value=0.35)

    # Create test results that should pass our criteria
    test_results = [
        {
            "bandgap": 2.0,  # Within range [1.0, 3.0]
            "hhi_production": 0.3,  # Below 0.35
            "hhi_reserve": 0.3,  # Below 0.35
        },
        {
            "bandgap": 1.5,  # Within range [1.0, 3.0]
            "hhi_production": 0.2,  # Below 0.35
            "hhi_reserve": 0.2,  # Below 0.35
        },
    ]

    result = metric.aggregate_results(test_results)

    assert "metrics" in result
    assert "success_rate" in result["metrics"]
    assert result["metrics"]["success_rate"] == 1.0  # All samples pass
    assert result["primary_metric"] == "success_rate"


def test_stable_magnets_failing_conditions():
    """Test that aggregate_results correctly identifies failing conditions."""
    metric = StableMagnets(min_bandgap=1.0, max_bandgap=3.0, hhi_value=0.35)

    # Create test results where some fail our criteria
    test_results = [
        {
            "bandgap": 0.5,  # Below min_bandgap
            "hhi_production": 0.3,
            "hhi_reserve": 0.3,
        },
        {
            "bandgap": 2.0,  # Good
            "hhi_production": 0.4,  # Above hhi_value
            "hhi_reserve": 0.3,
        },
        {
            "bandgap": 2.0,  # Good
            "hhi_production": 0.3,
            "hhi_reserve": 0.4,  # Above hhi_value
        },
    ]

    result = metric.aggregate_results(test_results)

    assert result["metrics"]["success_rate"] == 0.0  # No samples pass all criteria
