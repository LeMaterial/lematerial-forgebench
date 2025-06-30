"""Tests for conditional generation metrics."""

import numpy as np
import pytest
from pymatgen.core.structure import Lattice, Structure

from lematerial_forgebench.metrics.conditional_generation import (
    BandgapPropertyTargetMetric,
    ContinuousTargetMetric,
    DiscreteTargetMetric,
    SpacegroupTargetMetric,
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

    assert result.metrics["success_rate"] == 1.0  # All structures are simple cubic
    assert result.metrics["mean_value"] == 221.0


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
    assert all(v is None for v in result.individual_values)
    assert len(result.failed_indices) == len(structures)
    assert all(i in result.failed_indices for i in range(len(structures)))
