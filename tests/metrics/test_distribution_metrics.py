"""Tests for validity metrics implementation."""

import pickle

import numpy as np
import pytest
from pymatgen.util.testing import PymatgenTest

from lematerial_forgebench.metrics.distribution_metrics import MMD, JSDistance
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


def test_JSDistance_metric(valid_structures, reference_data):
    """Test JSDistance_metric on valid structures."""
    distribution_preprocessor = DistributionPreprocessor()
    preprocessor_result = distribution_preprocessor(valid_structures)

    metric = JSDistance(reference_data)
    result = metric.compute(
        preprocessor_result.processed_structures, reference_df=reference_data
    )

    # Check computation didn't fail
    assert len(result.failed_indices) == 0

    # Check result structure
    assert "Average_Jensen_Shannon_Distance" in result.metrics

    # Check values
    values = [val for key, val in result.metrics.items() if "Average" not in key]
    assert not np.any(np.isnan(values))
    for val in values:
        assert 0.0 <= val <= 1.0


def test_MMD_metric(valid_structures, reference_data):
    """Test MMD_metric on valid structures."""
    distribution_preprocessor = DistributionPreprocessor()
    preprocessor_result = distribution_preprocessor(valid_structures)

    metric = MMD(reference_data)
    result = metric.compute(
        preprocessor_result.processed_structures, reference_df=reference_data
    )
    # Check computation didn't fail
    assert len(result.failed_indices) == 0

    # Check result structure
    assert "Average_MMD" in result.metrics

    # Check values
    values = [val for key, val in result.metrics.items() if "Average" not in key]
    assert not np.any(np.isnan(values))
    for val in values:
        assert 0.0 <= val <= 1.0
