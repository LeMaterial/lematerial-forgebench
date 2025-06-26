"""Tests for validity metrics implementation."""

import numpy as np
import pytest
from pymatgen.core.structure import Structure
from pymatgen.util.testing import PymatgenTest

from lematerial_forgebench.metrics.distribution_metrics import (
    JSDistance, MMD
)
from lematerial_forgebench.preprocess.distribution_preprocess import (
    DistributionPreprocessor,
)
import pandas as pd 
import pickle 

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


def test_JSDistance_metric(valid_structures):
    """Test JSDistance_metric on valid structures."""
    distribution_preprocessor = DistributionPreprocessor()
    preprocessor_result = distribution_preprocessor(valid_structures)
    test_df = pd.DataFrame(preprocessor_result.processed_structures, columns = ["Volume", "Density(g/cm^3)", "Density(atoms/A^3)", 
                                                                            "SpaceGroup", "CrystalSystem", "CompositionCounts",
                                                                            "Composition"])
    with open("data/lematbulk_properties.pkl", "rb") as f:
        test_lemat = pickle.load(f)
    metric = JSDistance()
    result = metric([test_df], test_lemat)

    # Check computation didn't fail
    assert len(result.failed_indices) == 0

    # Check result structure
    assert "Jensen_Shannon_Distance" in result.metrics

    # Check values
    values = list(result.metrics["Jensen_Shannon_Distance"].values())
    assert not np.any(np.isnan(values))
    for val in result.metrics["Jensen_Shannon_Distance"].values():
        assert 0.0 <= val <= 1.0



def test_MMD_metric(valid_structures):
    """Test MMD_metric on valid structures."""
    distribution_preprocessor = DistributionPreprocessor()
    preprocessor_result = distribution_preprocessor(valid_structures)
    test_df = pd.DataFrame(preprocessor_result.processed_structures, columns = ["Volume", "Density(g/cm^3)", "Density(atoms/A^3)", 
                                                                            "SpaceGroup", "CrystalSystem", "CompositionCounts",
                                                                            "Composition"])

    with open("data/lematbulk_properties.pkl", "rb") as f:
        test_lemat = pickle.load(f)
    metric = MMD()
    result = metric([test_df], test_lemat)
    # Check computation didn't fail
    assert len(result.failed_indices) == 0

    # Check result structure
    assert "MMD" in result.metrics

    # Check values
    values = list(result.metrics["MMD"].values())
    assert not np.any(np.isnan(values))
    for val in result.metrics["MMD"].values():
        assert 0.0 <= val <= 1.0
