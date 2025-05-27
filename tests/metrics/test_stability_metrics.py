"""Tests for stability metrics implementation."""

import pytest
from pymatgen.util.testing import PymatgenTest

from lematerial_forgebench.metrics.stability_metrics import StabilityMetric


@pytest.fixture
def test_structures():
    """Create test structures for stability evaluation."""
    test = PymatgenTest()
    structures = [
        test.get_structure("Si"),
        test.get_structure("LiFePO4"),
        test.get_structure("CsCl"),
    ]
    return structures


def test_stability_metric(test_structures):
    metric = StabilityMetric(
        relaxer_type="orb",
        relaxer_config={"steps": 500, "fmax": 0.02},
        mp_entries_file="src/lematerial_forgebench/utils/relaxers/2023-02-07-ppd-mp.pkl.gz",
    )

    result = metric(test_structures)

    # Check that computation ran (may have failed indices due to no MP entries)
    assert len(result.individual_values) == len(test_structures)

    # Check result structure
    assert "mean_e_above_hull" in result.metrics
    assert "stable_ratio" in result.metrics
    assert "metastable_ratio" in result.metrics
    assert result.primary_metric == "stable_ratio"

    assert result.metrics["stable_ratio"] >= 0.0
    assert result.metrics["metastable_ratio"] >= 0.0
