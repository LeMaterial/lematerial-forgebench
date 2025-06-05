"""
Unit Test File for Diversity Metrics. File path at src/lematerial_forgebench/metrics/diversity_metric.py
Note: Diversity Assumes stable Materials, and ignores unstable or invalid materials
"""

import csv
from io import StringIO

import pytest
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser

from lematerial_forgebench.metrics.base import MetricResult
from lematerial_forgebench.metrics.diversity_metric import (
    _ElementDiversity,
    _SpaceGroupDiversity,
    )

trial_data_file_path = "trial_data/crystal_symmcd_mp20.csv"

def load_structures_from_csv(filepath: str, n_samples: int) -> list[Structure]:
    """Load the first `n` CIF structures from a CSV where each row is a quoted CIF string.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    n_samples : int
        Number of structures to load.

    Returns
    -------
    list[Structure]
        List of pymatgen Structure objects.
    """
    structures = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            cif_text = row[0]
            parser = CifParser(StringIO(cif_text))
            structure = parser.parse_structures()[0]
            structures.append(structure)
            if len(structures) >= n_samples:
                break
    return structures

@pytest.fixture
def fetch_sample_from_trial_data(
    path:str = trial_data_file_path,
    n_samples:int = 10 
    ):
    """Creating a Fixture of 10 data points from the trial data to test metrics"""
    sampled_structures = load_structures_from_csv(filepath=trial_data_file_path, n_samples=10)
    return sampled_structures


def test_trial_data_load(fetch_sample_from_trial_data):
    """Sanity Check for pytest fixture : fetch_sample_from_trial_data """
    for structure in fetch_sample_from_trial_data:
        assert isinstance(structure, Structure)


class TestElementDiversityMetric:
    
    """Tests the Elemental Diversity Metric component in overall Diversity Metrics"""

    def test_diversity_metrics_initialization(self):
        """ Tests the initialization of Diversity Metric"""

        metric = _ElementDiversity()
        assert metric.name == "Element Diversity"

    def test_element_diversity_score(self, fetch_sample_from_trial_data):
        """Runs a Sanity check on the elemental diversity score
        
            Parameters
            ----------
            fetch_sample_from_trial_data : pytest.fixture
                PyTest Fixture which locally initializes a list of 10 structures from the trial data
        """

        metric = _ElementDiversity()
        metric_score = metric.compute(fetch_sample_from_trial_data)
        
        # Sanity Checks
        assert isinstance(metric_score, MetricResult)
        assert len(metric_score.failed_indices) == 0

        # Check Result Structure
        assert "element_diversity_vendi_score" in metric_score.metrics
        assert "element_diversity_shannon_entropy" in metric_score.metrics
        assert "invalid_computations_in_batch" in metric_score.metrics
        assert metric_score.primary_metric == "element_diversity_vendi_score"

        # Check Value Range
        assert 0.0 <= metric_score.metrics[metric_score.primary_metric] <= 118
        
        #Check Uncertainty Sanity
        assert isinstance(metric_score.uncertainties["shannon_entropy_std"], float)
        assert isinstance(metric_score.uncertainties["shannon_entropy_variance"], float)


class TestSpaceGroupDiversityMetric:
    
    """Tests the SpaceGroup Diversity Metric component in overall Diversity Metrics"""

    def test_diversity_metrics_initialization(self):
        """ Tests the initialization of Diversity Metric"""

        metric = _SpaceGroupDiversity()
        assert metric.name == "Space Group Diversity"

    def test_element_diversity_score(self, fetch_sample_from_trial_data):
        """Runs a Sanity check on the elemental diversity score
        
            Parameters
            ----------
            fetch_sample_from_trial_data : pytest.fixture
                PyTest Fixture which locally initializes a list of 10 structures from the trial data
        """

        metric = _SpaceGroupDiversity()
        metric_score = metric.compute(fetch_sample_from_trial_data)
        
        # Sanity Checks
        assert isinstance(metric_score, MetricResult)
        assert len(metric_score.failed_indices) == 0

        # Check Result Structure
        assert "spacegroup_diversity_vendi_score" in metric_score.metrics
        assert "spacegroup_diversity_shannon_entropy" in metric_score.metrics
        assert "mean_symmetry_rating" in metric_score.metrics
        assert metric_score.primary_metric == "spacegroup_diversity_vendi_score"

        # Check Value Range
        assert 0.0 <= metric_score.metrics[metric_score.primary_metric] <= 230.0
        assert 0.0 <= metric_score.metrics["mean_symmetry_rating"] <= 1.0

        
        #Check Uncertainty Sanity
        assert isinstance(metric_score.uncertainties["shannon_entropy_std"], float)
        assert isinstance(metric_score.uncertainties["shannon_entropy_variance"], float)
