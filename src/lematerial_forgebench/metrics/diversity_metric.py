"""
Diversity: distribution of space group, element species, lattice params, etc. And potential comparison with
training set. Ensuring the generative process explores a wide range of chemical and structural space, rather
than collapsing to a few known motifs.

Metrics include: 
(i) Comparing distributions of elements, stoichiometry,
space groups, or lattice parameters between generated and training/reference sets[Kazeev et al., 2025]. 

(ii) Calculating pairwise similarities (e.g., using structure matchers or adapted molecular metrics like Tanimoto
similarity[Gubina et al., 2024]) within the generated set (internal diversity) and between generated and
known structures (novelty/external diversity).

(iii) Techniques like Density and Coverage[Naeem et al.,2020], originally from image generation, aim to 
separately quantify sample fidelity and the extent to which the generated distribution covers the true
distribution. The Fréchet Wrenformer Distance (FWD) has been proposed specifically for crystals to account
 for symmetry[Kelvinius et al., 2025].

(iv) Algorithms and metrics like the moqd-score explicitly reward finding diverse sets of
high-performing solutions across different feature dimensions (e.g., different conductivity
or deformation resistance values)[Janmohamed et al., 2024

"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from pymatgen.core import Structure

from lematerial_forgebench.metrics import BaseMetric
from lematerial_forgebench.metrics.base import MetricConfig
from lematerial_forgebench.utils.logging import logger

"""
-------------------------------------------------------------------------------
Elemental Diversity
-------------------------------------------------------------------------------
"""

@dataclass
class ElementComponentConfig(MetricConfig):
    """Configuration for the Elemental Diversity metric.

    This configuration extends the base MetricConfig to include
    weights for evaluating Elemental components of structural diversity.
    """

class _ElementDiversity(BaseMetric):
    """
    Calculates a scalar score capturing elemental diversity across the structures compared to reference distribution
    """
    def __init__(
        self,
        name: str | None = "Element Diversity",
        description: str | None = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "Element Diversity",
            description=description 
            or "Scalar Score of Elemental Diversity in Generated Set",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )
        self.config = ElementComponentConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            )
        
        # Initialize element Histogram
        self._init_element_histogram()
    
    def _init_element_histogram(self):
        """
        Initialize an empty dictionary to function as a histogram counter for element.
        Dictionary mapping is Element -> total count across all structure
        Note: Before compute, Dictionary Values are normalized 
        """

        self.element_histogram = defaultdict(int)


    def _compute_vendi_score_with_uncertainty(self) -> dict[str, float]:
        """
        Compute the Vendi score (effective diversity) from an elemental distribution,
        along with Shannon entropy, variance, and standard deviation.

        Parameters
        ----------
        elemental_distribution : dict[str, int]
            A dictionary where keys are atomic numbers (or categories)
            and values are counts or frequencies.

        Returns
        -------
        dict[str, float]
            Dictionary containing:
            - vendi_score: Effective number of categories
            - shannon_entropy: Raw entropy in nats
            - entropy_variance: Estimated variance of entropy (multi-nomial approx.)
            - entropy_std: Standard deviation (sqrt of variance)
        
        References
        ----------
        Friedman, D., & Dieng, A. B. (2023). 
        The Vendi Score: A Diversity Evaluation Metric for Machine Learning. 
        Transactions on Machine Learning Research. https://openreview.net/forum?id=aNVLfhU9pH

        """
        values = np.array(list(self.element_histogram.values()), dtype=float)
        total = np.sum(values)

        if total == 0:
            return {
                "vendi_score": 0.0,
                "shannon_entropy": 0.0,
                "entropy_variance": 0.0,
                "entropy_std": 0.0,
            }

        # Normalize to probability distribution
        probs = values / total

        # Shannon entropy (in nats)
        entropy = -np.sum(probs * np.log(probs + 1e-12))  # add epsilon to avoid log(0)

        # Vendi score
        vendi_score = np.exp(entropy)

        # Variance of entropy estimate (asymptotic approximation)
        second_moment = np.sum(probs * (np.log(probs + 1e-12)) ** 2)
        entropy_variance = (1 / total) * (second_moment - entropy ** 2)
        entropy_std = np.sqrt(entropy_variance)

        return {
            "vendi_score": vendi_score,
            "shannon_entropy": entropy,
            "entropy_variance": entropy_variance,
            "entropy_std": entropy_std,
        }


    def _get_compute_attributes(self) -> dict[str, Any]:
        return {
            "elemental_histogram" : self.element_histogram
        }

    @staticmethod
    def compute_structure(structure: Structure, elemental_histogram: Dict[str, int]) -> float:
        """
        Retrieves all elements present in Structure and adds count to internal elemental distribution
        Parameters
        ----------
        structure: Structure
            A pymatgen Structure object to evaluate.
        elemental_histogram: dict[str:int]
            Class variable for storing the current histogram/distribution of elements across all structures

        Returns:
        -------
        float:
            This value serves as a Binary Indicator representing if the structure was successfully evaluated or not. 

        """
        try:
            all_sites_in_structure = structure.sites
            for site in all_sites_in_structure:
                atom = site.species_string
                elemental_histogram[atom] += 1
            return 0.0
    
        except Exception as e:
            logger.debug(f"Could not determine Elements in {structure.formula} : {str(e)}")
            return 1.0

    def aggregate_results(self, values: list[float]) -> dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            1 or 0 indicators showcasing if the compute function was able to parse through a structure
            - values of 1 represent erroneous calculations and is used for debugging only

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        invalid_computations = sum(values)
        elemental_diversity_metric = self._compute_vendi_score_with_uncertainty()

        return {
            "metrics": {
                "element_diversity_vendi_score": elemental_diversity_metric["vendi_score"],
                "element_diversity_shannon_entropy": elemental_diversity_metric["shannon_entropy"],
                "invalid_computations_in_batch":invalid_computations,
            },
            "primary_metric": "element_diversity_vendi_score",
            "uncertainties": {
                "shannon_entropy_std" : elemental_diversity_metric["entropy_std"],
                "shannon_entropy_variance": elemental_diversity_metric["entropy_variance"],
            }
        }

"""
-------------------------------------------------------------------------------
SpaceGroup Diversity
-------------------------------------------------------------------------------
"""

@dataclass
class SpaceGroupComponentConfig(MetricConfig):
     """Configuration for the DiversityScore metric.

    This configuration extends the base MetricConfig to include
    weights for evaluating different components of structural diversity.

    """

class _SpaceGroupDiversity(BaseMetric):
    """
    Calculates a scalar score capturing Spacegroup Diversity across the structures
    """

    def __init__(
        self,
        name: str | None = "Space Group Diversity",
        description: str | None = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "Space Group Diversity",
            description=description 
            or "Scalar Score of Space Group Diversity in Generated Set",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )
        self.config = SpaceGroupComponentConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            )
        
        self._init_spacegroup_histogram()

    def _init_spacegroup_histogram(self):
        """
        Initialize an empty dictionary to function as a histogram counter for element.
        Dictionary mapping is Element -> total count across all structure
        Note: Before compute, Dictionary Values are normalized 
        """

        self.spacegroup_histogram = defaultdict(int)
    
    def _compute_vendi_score_with_uncertainty(self) -> dict[str, float]:
        """
        Compute the Vendi score (effective diversity) from the spacegroup distribution,
        along with Shannon entropy, variance, and standard deviation.

        Parameters
        ----------
        spacegroup_distribution : dict[str, int]
            A dictionary where keys are atomic numbers (or categories)
            and values are counts or frequencies.

        Returns
        -------
        dict[str, float]
            Dictionary containing:
            - vendi_score: Effective number of categories
            - shannon_entropy: Raw entropy in nats
            - entropy_variance: Estimated variance of entropy (multi-nomial approx.)
            - entropy_std: Standard deviation (sqrt of variance)
        
        References
        ----------
        Friedman, D., & Dieng, A. B. (2023). 
        The Vendi Score: A Diversity Evaluation Metric for Machine Learning. 
        Transactions on Machine Learning Research. https://openreview.net/forum?id=aNVLfhU9pH

        """
        values = np.array(list(self.spacegroup_histogram.values()), dtype=float)
        total = np.sum(values)

        if total == 0:
            return {
                "vendi_score": 0.0,
                "shannon_entropy": 0.0,
                "entropy_variance": 0.0,
                "entropy_std": 0.0,
            }

        # Normalize to probability distribution
        probs = values / total

        # Shannon entropy (in nats)
        entropy = -np.sum(probs * np.log(probs + 1e-12))  # add epsilon to avoid log(0)

        # Vendi score
        vendi_score = np.exp(entropy)

        # Variance of entropy estimate (asymptotic approximation)
        second_moment = np.sum(probs * (np.log(probs + 1e-12)) ** 2)
        entropy_variance = (1 / total) * (second_moment - entropy ** 2)
        entropy_std = np.sqrt(entropy_variance)

        return {
            "vendi_score": vendi_score,
            "shannon_entropy": entropy,
            "entropy_variance": entropy_variance,
            "entropy_std": entropy_std,
        }

    def _get_compute_attributes(self) -> dict[str, Any]:
        return {
            "spacegroup_histogram" : self.spacegroup_histogram
        }
    
    @staticmethod
    def compute_structure(structure: Structure, spacegroup_histogram: Dict[str, int]) -> float:
        """
        Retrieves all elements present in Structure and adds count to internal elemental distribution
        Parameters
        ----------
        structure: Structure
            A pymatgen Structure object to evaluate.
        elemental_histogram: dict[str:int]
            Class variable for storing the current histogram/distribution of elements across all structures

        Returns:
        -------
        float:
            This value serves as a Binary Indicator representing if the structure was successfully evaluated or not. 

        """
        try:
            spacegroup_symbol, spacegroup_number = structure.get_space_group_info()
            spacegroup_histogram[spacegroup_symbol] += 1
            return spacegroup_number/230
    
        except Exception as e:
            logger.debug(f"Could not determine Spacegroup in {structure.formula} : {str(e)}")
            return 0
        
    def aggregate_results(self, values: list[float]) -> dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            the structure-wise symmetry score computed based off spacegroup number / 230. 
            This normalized score gives a sense of low vs high symmetry

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        mean_symmetry_rating = np.mean(values)
        spacegroup_diversity_metric = self._compute_vendi_score_with_uncertainty()

        return {
            "metrics": {
                "spacegroup_diversity_vendi_score": spacegroup_diversity_metric["vendi_score"],
                "spacegroup_diversity_shannon_entropy": spacegroup_diversity_metric["shannon_entropy"],
                "mean_symmetry_rating":mean_symmetry_rating,
            },
            "primary_metric": "spacegroup_diversity_vendi_score",
            "uncertainties": {
                "shannon_entropy_std" : spacegroup_diversity_metric["entropy_std"],
                "shannon_entropy_variance": spacegroup_diversity_metric["entropy_variance"],
            }
        }

        
"""
-------------------------------------------------------------------------------
Density Diversity
-------------------------------------------------------------------------------
"""
