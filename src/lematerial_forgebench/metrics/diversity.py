"""
Diversity: distribution of space group, element species, lattice params, etc. And potential comparison with
training set. Ensuring the generative process explores a wide range of chemical and structural space, rather
than collapsing to a few known motifs. Metrics include: (i) Comparing distributions of elements, stoichiometry,
space groups, or lattice parameters between generated and training/reference sets[Kazeev et al., 2025]. (ii)
Calculating pairwise similarities (e.g., using structure matchers or adapted molecular metrics like Tanimoto
similarity[Gubina et al., 2024]) within the generated set (internal diversity) and between generated and
known structures (novelty/external diversity). (iii) Techniques like Density and Coverage[Naeem et al.,
2020], originally from image generation, aim to separately quantify sample fidelity and the extent to which
the generated distribution covers the true distribution. The Fréchet Wrenformer Distance (FWD) has been
proposed specifically for crystals to account for symmetry[Kelvinius et al., 2025]. (iv) Algorithms and metrics
like the moqd-score explicitly reward finding diverse sets of high-performing solutions across different feature
dimensions (e.g., different conductivity or deformation resistance values)[Janmohamed et al., 2024

"""
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict

from pymatgen.core import Structure

from lematerial_forgebench.metrics import BaseMetric
from lematerial_forgebench.metrics.base import MetricConfig, MetricResult

"""
-------
This script and class look to create a metric to quantify the diversity of a given dataset of materials.
-------
"""

@dataclass
class ElementComponentyConfig(MetricConfig):
    """Configuration for the DiversityScore metric.

    This configuration extends the base MetricConfig to include
    weights for evaluating different components of structural diversity.

    Parameters
    ----------
    comparison_distribtuion
    """

class _ElementDiversity(BaseMetric):

    """
    Calculates a score/ vector capturing elemental diversity across the strucutres
    """
    def __init__(
        self,
        name: str | None = None,
        description: str | None = "Measures the diversity of a set of structures.",
        lower_is_better: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name,
            description=description,
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )
        self.config = ElementComponentyConfig(
            name=name or self.__class__.__name__,
            description=description,
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
                    )
        self._element_historgram = defaultdict

    @staticmethod
    def _compute_vendi_score_with_uncertainty(elemental_distribution: dict[int, int | float]) -> dict[str, float]:
        """
        Compute the Vendi score (effective diversity) from an elemental distribution,
        along with Shannon entropy, variance, and standard deviation.

        Parameters
        ----------
        elemental_distribution : dict[int, int or float]
            A dictionary where keys are atomic numbers (or categories)
            and values are counts or frequencies.

        Returns
        -------
        dict[str, float]
            Dictionary containing:
            - vendi_score: Effective number of categories
            - shannon_entropy: Raw entropy in nats
            - entropy_variance: Estimated variance of entropy (multinomial approx.)
            - entropy_std: Standard deviation (sqrt of variance)
        """
        values = np.array(list(elemental_distribution.values()), dtype=float)
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


    @staticmethod
    def _normalize_histogram( hist: dict[int, int | float]) -> dict[int, float]:
        total = sum(hist.values())
        return {k: v / total for k, v in hist.items()}


    def _get_compute_attributes(self) -> dict[str, Any]:
        return {
            "elemental_histogram" : self._element_historgram
        }

    @staticmethod
    def compute_structure(structure: Structure, elemental_histogram: Dict[int:int]) -> float:
        """
        Computes the pre-aggregate values relevant to the diversity of the strucutre.
        These would include the density, lattice params, element type,
        Parameters
        ----------
        strucutre: Structure

        Returns
        -------


        """
        atomic_numbers = structure.atomic_numbers
        for atoms in atomic_numbers:
            elemental_histogram[atoms] += 1

        return 1.0

    def aggregate_results(self, values: list[float]) -> dict[str, Any]:

        elemental_diversity_metric = self._compute_vendi_score_with_uncertainty(self._element_historgram)

        return {
            "metrics": {
                "element_diversity_vendi_score": elemental_diversity_metric["vendi_score"],
                "element_diversity_shannon_entropy": elemental_diversity_metric["shannon_entropy"],
            },
            "primary_metric": "element_diversity_vendi_score",
            "uncertainties": {
                "shannon_entropy_std" : elemental_diversity_metric["entropy_std"],
                "shannon_entropy_variance": elemental_diversity_metric["entropy_variance"],
            }
        }



@dataclass
class DiversityConfig(MetricConfig):
    """Configuration for the DiversityScore metric.

    This configuration extends the base MetricConfig to include
    weights for evaluating different components of structural diversity.

    Parameters
    ----------
    elemental_diversity_weight : float, default=0.3
        Weight assigned to the elemental diversity component in the final score.
        Should be a value between 0 and 1. The total of all component weights
        should ideally sum to 1.0.

    symmetrical_diversity_weight : float, default=0.7
        Weight assigned to the symmetry (e.g., space group) diversity component
        in the final score. Higher values emphasize symmetry-based diversity more
        heavily.
    """

    elemental_diversity_weight: float = 0.3
    symmetrical_diversity_weight:float = 0.7


class DiversityMetric(BaseMetric):
    """Metric that quantifies the diversity of a set of structures.

    This metric computes structure-level diversity of Elements and Symmetrical components and aggregates them into
    a single diversity score, based on configurable weights for elemental contribution and symmetrical contribution.

    Assuming the full dataset is on the scale of 10e6, this would lead to too much compute utilization,
    so we randomly sample ~10e4 to quickly calculate diversity. we can take 3 rounds of compute to get a good value.

    Diversity calculated across:
        - Element Distribution
        - Space Group Distribution
        - Mean/Median Symmetry Quantification
        - Number of Atoms in each strucuture
        - Maybe the molecular weight (calculated from density and volume)
        - Adapting Vendi Score

    Parameters
    ----------
    name : str, optional
        Custom name for the metric. If None, the class name will be used.
    description : str, optional
        Description of what the metric measures.
    lower_is_better : bool, default=False
        Whether lower values indicate better performance (typically False for diversity).
    n_jobs : int, default=1
        Number of parallel jobs to run.
    element_div_weight : float, default=1.0
        Weight applied to the aggregated Element diversity score.
    symm_div_weight : float, default=1.0
        Weight applied to the aggregated Symmetry diversity score.
    """

    def __init__(
        self,
        name: str | None = None,
        description: str | None = "Measures the diversity of a set of structures.",
        lower_is_better: bool = False,
        n_jobs: int = 1,
        element_div_weight: float = 0.7, # placeholder value
        symm_div_weight: float = 0.3, # placeholder value
    ):
        super().__init__(
            name=name,
            description=description,
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )
        self.config = DiversityConfig(
            name=name or self.__class__.__name__,
            description=description,
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
            elemental_diversity_weight=element_div_weight,
            symmetrical_diversity_weight=symm_div_weight,
        )

    def compute_structure(structure: Structure) -> float:
        """
        Computes the pre-aggregate values relevant to the diversity of the strucutre.
        These would include the density, lattice params, element type,
        Parameters
        ----------
        strucutre: Structure
            A Given structure

        Returns
        -------

        """



