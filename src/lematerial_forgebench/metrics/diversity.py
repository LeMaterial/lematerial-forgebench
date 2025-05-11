from dataclasses import dataclass

from lematerial_forgebench.metrics import BaseMetric
from lematerial_forgebench.metrics.base import MetricConfig

"""
-------
This script and class look to create a metric to quantify the diversity of a given dataset of materials.
-------
"""

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
