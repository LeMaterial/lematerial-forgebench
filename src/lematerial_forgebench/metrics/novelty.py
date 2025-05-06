import numpy as np
from material_hasher.dataset_store import DatasetStore
from material_hasher.hasher import HASHERS
from material_hasher.similarity import SIMILARITY_MATCHERS
from pymatgen.core import Structure

from lematerial_forgebench.metrics.base import BaseMetric

ALGORITHMS = {
    **HASHERS,
    **SIMILARITY_MATCHERS,
}


class NoveltyMetric(BaseMetric):
    def __init__(
        self, dataset_store_path: str, threshold: float | None = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset_store = DatasetStore.load(dataset_store_path)
        self.threshold = threshold

    @staticmethod
    def compute_structure(structure: Structure, **compute_args) -> float:
        # Novelty means that the structure is not in the dataset
        # ie it is not equivalent to any of the structures in the dataset
        return 1 - np.mean(
            compute_args["dataset_store"].is_equivalent(
                structure, compute_args["threshold"]
            )
        )

    def _get_compute_attributes(self) -> dict:
        return {
            "dataset_store": self.dataset_store,
            "threshold": self.threshold,
        }

    def aggregate_results(self, values: list[float]) -> dict:
        return {
            "novelty_rate": np.mean(values),
        }
