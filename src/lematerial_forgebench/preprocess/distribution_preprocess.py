from dataclasses import dataclass, field

import numpy as np
from datasets import load_dataset
from pymatgen.core import Composition, Element, Structure

from lematerial_forgebench.preprocess.base import BasePreprocessor, PreprocessorConfig
from lematerial_forgebench.utils.distribution_utils import (
    map_space_group_to_crystal_system,
    one_hot_encode_composition,
)


@dataclass
class DistributionPreprocessorConfig(PreprocessorConfig):
    name: str
    description: str
    n_jobs: int


class DistributionPreprocessor(BasePreprocessor):
    def __init__(
        self,
        distribution_metric: str = "all",
        crystal_descriptor: str = "SpaceGroup",
        name: str | None = None,
        description: str | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "DistributionPreprocessor",
            description=description
            or "Preprocesses structures for distribution analysis",
            n_jobs=n_jobs,
        )
        self.config = DistributionPreprocessorConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
        )

    @staticmethod
    def process_structure(
        structure: list,
    ) -> list:
        """Process a sample of structures by turning them into a dataframe
        built of labeled structural properties for comparison to other distributions.

        Parameters
        ----------
        structure : list
            A list of pymatgen Structure objects to process.


        Returns
        -------
        list
            The list of extracted properties from each structure.

        """

        one_hot_vectors = one_hot_encode_composition(structure.composition)
        row = [
            structure.volume,
            structure.density,
            structure.num_sites / structure.volume,
            structure.get_space_group_info()[1],
            map_space_group_to_crystal_system(structure.get_space_group_info()[1]),
            one_hot_vectors[0],
            one_hot_vectors[1],
        ]

        return row
