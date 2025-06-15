import numpy as np
from datasets import load_dataset
from dataclasses import dataclass, field
from lematerial_forgebench.preprocess.base import BasePreprocessor, PreprocessorConfig
from lematerial_forgebench.utils.distribution_utils import one_hot_encode_composition 
from pymatgen.core import Composition, Element, Structure

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
            name = name or "DistributionPreprocessor",
            description = description or "Preprocesses structures for distribution analysis",
            n_jobs=n_jobs,
        )
        self.config = DistributionPreprocessorConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
        )


    @staticmethod
    def process_structure(
        structure: Structure,
    ) -> list:
        """Process a single structure by relaxing it and computing e_above_hull.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to process.
        relaxer : BaseRelaxer
            Relaxer object to use.

        Returns
        -------
        Structure
            The processed Structure with relaxed geometry and e_above_hull in properties.

        Raises
        ------
        Exception
            If relaxation fails or other processing errors occur.
        """

        one_hot_vectors = one_hot_encode_composition(structure.composition)
        row = [structure.get_space_group_info()[1], structure.volume, structure.num_sites/structure.volume, one_hot_vectors[0],
               one_hot_vectors[1]]
        
        return row
