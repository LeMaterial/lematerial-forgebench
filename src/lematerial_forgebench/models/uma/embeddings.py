"""UMA embedding extraction utilities."""

import numpy as np
import torch
from pymatgen.core.structure import Structure

try:
    import torch_scatter
    from fairchem.core.utils.ase_conversion import atoms_to_data

    UMA_AVAILABLE = True
except ImportError:
    UMA_AVAILABLE = False

from lematerial_forgebench.models.base import BaseEmbeddingExtractor


class UMAEmbeddingExtractor(BaseEmbeddingExtractor):
    """Embedding extractor for UMA models."""

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def extract_node_embeddings(self, structure: Structure) -> np.ndarray:
        """Extract per-atom embeddings from UMA model.

        Parameters
        ----------
        structure : Structure
            Input structure

        Returns
        -------
        np.ndarray
            Node embeddings with shape (n_atoms, hidden_dim)
        """
        atoms = structure.to_ase_atoms()

        # Convert to FAIRChem's AtomicData format
        adata = atoms_to_data(atoms, device=self.device)

        out = self.model(adata, return_embeddings=True)
        node_embeddings = out["node_embeddings"]

        return node_embeddings.cpu().numpy()

    def extract_graph_embedding_with_pooling(
        self, structure: Structure, pooling_method: str = "mean"
    ) -> np.ndarray:
        """Extract graph embedding using explicit pooling.

        Parameters
        ----------
        structure : Structure
            Input structure
        pooling_method : str
            Pooling method: "mean", "sum", "max"

        Returns
        -------
        np.ndarray
            Graph-level embedding
        """
        atoms = structure.to_ase_atoms()
        adata = atoms_to_data(atoms, device=self.device)

        out = self.model(adata, return_embeddings=True)
        node_embeddings = out["node_embeddings"]

        # Pool over atoms to get graph representation
        if pooling_method == "mean":
            graph_emb = torch_scatter.scatter_mean(node_embeddings, adata.batch, dim=0)
        elif pooling_method == "sum":
            graph_emb = torch_scatter.scatter_sum(node_embeddings, adata.batch, dim=0)
        elif pooling_method == "max":
            graph_emb = torch_scatter.scatter_max(node_embeddings, adata.batch, dim=0)[
                0
            ]
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")

        return graph_emb.detach().cpu().numpy().squeeze()


class UMAASECalculator:
    """ASE calculator wrapper for UMA."""

    def __init__(self, predictor, device="cpu"):
        self.predictor = predictor
        self.device = device
        self.implemented_properties = ["energy", "forces"]
        self.results = {}

    def calculate(self, atoms, properties=None, system_changes=None):
        """Calculate properties using UMA."""
        # Convert to FAIRChem format
        adata = atoms_to_data(atoms, device=self.device)

        # Predict using UMA
        predictions = self.predictor.predict(adata)

        # Store results
        self.results = {}
        if "energy" in predictions:
            self.results["energy"] = predictions["energy"].cpu().numpy().item()
        if "forces" in predictions:
            self.results["forces"] = predictions["forces"].cpu().numpy()

    def get_potential_energy(self, atoms=None):
        if atoms is not None:
            self.calculate(atoms)
        return self.results.get("energy", 0.0)

    def get_forces(self, atoms=None):
        if atoms is not None:
            self.calculate(atoms)
        return self.results.get("forces", np.zeros((len(atoms), 3)))
