"""ORB embedding extraction utilities."""

import numpy as np
import torch
from orb_models.forcefield import atomic_system
from pymatgen.core.structure import Structure

from lematerial_forgebench.models.base import BaseEmbeddingExtractor


class ORBEmbeddingExtractor(BaseEmbeddingExtractor):
    """Embedding extractor for ORB models."""

    def __init__(self, model, device="cpu"):
        super().__init__(model, device)
        self.system_config = model._system_config

    def extract_node_embeddings(self, structure: Structure) -> np.ndarray:
        """Extract per-atom embeddings from ORB model.

        Parameters
        ----------
        structure : Structure
            Input structure

        Returns
        -------
        np.ndarray
            Node embeddings with shape (n_atoms, 1024)
        """
        # Convert to ASE atoms
        atoms = structure.to_ase_atoms()

        # Convert to ORB graph format
        graph = atomic_system.ase_atoms_to_atom_graphs(atoms, self.system_config)
        graph = graph.to(self.device)

        # Forward pass to get embeddings
        out = self.model(graph)
        node_features = out["node_features"]  # Shape: (N_atoms, 1024)

        return node_features.detach().cpu().numpy()

    def extract_graph_embedding_with_learned_pooling(
        self, structure: Structure
    ) -> np.ndarray:
        """Extract graph embedding using ORB's learned global pooling.

        This uses the same pooling layer that ORB uses before its energy head.

        Parameters
        ----------
        structure : Structure
            Input structure

        Returns
        -------
        np.ndarray
            Graph embedding from learned pooling
        """
        import torch_scatter

        atoms = structure.to_ase_atoms()
        graph = atomic_system.ase_atoms_to_atom_graphs(atoms, self.system_config)
        graph = graph.to(self.device)

        # Forward pass through the full model to get pooled representation
        out = self.model(graph)

        # The model should have a global pooling layer before energy prediction
        # This varies by ORB version, so we'll use the manual pooling as fallback
        if "graph_features" in out:
            graph_features = out["graph_features"]
        else:
            # Manual pooling as fallback
            node_features = out["node_features"]
            graph_features = torch_scatter.scatter_mean(
                node_features, graph.batch, dim=0
            )

        return graph_features.detach().cpu().numpy().squeeze()
