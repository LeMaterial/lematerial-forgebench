"""Equiformer v2 embedding extraction utilities."""

import numpy as np
import torch
from pymatgen.core.structure import Structure

from lematerial_forgebench.models.base import BaseEmbeddingExtractor


class EquiformerEmbeddingExtractor(BaseEmbeddingExtractor):
    """Embedding extractor for Equiformer v2 models."""

    def __init__(self, model, converter, device="cpu"):
        self.model = model
        self.converter = converter
        self.device = device

    def extract_node_embeddings(self, structure: Structure) -> np.ndarray:
        """Extract per-atom embeddings from Equiformer v2.

        Parameters
        ----------
        structure : Structure
            Input structure

        Returns
        -------
        np.ndarray
            Node embeddings with shape (n_atoms, (1+L_max)^2, C)
            Flattened to (n_atoms, (1+L_max)^2 * C)
        """
        atoms = structure.to_ase_atoms()
        data = self.converter.convert(atoms).to(self.device)

        with torch.no_grad():
            out = self.model(data, return_embeddings=True)
            node_irreps = out["node_embeddings"]  # Shape: (N, (1+L_max)^2, C)

        # Flatten the irreps dimension for easier use
        n_atoms, irreps_dim, channels = node_irreps.shape
        node_embeddings = node_irreps.view(n_atoms, -1)  # (N, irreps_dim * C)

        return node_embeddings.cpu().numpy()


class EquiformerASECalculator:
    """ASE calculator wrapper for Equiformer v2."""

    def __init__(self, model, converter, device="cpu"):
        self.model = model
        self.converter = converter
        self.device = device
        self.implemented_properties = ["energy", "forces"]
        self.results = {}

    def calculate(self, atoms, properties=None, system_changes=None):
        """Calculate properties using Equiformer v2."""
        data = self.converter.convert(atoms).to(self.device)

        with torch.no_grad():
            out = self.model(data)

        # Store results
        self.results = {}
        if "energy" in out:
            self.results["energy"] = out["energy"].cpu().numpy().item()
        if "forces" in out:
            self.results["forces"] = out["forces"].cpu().numpy()

    def get_potential_energy(self, atoms=None):
        if atoms is not None:
            self.calculate(atoms)
        return self.results.get("energy", 0.0)

    def get_forces(self, atoms=None):
        if atoms is not None:
            self.calculate(atoms)
        return self.results.get("forces", np.zeros((len(atoms), 3)))
