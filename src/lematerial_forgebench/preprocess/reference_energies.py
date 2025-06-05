import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition
import numpy as np

from functools import lru_cache

from pymatgen.core.periodic_table import Element

import pickle

from pymatgen.core import Element
from tqdm import tqdm

CURRENT_FOLDER = os.path.dirname(Path(__file__).resolve())


def build_formation_energy_reference_file():
    from datasets import load_dataset

    ds_pbe = load_dataset("LeMaterial/LeMat-Bulk", "compatible_pbe")
    # ds_pbesol = load_dataset("LeMaterial/LeMat-Bulk", "compatible_pbesol")
    # ds_scan = load_dataset("LeMaterial/LeMat-Bulk", "compatible_scan")

    data = {
        "energy": [
            *[
                x / z
                for x, y, z in zip(
                    ds_pbe["train"]["energy"],
                    ds_pbe["train"]["nelements"],
                    ds_pbe["train"]["nsites"],
                )
                if y == 1
            ],
        ],
        "composition": [
            *[
                [x for x in Composition(Counter(y)).chemical_system_set][0]
                for x, y in zip(
                    ds_pbe["train"]["nelements"], ds_pbe["train"]["species_at_sites"]
                )
                if x == 1
            ],
        ],
    }

    element_chem_pot = {}
    for element, energy in (
        pd.DataFrame(data).groupby("composition").min().to_dict()["energy"].items()
    ):
        if element not in element_chem_pot:
            element_chem_pot[element] = {}
        element_chem_pot[element]["pbe"] = energy

    json.dump(
        element_chem_pot,
        open(os.path.join(CURRENT_FOLDER, "element_chem_pot.json"), "w"),
    )


def get_formation_energy_from_composition_energy(
    total_energy, composition, functional="pbe"
):
    element_chem_pot = json.load(
        open(os.path.join(CURRENT_FOLDER, "element_chem_pot.json"))
    )
    try:
        res = 0
        res = total_energy - sum(
            [
                element_chem_pot[k][functional] * v
                for k, v in composition.as_dict().items()
            ]
        )
        return res / len(composition)
    except Exception as e:
        print("Error in get_formation_energy_from_composition_energy: ", e)
        return None

def one_hot_encode_composition(elements):
    one_hot = np.zeros(118)
    for element in elements:
        one_hot[int(Element(element).number) - 1] = 1
    return one_hot


def process_chunk(chunk):
    one_hot_compositions = []
    for elements in tqdm(chunk):
        one_hot_compositions.append(one_hot_encode_composition(elements))
    return one_hot_compositions


@lru_cache(maxsize=None)
def _retrieve_df():
    csv_path = "dataset_filtered.csv"
    return pd.read_csv(csv_path)


@lru_cache(maxsize=None)
def _retrieve_matrix():
    npy_path = "all_compositions.npy"
    return np.load(npy_path)


def filter_df(df, matrix, composition):
    structure_vector = one_hot_encode_composition(composition.elements).reshape(-1, 1)
    forbidden_elements = 1 - structure_vector
    intersection_elements = df.loc[(matrix @ forbidden_elements) == 0]

    # print(intersection_elements)

    return intersection_elements


def get_energy_above_hull(total_energy, composition):
    intersection_elements = filter_df(_retrieve_df(), _retrieve_matrix(), composition)


    # Create PDEntries from the filtered DataFrame
    pd_entries = [
        PDEntry(Composition(row["chemical_formula_descriptive"]), row["energy"])
        for _, row in intersection_elements.iterrows()
    ]

    if not pd_entries:
        raise ValueError(
            f"No entries found in dataset containing any of the elements in: {composition.elements}"
        )

    # Construct phase diagram
    pd = PhaseDiagram(pd_entries)


    # Compute energy above hull
    entry = PDEntry(composition, total_energy)
    e_above_hull = pd.get_decomp_and_e_above_hull(entry, allow_negative=True)[1]

    print(e_above_hull)
    return e_above_hull
