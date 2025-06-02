import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

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


def get_energy_above_hull(total_energy, composition, ds):
    chemical_system_set = tuple(sorted({el.symbol for el in composition.elements}))

    relevant_entries = {
        "composition": [
            Composition(x)
            for x in ds["train"]["chemical_formula_descriptive"]
            if Composition(x).chemical_system_set.issubset(chemical_system_set)
        ],
        "energy": [
            y
            for x, y in zip(
                ds["train"]["chemical_formula_descriptive"],
                ds["train"]["energy"],
            )
            if Composition(x).chemical_system_set.issubset(chemical_system_set)
        ],
    }

    pd_entries = []

    for composition, energy in zip(
        relevant_entries["composition"], relevant_entries["energy"]
    ):
        pd_entries.append(
            PDEntry(
                composition,
                energy,
            )
        )
    pd = PhaseDiagram(pd_entries)

    energy_above_hull = pd.get_decomp_and_e_above_hull(
        PDEntry(
            composition,
            total_energy,
        ),
        allow_negative=True,
    )[1]
    return energy_above_hull
