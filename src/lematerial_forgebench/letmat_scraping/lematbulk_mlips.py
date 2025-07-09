import os
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset
from pymatgen.core import Structure

from lematerial_forgebench.preprocess.universal_stability_preprocess import (
    UniversalStabilityPreprocessor,
)


@dataclass
class StabilityPreprocessors:
    """
    
    """

    stability_preprocessor_orb: UniversalStabilityPreprocessor | None = None 
    stability_preprocessor_mace: UniversalStabilityPreprocessor | None = None
    stability_preprocessor_uma: UniversalStabilityPreprocessor | None = None 
    stability_preprocessor_equiformer: UniversalStabilityPreprocessor | None = None


def lematbulk_item_to_structure(item: dict) -> Structure:
    """Convert a LeMat-Bulk item to a pymatgen Structure object.

    Parameters
    ----------
    item : dict
        The item to convert.

    Returns
    -------
    Structure
        The pymatgen Structure object.
    """
    sites = item["species_at_sites"]
    coords = item["cartesian_site_positions"]
    cell = item["lattice_vectors"]

    structure = Structure(
        species=sites, coords=coords, lattice=cell, coords_are_cartesian=True
    )

    return structure

def stability_calculations(structures, stability_preprocessor):
        graph_embeddings = []
        node_embeddings = []
        energy = []
        forces = []
        formation_energy = []
        e_above_hull = []

        result = stability_preprocessor(structures)

        for processed_strut in result.processed_structures:
            graph_embeddings.append(processed_strut.properties.get("graph_embedding"))
            node_embeddings.append(processed_strut.properties.get("node_embeddings"))
            energy.append(processed_strut.properties.get("energy"))  
            forces.append(processed_strut.properties.get("forces"))  
            formation_energy.append(processed_strut.properties.get("formation_energy"))  
            e_above_hull.append(processed_strut.properties.get("e_above_hull"))

        return {"GraphEmbeddings":graph_embeddings, 
                "NodeEmbeddings":node_embeddings, 
                "Energy":energy, 
                "Forces":forces, 
                "FormationEnergy":formation_energy, 
                "EAboveHull":e_above_hull}

def process_item_action(dataset, stability_processors):

    LeMatIDs = []
    struts = []
    for i in range(0, len(dataset)):
        struts.append(lematbulk_item_to_structure(dataset[i]))
        LeMatIDs.append(dataset[i]["immutable_id"])
    
    output = {"orb": [],
              "mace": [],
              "uma": [],
              "equiformer": []}

    if stability_processors.stability_preprocessor_orb is None:
        pass
    else:
        print("starting orb calculation")
        print(stability_processors.stability_preprocessor_orb.config.name)
        output["orb"] = stability_calculations(struts, 
                                        stability_processors.stability_preprocessor_orb)  
        print("finished orb calculation")

    if stability_processors.stability_preprocessor_mace is None:
        pass
    else:
        print("starting mace calculation")
        print(stability_processors.stability_preprocessor_mace.config.name)
        output["mace"] = stability_calculations(struts, 
                                        stability_processors.stability_preprocessor_mace) 
        print("finished mace calculation")

         
    if stability_processors.stability_preprocessor_uma is None:
        pass
    else:
        print(stability_processors.stability_preprocessor_uma.config.name)
        print("starting uma calculation")
        output["uma"] = stability_calculations(struts, 
                                        stability_processors.stability_preprocessor_uma)
        print("finished uma calculation")

        
    if stability_processors.stability_preprocessor_equiformer is None:
        pass
    else:
        print("starting equiformer calculation")
        print(stability_processors.stability_preprocessor_equiformer.config.name)
        output["equiformer"] = stability_calculations(struts, 
                                        stability_processors.stability_preprocessor_equiformer)  
        print("finished equiformer calculation")

    return output


if __name__ == "__main__":
    import pandas as pd

    full_dataset = True
    vals_spacing = 100000
    # vals = np.arange(0, 1000, vals_spacing)
    dir_name = "test_small_lematbulk"
    
    dataset_name = "Lematerial/LeMat-Bulk"
    name = "compatible_pbe"
    split = "train"
    dataset = load_dataset(dataset_name, name=name, split=split, streaming=False)

    mlips = ["orb", "mace"]

    timeout = 60 # seconds, for one MLIP calculation

    try:
        if "orb" not in mlips:
            raise ValueError
        orb_stability_preprocessor = UniversalStabilityPreprocessor(model_name="orb", timeout=timeout, relax_structures=False)
    except (ImportError, ValueError): 
        orb_stability_preprocessor = None
    
    try:
        if "mace" not in mlips:
            raise ValueError
        mace_stability_preprocessor = UniversalStabilityPreprocessor(model_name="mace", timeout=timeout, relax_structures=False)
    except (ImportError, ValueError): 
        mace_stability_preprocessor = None

    try:
        if "uma" not in mlips:
            raise ValueError
        uma_stability_preprocessor = UniversalStabilityPreprocessor(model_name="uma", timeout=timeout, relax_structures=False)
    except (ImportError, ValueError): 
        uma_stability_preprocessor = None

    try:
        if "equiformer" not in mlips:
            raise ValueError
        equiformer_stability_preprocessor = UniversalStabilityPreprocessor(model_name="equiformer", timeout=timeout, relax_structures=False)
    except (ImportError, ValueError): 
        equiformer_stability_preprocessor = None


    preprocessors = StabilityPreprocessors(
        orb_stability_preprocessor,
        mace_stability_preprocessor,
        uma_stability_preprocessor,
        equiformer_stability_preprocessor,
    )

    if full_dataset: 
        vals = np.arange(0, len(dataset), vals_spacing) # example for running on all of LeMatBulk

    if os.path.exists("data/"+dir_name):
        pass
    else:
        os.mkdir("data/"+dir_name)

    for i in vals:
        dataset_temp = dataset.select(range(i, min(len(dataset), i + vals_spacing))) 

        output = process_item_action(dataset=dataset_temp, 
                                            stability_processors=preprocessors)
        print("Creating DataFrame and saving to pkl...")
        for key in output.keys():
            if isinstance(output[key], dict): 
                temp_df = pd.DataFrame(output[key], columns = output[key].keys())
                if os.path.exists("data/"+dir_name+"/"+key):
                    pass
                else:
                    os.mkdir("data/"+dir_name+"/"+key)

                temp_df.to_pickle("data/"+dir_name+"/"+key+"/"+key+"_lematbulk_MLIP_full_" + str(i) + ".pkl")
            else:
                pass
