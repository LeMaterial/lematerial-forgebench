import os
from dataclasses import dataclass
from multiprocessing import cpu_count

import numpy as np
from datasets import load_dataset
from func_timeout import FunctionTimedOut, func_timeout
from pymatgen.core import Structure
from tqdm import tqdm

from lematerial_forgebench.preprocess.universal_stability_preprocess import (
    UniversalStabilityPreprocessor,
)


@dataclass
class StabilityPreprocessors:
    """
    
    """

    stability_preprocessor_orb: UniversalStabilityPreprocessor
    stability_preprocessor_mace: UniversalStabilityPreprocessor


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

def process_item_action(dataset):
    if not hasattr(process_item_action, "stability_processors"):
        process_item_action.stability_processors = StabilityPreprocessors(
                                            UniversalStabilityPreprocessor(model_name="orb"), 
                                            UniversalStabilityPreprocessor(model_name="mace"))
    LeMatIDs = []
    struts = []
    for i in tqdm(range(0, len(dataset))):
        struts.append(lematbulk_item_to_structure(dataset[i]))
        LeMatIDs.append(dataset[i]["immutable_id"])
    
    stability_preprocessor = process_item_action.stability_processors

    orb_result = stability_preprocessor.stability_preprocessor_orb(struts)
    mace_result = stability_preprocessor.stability_preprocessor_mace(struts)
    
    orb_graph_embeddings = []
    orb_node_embeddings = []
    
    mace_graph_embeddings = []
    mace_node_embeddings = []

    for orb_strut in orb_result.processed_structures:
        orb_graph_embeddings.append(orb_strut.properties.get("graph_embedding"))
        orb_node_embeddings.append(orb_strut.properties.get("node_embedding"))
    
    for mace_strut in mace_result.processed_structures:
        mace_graph_embeddings.append(mace_strut.properties.get("graph_embedding"))
        mace_node_embeddings.append(mace_strut.properties.get("node_embedding"))

    return {
        "LeMatIDs": LeMatIDs,
        "orb_graph_embeddings": orb_graph_embeddings, 
        "orb_node_embeddings": orb_node_embeddings, 
        "mace_graph_embeddings": mace_graph_embeddings, 
        "mace_node_embeddings": mace_node_embeddings, 
    }



def process_structures_wrapper(dataset):
    """Process an item.

    Parameters
    ----------
    item : dict 
        The item to process.

    Returns
    -------
    list[Any]
        The result of the processing of the item.
    """

    try:
        result = func_timeout(
            60_000, process_item_action, [dataset]
        )  # TODO This should not be hardcoded!
        return result
    except FunctionTimedOut:
        timeout_list = []
        for i in range(0, len(dataset)):
            timeout_list.append("TimedOut>100min")
        print("Function timed out and was skipped")
        return [dataset.immutable_id, 
                np.ones(len(dataset)), 
                np.ones(len(dataset)), 
                timeout_list]



if __name__ == "__main__":
    import pandas as pd

    full_dataset = False
    vals_spacing = 10
    vals = np.arange(0, 100, vals_spacing)
    dir_name = "test_small_lematbulk"
    
    dataset_name = "Lematerial/LeMat-Bulk"
    name = "compatible_pbe"
    split = "train"
    dataset = load_dataset(dataset_name, name=name, split=split, streaming=False)
    target_columns = ["LeMatID", "OrbGraphEmbeddings", "OrbNodeEmbeddings",
                        "MaceGraphEmbeddings", "MaceNodeEmbeddings"]


    if full_dataset: 
        vals = np.arange(0, len(dataset), vals_spacing) # example for running on all of LeMatBulk

    for i in tqdm(vals):
        dataset_temp = dataset.select(range(i, min(len(dataset), i + vals_spacing))) 

        # Process and handle results as they come
        print(
            f"Processing {len(dataset_temp)} structures using {cpu_count()} workers..."
        )


        output = process_structures_wrapper(dataset=dataset_temp)
        print("Creating DataFrame and saving to pkl...")

        df = pd.DataFrame(output, columns=target_columns)
        if os.path.exists("data/"+dir_name):
            df.to_pickle("data/"+dir_name+"/lematbulk_embeddings_full_" + str(i) + ".pkl")
        else:
            os.mkdir("data/"+dir_name)
            df.to_pickle("data/"+dir_name+"/lematbulk_embeddings_full_" + str(i) + ".pkl")
