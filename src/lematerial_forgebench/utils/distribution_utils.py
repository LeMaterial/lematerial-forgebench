import json

import matplotlib.pyplot as plt
import numpy as np
from frechetdist import frdist
from pymatgen.core import Composition, Element, Structure
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity


def safe_float(value):
    """Currently a no-op function.

    This is a placeholder for a function that will safely convert a value to a float,
    handling None and NaN.
    """
    return value


def map_space_group_to_crystal_system(space_group: int):
    if space_group <= 2 and space_group > 0:
        return 1  # "triclinic"
    elif space_group <= 15 and space_group > 2:
        return 2  # "monoclinic"
    elif space_group <= 74 and space_group > 15:
        return 3  # "orthorhombic"
    elif space_group <= 142 and space_group > 74:
        return 4  # "tetragonal"
    elif space_group <= 167 and space_group > 142:
        return 5  # "trigonal"
    elif space_group <= 194 and space_group > 167:
        return 6  # "hexagonal"
    elif space_group <= 230 and space_group > 194:
        return 7  # "cubic"
    else:
        raise ValueError


def one_hot_encode_composition(composition):
    one_hot_counts = np.zeros(118)
    one_hot_bool = np.zeros(118)
    for element in composition.elements:
        one_hot_bool[int(Element(element).number) - 1] = 1
        one_hot_counts[int(Element(element).number) - 1] = composition.as_dict()[
            element.as_dict()["element"]
        ]
    return [one_hot_counts, one_hot_bool]


def generate_probabilities(df, metric, metric_type=np.int64, return_2d_array=False):
    # create an empty list of space groups/crystal systems/compositions and fill in proportions/counts
    # depending on the application (as some samples will have zero of space group 1 etc)

    if metric_type == np.int64:
        if metric == "SpaceGroup":
            prob_dict = {}
            for i in range(0, 230):
                prob_dict[str(i + 1)] = 0
        if metric == "CrystalSystem":
            prob_dict = {}
            for i in range(0, 7):
                prob_dict[str(i + 1)] = 0

        probs = np.asarray(df.value_counts(metric) / len(df))
        indices = np.asarray(df.value_counts(metric).index)
        strut_list = np.concatenate(([indices], [probs]), axis=0).T
        strut_list = strut_list[strut_list[:, 0].argsort()]
        if return_2d_array:
            return strut_list

        for row in strut_list:
            prob_dict[str(int(row[0]))] = row[1]

    if metric_type == np.ndarray:
        prob_dict = {}
        for i in range(0, 118):
            prob_dict[str(i + 1)] = 0

        one_hots = np.zeros(118)
        for i in range(0, len(df)):
            one_hots += df.iloc[i][metric]

        one_hots = one_hots / sum(one_hots)

        for i in range(0, 118):
            prob_dict[str(i + 1)] = one_hots[i]

    return prob_dict


def compute_shannon_entropy(probability_vals):
    H = 0
    for i in range(len(probability_vals)):
        val = probability_vals[i]
        if val < 10**-14:
            pass
        else:
            H += val * np.log(val)
    H = -H
    return H


def compute_jensen_shannon_distance(
    reference_data, generated_crystals, crystal_param, metric_type
):
    """
    reference_data - letmatbulk dataframe
    generated_crystals - dataframe of generated crystals
    crystal_param - CrystalSystem, SpaceGroup, or LatticeConstant
    """
    generated_crystals_dist = generate_probabilities(
        generated_crystals, metric=crystal_param, metric_type=metric_type
    )
    if crystal_param not in ["CompositionCounts", "Composition"]:
        reference_data_dist = generate_probabilities(
            reference_data, metric=crystal_param, metric_type=metric_type
        )

    elif crystal_param == "CompositionCounts":
        with open("data/lematbulk_composition_counts_distribution.json", "r") as file:
            reference_data_dist = json.load(file)
    elif crystal_param == "Composition":
        with open("data/lematbulk_composition_distribution.json", "r") as file:
            reference_data_dist = json.load(file)
    gen_vals = np.array(list(generated_crystals_dist.values()))
    ref_vals = np.array(list(reference_data_dist.values()))

    return jensenshannon(gen_vals, ref_vals)


def gaussian_kernel(x, y, sigma=1.0):
    pairwise_dists = cdist(x, y, "sqeuclidean")
    return np.exp(-pairwise_dists / (2 * sigma**2))


def compute_mmd(reference_data, generated_crystals, crystal_param, sigma=1.0):
    generated_crystals_dist = np.atleast_2d(
        generated_crystals[crystal_param].to_numpy()
    ).T
    reference_data_dist = np.atleast_2d(reference_data[crystal_param].to_numpy()).T

    k_xx = gaussian_kernel(generated_crystals_dist, generated_crystals_dist, sigma)
    k_yy = gaussian_kernel(reference_data_dist, reference_data_dist, sigma)
    k_xy = gaussian_kernel(generated_crystals_dist, reference_data_dist, sigma)

    mmd = np.mean(k_xx) + np.mean(k_yy) - 2 * np.mean(k_xy)
    return mmd


def compute_frechetdist(reference_data, generated_crystals, crystal_param):
    generated_crystals_dist = generate_probabilities(
        generated_crystals, metric=crystal_param, return_2d_array=True
    )
    reference_data_dist = generate_probabilities(
        reference_data, metric=crystal_param, return_2d_array=True
    )

    distance = frdist(reference_data_dist, generated_crystals_dist)
    return distance
