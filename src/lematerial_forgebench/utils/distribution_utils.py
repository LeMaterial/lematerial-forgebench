import numpy as np 
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
from pymatgen.core import Composition, Element, Structure
from scipy.spatial.distance import cdist


def one_hot_encode_composition(composition):
    one_hot_counts = np.zeros(118)
    one_hot_bool = np.zeros(118)
    for element in composition.elements:
        one_hot_bool[int(Element(element).number) - 1] = 1
        one_hot_counts[int(Element(element).number) - 1] = composition.as_dict()[element.as_dict()['element']]
    return [one_hot_counts, one_hot_bool]


def generate_probabilities(df, metric, metric_type = np.int64, return_2d_array = False):
    # create an empty list of space groups/crystal systems/compositions and fill in proportions/counts
    # depending on the application (as some samples will have zero of space group 1 etc)
    if metric_type == np.int64:
        prob_dict = {}
        for i in range(0, 230):
            prob_dict[str(i+1)] = 0
        
        probs = np.asarray(df.value_counts(metric) / len(df))
        indicies = np.asarray(df.value_counts(metric).index)
        strut_list = np.concatenate(([indicies], [probs]), axis=0).T
        strut_list = strut_list[strut_list[:, 0].argsort()]
        if return_2d_array:
            return strut_list

        for row in strut_list: 
            prob_dict[str(int(row[0]))] = row[1]

    if metric_type == np.ndarray:
        prob_dict = {}
        for i in range(0, 118):
            prob_dict[str(i+1)] = 0
        
        one_hots = np.zeros(118)
        for i in range(0, len(df)):
            one_hots += df.iloc[i][metric]

        one_hots = one_hots/sum(one_hots)

        for i in range(0, 118):
            prob_dict[str(i+1)] = one_hots[i]

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
    reference_data, generated_crystals, crystal_param, subcategory=None
):
    """
    reference_data - letmatbulk dataframe
    generated_crystals - dataframe of generated crystals
    crystal_param - CrystalSystem, SpaceGroup, or LatticeConstant
    subcategory - if looking at SpaceGroup, can specify only cubic crystal systems, for example. Defaults to None (and would therefore look at the distribution across all the space groups)
    """
    # TODO this method will not work for lattice constants as it is currently written

    generated_crystals_dist = generate_probabilities(generated_crystals, metric=crystal_param)
    reference_data_dist = generate_probabilities(reference_data, metric=crystal_param)

    gen_vals = np.array(list(generated_crystals_dist.values()))
    ref_vals = np.array(list(reference_data_dist.values()))

    return jensenshannon(gen_vals, ref_vals)

def gaussian_kernel(x, y, sigma=1.0):
    pairwise_dists = cdist(x, y, 'sqeuclidean')
    return np.exp(-pairwise_dists / (2 * sigma ** 2))

def compute_mmd(reference_data, generated_crystals, crystal_param, sigma=1.0):


    generated_crystals_dist = generate_probabilities(generated_crystals, metric=crystal_param, return_2d_array=True)
    reference_data_dist = generate_probabilities(reference_data, metric=crystal_param, return_2d_array=True)

    k_xx = gaussian_kernel(generated_crystals_dist, generated_crystals_dist, sigma)
    k_yy = gaussian_kernel(reference_data_dist, reference_data_dist, sigma)
    k_xy = gaussian_kernel(generated_crystals_dist, reference_data_dist, sigma)
    
    mmd = np.mean(k_xx) + np.mean(k_yy) - 2 * np.mean(k_xy)
    return mmd