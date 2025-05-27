import numpy as np 
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

def generate_probabilities(df, show_hist = True):
    # create an empty list of space groups/crystal systems/compositions and fill in proportions/counts
    # depending on the application (as some samples will have zero of space group 1 etc) 
    
    probs = np.asarray(df.value_counts("SpaceGroup")/len(df))
    indicies = np.asarray(df.value_counts("SpaceGroup").index)
    strut_list = np.concatenate(([indicies], [probs]), axis = 0).T
    strut_list = strut_list[strut_list[:, 0].argsort()]
    # strut_list = np.flip(strut_list)
    if show_hist:     
        plt.bar(strut_list.T[0], strut_list.T[1])
        plt.show()

    return strut_list # 2d array with col1 = crystal descriptor and col2 = probability

def compute_shannon_entropy(probability_vals):
    H = 0
    for i in range(len(probability_vals)): 
        H += probability_vals[i]*np.log(probability_vals[i])
    H = -H
    return H


def compute_jensen_shannon_distance(reference_data, generated_crystals, crystal_param, subcategory = None):
    """
    reference_data - letmatbulk dataframe 
    generated_crystals - dataframe of generated crystals
    crystal_param - CrystalSystem, SpaceGroup, or LatticeConstant
    subcategory - if looking at SpaceGroup, can specify only cubic crystal systems, for example. Defaults to None (and would therefore look at the distribution across all the space groups) 
    """
    # TODO this method will not work for lattice constants as it is currently written 
    
    generated_crystals_dist = generated_crystals.value_counts(crystal_param)/len(generated_crystals) # TODO - order is not always going to be the same, since its sorted from higest to lowest for value counts. Need to standardize this 
    generated_crystals_dist.sort_index(ascending=True, inplace = True)
    reference_data_dist = reference_data.value_counts(crystal_param)/len(reference_data)
    reference_data_dist.sort_index(ascending=True, inplace = True)

    return jensenshannon(generated_crystals_dist, reference_data_dist)