from sklearn.model_selection import ParameterGrid
from copy import deepcopy

import numpy as np 
import joblib
import itertools


"""
utilities for 02_Explore_interaction_term
"""


def write_configure(logger, configure_file):
    """
    This function logs the directory of the configuration file and saves the model hyperparameters/metadata 
    to the output directory specified by the configure_file parameter.

    :param logger: Logger object for logging information.
    :param configure_file: Path to the configuration file containing model hyperparameters/metadata.
    """

    logger.info('+++++++++++ File INFORMATION +++++++++++')
    logger.info("Parameter file dir: {}".format(configure_file))


    

def FeatureEncoder(np_genotype_rsid, np_genotype, int_dim):
    """
    Implementation of the two-element combinatorial encoding.

    :param np_genotype_rsid (ndarray): 1D array containing rsid of genotype data with `str` type
    :param np_genotype (ndarray): 2D array containing genotype data with `int8` type
    :param int_dim (int): The dimension of a variant (default: 3. AA, AB and BB)

    :return: list_interaction_rsid (ndarray): 1D array containing rsid of genotype data with `str` type
    :return: np_interaction (ndarray): 2D array containing genotype data with `int8` type
    """
    
    np_interaction = np_genotype
    list_interaction_rsid = np_genotype_rsid
    list_combs = list(itertools.combinations(range(int(np_interaction.shape[1]/int_dim)), 2))[0]

    

    np_this_interaction = np.zeros([np_genotype.shape[0], int_dim**2], dtype='int8')
    list_this_interaction_id = []
    for idx_x in range(int_dim):
            for idx_y in range(int_dim):
                np_this_interaction_term = (np_genotype[:, list_combs[0] * int_dim + idx_x] * np_genotype[:, list_combs[1] * int_dim + idx_y]).astype(np.int8)
                if not(np.array_equal(np_this_interaction_term, np_genotype[:, list_combs[0] * int_dim + idx_x])) and not(np.array_equal(np_this_interaction_term, np_genotype[:, list_combs[1] * int_dim + idx_y])):
                        np_this_interaction[:, idx_x * int_dim + idx_y] = np_this_interaction_term
                list_this_interaction_id.append(np_genotype_rsid[list_combs[0] * int_dim + idx_x] + "*" + np_genotype_rsid[list_combs[1] * int_dim + idx_y])
            
    
    np_this_interaction_id = np.array(list_this_interaction_id)
    np_interaction_append = np.empty((np_interaction.shape[0], np_interaction.shape[1] + np_this_interaction.shape[1]), dtype='int')
    np_interaction_append[:,:-(np_this_interaction.shape[1])] = np_interaction
    np_interaction_append[:,-(np_this_interaction.shape[1]):] = np_this_interaction
    np_interaction = np_interaction_append
    list_interaction_rsid.extend(list(np_this_interaction_id))
        
    return list_interaction_rsid, np_interaction 


def filter_leaves_classifier(dtree_data,
                             bin_class_type):
        
        # following https://github.com/sumbose/iRF/blob/master/R/gRIT.R to convert leaf nodes in regression to binary classes.
        

        # Filter based on the specific value of the leaf node classes
        leaf_node_classes = dtree_data['all_leaf_node_classes']
        # perform the filtering and return list
        if bin_class_type is not None:      

            # unique feature paths from root to leaf node
            unique_feature_paths = [
                    i for i, j in zip(dtree_data['all_uniq_leaf_paths_features'],
                                  leaf_node_classes) if j == bin_class_type
            ]

            # total number of training samples ending up at each node
            tot_leaf_node_values = [
                i for i, j in zip(dtree_data['tot_leaf_node_values'],
                              leaf_node_classes) if j == bin_class_type
            ]
            
            all_filtered_output = {
                "Unique_feature_paths": unique_feature_paths,
                "tot_leaf_node_values": tot_leaf_node_values
            }
        
        else:
            all_filtered_output = {
                "Unique_feature_paths": list(dtree_data['all_uniq_leaf_paths_features']),
                "tot_leaf_node_values": list(dtree_data['tot_leaf_node_values'])
            }
        
        return all_filtered_output


def generate_all_samples(all_rf_tree_data, 
                         bin_class_type=1):
    
    n_estimators = all_rf_tree_data['rf_obj'].n_estimators

    all_paths = []
    for dtree in range(n_estimators):
        filtered = filter_leaves_classifier(
            dtree_data=all_rf_tree_data['dtree{}'.format(dtree)],
            bin_class_type=bin_class_type)
        all_paths.extend(filtered['Unique_feature_paths'])
    return all_paths
