from sklearn.model_selection import ParameterGrid
from copy import deepcopy
from sklearn.tree import _tree
from scipy import stats

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


def get_rss(y_np, fixedEff_np):
    """
    compute residual sum of squares of H0: marker has no effect on phenotype
    
    :param y: phenotype vector
    :param fixedEff: vector or matrix of fixed effects
    
    :return: residual sum of squares
    """
    
    # Compute the transpose of fixedEff
    fixedT = fixedEff_np.T   
    # Compute beta, the coefficients for the fixed effects
    beta = np.linalg.lstsq(fixedT @ fixedEff_np, fixedT @ y_np, rcond=None)[0]
    # Compute the difference between the actual y and the predicted values
    dif = y_np - fixedEff_np @ beta
    # Compute the residual sum of squares
    rss = dif.T @ dif
    
    return rss

def get_f_score(rss0, rss1, n, freedom_deg):
    """
    compute test statistics in batches
    
    :param rss0: residual sum of squares of H0: marker has no effect on phenotype
    :param rss1: residual sum of squares of H1: marker has effect on phenotype
    :param n: number of samples
    :param freedom_deg: degrees of freedom
    
    :return: F1 score
    """
    return (n-freedom_deg)*(rss0-rss1)/rss1

def get_p_value(f1, n: int, freedom_deg: int):
    """
    compute p-value using survival function of f distribution
    :param f1: F1 score
    :param n: number of samples
    :param freedom_deg: degrees of freedom
    :return: p-value
    """
    return stats.f.sf(f1, 1, n-freedom_deg)



class RF_DataModel(object):

    def __init__(self):
        """
        Initialize  prediction model/data
        """
        pass


    def all_tree_paths(self, dtree, node_id=0, depth=0):
        """
        Get all the individual tree paths from root node to the leaves
        for a decision tree classifier object
        """

        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

        left_child = dtree.tree_.children_left[node_id]
        right_child = dtree.tree_.children_right[node_id]

        if left_child != _tree.TREE_LEAF:

            left_paths = [
                np.append(node_id, path)
                for path in self.all_tree_paths(dtree, left_child, depth=depth + 1)
            ]

            right_paths = [
                np.append(node_id, path)
                for path in self.all_tree_paths(dtree, right_child, depth=depth + 1)
            ]

            paths = left_paths + right_paths

        else:
            paths = [[node_id]]

        return paths


    def get_rf_tree_data(self, rf, X_train, X_test, y_test, predictor="regress"):
        """
        Get the entire fitted random forest and its decision tree data
        as a convenient dictionary format
        """


        # random forest feature importances i.e. next iRF iteration weights
        feature_importances = rf.feature_importances_

        # standard deviation of the feature importances
        feature_importances_std = np.std(
            [dtree.feature_importances_ for dtree in rf.estimators_], axis=0)
        feature_importances_rank_idx = np.argsort(feature_importances)[::-1]


        # Create a dictionary with all random forest metrics
        # This currently includes the entire random forest fitted object
        all_rf_tree_outputs = {
            "rf_obj": rf,
            "get_params": rf.get_params,
            "feature_importances": feature_importances,
            "feature_importances_std": feature_importances_std,
            "feature_importances_rank_idx": feature_importances_rank_idx
        }


        for idx, dtree in enumerate(rf.estimators_):
            dtree_out = self.get_tree_data(
                X_train=X_train,
                X_test=X_test,
                y_test=y_test,
                dtree=dtree,
                node_id=0,
                predictor=predictor
            )

            # Append output to our combined random forest outputs dict
            all_rf_tree_outputs["dtree{}".format(idx)] = dtree_out

        return all_rf_tree_outputs

    def get_tree_data(self, X_train, X_test, y_test, dtree, predictor, node_id=0):

        """
        This returns all of the required summary results from an
        individual decision tree
        """

        value = dtree.tree_.value
        
        n_node_samples = dtree.tree_.n_node_samples

        # Get the total number of features in the training data
        tot_num_features = X_train.shape[1]

        # Get indices for all the features used - 0 indexed and ranging
        # to the total number of possible features in the training data
        all_features_idx = np.array(range(tot_num_features), dtype='int64')

        # Get the raw node feature indices from the decision tree classifier
        node_features_raw_idx = dtree.tree_.feature

        # Get the refined non-negative feature indices for each node
        node_features_idx = all_features_idx[np.array(node_features_raw_idx)]


        # Get all of the paths used in the tree
        all_leaf_node_paths = self.all_tree_paths(
                dtree=dtree, node_id=node_id
        )

        # Get all of the features used along the leaf node paths
        all_leaf_paths_features = [
            node_features_idx[path[:-1]]
            for path in all_leaf_node_paths
        ]

        # Get the unique list of features along a path
        all_uniq_leaf_paths_features = [
                np.unique(feature_path)
                for feature_path in all_leaf_paths_features
        ]

        # Get list of leaf nodes
        # In all paths it is the final node value
        all_leaf_nodes = [
            path[-1]
            for path in all_leaf_node_paths
        ]

        # Final predicted values in each class at each leaf node
        all_leaf_node_values = [
            value[node_id].astype(int)
            for node_id in all_leaf_nodes
        ]


        # Get the total number of training samples used in each leaf node
        tot_leaf_node_values = [n_node_samples[node_id].astype(int)
                                for node_id in all_leaf_nodes]

        # Get all feature index
        all_features_idx = np.array(range(tot_num_features), dtype='int64')
        
        if predictor == "regress":
            # Predicted Classes
            all_leaf_node_classes = [
                np.unique(y_test)[np.argmax(value)] 
                for value in all_leaf_node_values
            ]
        else:
            all_leaf_node_classes = [all_features_idx[np.argmax(value)] for value in all_leaf_node_values]
            

        # Dictionary of all tree values
        tree_data = {
            "node_features_idx": node_features_idx,
            "all_leaf_paths_features": all_leaf_paths_features,
            "all_uniq_leaf_paths_features": all_uniq_leaf_paths_features,
            "tot_leaf_node_values": tot_leaf_node_values,
            "all_leaf_node_classes": all_leaf_node_classes,
            "all_leaf_node_values": all_leaf_node_values
        }

        return tree_data


class IterativeRF:
    def __init__(self, estimator, param_grid, n_jobs=-1):
        """
        Initializes the IterativeRF class.

        :param estimator (object): The base estimator to be used.
        :param param_grid (dict or list of dicts): The parameter grid to search over.
        :param n_jobs (int, optional): The number of jobs to run in parallel. Defaults to -1.
        """
        self.n_jobs = n_jobs
        self.estimator = estimator
        self.param_grid = param_grid

    def fit(self, X, y):
        """
        Fits the model with the given training data using the parameter grid search.

        :param X (array-like): The input features for training.
        :param y (array-like): The target values for training.

        :return self (object): Returns self.
        """
        self.params_iterable = list(ParameterGrid(self.param_grid))
        parallel = joblib.Parallel(self.n_jobs)

        output = parallel(
            joblib.delayed(self.fit_and_score)(deepcopy(self.estimator), X, y, parameters)
            for parameters in self.params_iterable
        )

        self.output_array = np.array(output)
        return self

    def fit_and_score(self, estimator, X, y, parameters):
        """
        Fits the model and calculates the out-of-bag (OOB) error score.

        :param estimator (object): The estimator object.
        :param X (array-like): The input features for training.
        :param y (array-like): The target values for training.
        :param parameters (dict): The hyperparameters to use for fitting the model.

        :return oob_error (float): The calculated out-of-bag error score.
        """
        all_rf_weights = {}
        initial_weights = None

        for k in range(1, int(parameters['K'])):
            if k == 1:
                feature_importances = initial_weights
                all_rf_weights[f"rf_weight{k}"] = feature_importances

                estimator.fit(X, y, feature_weight=None)
                feature_importances = estimator.feature_importances_
                all_rf_weights[f"rf_weight{k + 1}"] = feature_importances
            else:
                estimator.fit(X, y, feature_weight=all_rf_weights[f"rf_weight{k}"])
                feature_importances = estimator.feature_importances_
                all_rf_weights[f"rf_weight{k + 1}"] = feature_importances

       
        return all_rf_weights


def FeatureEncoder(np_genotype_rsid, np_genotype):
    np_interaction = np_genotype
    list_interaction_rsid = np_genotype_rsid
    # list the postion of feature to combine
    list_combs = list(itertools.combinations(range(int(np_interaction.shape[1]/1)), 2))[0]

    # Create array to store interaction term
    np_this_interaction = np.zeros([np_genotype.shape[0], 1**2])
    np_this_interaction_term = (np_interaction[:, list_combs[0]] * np_interaction[:, list_combs[1]])
    np_this_interaction[:, 0] = np_this_interaction_term
    # Store the name of interaction term
    list_this_interaction_id = []
    list_this_interaction_id.append(list_interaction_rsid[list_combs[0]] + "*" + list_interaction_rsid[list_combs[1]])
            
    np_interaction_append = np.empty((np_interaction.shape[0], np_interaction.shape[1] + np_this_interaction.shape[1]))
    # retain original of feature
    np_interaction_append[:,:-(np_this_interaction.shape[1])] = np_interaction
    # intsert new interaction term
    np_interaction_append[:,-(np_this_interaction.shape[1]):] = np_this_interaction
    np_interaction = np_interaction_append
    list_interaction_rsid.extend(list_this_interaction_id)

    return list_interaction_rsid, np_interaction