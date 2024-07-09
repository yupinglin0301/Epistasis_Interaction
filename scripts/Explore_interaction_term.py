from pathlib import Path
from sklearn.model_selection import ParameterGrid
from copy import deepcopy
from RF_dataset_model import RF_DataModel, RIT_DataModel
from sklearn.utils import resample


import numpy as np 
import pandas as pd
import joblib
import itertools
from scipy import stats


"""
utilities for 02_Explore_interaction_term
"""


def get_dataset(feature_data, phen_data, data_split_indexes, label):
    """
    Retrieve a subset of the dataset based on specified label for data splitting.
    """
    
    indexes = data_split_indexes.loc[data_split_indexes["label"] == label]
    indexes = indexes["labeled_data_index"]
    
    data = feature_data.loc[indexes]
    phenotype = phen_data.loc[indexes]
    
    return data, phenotype


def write_configure_run_IRF(logger, 
                            configure_file, 
                            interaction_pvalue_rst_file, 
                            interaction_rst_file,
                            phen_file,
                            predictor_file):
    """
    Save model hyperparameters/metadata to output directory.
    """

   
    logger.info('+++++++++++ File INFORMATION +++++++++++')
    logger.info("Parameter file dir: {}".format(configure_file))
    logger.info("Interaction pvalue result dir: {}".format(interaction_pvalue_rst_file))
    logger.info("Interaction result dir: {}".format(interaction_rst_file))
    logger.info("Phenotype file dir: {}".format(phen_file))
    logger.info("Predictor file dir: {}".format(predictor_file))


def write_configure_permutation(logger, 
                                configure_file, 
                                interaction_rst_file,
                                phen_file,
                                predictor_file):
    """
    Save model hyperparameters/metadata to output directory.
    """

   
    logger.info('+++++++++++ File INFORMATION +++++++++++')
    logger.info("Parameter file dir: {}".format(configure_file))
    logger.info("Interaction result dir: {}".format(interaction_rst_file))
    logger.info("Phenotype file dir: {}".format(phen_file))
    logger.info("Predictor file dir: {}".format(predictor_file))
            

def group_shuffle_split(X, y, groups, seed, n_splits=1 , test_size=0.2):
    """
    Split the data into train and test sets
    """
        
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)  
    split = gss.split(X, y, groups=groups)
    train_ids, test_ids = next(split)
    
    
        
    X_train, X_test = X[train_ids], X[test_ids]
    y_train, y_test = y[train_ids], y[test_ids]
        
    return X_train, X_test, y_train, y_test, train_ids, test_ids

def get_rit_counts(b, all_rit_bootstrap_output, column_name):
    """
    Get each bootstrapping  interaction term
    """  
    from RF_dataset_model import RF_DataModel, RIT_DataModel 
    rit_counts = RIT_DataModel().rit_interactions(all_rit_bootstrap_output['rf_bootstrap{}'.format(b)], column_name)
    
    
    return list(rit_counts)

def rit_interactions(tree_data, column_name):
    """
    Get interaction term for each bootstrapping
    """
    intersected_values = tree_data['rit_intersected_values']
    # loop through all found interactions
    for value_list in intersected_values:
        values = [key for key, value in column_name.items() if value in value_list]
        if len(values) > 1:
            intersection_str = "_".join(map(str, values))
            return intersection_str


class OOB_Search(object):
    def __init__(self, 
                 estimator, 
                 param_grid,
                 n_jobs=-1):
        """
        Initializes the OOB_Search class.

       
        :param estimator (object): The base estimator to be used.
        :param param_grid (dict or list of dicts): The parameter grid to search over.
        :param n_jobs (int, optional): The number of jobs to run in parallel. Defaults to -1.
        """
        self.n_jobs = n_jobs
        self.estimator = estimator
        self.param_grid = param_grid

    def fit(self, 
            X_train, 
            y_train):
        """
        Fits the model with the given training data using the parameter grid search.

        :param X_train (array-like): The input features for training.
        :param y_train (array-like): The target values for training.

        :return self (object): Returns self.
        """
        self.params_iterable = list(ParameterGrid(self.param_grid))
        parallel = joblib.Parallel(self.n_jobs)

        output = parallel(
            joblib.delayed(self.fit_and_score)(deepcopy(self.estimator), X_train, y_train, parameters)
            for parameters in self.params_iterable)

        self.output_array = np.array(output)
        

        return self

    def fit_and_score(self, 
                      estimator, 
                      X_train, 
                      y_train,
                      parameters):
        """
        Fits the model and calculates the out-of-bag (OOB) error score.

        :param estimator (object): The estimator object.
        :param X_train (array-like): The input features for training.
        :param y_train (array-like): The target values for training.
        :param parameters (dict): The hyperparameters to use for fitting the model.

        :return oob_error (float): The calculated out-of-bag error score.
        """
        
        
        # Initialize dictionary of rf weights
        all_rf_weights = {}
        initial_weights = None
        # Loop through number of iteration
        for k in range(int(parameters['K'])):
           
            if k == 0:
                # Initially feature weights are None
                feature_importances = initial_weights

                # Update the dictionary of all our RF weights
                all_rf_weights["rf_weight{}".format(k)] = feature_importances

                # fit the model
                estimator.fit(X_train=X_train,
                              Y_train=y_train,
                              feature_weight=None)

                # Update feature weights using the
                # new feature importance score
                feature_importances = getattr(estimator,"model").feature_importances_

                # Load the weights for the next iteration
                all_rf_weights["rf_weight{}".format(k + 1)] = feature_importances

            else:

                # fit weighted RF
                # Use the weights from the previous iteration
                estimator.fit(
                    X_train=X_train,
                    Y_train=y_train,
                    feature_weight=all_rf_weights["rf_weight{}".format(k)])

                # Update feature weights using the
                # new feature importance score
                feature_importances = getattr(estimator, "model").feature_importances_

                # Load the weights for the next iteration
                all_rf_weights["rf_weight{}".format(k + 1)] = feature_importances
        
     
        oob_error = 1 - estimator.model.oob_score_
        return oob_error, all_rf_weights

    @staticmethod
    def extract_oob_result(output_array, params_iterable):
        """
        Extracts the out-of-bag (OOB) results from an output array and a params iterable.
        
        :param output_array (list): List of output values.
        :param params_iterable (list): List of parameter values.

        :return  A tuple containing the list of RF weights and a DataFrame with OOB error scores and parameters.
        """
        # Extract OOB error scores from output array
        oob_error_score = [i[0] for i in output_array]

        # Find the index of the best OOB error score
        best_index = np.argmin(oob_error_score)
        best_param_ = params_iterable[best_index]

        # Create a DataFrame with OOB error scores and parameters
        cv_results = pd.DataFrame(oob_error_score, columns=['OOB_Error_Score'])
        df_params = pd.DataFrame(params_iterable)
        cv_results = pd.concat([cv_results, df_params], axis=1)
        cv_results["params"] = params_iterable
        cv_results = (cv_results.
                      sort_values(['OOB_Error_Score'], ascending=True).
                      reset_index(drop=True))
        
        if len(oob_error_score) ==1:
            # Extract RF weights for the best parameter value when only have one parameter
            all_rf_weights = [j[1]["rf_weight{}".format(best_param_['K'])] for i, j in enumerate(output_array)]
        else:
            # Extract RF weights for the best parameter value
            all_rf_weights = [j[1]["rf_weight{}".format(best_param_['K'])] for i, j in enumerate(output_array) if(i+1)==best_param_['K']]

        return all_rf_weights, cv_results



def run_RIT(rf_bootstrap,
            X_train,
            y_train,
            X_test,
            y_test,
            n_samples,
            all_rf_weights,
            **parameters):
    
  
    
    X_train_rsmpl, y_rsmpl = resample(X_train, 
                                      y_train, 
                                      n_samples=n_samples)
    
    # Set up the weighted random forest
    # Using the weight from the (K-1)th iteration
    rf_bootstrap.fit(
            X_train=X_train_rsmpl,
            Y_train=y_rsmpl,
            feature_weight=all_rf_weights
    )  
    
    # All RF tree data
    all_rf_tree_data = RF_DataModel().get_rf_tree_data(
            rf=rf_bootstrap.model,
            X_train=X_train_rsmpl,
            X_test=X_test,
            y_test=y_test
    )

    # Run RIT on the interaction rule set
    all_rit_tree_data = RIT_DataModel().get_rit_tree_data(
            all_rf_tree_data=all_rf_tree_data,
            bin_class_type=y_test,
            M=parameters['n_intersection_tree'],  # number of RIT 
            max_depth=parameters['max_depth'], # Tree depth for RIT
            noisy_split=False,
            num_splits=parameters['num_splits']  # number of children to add
    ) 
 
    return all_rit_tree_data


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