from pathlib import Path
from sklearn.model_selection import ParameterGrid
from copy import deepcopy
from math import ceil
from RF_dataset_model import RF_DataModel, RIT_DataModel
from sklearn.utils import resample



import pandas as pd
import model 
import numpy as np
import dataset_model as dm
import argparse
import utils
import sys
import yaml
import joblib
import time
import datetime


"""
Functions for conducting iterative random forest model

Usage:
    
    python run_IRF.py \
      --work_dir "/exeh_4/yuping/Epistasis_Interaction/02_Select_Parameter_Model" \
      --weight_tissue "Brain_Amygdala" \
      --phen_name  "CWR_Total" > /exeh_4/yuping/Epistasis_Interaction/02_Select_Parameter_Model/Log/nohup.txt &
          
Output:
out-of bag (OOB) error score for each iteration and interaction term.
"""

class OOB_ParamGridSearch(object):
    def __init__(self, 
                 estimator, 
                 param_grid,
                 n_jobs=-1):
        """
        Initializes the OOB_ParamGridSearch class.

       
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
            feature_weight=all_rf_weights[0]
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


def process_args():
    """
    Parse and format and command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weight_tissue',
        action="store",
        help="Data directory where the phenotype and genotype matrix are stored."
    )
    
    parser.add_argument(
        '--phen_name',
        action="store",
        help="Data directory where the phenotype and genotype matrix are stored."
    )
    
    parser.add_argument(
        '--work_dir',
        action="store",
        help="Data directory where the phenotype and genotype matrix are stored."
    )
    
    parser.add_argument(
        '--configure_file',
        action="store",
        help="Data directory where the phenotype and genotype matrix are stored."
    )

    
    args = parser.parse_args()
    return(args)


if __name__ == '__main__':
    startTime=time.time()
    
    # process command line arguments
    input_arguments = process_args()
    
    timestamp = datetime.datetime.now().today().isoformat()
    # set up logging
    logger = utils.logging_config(input_arguments.weight_tissue + input_arguments.phen_name + "run_IRF", timestamp)
     # set up directory
    repo_root = Path(__file__).resolve().parent.parent
    # set up working directory
    work_dir = Path.joinpath(repo_root, "02_Select_Parameter_Model")
    # set up save directory
    save_dir = Path.joinpath(work_dir, "results")
   
    # loading configure file
    configure_file = Path(input_arguments.work_dir, "model_configure", input_arguments.configure_file)
    try:
        with open(configure_file) as infile:
            load_configure = yaml.safe_load(infile)
    except Exception:
            sys.stderr.write("Please specify valid yaml file.")
            sys.exit(1)
            
  
    logger.info("Check if all files and directories exist ... ")
   
    transformed_data_dir = Path(load_configure['dataset']['preprocess_data_dir']) / input_arguments.weight_tissue 
    if not utils.check_exist_directories([save_dir, transformed_data_dir]):
        raise Exception("See output above. Problems with specified directories")
    else:
        # Create experiment directory
        experiment_dir = save_dir / input_arguments.weight_tissue
        experiment_dir.mkdir(parents=True,  exist_ok=True)
        
        timestamp = datetime.datetime.now().today().isoformat()
        oob_score_filename = utils.construct_filename(experiment_dir, "result", ".csv", timestamp, input_arguments.weight_tissue, "oob_score")
        interaction_result_filename = utils.construct_filename(experiment_dir, "result",".csv",  timestamp, input_arguments.weight_tissue, "interaction")
      
        # Check output file is exist or not
        if utils.check_exist_files([oob_score_filename, interaction_result_filename]):
            raise Exception("See output above. Problems with specified directories")
        
    
    logger.info("Prepareing feature ... ") 
    GTex_raw_dataset = dm.GTEX_raw_Dataset.from_config(config_file=load_configure, 
                                                       weight_tissue=input_arguments.weight_tissue)
    # generate gene expression dataset
    X_raw_df = GTex_raw_dataset.all_gen_df
    
    # generate phenotype label
    y_tran_df_dir = transformed_data_dir / (input_arguments.phen_name + "_imputed.csv")
    y_tran_df = GTex_raw_dataset.load(y_tran_df_dir)
    
    # generate group label
    group_raw_df = pd.read_csv(Path(load_configure['dataset']['group_dir']))
 

    
    # Set the random state for reproducibility
    np.random.seed(load_configure['default_seed'])
    
    logger.info("Train-Test Splitting ... ")
    X_train_raw_df, X_test_raw_df, y_train_tran_df, y_test_tran_df, train_index, test_index = GTex_raw_dataset.group_shuffle_split(X_raw_df.values, 
                                                                                                                                   y_tran_df.values, 
                                                                                                                                   seed=load_configure['default_seed'],
                                                                                                                                   test_size=0.2,
                                                                                                                                   groups=group_raw_df)     
    
    
    # hyperparameter to for RF
    model_params = {
        'n_estimators': load_configure['model_params']['n_estimators'][0],
        'max_features': load_configure['model_params']['max_features'][0],
        'oob_score': load_configure['model_params']['oob_score'],  
    }
    
    # hyperparameter to for RIT
    rit_params = {
        'n_intersection_tree' :  load_configure['model_params']['n_intersection_tree'][0],
        'max_depth' : load_configure['model_params']['max_depth'][0],
        'num_splits' : load_configure['model_params']['num_splits'][0],
        'n_bootstrapped': load_configure['model_params']['n_bootstrapped'][0],
        'propn_n_samples' : load_configure['model_params']['propn_n_samples'][0]
    }
    
    # number of iteration to test 
    iteration_grid = {
        'K': [1,2,3]
    }
     
    # Create the model
    train_model = model.IterativeRFRegression(rseed=load_configure['default_seed'], **model_params)
    
    
    logger.info("Find the random forest from the iteration with lowest OOB error score ... ")
    oob_gridsearch = OOB_ParamGridSearch(n_jobs=1,
                                         estimator=train_model,
                                         param_grid=iteration_grid)

    oob_gridsearch.fit(X_train=X_train_raw_df, y_train=y_train_tran_df.flatten())
    # get iteration of feature weights and cv_results
    all_rf_weights, cv_results = oob_gridsearch.extract_oob_result(oob_gridsearch.output_array, 
                                                                   oob_gridsearch.params_iterable)

    # output oob_error_score result
    cv_results.to_csv(oob_score_filename, sep="\t", index=False)
    
    
    
    logger.info("Run Random Intersection Tree ...")
    # Convert the bootstrap resampling proportion to the number
    # of rows to resample from the training data
    n_samples = ceil(rit_params['propn_n_samples']* X_train_raw_df.shape[0])

    all_rit_bootstrap_output = {}
    output = joblib.Parallel(n_jobs=2)(
        joblib.delayed(run_RIT)(deepcopy(train_model), 
                                X_train_raw_df, 
                                y_train_tran_df.flatten(), 
                                X_test_raw_df, 
                                y_test_tran_df.flatten(), 
                                n_samples, 
                                all_rf_weights, 
                                **rit_params)
        for _ in range(0, rit_params['n_bootstrapped'])) 

    for i in (range(0, rit_params['n_bootstrapped'])):
            all_rit_bootstrap_output['rf_bootstrap{}'.format(i)] = output[i]

    
    logger.info("Compute stability score for each interaction term ...")
    stability_score = RIT_DataModel().get_stability_score(all_rit_bootstrap_output=all_rit_bootstrap_output)  
    
    # output oob_error_score result
    stability_score_df = pd.DataFrame(stability_score.items(), columns=["Pattern", "Value"])
    stability_score_df.to_csv(interaction_result_filename, sep="\t", index=False)
    
    
    totalTime = time.time() - startTime
    logger.info("Computing done = {} min.".format(float(totalTime)/60))  