from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from copy import deepcopy
from sklearn.pipeline import Pipeline



import pandas as pd
import model as im
import numpy as np
import dataset_model as dm
import argparse
import utils
import sys
import yaml
import joblib
import IRF_stability_score as IRF
import time

"""
Functions for computing line with interaction terms and LR only with covariates.

Usage:
    
    python run_oob_pipline.py \
      --work_dir "/exeh_4/yuping/Epistasis_Interaction/02_Select_Parameter_Model" \
      --weight_tissue "Brain_Amygdala" \
      --phen_name  "CWR_Total" \
      --FE_pipeline "/exeh_4/yuping/Epistasis_Interaction/01_Preprocessing/results"
      
          
      
Output:
Feature engineering pipeline for specific imputed brain tissue and environmental factors

"""

def transform(X, stability_term, threshold, num_term):
    """
    Perform feature transformation based on stability terms and select top interactions.

    
    :param X : Input array of shape (n_samples, n_features) containing the data.
    :param threshold : Cut-off to select interaction term.
    :param stability_term : Dictionary with stability terms as keys and their corresponding values.
    :param num_term:  Number of top interactions to select.

    :return: Array of shape (n_samples, num_term) containing the calculated interaction values for the top interactions with stability terms greater than 0.5.
    """
    
    sorted_stability_term = {k: v for k, v in sorted(stability_term.items(), key=lambda x: x[1], reverse=True)}

    interaction = {}
    count = 0
    for key, value in sorted_stability_term.items():
        if value > threshold and count <= num_term:
            cols = list(map(int, key.split('_')))
            values = np.prod(X[:, cols], axis=1)
            interaction[key] = values
            count += 1

    select_interaction = pd.DataFrame(interaction).to_numpy()

    return select_interaction

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
        '--FE_pipeline',
        action="store",
        help="Data directory where the phenotype and genotype matrix are stored."
    )
    
    args = parser.parse_args()
    return(args)


if __name__ == '__main__':
    
    # process command line arguments
    input_arguments = process_args()
    # set up logging
    logger = utils.logging_config(input_arguments.weight_tissue)
    # set up repo_directory
    repo_root = Path(__file__).resolve().parent.parent
    # set up working directory
    work_dir = Path.joinpath(repo_root, "02_Select_Parameter_Model")
    # set up savinf directory
    save_dir = Path.joinpath(work_dir, "results")
    
    # loading configure file
    configure_file = Path(input_arguments.work_dir, "config.yaml")
    try:
        with open(configure_file) as infile:
            load_configure = yaml.safe_load(infile)
    except Exception:
            sys.stderr.write("Please specify valid yaml file.")
            sys.exit(1)
       
    logger.info("Check if all files and directories exist ... ")
    
    save_dir = Path(input_arguments.work_dir) / "results"
    transformed_data_dir = Path(load_configure['dataset']['preprocess_data_dir']) / input_arguments.weight_tissue 
    if not utils.check_exist_directories([save_dir, transformed_data_dir]):
        raise Exception("See output above. Problems with specified directories")
    else:
        # Create experiment directory
        experiment_dir = save_dir / input_arguments.weight_tissue
        experiment_dir.mkdir(parents=True,  exist_ok=True)
        best_model_filename = utils.construct_filename(experiment_dir, "model", ".pkl", input_arguments.weight_tissue, "best_param")
        cv_result_filename = utils.construct_filename(experiment_dir, "result",".csv", input_arguments.weight_tissue, "cv")
        interaction_result_filename = utils.construct_filename(experiment_dir, "result",".csv", input_arguments.weight_tissue, "interaction")
        
        # Check output file is exist or not
        if utils.check_exist_files([best_model_filename, cv_result_filename, interaction_result_filename]):
            raise Exception("See output above. Problems with specified directories")
        
        
    GTex_raw_dataset = dm.GTEX_raw_Dataset.from_config(config_file=load_configure, 
                                                       weight_tissue=input_arguments.weight_tissue)
    # generate gene expression dataset
    X_raw_df = GTex_raw_dataset.all_gen_df
    
    # generate phenotype label
    y_tran_df_dir = transformed_data_dir / (input_arguments.phen_name + "_imputed.csv")
    y_tran_df = GTex_raw_dataset.load(y_tran_df_dir)
    
    # generate group label
    group_raw_df = pd.read_csv(Path(load_configure['dataset']['group_dir']))
 

    logger.info("Train-Test Splitting ... ")
    X_train_raw_df, X_test_raw_df, y_train_tran_df, y_test_tran_df, train_index, test_index = GTex_raw_dataset.group_shuffle_split(X_raw_df.values, 
                                                                                                                                   y_tran_df.values, 
                                                                                                                                   seed=load_configure['default_seed'],
                                                                                                                                   test_size=0.2,
                                                                                                                                   groups=group_raw_df)
    # generate train group label
    group_train_id = [group_raw_df["FID"].values[i] for i in train_index]
  
    
    
    logger.info("Training ...")
    start_training = time.time()
    
    supervised_params = {
         'predictor': load_configure['model_params']['name'],
         'n_estimators' :  load_configure['model_params']['n_estimators'],
         'max_features' : load_configure['model_params']['max_features']
    }
    
    maximize_score = True
    best_score = -1e6 if maximize_score else 1e6
    best_cv_param = None
    for cv_param in [4,6,8,10,20]:
        
        logger.info("Trying {} = {}".format("number interaction", cv_param))
        
        
        mean_score = 0
        
        # Create the model
        full_model = im.hiPRS()
        # Defining the number of folds for cross-validation
        k = 3
        # Creating the group k-fold cross-validation on the training set
        group_kf = GroupKFold(n_splits=k)
        # Iterating over the folds
        for train_index, val_index in group_kf.split(X_train_raw_df, y_train_tran_df, groups=group_train_id):
         
            X_train_fold, X_val_fold = X_train_raw_df[train_index], X_train_raw_df[val_index]
            y_train_fold, y_val_fold = y_train_tran_df[train_index].flatten(), y_train_tran_df[val_index].flatten()
            
            
            # Get interaction term
            stability_term = IRF.run_iRF(X_train=X_train_fold,
                                         X_test=X_val_fold,
                                         y_train=y_train_fold,
                                         y_test=y_test_tran_df,
                                         K=5, #number of iteration
                                         B=50, #number of bootstrap 
                                         random_state_classifier=load_configure['default_seed'],
                                         propn_n_samples=.2,  # The proportion of samples drawn for bootstrap
                                         bin_class_type=None,
                                         M=500, # number of intersection tree 
                                         max_depth=5, # intersection tree depth 
                                         noisy_split=False,
                                         num_splits=2, # the maximum number of children to be added.
                                         **supervised_params
                                        )
            
            df = pd.DataFrame(stability_term.items(), columns=["Pattern", "Value"])
            df.to_csv(interaction_result_filename, index=False)
            
            

    
    


    
  
    
    
  
    
    
        
    
    


