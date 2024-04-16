from pathlib import Path
from copy import deepcopy
from math import ceil
from select_parameter_utils import write_configure_run_IRF, group_shuffle_split, run_RIT, rit_interactions, OOB_Search


import pandas as pd
import model 
import numpy as np
import argparse
import utils
import sys
import yaml
import time
import datetime
import sqlite3
import joblib



"""
Functions for conducting iterative random forest model

Usage:
    
    nohup python run_IRF.py \
      --work_dir "/exeh_4/yuping/Epistasis_Interaction/02_Select_Parameter_Model" \
      --weight_tissue "Brain_Amygdala" \
      --phen_name  "CWR_Total" \
      --phen_df_name  "2024-03-27T11:03:04.174838_phenotype_residualed.db"   \
      --pred_df_name "2024-03-25T11:01:20.810692_predictor_feature.csv" \
      --configure_file IRF_RF_test.yaml > /exeh_4/yuping/Epistasis_Interaction/02_Select_Parameter_Model/Log/nohup.txt &
          
Output:
out-of bag (OOB) error score for each iteration and interaction term.
"""



def process_args():
    """
    Parse and format and command line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--weight_tissue',
        action="store",
        help="Specify the weight tissue."
    )

    parser.add_argument(
        '--phen_name',
        action="store",
        help="Specify the phen name."
    )

    parser.add_argument(
        '--work_dir',
        action="store",
        help="Specify the working directory."
    )

    parser.add_argument(
        '--configure_file',
        action="store",
        help="Specify the configuration file."
    )

    parser.add_argument(
        '--phen_df_name',
        action="store",
        help="Specify the phen df name."
    )

    parser.add_argument(
        '--pred_df_name',
        action="store",
        help="Specify the pred df name."
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
    
    
    logger.info("Parameter setting ... ")
    # hyperparameter to for RF
    model_params = {
        'n_estimators': load_configure['model_params']['n_estimators'][0],
        'max_features': load_configure['model_params']['max_features'][0],
        'oob_score': load_configure['model_params']['oob_score'],
        'max_depth': load_configure['model_params']['max_depth'][0]
    }
    
    # hyperparameter to for RIT
    rit_params = {
        'n_intersection_tree' :  load_configure['model_params']['n_intersection_tree'][0],
        'max_depth' : load_configure['model_params']['max_depth_rit'][0],
        'num_splits' : load_configure['model_params']['num_splits'][0],
        'n_bootstrapped': load_configure['model_params']['n_bootstrapped'][0],
        'propn_n_samples' : load_configure['model_params']['propn_n_samples'][0]
    }
    

    logger.info("Check if all files and directories exist ... ")
    if not utils.check_exist_directories([save_dir]):
        raise Exception("See output above. Problems with specified directories")
    else:
        # Create experiment directory
        experiment_dir = save_dir / input_arguments.weight_tissue
        experiment_dir.mkdir(parents=True,  exist_ok=True)
        
        oob_score_filename = utils.construct_filename(experiment_dir, "result", ".csv", timestamp, input_arguments.weight_tissue, input_arguments.phen_name, "oob_score","run_IRF")
        interaction_filename = utils.construct_filename(experiment_dir, "result",".csv",  timestamp, input_arguments.weight_tissue, input_arguments.phen_name, "interaction", "run_IRF")
        # Check output file is exist or not
        if utils.check_exist_files([oob_score_filename, interaction_filename]):
            raise Exception("See output above. Problems with specified directories")
    
   
    logger.info("Prepareing feature ... ") 
    # generate phenotype
    phen_dir =  Path(load_configure['dataset']['data_dir']) / input_arguments.phen_df_name
    # Connect to the SQLite database
    conn = sqlite3.connect(phen_dir)
    # Read data from the "Phenotype" table into a DataFrame
    query = "SELECT * FROM Phenotype"
    phen_df = pd.read_sql_query(query, conn, index_col=None)
    # generate predictor
    predictor_dir =  Path(load_configure['dataset']['data_dir']) / input_arguments.weight_tissue / input_arguments.pred_df_name
    except_column = ["MotherEducation", "FatherEducation"]
    pred_df = pd.read_csv(predictor_dir , sep="\t")
    pred_df = pred_df.drop(except_column, axis=1)
    # generate group label
    group_raw_df = pd.read_csv(Path(load_configure['dataset']['group_dir']))
    # specify the configure and result file dir
    write_configure_run_IRF(logger, 
                            configure_file=configure_file, 
                            oob_rst_file=oob_score_filename, 
                            interaction_rst_file=interaction_filename,
                            phen_file=phen_dir,
                            predictor_file=predictor_dir)

    
    # Set the random state for reproducibility
    np.random.seed(load_configure['default_seed'])
    
    logger.info("Train-Test Splitting ... ")
    X_train_raw_df, X_test_raw_df, y_train_tran_df, y_test_tran_df, train_index, test_index = group_shuffle_split(pred_df.values, 
                                                                                                                  phen_df.values, 
                                                                                                                  seed=load_configure['default_seed'],
                                                                                                                  test_size=0.2,
                                                                                                                  groups=group_raw_df)
    
    logger.info("Find the random forest from the iteration with lowest OOB error score ... ")
    # Create the model
    train_model = model.IterativeRFRegression(rseed=load_configure['default_seed'], **model_params)
    oob_gridsearch = OOB_Search(n_jobs=1,
                                estimator=train_model,
                                param_grid={'K':[3]})
    
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
    output = joblib.Parallel(n_jobs=5)(
        joblib.delayed(run_RIT)(deepcopy(train_model), 
                                X_train_raw_df, 
                                y_train_tran_df.flatten(), 
                                X_test_raw_df, 
                                y_test_tran_df.flatten(), 
                                n_samples, 
                                all_rf_weights[0], 
                                **rit_params)
        for b in range(0, rit_params['n_bootstrapped'])) 
    for i in (range(0,rit_params['n_bootstrapped'])):
          all_rit_bootstrap_output['rf_bootstrap{}'.format(i)] = output[i]
          
    
    logger.info("Compute stability score for each interaction term ...")
    feature_name = pred_df.columns.to_list()
    nub_feature = list(range(len(feature_name)))
    feature_dict = {feature_name[i]: nub_feature[i] for i in range(len(feature_name))}
    
    bootstrap_interact = []
    with joblib.Parallel(n_jobs=5) as parallel:
        # Loop over the data and make multiple parallel calls
        for b in range(rit_params['n_bootstrapped']):
            # Execute the parallel function using the context manager
            processed_results = parallel(joblib.delayed(rit_interactions)(tree_data, 
                                                                          feature_dict)
                                        for tree_data in all_rit_bootstrap_output['rf_bootstrap{}'.format(b)].values())
            
            bootstrap_interact.append(set(processed_results))
    
    
    all_rit_interactions = [item for sublist in (bootstrap_interact) for item in sublist if item is not None]
    stability_score = {m: all_rit_interactions.count(m) / rit_params['n_bootstrapped'] for m in all_rit_interactions}
    
    # output stability_score result
    stability_score_df = pd.DataFrame(stability_score.items(), columns=["Pattern", "Value"])
    stability_score_df.to_csv(interaction_filename, sep="\t", index=False)
    
    
    totalTime = time.time() - startTime
    logger.info("Computing done = {} min.".format(float(totalTime)/60))  