from pathlib import Path
from math import ceil
from Explore_interaction_term import write_configure, generate_all_samples, RF_DataModel, FeatureEncoder, get_rss, get_f_score, get_p_value, IterativeRF
from preprocessing_utils import get_features_data, get_phen_data
from sklearn.utils import resample
from irf.ensemble import RandomForestRegressor
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from tqdm.contrib.concurrent import process_map


import pandas as pd
import numpy as np
import argparse
import utils
import sys
import yaml
import time
import datetime


"""
Functions for conducting iterative random forest modlel

Usage:
    
    nohup python run_IRF.py \
      --work_dir "02_Explore_interaction_term" \
      --weight_tissue "Brain_Amygdala" \
      --phen_name  "CWR_Total" \
      --phen_df_name "2024-11-20T17:24:46.601289_phenotype_residualed.csv" \
      --pred_df_name "2024-11-20T17:27:02.283360_predictor_feature.csv" \
      --split_df_name "2024-11-20T17:51:00.127160_CWR_Total_Data_splits_index.csv" \
      --configure_file IRF_RF_test.yaml > /exeh_4/yuping/Epistasis_Interaction/02_Explore_interaction_term/Log/CWR_Total.txt &
          
Output:
stability score and p-value for interaction term.
"""


def process_bootstrap_iteration(args):

    b, X_train_raw_df, y_train_raw_df, X_test_raw_df, y_test_raw_df, n_samples, all_rf_weights, phen_name, seed = args
    # Train the model
    rf_bootstrap = train_model
    X_train_rsmpl, y_rsmpl = resample(X_train_raw_df.to_numpy(), 
                                      y_train_raw_df[phen_name].to_list(), 
                                      n_samples=n_samples,
                                      random_state=seed
    )
    
    # Using the weight from the (K-1)th iteration
    rf_bootstrap.fit(X=X_train_rsmpl,
                     y=y_rsmpl,
                     feature_weight=all_rf_weights
    )  
    
    # All RF tree data
    all_rf_tree_data = RF_DataModel().get_rf_tree_data(
                     rf=rf_bootstrap,
                     X_train=X_train_rsmpl,
                     X_test=X_test_raw_df.to_numpy(),
                     y_test=y_test_raw_df[phen_name].to_list(),
                     predictor="regress"
    )

    # Run FP-Growth on interaction rule set
    all_FP_Growth_data = generate_all_samples(all_rf_tree_data, bin_class_type=None)
    
    input_list = [all_FP_Growth_data[i].tolist() for i in range(len(all_FP_Growth_data))]
   
    te = TransactionEncoder()
    te_ary = te.fit(input_list).transform(input_list)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    item_sets = fpgrowth(df, min_support=1/len(input_list))
    
    # Return the result for this iteration
    return (b, item_sets)


def process_itemsets(args):
    label, item_sets = args
    top_itemsets = [set(x) for x in sorted(item_sets["itemsets"]) if len(x) ==2]
    filtered_itemsets = [item for item in top_itemsets if item]
    return filtered_itemsets


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
    
    parser.add_argument(
        '--split_df_name',
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
    # set up directory
    repo_root = Path(__file__).resolve().parent.parent
    # set up working directory
    full_work_dir = Path.joinpath(repo_root, input_arguments.work_dir)
    # set up save directory
    save_dir = Path.joinpath(full_work_dir, "results")
    # set up logging
    logger = utils.logging_config(full_work_dir, f"run_IRF:{input_arguments.weight_tissue}:{input_arguments.phen_name}", timestamp)
    # loading configure file
    configure_file = Path(full_work_dir, "model_configure", input_arguments.configure_file)
    try:
        with open(configure_file) as infile:
            load_configure = yaml.safe_load(infile)
    except Exception:
            sys.stderr.write("Please specify valid yaml file.")
            sys.exit(1)
    
    
    logger.info("Check if all files and directories exist ... ")
    if not utils.check_exist_directories([save_dir]):
        raise Exception("See output above. Problems with specified directories")
    else:
        # Create experiment directory
        experiment_dir = save_dir / input_arguments.weight_tissue
        experiment_dir.mkdir(parents=True,  exist_ok=True)
        # Create output file name
        interaction_pvalue_filename = utils.construct_filename(experiment_dir, "pvalue",".csv",  timestamp, input_arguments.weight_tissue, input_arguments.phen_name, "interaction", "run_IRF")
        stability_filename = utils.construct_filename(experiment_dir, "stability",".csv",  timestamp, input_arguments.weight_tissue, input_arguments.phen_name, "interaction", "run_IRF")
        # Check output file is exist or not
        if utils.check_exist_files([interaction_pvalue_filename, stability_filename]):
            raise Exception("See output above. Problems with specified directories")
    
    logger.info("Prepareing feature ... ") 
    # specify the configure and result file dir
    write_configure(logger, 
                    configure_file=configure_file)
    
    # load x (features) dataframes
    feature_path = Path(load_configure['dataset']['data_dir']) / input_arguments.weight_tissue / input_arguments.pred_df_name 
    feature_df = get_features_data(feature_path)
    # load y (label) dataframes
    phen_path =  Path(load_configure['dataset']['data_dir']) / input_arguments.phen_df_name
    phen_df = get_phen_data(phen_path)


    # Set the random state for reproducibility
    np.random.seed(load_configure['default_seed'])
    logger.info("Train-Test Splitting ... ")
    # load training data, testing data from indexes and features dataframe
    data_split_path = repo_root / "01_Preprocessing/results" / input_arguments.weight_tissue / input_arguments.split_df_name
    data_split_indexes = pd.read_csv(data_split_path, sep="\t", index_col=0)
    X_train_raw_df, y_train_raw_df = utils.get_dataset(feature_df, phen_df, data_split_indexes, "train")
    X_test_raw_df, y_test_raw_df = utils.get_dataset(feature_df, phen_df, data_split_indexes, "test")
        

    logger.info("Parameter setting ... ")
    # hyperparameter to for IRF
    model_params = {
        'n_estimators': load_configure['model_params']['n_estimators'][0],
        'max_features': load_configure['model_params']['max_features'][0],
        'max_depth': load_configure['model_params']['max_depth'][0],
        'n_bootstrapped': load_configure['model_params']['n_bootstrapped'][0],
        'propn_n_samples' : load_configure['model_params']['propn_n_samples'][0]
    }
    
   
    logger.info("Iterative RandomForest")
    # Initialize RF model
    train_model = RandomForestRegressor(random_state=load_configure['default_seed'], 
                                        n_estimators=model_params['n_estimators'],
                                        max_features=model_params['max_features'])
    
    oob_gridsearch = IterativeRF(n_jobs=2, estimator=train_model, param_grid={'K': [2]})
    oob_gridsearch.fit(X_train_raw_df.to_numpy(), y_train_raw_df[input_arguments.phen_name].to_list())
    all_rf_weights = oob_gridsearch.output_array[0]["rf_weight{}".format(2)]


    logger.info("FP_Growth pattern")
    # Convert the bootstrap resampling proportion to the number
    # of rows to resample from the training data
    n_samples = ceil(model_params["propn_n_samples"] * X_train_raw_df.shape[0])
    FP_Growth_args = [(b, X_train_raw_df, y_train_raw_df, X_test_raw_df, y_test_raw_df, n_samples, all_rf_weights, input_arguments.phen_name, load_configure['default_seed']) 
                      for b in range(model_params['n_bootstrapped'])]
    # Run for FPGrowth
    all_FP_Growth_bootstrap_output = process_map(process_bootstrap_iteration, FP_Growth_args, max_workers=5)
    # Extract pairwise interaction
    filtered_itemsets_list = process_map(process_itemsets, all_FP_Growth_bootstrap_output, max_workers=5)


    logger.info("Compute stability_score")
    def flatten(l): return ['_'.join(map(str, sorted(item))) for sublist in l for item in sublist]
    # Function to map indices to words
    def map_indices_to_words(index_str, word_list):
        indices = index_str.split('_')  # Split the string into individual indices
        mapped_words = [word_list[int(index)] for index in indices]  # Map indices to words
        return '_'.join(mapped_words)  # Join words with an underscore
    all_FP_Growth_interactions = flatten(filtered_itemsets_list)
    all_FP_Growth_interaction_transformed = [map_indices_to_words(item, feature_df.columns.to_list()) for item in all_FP_Growth_interactions]
    stability = {m: all_FP_Growth_interaction_transformed.count(m) / model_params['n_bootstrapped'] for m in all_FP_Growth_interaction_transformed}
    stability_score_df = pd.DataFrame(stability.items(), columns=["Pattern", "Value"])
    stability_score_df.to_csv(stability_filename, index=0)


    logger.info("Compute P-value")
    genotype_df = pd.DataFrame(X_test_raw_df, columns=feature_df.columns.to_list())
    P_value_list = []
    F_score_list = []
    Stability_term_list = []
    for i in range(stability_score_df.shape[0]): 
        # analysis feature term
        select_column = stability_score_df['Pattern'][i].split("_")
        # analysis genotype
        test_genotype = np.array(genotype_df[select_column])
        np_genotype_rsid, np_genotype = FeatureEncoder(select_column, test_genotype)
     
        if "Gender" in select_column and "Ageyr" not in select_column:
                covariate = np.array(genotype_df["Ageyr"])
                # covaraite and analysis feature term
                reduce_genotype = np.hstack((covariate[:,np.newaxis], test_genotype))
                # covaraite, analysis feature term and interaction term
                full_genotype = np.hstack((covariate[:,np.newaxis], np_genotype))
        if "Ageyr" in select_column and "Gender" not in select_column:
                covariate = np.array(genotype_df["Gender"])
                # covaraite and analysis feature term
                reduce_genotype = np.hstack((covariate[:,np.newaxis], test_genotype))
                # covaraite, analysis feature term and interaction term
                full_genotype = np.hstack((covariate[:,np.newaxis], np_genotype))
    
        if "Gender" not in select_column and "Ageyr" not in select_column:
                covariate = np.array(genotype_df[["Gender","Ageyr"]])
                # covaraite and analysis feature term
                reduce_genotype = np.hstack((covariate,test_genotype))
                full_genotype = np.hstack((covariate, np_genotype))
        else:
                pass    
    
        # number of sample
        n = len(y_test_raw_df.to_numpy())
        # degrees of freedom for full_genotype
        freedom_deg = full_genotype.shape[1]
        # get error sum of squares for reduce and full model
    
        get_rss_ho = get_rss(y_test_raw_df.to_numpy(), reduce_genotype)
        get_rss_h1 = get_rss(y_test_raw_df.to_numpy(), full_genotype)
        # get f-score and p-value
        f_score = get_f_score(get_rss_ho, get_rss_h1, n, freedom_deg)[0][0]
        p_value = get_p_value(f_score, n, freedom_deg)
        # append cacluated result to list
        P_value_list.append(p_value)
        F_score_list.append(f_score)
        Stability_term_list.append(np_genotype_rsid)

    output = pd.DataFrame(Stability_term_list, columns=["Feature_1", "Feature_2", "Interaction_term"])
    output['F_Score'] = F_score_list
    output["P_value"] = P_value_list
    output.to_csv(interaction_pvalue_filename, index=0)