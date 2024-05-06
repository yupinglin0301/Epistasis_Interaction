from pathlib import Path
from sklearn.pipeline import make_pipeline
from numpy_sugar.linalg import economic_qs_linear
from glimix_core.lmm import LMM
from numpy_sugar import is_all_finite

import pandas as pd
import numpy as np
import argparse
import utils
import sys
import yaml
import datetime
import argparse
import dataset_model as dm
import preprocessors as pp





"""
Functions for compute hertiability from predicted gene expression

Usage:
    
    nohup python estimate_hsq.py \
         --weight_tissue "Brain_Amygdala" \
         --phen_name CWR_Total \
         --work_dir 03_Compute_heritability > /exeh_4/yuping/Epistasis_Interaction/03_Compute_heritability/Log/nohup.txt &
     
Output:
heritability score
"""



def estimate_variance_components(y, K, cov):
    """
    Function to estimate variance components, using the packages numpy_sugar and glimix_core
    """
   
    if not is_all_finite(y):
        raise ValueError("Outcome must have finite values only.")
    if not is_all_finite(K):
        raise ValueError("Outcome must have finite values only.")
    
    fixed_effect = get_fixed_effects(cov)
    QS = economic_qs_linear(K)

    method = LMM(y, fixed_effect, QS)
    method.fit(verbose=False)

    v_g = method.v0
    v_e = method.v1

    return v_g, v_e


def standardize_expression(expression_matrix):
    
    # Calculate the mean and standard deviation along each gene (column)
    mean = expression_matrix.mean(axis=0)
    std = expression_matrix.std(axis=0)
    # Subtract the mean and divide by the standard deviation for each gene
    standardized_matrix = (expression_matrix - mean) / std

    return standardized_matrix


def get_kniship(standarized_df):
    """
    compute realized relationship matrix
    """    
    M = standarized_df.shape[1]
    np_standarized_matrix = standarized_df.to_numpy()
    np_standarized_matrix_T = standarized_df.to_numpy().T
    
    grm = (np_standarized_matrix @ np_standarized_matrix_T) / M

    return grm


def get_fixed_effects(covs):
    """
    check for covariates and create fixed effects vector
    """
    if covs is None:
        return np.ones(covs.shape[0], dtype=np.float64)
    else:
        return np.column_stack((np.ones(covs.shape[0], dtype=np.float64), covs.to_numpy()))
    

def estimate_heritability(v_g, v_e):
    """
    compute narrow sense heritability
    """
    return v_g / (v_g + v_e)


def process_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weight_tissue',
        action="store",
        help="Predicted gene expression"
    )

    parser.add_argument(
        '--phen_name', 
        action="store", 
        help="Phenotype TSV"
    )
    
    parser.add_argument(
        '--work_dir', 
        action="store", 
        help="Working directory"
    )
    
    args = parser.parse_args()
    return(args)


if __name__ == '__main__':

    # process command line arguments
    input_arguments = process_args()
    
    timestamp = datetime.datetime.now().today().isoformat()
    # set up logging
    logger = utils.logging_config(input_arguments.weight_tissue + input_arguments.phen_name + "Compute_heritability", timestamp)
    # set up directory
    repo_root = Path(__file__).resolve().parent.parent
    # set up working directory
    work_dir = Path.joinpath(repo_root, input_arguments.work_dir)
    # set up save directory
    save_dir = Path.joinpath(work_dir, "results")
    # loading configure file
    configure_file = Path(work_dir, "config.yaml")
    
    try:
        with open(configure_file) as infile:
            load_configure = yaml.safe_load(infile)
    except Exception:
            sys.stderr.write("Please specify valid yaml file.")
            sys.exit(1)
    
    
    logger.info("Check if all files and directories exist ...")
    #Check save directory if is exist or not
    if not utils.check_exist_directories(save_dir):
        raise Exception("The directory" + str(save_dir) + "not exist. Please double-check.")
    else:
        # Create experiment directory
        experiment_dir = save_dir / input_arguments.weight_tissue
        experiment_dir.mkdir(parents=True,  exist_ok=True)
       
        output_filename = utils.construct_filename(experiment_dir, "feature", ".csv", timestamp, input_arguments.weight_tissue)
        # Check output file is exist or not
        if utils.check_exist_files(output_filename):
            raise Exception("Results files exist already. Please double-check.")
        
    
    GTEX_Dataset = dm.GTEX_raw_Dataset.from_config(config_file=load_configure, 
                                                   weight_tissue=input_arguments.weight_tissue)
    
    logger.info('standarized gene expression each column')
    inv_norm_pred_expr_mat = standardize_expression(GTEX_Dataset.all_gen_df)
    
    
    logger.info('Build GRM')
    columns_with_na  = inv_norm_pred_expr_mat.isna().any()
    columns_with_na_values = inv_norm_pred_expr_mat.columns[columns_with_na]
    # remove out na column
    inv_norm_pred_exper_mat_without_na = inv_norm_pred_expr_mat.drop(columns_with_na_values, axis=1)
    grm = get_kniship(inv_norm_pred_exper_mat_without_na)
    
    
    
    logger.info('Compute heritability')
    # impute missing value mean value in each column
    y = pd.read_csv("/mnt/data/share/yuping/data/phenotype_info.csv", sep="\t")
    pipeline = make_pipeline(
       pp.NumericalImputer(variables=[input_arguments.phen_name])
    )
    transformed_y = pipeline.fit_transform(y)
    # varaince of genetic effect and error term
    v_g, v_e = estimate_variance_components(y=transformed_y[input_arguments.phen_name].to_list(),
                                            K=grm,
                                            cov=GTEX_Dataset.all_gwas_df.loc[:,["Ageyr", "Gender"]])
    
    
    heritability = estimate_heritability(vg=v_g, ve=v_e)                                                                 