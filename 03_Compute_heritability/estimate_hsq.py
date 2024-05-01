from pathlib import Path


import pandas as pd

import numpy as np
import argparse
import utils
import sys
import yaml

import time
import datetime

import argparse
import dataset_model as dm



def standardize_expression(expression_matrix):
    # Calculate the mean and standard deviation along each gene (column)
    mean = np.mean(expression_matrix, axis=0)
    std = np.std(expression_matrix, axis=0)

    # Subtract the mean and divide by the standard deviation for each gene
    standardized_matrix = (expression_matrix - mean) / std

    return standardized_matrix



"""
Functions for remove out genetic relationship structure

Usage:
    
    nohup python estimate_hsq.py \
         --weight_tissue "Brain_Amygdala" \
         --phen_name CWR_Total \
         --work_dir 03_Compute_heritability 
     
Output:
residual phenotype

"""


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
    logger = utils.logging_config(input_arguments.weight_tissue + input_arguments.phen_name + "Permutation_Test", timestamp)
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
    # Check save directory if is exist or not
    #if not utils.check_exist_directories(save_dir):
    #    raise Exception("The directory" + str(save_dir) + "not exist. Please double-check.")
    #else:
    #    # Create experiment directory
    #    experiment_dir = save_dir / input_arguments.weight_tissue
    #    experiment_dir.mkdir(parents=True,  exist_ok=True)
       
    #    output_filename = utils.construct_filename(experiment_dir, "feature", ".csv", timestamp, "predictor")
    #    # Check output file is exist or not
    #    if utils.check_exist_files(output_filename):
    #        raise Exception("Results files exist already. Please double-check.")
        
    #logger.info("Conduct ETL pipeline ...")
    #GTEX_Dataset = dm.GTEX_raw_Dataset.from_config(config_file=load_configure, 
    #                                               weight_tissue=input_arguments.weight_tissue)
    
    
    
    
    #print(GTEX_Dataset.all_gen_df)
    
    
    
    

    #logger.info('Read predicted expression')
    #pred_expr = pd.read_csv(input_arguments.pred_expr, sep = '\t', header = 0)
    #gene_in_pred_expr = pred_expr['gene'].to_list()
    #pred_expr_mat = pred_expr.drop(columns = ['gene'])

    #logger.info('standarized gene expression each column')
    #inv_norm_pred_expr_mat = standardize_expression(pred_expr_mat)
    
    
    #logging.info('Build GRM')
    #grm = ghelper.format_to_gcta_grm(inv_norm_pred_expr_mat)
    