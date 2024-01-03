import argparse
import numpy as np
import pandas as pd
import utils
import sys
import yaml
import dataset_model as dm
from pathlib import Path


"""
Functions for remove out genetic relationship structure

Usage:
    
    nohup python regress_out_grm.py \
      --work_dir "/exeh_4/yuping/Epistasis_Interaction/01_Preprocessing" \
      --phen_name "CCR_Total" \
      --weight_tissue "Brain_Amygdala" >  /exeh_4/yuping/Epistasis_Interaction/01_Preprocessing/Log/nohup.txt &
      
Output:
residual phenotype

"""

def compute_expected_value(grm, y):
    """
    Compute the expected value using GBLUP (Genomic Best Linear Unbiased Prediction)
    """
    
    ones = np.ones(grm.shape[0])
    # The next line adds a small amount to the diagonal of G,
    # otherwise G is not invertable in this small example!
    grm += np.diag(np.ones(grm.shape[0]) * 0.01)
    # Compute the inverse of GRM
    grm_inv = np.linalg.inv(grm)

    # Construct Z
    Z = np.diag(np.ones(grm.shape[0]))
    # Build mixed model solution equations
    coeff = np.zeros((grm.shape[0] + 1, grm.shape[0] + 1))
    coeff[0, 0] = np.matmul(ones.T, ones)
    coeff[0, 1:] = np.matmul(ones.T, Z)
    coeff[1:, 0] = np.matmul(Z.T, ones)
    coeff[1:, 1:] = np.matmul(Z.T, Z) + grm_inv
    
    # Compute the right-hand side
    rhs = np.vstack((np.matmul(ones.T, y), np.matmul(Z.T, y)))
    gblup = np.linalg.solve(coeff, rhs)
    # Compute expected value
    expected_value = np.ones((len(y),1)) * gblup[0] + np.matmul(Z, gblup[1:])
    
    return expected_value
    


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
    
    args = parser.parse_args()
    return(args)


if __name__ == '__main__':
   
    # process command line arguments
    input_arguments = process_args()
    # set up logging
    logger = utils.logging_config(input_arguments.weight_tissue + input_arguments.phen_name)
    
    
    # loading configure file
    work_dir = Path(input_arguments.work_dir)
    configure_file = work_dir.joinpath("config.yaml")
    try:
        with open(configure_file) as infile:
            load_configure = yaml.safe_load(infile)
    except Exception:
            sys.stderr.write("Please specify valid yaml file.")
            sys.exit(1)
            
    save_dir = work_dir.joinpath("results")
    if not utils.check_exist_directories(save_dir):
        raise Exception("The directory" + str(save_dir) + "not exist. Please double-check.")
    else:
        # Create experiment directory
        experiment_dir = Path(save_dir, input_arguments.weight_tissue).resolve()
        experiment_dir.mkdir(parents=True,  exist_ok=True)
        output_filename = utils.construct_filename(experiment_dir, "normalized_gene_expression", ".csv", input_arguments.weight_tissue)
        # Check output file is exist or not
        if utils.check_exist_files(output_filename):
            raise Exception("Results files exist already. Please double-check.")
    
    GTEX_Dataset = dm.GTEX_raw_Dataset.from_config(config_file=load_configure, 
                                                   weight_tissue=input_arguments.weight_tissue)

    # generate phenotype label
    y_given_raw_df = GTEX_Dataset.generate_labels(input_arguments.phen_name)
    # impute missing value with mean value
    mean_value = y_given_raw_df[input_arguments.phen_name].mean()
    y_given_raw_df[input_arguments.phen_name].fillna(mean_value, inplace=True)
    y_raw = y_given_raw_df.values if isinstance(y_given_raw_df, pd.DataFrame) else y_given_raw_df
    
    # load GRM
    grm = GTEX_Dataset.gene_cor_matrix 
    # get expected_value 
    expected_value = compute_expected_value(grm, y_raw)
    # substract genetic relationship structure from phenotype
    y_residual = y_raw - expected_value
    y_residual_df = pd.DataFrame(y_residual, columns=[input_arguments.phen_name])
    # save residual phenotype file 
    dm.GTEX_raw_Dataset.save(y_residual_df, output_filename)
    
  
    

    
    
    
    
    
    
    
        
    
    
    
   
    
   
    
    

    
    
   
    
    
    
    
    
 





