import argparse
import numpy as np
import pandas as pd
import utils
import sys
import yaml
import preprocessing_utils as pp
import sqlite3
import pickle
import gzip
import datetime

from pathlib import Path
from sklearn.pipeline import make_pipeline



"""
Functions for remove out genetic relationship structure

Usage:
    
    nohup python etl_for_phenotype.py \
      --phen_name "CWR_Total" \
      --gene_cor_name "genetic_correlation.pkl.gz"
      >  /exeh_4/yuping/Epistasis_Interaction/01_Preprocessing/Log/nohup.txt &
      
Output:
residual phenotype

"""

def compute_expected_value(y, grm):
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
    crossprod_ones_y = np.dot(ones.T, y).flatten()
    crossprod_Z_y = np.dot(Z.T, y).flatten()
    RHS = np.concatenate((crossprod_ones_y, crossprod_Z_y), axis=0)
    
    gblup = np.linalg.solve(coeff, RHS)
    # Compute expected value
    expected_value = gblup[0] + np.matmul(Z, gblup[1:])
    
    return expected_value

# Extraction
def extract(phen_file, columns):
    # Read the data file and extract specified columns
    df = pd.read_csv(phen_file, usecols=[columns], sep="\t")
    return df

# Transform
def transform_data(df, columns, grm):
    # Define a pipeline for data transformation
    pipeline = make_pipeline(
        pp.NumericalImputer(variables=columns)
    )
   
    # Apply the transformation pipeline on the DataFrame
    df = pipeline.fit_transform(df)
    
    # Compute the expected value using a function 'compute_expected_value' and the 'grm'
    result = df.apply(lambda column: compute_expected_value(column, grm), axis=0)
    
    # Compute the residual by subtracting transformed data from the expected_value
    residual_phen = df - result
    
    return residual_phen
   
# Loading
def load_data(db_name, file, columns, grm):
    # Extract the data
    load_phen = extract(file, columns)
    
    # Transform the data
    tran_phen = transform_data(load_phen, columns, grm)
    
    # Connect to the database and load the residual phenotype data into a table named "Phenotype"
    connection = sqlite3.connect(db_name)
    cur = connection.cursor()
    tran_phen.to_sql("Phenotype", connection, index=False, if_exists="replace")
    cur.close()
    connection.close()

def process_args():
    """
    Parse and format and command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--phen_name',
        action="store", 
        help="Data directory where the phenotype and genotype matrix are stored."
    )
    parser.add_argument(
        '--gene_cor_name',
        action="store", 
        help="Data directory where the phenotype and genotype matrix are stored."
    )
    
    
    args = parser.parse_args()
    return(args)


if __name__ == '__main__':
   
    # process command line arguments
    input_arguments = process_args()
    timestamp = datetime.datetime.now().today().isoformat()
    # set up repo_directory
    repo_root = Path(__file__).resolve().parent.parent
    # set up working directory
    work_dir = repo_root / str("01_Preprocessing")
    # set up result directory
    save_dir = work_dir.joinpath("results")
    # set up logging
    logger = utils.logging_config(work_dir, f"etl_for_phenotype:{input_arguments.phen_name}", timestamp)
    # loading configure file
    configure_file = work_dir.joinpath("config.yaml")
    try:
        with open(configure_file) as infile:
            load_configure = yaml.safe_load(infile)
    except Exception:
            sys.stderr.write("Please specify valid yaml file.")
            sys.exit(1)
    
    logger.info("Check if all files and directories exist ... ")
    if not utils.check_exist_directories(save_dir):
        raise Exception("The directory" + str(save_dir) + "not exist. Please double-check.")
    else:
        # Create experiment directory
        output_filename = utils.construct_filename(save_dir, "residualed", ".db", timestamp, "phenotype")
        # Check output file is exist or not
        if utils.check_exist_files(output_filename):
            raise Exception("Results files exist already. Please double-check.")
    
    logger.info("Conduct ETL pipeline for format {}...".format(input_arguments.phen_name))
    
    data_dir = Path(load_configure['dataset']['data_dir'])
    gene_cor_dir = data_dir / input_arguments.gene_cor_name
    with gzip.open(gene_cor_dir, 'rb') as f:
        gene_cor_matrix = pickle.load(f)  
    
    load_data(db_name=output_filename, 
              file=load_configure['dataset']['phentoype_dir'], 
              grm=gene_cor_matrix, 
              columns=input_arguments.phen_name)
    
    logger.info("Save predictor feature to {}".format(output_filename))