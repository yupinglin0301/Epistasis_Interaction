
import argparse
import utils
import numpy as np
import pickle
import gzip 
import yaml
import sys
from pathlib import Path

"""
Functions for compute genetic relationship matrix

Usage:

    nohup python compute_grm.py \
      --data_config "/exeh_4/yuping/Epistasis_Interaction/00_Generate_Data/config.yaml" \
      --work_dir "/exeh_4/yuping/Epistasis_Interaction/00_Generate_Data" \
      --dosage_prefix "chr" \
      --dosage_end_prefix ".plink.dosage" > /exeh_4/yuping/Epistasis_Interaction/00_Generate_Data/Log/nohup.txt &

Output:
Gene relationship matrix(GRM)
"""

def load_genotype_from_dosage(chrfile):
    """
    Load genotype data from a dosage file and performs preprocessing.
    """
    
    genotype_matrix = []
    with open(str(chrfile), "rt") as file:
        for line_index, line in enumerate(file):
            
            if line_index <= 0:
                continue

            arr = line.strip().split()
            dosage_row = np.array(arr[3:], dtype=np.float64)
            genotype_matrix.append(dosage_row)
            
        genotype_matrix = np.transpose(np.array(genotype_matrix))        
        p = np.mean(genotype_matrix, axis=0) / 2 # Row menas over 2 
        var_geno = 2 * p * (1 - p)
        genotype_matrix = (genotype_matrix - 2 * p) / np.sqrt(var_geno)
        
    return genotype_matrix

def compute_grm_from_dosage(genotype_dir, dosage_prefix, dosage_end_prefix, logger):
    """
    Computes the Genetic Relationship Matrix (GRM) from dosage files in the given directory.
    """
    
    nsnp = 0
    grm = None
    for chrfile in [
        x for x in sorted(genotype_dir.iterdir())
        if x.name.startswith(str(dosage_prefix))
        and x.name.endswith(str(dosage_end_prefix))
    ]:
          
        logger.info(f"Computing GRM for {chrfile} ... ")
        geno = load_genotype_from_dosage(chrfile)
        # calc sub-GRM
        M = geno.shape[1]
        grm_now = geno @ (geno.T / M)
        
        # update GRM
        if grm is None:
            grm = np.zeros((geno.shape[0], geno.shape[0]))
        w1 = nsnp / (M + nsnp)
        w2 = 1 - w1
        grm = grm * w1 + grm_now * w2
        nsnp += M
    
    return grm


def process_args():
    """
    Parse and format and command line arguments.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_configure',
        action="store",
        help="The yaml formatted dataset configuration file."
    )

    parser.add_argument(
        '--work_dir',
        action="store",
        help="Working directory where the experiment will conducted."
    )
    
    parser.add_argument(
        '--dosage_prefix',
        action="store",
        help="Specify the prefix of filenames of dosage files."
    )

    parser.add_argument(
        '--dosage_end_prefix',
        action="store",
        help="Specify the end prefix of filenames of dosage files."
    )

    args = parser.parse_args()
    return(args)


if __name__ == '__main__':

    # process command line arguments
    input_arguments = process_args()
    # set up logging
    logger = utils.logging_config("GRM")
    
    
    logger.info("Check if all files and directories exist ... ")
    # Loading configuration file
    try:
        with open(input_arguments.data_configure) as infile:
            load_configure = yaml.safe_load(infile)
    except Exception:
        sys.stderr.write("Please specify valid yaml file.")
        sys.exit(1)

    save_dir = Path(input_arguments.work_dir) / "results"
    genotype_dir = Path("/mnt/data/share/yuping") / load_configure['genotype_data_path']

    if not utils.check_exist_directories(genotype_dir):
        raise Exception("The directory" + str(genotype_dir) + "not exist. Please double-check.")
    
    if not utils.check_exist_directories(save_dir):
        raise Exception("The directory" + str(save_dir) + "not exist. Please double-check.")
    else:
        output_filename = utils.construct_filename(save_dir, "correlation.pkl", ".gz", "genetic")
        #Check output file is exist or not
        if utils.check_exist_files(output_filename):
            raise Exception("Results file " + str(output_filename) + " exist already. Please double-check.")
    
        
    logger.info("Computing Genetic Relationship Matrix  ... ")
    grm = compute_grm_from_dosage(genotype_dir, 
                                  input_arguments.dosage_prefix, 
                                  input_arguments.dosage_end_prefix,
                                  logger)
    with gzip.open(output_filename, 'wb') as f:
        pickle.dump(grm, f)
    




