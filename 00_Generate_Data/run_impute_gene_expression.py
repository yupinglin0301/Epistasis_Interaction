"""
Functions for impute gene expression for different tissue.

Usage:

    nohup python run_impute_gene_expression.py \
      --data_config "/exeh_4/yuping/Epistasis_Interaction/00_Generate_Data/config.yaml" \
      --data_dir "/mnt/data/share/yuping" \
      --work_dir "/exeh_4/yuping/Epistasis_Interaction/00_Generate_Data" \
      --weight_tissue "Brain_Amygdala" \
      --weight_end_prefix ".db" \
      --weight_prefix "gtex_v7" \
      --dosage_prefix "chr" \
      --dosage_end_prefix ".plink.dosage" > /exeh_4/yuping/Epistasis_Interaction/00_Generate_Data/Log/nohup.txt &

Output:
gene expression for specifice tissue

"""

import argparse
import utils
import impute_predixcan as im
import sys
import yaml
from pathlib import Path

  
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
        '--data_dir',
        action="store",
        help="Data directory where the phenotype and genotype matrix are stored."
    )

    parser.add_argument(
        '--work_dir',
        action="store",
        help="Working directory where the experiment will conducted."
    )

    parser.add_argument(
        '--weight_tissue',
        action="store",
        help="Specify the tissue SQLite database."
    )

    parser.add_argument(
        '--weight_prefix',
        action="store",
        help="Specify the prefix of the tissue SQLite database."
    )

    parser.add_argument(
        '--weight_end_prefix',
        action="store",
        help="Specify the end prefix of the tissue SQLite database."
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

    # Process command line arguments
    input_arguments = process_args()
    # Set up logging
    logger = utils.logging_config(input_arguments.weight_tissue)
    
    logger.info("Check if all files and directories exist ... ")
    
    # Loading configuration file
    try:
        with open(input_arguments.data_configure) as infile:
            load_configure = yaml.safe_load(infile)
    except Exception:
            sys.stderr.write("Please specify valid yaml file.")
            sys.exit(1)

    data_dir = Path(input_arguments.data_dir)
    work_dir = Path(input_arguments.work_dir)

    save_dir = work_dir.joinpath("results")
    genotype_dir = data_dir.joinpath(load_configure['genotype_data_path'])
    weight_tissue_dir = data_dir.joinpath(load_configure['weights_data_path'])
    
    dosage_prefix = input_arguments.dosage_prefix
    weight_prefix = input_arguments.weight_prefix
    dosage_end_prefix = input_arguments.dosage_end_prefix
    weight_end_prefix = input_arguments.weight_end_prefix
    weight_tissue = input_arguments.weight_tissue
    
    sample_data = data_dir.joinpath(load_configure['sample_data'])
    reference_data = data_dir.joinpath(load_configure['reference_data'])
    beta_file = weight_tissue_dir / (weight_prefix + "_" +  weight_tissue + "_imputed_europeans_tw_0.5_signif" +  weight_end_prefix)
    
    
    if not utils.check_exist_files([sample_data, reference_data, beta_file]):
        raise Exception("See output above. Problems with specified directories")
    if not utils.check_exist_directories(save_dir):
        raise Exception("The directory" + str(save_dir) + "not exist. Please double-check.")
    else:
        # Create experiment directory
        experiment_dir = Path(save_dir, input_arguments.weight_tissue).resolve()
        experiment_dir.mkdir(parents=True,  exist_ok=True)
        
        output_filename = utils.construct_filename(experiment_dir, "imputed", ".txt", input_arguments.weight_tissue)
        # Check output file is exist or not
        if utils.check_exist_files(output_filename):
            raise Exception("Results file " + str(output_filename) + " exist already. Please double-check.")
        
    
    logger.info("Imputing Gene Expression ... ")

    unqiue_rsids = im.GenotypeDataset.UniqueRsid(beta_file)
    TranscriptionMatrix = im.TranscriptionMatrix(beta_file, sample_data)
    reference_file = im.GenotypeDataset.get_reference(reference_data)

    for rsid, allele, dosage_row in im.GenotypeDataset.get_all_dosages(genotype_dir, 
                                                                       dosage_prefix, 
                                                                       dosage_end_prefix,
                                                                       unqiue_rsids,
                                                                       reference_file,
                                                                       logger):
        
        for gene, weight, ref_allele in im.WeightsDB(beta_file).query("SELECT gene, weight, eff_allele FROM weights WHERE rsid=?",(rsid,)):
            TranscriptionMatrix.update(gene, weight, ref_allele, allele, dosage_row)
    
    TranscriptionMatrix.save(output_filename, logger)