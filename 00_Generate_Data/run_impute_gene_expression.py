"""
Functions for impute gene expression for different tissue.

Usage:

    nohup python run_impute_gene_expression.py \
      --data_config "/exeh_4/yuping/Epistasis_Interaction/00_Generate_Data/config.yaml" \
      --data_dir "/mnt/data/share/yuping" \
      --work_dir "/exeh_4/yuping/Epistasis_Interaction" \
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
    
    logger.info("Checking arguments and creating experiment dir/file ... ")
    # Loading configure file and checking all arguments
    data_configure = utils.load_config_and_check_arg(input_arguments)
    # Create experiment dir 
    experiment_dir = Path(data_configure['save_dir'], data_configure['weight_tissue']).resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    output_filename = utils.construct_filename(experiment_dir, "imputed", ".txt", data_configure['weight_tissue'])
    # Check if the output filename already exists
    if utils.check_exist_files(output_filename):
        raise Exception("Results file " + str(output_filename) + " exist already. Please double-check.")

    
    logger.info("Imputing Gene Expression ... ")

    beta_file = data_configure['weight_tissue_dir'] / (data_configure['weight_prefix'] + "_" + data_configure['weight_tissue'] +"_imputed_europeans_tw_0.5_signif" + data_configure['weight_end_prefix'])
    sample_file =  data_configure['sample_data']
    reference_file = data_configure['reference_data']

    unqiue_rsids = im.GenotypeDataset.UniqueRsid(beta_file)
    TranscriptionMatrix = im.TranscriptionMatrix(beta_file, sample_file)
    reference_file = im.GenotypeDataset.get_reference(reference_file)

    for rsid, allele, dosage_row in im.GenotypeDataset.get_all_dosages(data_configure['genotype_dir'],
                                                                       data_configure['dosage_prefix'],
                                                                       data_configure['dosage_end_prefix'],
                                                                       unqiue_rsids,
                                                                       reference_file,
                                                                       logger):
        
        for gene, weight, ref_allele in im.WeightsDB(beta_file).query("SELECT gene, weight, eff_allele FROM weights WHERE rsid=?",(rsid,)):
            TranscriptionMatrix.update(gene, weight, ref_allele, allele, dosage_row)
    
    TranscriptionMatrix.save(output_filename, logger)