from sklearn.pipeline import make_pipeline
from pathlib import Path
import preprocessors as pp
import argparse
import utils
import joblib


"""
Functions for making pipline for feature engineering

Usage:
    
    nohup python run_making_pipline.py \
      --weight_tissue "Brain_Amygdala" \
      --normalized  >  /exeh_4/yuping/Epistasis_Interaction/01_Preprocessing/Log/nohup.txt &
      
Output:
Feature engineering pipeline for specific imputed brain tissue and environmental factors

"""
def process_args():
    """
    Parse and format and command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--work_dir',
        action="store",
        help="The yaml formatted dataset configuration file."
    )

    parser.add_argument(
        '--weight_tissue',
        action="store",
        help="Data directory where the phenotype and genotype matrix are stored."
    )
    
    parser.add_argument(
        '--normalized',
        action="store_true",
        help="Normalized the gene expression dataset or not."
    )
    
    args = parser.parse_args()
    return(args)

if __name__ == '__main__':
    
    # process command line arguments
    input_arguments = process_args()
    # set up logging
    if input_arguments.normalized:
        logger = utils.logging_config(input_arguments.weight_tissue + "Normalized")
    else:
        logger = utils.logging_config(input_arguments.weight_tissue + "Feature_Engineering")
    # set up repo_directory
    repo_root = Path(__file__).resolve().parent.parent
    
 
    
    logger.info("Check if all files and directories exist ... ")
    # Check save directory if is exist or not
    work_dir = repo_root / str("01_Preprocessing")
    save_dir = work_dir.joinpath("results")
    if not utils.check_exist_directories(save_dir):
        raise Exception("The directory" + str(save_dir) + "not exist. Please double-check.")
    else:
        # Create experiment directory
        experiment_dir = save_dir / input_arguments.weight_tissue
        experiment_dir.mkdir(parents=True,  exist_ok=True)
    
    logger.info("Creating pipline {}... ".format(input_arguments.weight_tissue))
    
    if input_arguments.normalized:
        
        logger.info("Normalized gene expression data")
        output_filename = utils.construct_filename(experiment_dir, "output", ".pkl", "normalized", input_arguments.weight_tissue)
        # Check output file is exist or not
        if utils.check_exist_files(output_filename):
            raise Exception("Results file " + str(output_filename) + " exist already. Please double-check.")
        
        pipeline = make_pipeline(
            pp.NormalizeDataTransformer()
        )
        joblib.dump(pipeline, output_filename)
    else:
        output_filename = utils.construct_filename(experiment_dir, "output", ".pkl", "feature_eng", "envir")
        # Check output file is exist or not
        if utils.check_exist_files(output_filename):
            raise Exception("Results file " + str(output_filename) + " exist already. Please double-check.")
        
        pipeline = make_pipeline(
            pp.CategoricalImputer_Education(variables=['FatherEducation', 'MotherEducation']),
            pp.CategoricalEncoder_Income(variables=['Income'])
        )
        joblib.dump(pipeline, output_filename)
        