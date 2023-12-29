from sklearn.pipeline import make_pipeline
from pathlib import Path
import dataset_model as dm
import preprocessors as pp
import argparse
import utils


"""
Functions for making pipline for feature engineering

Usage:
    
    nohup python run_making_pipline.py \
      --work_dir "/exeh_4/yuping/Epistasis_Interaction/01_Preprocessing" \
      --weight_tissue "Brain_Amygdala" >  /exeh_4/yuping/Epistasis_Interaction/01_Preprocessing/Log/nohup.txt &
      
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
    
    args = parser.parse_args()
    return(args)

if __name__ == '__main__':
    
    # process command line arguments
    input_arguments = process_args()
    # set up logging
    logger = utils.logging_config(input_arguments.weight_tissue)
    
    logger.info("Check if all files and directories exist ... ")
    # Check save directory if is exist or not
    save_dir = Path(input_arguments.work_dir) / "results"
    if not utils.check_exist_directories(save_dir):
        raise Exception("The directory" + str(save_dir) + "not exist. Please double-check.")
    else:
        # Create experiment directory
        experiment_dir = save_dir / input_arguments.weight_tissue
        experiment_dir.mkdir(parents=True,  exist_ok=True)
        output_filename = utils.construct_filename(experiment_dir, "output", ".pkl", "pipline", "version1")
        # Check output file is exist or not
        if utils.check_exist_files(output_filename):
            raise Exception("Results file " + str(output_filename) + " exist already. Please double-check.")
    
    logger.info("Creating pipline {}... ".format(input_arguments.weight_tissue))

    pipeline = make_pipeline(
         pp.CategoricalImputer_Education(variables=['FatherEducation', 'MotherEducation']),
         pp.CategoricalEncoder_Income(variables=['Income'])
    )
    dm.RF_OOB_Dataset.save(pipeline, output_filename)