from sklearn.pipeline import make_pipeline
from pathlib import Path
import preprocessing_utils as pp
import argparse
import utils
import yaml
import sys
import datetime



"""
Functions for making pipline for feature engineering

Usage:
    
    nohup python etl_for_predictor.py \
      --weight_tissue "Brain_Amygdala" > /exeh_4/yuping/Epistasis_Interaction/01_Preprocessing/Log/nohup.txt &
      
Output:
Feature engineering pipeline for specific imputed brain tissue and environmental factors

"""

# Extraction
def extract(data_set):
    """
    Read Excel file
    """
    df = data_set.all_gwas_df
    
    return df

# Transform
def transform_data(df):
    """
    Perform data transformation using a pipeline of preprocessing steps.

    :param df: DataFrame containing the data to be transformed.
    :return Transformed DataFrame.
    """
    pipeline = make_pipeline(
            pp.NormalizeDataTransformer(column_names=df.columns.tolist()[4:]),
            pp.CategorialEncoder_Education(variables=['FatherEducation', 'MotherEducation']),
            pp.CategorialEncoder_Income(variables=['Income'])   
    )
    
    transformed_data = pipeline.fit_transform(df)
    return transformed_data
   

# Loading
def load_data(dataset, output_filename):
    """
    Extracts data from the dataset, performs transformation, and saves the result as a csv file.
    
    :param dataset: The dataset object containing the data to be loaded.
    :param output_filename: The name of the output file to save the transformed data.
    """
    extracted_data = extract(dataset)
    transformed_data = transform_data(extracted_data)
    
    # Save transformed data as a csv file
    transformed_data.to_csv(output_filename, sep="\t", index=False)
   

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
    logger = utils.logging_config(work_dir, f"etl_for_predictor:{input_arguments.weight_tissue}", timestamp)
    #loading configure file
    configure_file = work_dir.joinpath("config.yaml")
    try:
        with open(configure_file) as infile:
            load_configure = yaml.safe_load(infile)
    except Exception:
            sys.stderr.write("Please specify valid yaml file.")
            sys.exit(1)
    
 
    logger.info("Check if all files and directories exist ...")
    # Check save directory if is exist or not
    if not utils.check_exist_directories(save_dir):
        raise Exception("The directory" + str(save_dir) + "not exist. Please double-check.")
    else:
        # Create experiment directory
        experiment_dir = save_dir / input_arguments.weight_tissue
        experiment_dir.mkdir(parents=True,  exist_ok=True)
       
        output_filename = utils.construct_filename(experiment_dir, "feature", ".csv", timestamp, "predictor")
        # Check output file is exist or not
        if utils.check_exist_files(output_filename):
            raise Exception("Results files exist already. Please double-check.")
        
    logger.info("Conduct ETL pipeline ...")
    GTEX_Dataset = pp.GTEX_raw_Dataset.from_config(config_file=load_configure, 
                                                   weight_tissue=input_arguments.weight_tissue)
    
    load_data(dataset=GTEX_Dataset, output_filename=output_filename)
    logger.info("Save predictor feature to {}".format(output_filename))  