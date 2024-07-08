from pathlib import Path
from sklearn.model_selection import train_test_split 
from pathlib import Path

import pandas as pd
import preprocessing_utils as pp
import argparse
import pandas as pd
import utils
import datetime
import numpy as np




"""
Functions for get train test split

Usage:
    
    python get_train_test_split.py \
      --phen_name CWR_Total  \
      --weight_tissue Brain_Amygdala \
      --phen_df_name  2024-07-08T15:04:53.628603_phenotype_residualed.db    \
      --pred_df_name  2024-07-08T14:42:32.079935_predictor_feature.csv \
      --default_seed  42  \
      --test_size 0.2 
           
Output:
index of train and test 
"""

def process_args():
    """
    Parse and format and command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--phen_name',
        action="store"
    )

    parser.add_argument(
        '--weight_tissue',
        action="store"
    )
    
    
    parser.add_argument(
        '--phen_df_name',
        action="store"
    )
    
    parser.add_argument(
        '--pred_df_name',
        action="store"
    )
    
    
    parser.add_argument(
        '--default_seed',
        type=int
    )
    
    parser.add_argument(
        '--test_size',
        type=float
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
    logger = utils.logging_config(work_dir, f"get_train_test_split:{input_arguments.weight_tissue}andphenotype:{input_arguments.phen_name}", timestamp)
    
    logger.info("Check if all files and directories exist ... ")
    if not utils.check_exist_directories(save_dir):
        raise Exception("The directory" + str(save_dir) + "not exist. Please double-check.")
    else:
        # Create experiment directory
        experiment_dir = save_dir / input_arguments.weight_tissue
        # Create experiment directory
        output_filename = utils.construct_filename(experiment_dir, "Data_splits_index", ".csv", timestamp, input_arguments.phen_name)
        # Check output file is exist or not
        if utils.check_exist_files(output_filename):
            raise Exception("Results files exist already. Please double-check.")
    

    # load x (features) dataframes
    feature_path = Path(save_dir) / input_arguments.weight_tissue / input_arguments.pred_df_name
    feature_data = pp.get_features_data(feature_path)
    # load y (label) dataframes
    phen_path =  Path(save_dir) / input_arguments.phen_df_name
    phen_data = pp.get_phen_data(phen_path)
    
    
    
    logger.info(f"Splitting data for tissue : {input_arguments.weight_tissue} and phenotype : {input_arguments.phen_name}")
    indices = np.arange(len(phen_data))
    indices_train, indices_test, training_data, testing_data = train_test_split(indices,
                                                                                phen_data["CWR_Total"].values, 
                                                                                random_state=input_arguments.default_seed,
                                                                                test_size=input_arguments.test_size)
  

    logger.info(f"Training data has shape: {training_data.shape}")
    logger.info(f"Testing data has shape: {testing_data.shape}")
    # get index of trainning and testing data
    train_indices = phen_data.index[indices_train]
    test_indices = phen_data.index[indices_test]
    # create pandas dataframe with all indexes and their respective labels, stratified by phenotypic class
    index_data = []
    for index in train_indices:
        index_data.append({"labeled_data_index": index, "label": "train"})
    for index in test_indices:
        index_data.append({"labeled_data_index": index, "label": "test"})

    # make index data a dataframe and sort it by labeled data index
    index_data = (
        pd.DataFrame(index_data)
        .sort_values(["labeled_data_index"])
    )
    
    # save indexes as csv file
    index_data.to_csv(output_filename, sep="\t")
    logger.info("Save index to {}".format(output_filename))  
    