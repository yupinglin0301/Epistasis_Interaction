from pathlib import Path
import pandas as pd
import model as im
import dataset_model as dm
import argparse
import utils
import sys
import yaml
import eval_metrics
import gridsearch_model

"""
Functions for computing out-of-bag scores to find the optimal parameter combination

Usage:
    
    python run_oob_pipline.py \
      --work_dir "/exeh_4/yuping/Epistasis_Interaction/02_Select_Parameter_Model" \
      --weight_tissue "Brain_Amygdala" \
      --phen_name  "BDS_Total" \
      --FE_pipeline "/exeh_4/yuping/Epistasis_Interaction/01_Preprocessing/results"
      
          
      
Output:
Feature engineering pipeline for specific imputed brain tissue and environmental factors

"""


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
    
    parser.add_argument(
        '--FE_pipeline',
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
    
    # loading configure file
    configure_file = Path(input_arguments.work_dir, "config.yaml")
    try:
        with open(configure_file) as infile:
            load_configure = yaml.safe_load(infile)
    except Exception:
            sys.stderr.write("Please specify valid yaml file.")
            sys.exit(1)
   
    logger.info("Check if all files and directories exist ... ")
    # Check save directory if is exist or not
    save_dir = Path(input_arguments.work_dir) / "results"
    if not utils.check_exist_directories(save_dir):
        raise Exception("The directory" + str(save_dir) + "not exist. Please double-check.")
    else:
        # Create experiment directory
        experiment_dir = save_dir / input_arguments.weight_tissue
        experiment_dir.mkdir(parents=True,  exist_ok=True)
        best_model_filename = utils.construct_filename(experiment_dir, "model", ".pkl", input_arguments.weight_tissue, "best_param")
        cv_result_filename = utils.construct_filename(experiment_dir, "result",".csv", input_arguments.weight_tissue, "cv")
        
        x_train_filename = utils.construct_filename(experiment_dir, "transformed",".csv", input_arguments.weight_tissue, "x_train")
        x_test_filename = utils.construct_filename(experiment_dir, "transformed",".csv", input_arguments.weight_tissue, "x_test")
        y_train_filename = utils.construct_filename(experiment_dir, input_arguments.phen_name, ".csv", input_arguments.weight_tissue, "y_train")
        y_test_filename = utils.construct_filename(experiment_dir, input_arguments.phen_name, ".csv", input_arguments.weight_tissue, "y_test")
         
        # Check output file is exist or not
        if utils.check_exist_files(best_model_filename):
            raise Exception("Results file " + str(best_model_filename) + " exist already. Please double-check.")
        if utils.check_exist_files(cv_result_filename):
                raise Exception("Results file " + str(cv_result_filename) + " exist already. Please double-check.")
            
        
    
    logger.info("Train-Test splitting ... ")
    RF_OOB_Dataset = dm.RF_OOB_Dataset.from_config(config_file=load_configure, 
                                                   weight_tissue=input_arguments.weight_tissue)
    # generate phenotype label
    y_given_raw_df = RF_OOB_Dataset.generate_labels(input_arguments.phen_name)
    X_raw_df = RF_OOB_Dataset.all_gwas_df.values if isinstance(RF_OOB_Dataset.all_gwas_df, pd.DataFrame) else RF_OOB_Dataset.all_gwas_df
    y_raw_df = y_given_raw_df.values if isinstance(y_given_raw_df, pd.DataFrame) else y_given_raw_df


    X_train_raw_df, X_test_raw_df, y_train_raw_df, y_test_raw_df = RF_OOB_Dataset.train_test_split(X_raw_df, 
                                                                                                   y_raw_df, 
                                                                                                   seed=load_configure['default_seed'],
                                                                                                   test_size=0.2)
    

    X_train_df = pd.DataFrame(X_train_raw_df, columns=RF_OOB_Dataset.all_gwas_df.columns)
    X_test_df = pd.DataFrame(X_test_raw_df, columns=RF_OOB_Dataset.all_gwas_df.columns)
    X_train_df['MotherEducation'].replace("#NULL!", pd.NA, inplace=True)
    X_train_df['FatherEducation'].replace("#NULL!", pd.NA, inplace=True)
    X_test_df['MotherEducation'].replace("#NULL!", pd.NA, inplace=True)
    X_test_df['FatherEducation'].replace("#NULL!", pd.NA, inplace=True)
    y_train_df = pd.DataFrame(y_train_raw_df, columns=[input_arguments.phen_name])
    y_test_df =  pd.DataFrame(y_test_raw_df, columns=[input_arguments.phen_name])
    
    
    logger.info("Feature Engineering ... ")
    # load pipeline for feature engineering
    pipeline_dir = Path(input_arguments.FE_pipeline, input_arguments.weight_tissue, "pipline_version1_output.pkl")
    pipeline = dm.RF_OOB_Dataset.load_pipeline(pipeline_dir)
    # fit and transform the training data using the pipeline
    X_train_transformed = pipeline.fit_transform(X_train_df)
    # transform the test data using the fitted pipeline
    X_test_transformed = pipeline.transform(X_test_df)
    
    
    logger.info("Feature Engineering ... ")
    # set up the hyper-parameter
    param_grid = {
        'n_estimators': load_configure['model_params']['n_estimators'],
        'max_features': load_configure['model_params']['max_features']
    }
    
    oob_gridsearch = gridsearch_model.OOB_ParamGridSearch(n_jobs=1,
                                                          estimator=im.IterativeRFRegression,
                                                          param_grid=param_grid,
                                                          seed=load_configure['default_seed'],
                                                          refit=True,
                                                          task="regression",
                                                          metric="mse")
    
   
    oob_gridsearch.fit(X_train=X_train_transformed.iloc[:, 1:] , y_train=y_train_df)
    #oob_gridsearch.save_model(best_model_filename)
    #oob_gridsearch.cv_result.to_csv(cv_result_filename, sep="\t")
    
    
    

    
  
   
    
    
    
  
    
    
  
    
    
        
    
    


