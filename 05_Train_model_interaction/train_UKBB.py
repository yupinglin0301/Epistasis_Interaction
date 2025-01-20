import numpy as np
import pandas as pd
import optuna
import utils
import argparse
import utils
import sys
import yaml
import time
import datetime

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import KFold

from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from Explore_interaction_term import write_configure


def sample_hyperparameters(model_fn, trial):
    """
    Define a range of values of the hyperparameters which we will be looking to optimise. 
    """
    if model_fn == XGBRegressor:
        return {
            "eta": trial.suggest_float("eta", 0.1, 1),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "lambda": trial.suggest_float("alpha", 0, 2),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, 1000)
        }

    elif model_fn == ElasticNet:
        return {
            "alpha": trial.suggest_float("alpha", 0.001, 10),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1)
        }
    else:
        raise NotImplementedError("This model has not been implemented")


def optimise_hyperparameters(model_fn, hyperparameter_trials, x, y, logger):
    """
    Take a sample of values for each hyperparameter, and define an objective function which is to be
    optimised in an attempt to approximate the minimal RMSE (within the set of hyperparameters sampled).
    """

    def objective(trial):
        """
        Perform Time series cross validation, fit a pipeline to it (equipped with the)
        selected model, and return the average error across all cross validation splits.
        """
        train_errors = []
        val_errors = []
        hyperparameters = sample_hyperparameters(model_fn=model_fn, trial=trial)
        tss = KFold()(n_splits=5)
        pipeline = make_pipeline(
            StandardScaler(),
            model_fn(**hyperparameters)
        )

        logger.warning(f"Starting Trial {trial.number}")
        # Use TSS to split the features and target variables for training and validation
        for split_number, (train_indices, val_indices) in enumerate(tss.split(x)):
            logger.info(f"Performing split number {split_number}")
            x_train, x_val = x.iloc[train_indices], x.iloc[val_indices]
            y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

            logger.info("Fitting model...")
            pipeline.fit(x_train, y_train)

            # Training error
            y_train_pred = pipeline.predict(X=x_train)
            train_error = root_mean_squared_error(y_true=y_train, y_pred=y_train_pred)
            train_errors.append(train_error)
        
            # Validation error
            y_val_pred = pipeline.predict(X=x_val)
            val_error = root_mean_squared_error(y_true=y_val, y_pred=y_val_pred)
            val_errors.append(val_error)

        
        avg_train_error = np.mean(train_errors)
        avg_val_error = np.mean(val_errors)
        # Log the errors for analysis
        logger.info(f"Average Train RMSE = {avg_train_error}")
        logger.info(f"Average Validation RMSE = {avg_val_error}")
            
        return avg_val_error

    logger.info("Beginning hyperparameter search")
    sampler = TPESampler(seed=69)
    study = optuna.create_study(study_name="study", direction="minimize", sampler=sampler, pruner=MedianPruner())
    study.optimize(func=objective, n_trials=hyperparameter_trials, n_jobs=4)

    # Get the dictionary of the best hyperparameters and the error that they produce
    best_hyperparams = study.best_params
    best_value = study.best_value

    logger.info(f"The best hyperparameters for the {model_fn} model are: {best_hyperparams}")
    logger.success(f"Best MAE Across Trials: {best_value}")

    return best_hyperparams

"""
Functions train model (xgboost, elasticnet)
    nohup python train_UKBB.py \
        --work_dir "05_Train_model_interaction" \
        --weight_tissue "Brain_Amygdala" \
        --phen_name  "Intell" \
        --phen_df_name "Phen_intell.csv" \
        --pred_df_name "Brain_Amygdala_imputed_cleaned.txt" \
        --split_df_name  "2025-01-04T14:54:43.097172_Intell_Data_splits_index.csv" \
        --model_type XGBRegressor \
        --configure_file  train_interaction_UKBB.yaml > /exeh_4/yuping/Epistasis_Interaction/05_Train_model_interaction/Log/train_interaction_UKBB.txt &
          
output:
    best parameter
"""

def process_args():
    """
    Parse and format and command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weight_tissue',
        action="store",
        help="Specify the weight tissue."
    )

    parser.add_argument(
        '--phen_name',
        action="store",
        help="Specify the phen name."
    )

    parser.add_argument(
        '--work_dir',
        action="store",
        help="Specify the working directory."
    )

    parser.add_argument(
        '--configure_file',
        action="store",
        help="Specify the configuration file."
    )

    parser.add_argument(
        '--phen_df_name',
        action="store",
        help="Specify the phen df name."
    )

    parser.add_argument(
        '--pred_df_name',
        action="store",
        help="Specify the pred df name."
    )
    
    parser.add_argument(
        '--split_df_name',
        action="store",
        help="Specify the pred df name."
    )

    parser.add_argument(
        '--model_type',
        action="store",
        help="Specify the pred df name."
    )



    args = parser.parse_args()

    return(args)


if __name__ == '__main__':
    startTime=time.time()
    
    # process command line arguments
    input_arguments = process_args()
    timestamp = datetime.datetime.now().today().isoformat()
    # set up directory
    repo_root = Path(__file__).resolve().parent.parent
    # set up working directory
    full_work_dir = Path.joinpath(repo_root, input_arguments.work_dir)
    # set up save directory
    save_dir = Path.joinpath(full_work_dir, "results")
    # set up logging
    logger = utils.logging_config(full_work_dir, f"train_UKBB:{input_arguments.model_type}:{input_arguments.weight_tissue}:{input_arguments.phen_name}", timestamp)
    # loading configure file
    configure_file = Path(full_work_dir, "model_configure", input_arguments.configure_file)
    try:
        with open(configure_file) as infile:
            load_configure = yaml.safe_load(infile)
    except Exception:
            sys.stderr.write("Please specify valid yaml file.")
            sys.exit(1)
    
    
    logger.info("Check if all files and directories exist ... ")
    if not utils.check_exist_directories([save_dir]):
        raise Exception("See output above. Problems with specified directories")
    else:
        # Create experiment directory
        experiment_dir = save_dir / input_arguments.weight_tissue 
        experiment_dir.mkdir(parents=True,  exist_ok=True)
        # Create output file name
        cv_result_filename = utils.construct_filename(experiment_dir, "cv_result",".pkl",  timestamp, input_arguments.weight_tissue, input_arguments.phen_name, "UKBB", "run_interaction")

        # Check output file is exist or not
        if utils.check_exist_files([cv_result_filename]):
            raise Exception("See output above. Problems with specified directories")
    
    logger.info("Prepareing feature ... ") 
    # specify the configure and result file dir
    write_configure(logger, 
                    configure_file=configure_file)
    
    # load x (features) dataframes
    feature_path = Path(load_configure['dataset']['data_dir']) / input_arguments.weight_tissue / input_arguments.pred_df_name 
    feature_df = pd.read_csv(feature_path)
    # load y (label) dataframes
    phen_path = Path(load_configure['dataset']['data_dir']) / input_arguments.weight_tissue / input_arguments.phen_df_name
    phen_df = pd.read_csv(phen_path)


    # Set the random state for reproducibility
    np.random.seed(load_configure['default_seed'])
    logger.info("Train-Test Splitting ... ")
    # load training data, testing data from indexes and features dataframe
    data_split_path = Path(load_configure['dataset']['data_dir']) / input_arguments.weight_tissue / input_arguments.split_df_name
    data_split_indexes = pd.read_csv(data_split_path, sep="\t", index_col=0)
    X_train_raw_df, y_train_raw_df = utils.get_dataset(feature_df, phen_df, data_split_indexes, "train")
    X_test_raw_df, y_test_raw_df = utils.get_dataset(feature_df, phen_df, data_split_indexes, "test")

    logger.info("Hyper-Parameter tunning ...")
    best_model_hyperparameters = optimise_hyperparameters(
          model_fn=,
          hyperparameter_trials=100,
          x=x_train_raw_df,
          y=y_train_raw_df["Intell"],
          logger=logger
    )
    logger.info("Done ...")