from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from pathlib import Path
import dataset_model as dm
import pandas as pd
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

class NumericalImputer(BaseEstimator, TransformerMixin):
        """
        Numerical missing value imputer
        """
      
        def __init__(self, variable=None):
            if not isinstance(variable, list):
                self.variables = [variable]
            else:
                self.variables = variable
                
        def fit(self, X, y=None):
            self.imputer_dict_ = {}
            for feature in self.variables:
                self.imputer_dict_[feature] = X[feature].mean()
            return self
        
        def transform(self, X):
            X =X.copy()
            for feature in self.variables:
                X[feature].fillna(self.imputer_dict_[feature], inplace=True)
            
            return X
            
class CategoricalImputer_Education(BaseEstimator, TransformerMixin):
    """
    Categorical missing value imputer for Education 
    """

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        self.imputer_dict = {}
        X['MotherEducation'].replace("#NULL!", pd.NA, inplace=True)
        X['FatherEducation'].replace("#NULL!", pd.NA, inplace=True)
        for feature in self.variables:
            self.imputer_dict[feature] = X[feature].mode()[0]
        return self
    
    def transform(self, X):
        
        for feature in self.variables:
            X[feature] = X[feature].fillna(self.imputer_dict[feature])
        
        X['TotalEducation'] = X.apply(lambda x: (int(x['MotherEducation']) + int(x['FatherEducation']))/2, axis=1)
        median = X.TotalEducation.median()
        X['TotalEducation'] = X['TotalEducation'].apply(lambda x: 0 if x < median else 1)
        
        return X

class CatgoricalEncoder_Income(BaseEstimator, TransformerMixin):
    """
    String to numbers categorical encoder for Income
    """
    
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self, y=None):
        self.imputer_dict = {}
        for feature in self.variables:
            self.imputer_dict[feature] = {"<1000": "0",
                                          "10001~20000": "1",
                                          "20001~30000": "2",
                                          "30001~40000": "3",
                                          "40001~50000": "4",
                                          ">50001": "5"}
        return self  
    
    def transform(self, X, y=None):
        for feature in self.variables:
            X[feature] = X[feature].map(self.imputer_dict[feature])
            
            if X[feature].isnull().any():
                X[feature].replace("#NULL!", pd.NA, inplace=True)
                X[feature].fillna(X[feature].mode()[0], inplace=True)
        
        return X

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
        output_filename = utils.construct_filename(experiment_dir, "output", ".pkl", input_arguments.weight_tissue, "pipline", "version1")
        # Check output file is exist or not
        if utils.check_exist_files(output_filename):
            raise Exception("Results file " + str(output_filename) + " exist already. Please double-check.")
    
    logger.info("Creating pipline {}... ".format(input_arguments.weight_tissue))

    pipeline = make_pipeline(
         CategoricalImputer_Education(variables=['FatherEducation', 'MotherEducation']),
         CatgoricalEncoder_Income(variables=['Income'])
    )
    dm.RF_OOB_Dataset.save_pipeline(pipeline, output_filename)