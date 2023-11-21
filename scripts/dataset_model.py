from abc import ABC, abstractmethod
from  pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
import sys


class ExpressionDataset(ABC):
    """ 
    The base dataset defining the API for datasets in this project
    """
    
    @abstractmethod
    def __init__(self):
        """
        Abstract initializer.
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(class_object):
        """
        A function to initialize a ExpressionDataset object
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_samples(self):
        """
        Return the sample ids for all samples in the dataset
        """
        raise NotImplementedError
        
    @abstractmethod
    def get_features(self):
        """
        Return the list of the ids of all the features in the dataset
        """
        raise NotImplementedError
    
    @abstractmethod
    def generate_labels(self):
        """
        Process the y matrix for the given phenotype trait
        """
        raise NotImplementedError
    
    @abstractmethod
    def save_pipeline(self):
         """
         Save the version of the pipline
         """
         raise NotImplementedError
     
    @abstractmethod
    def load_pipeline(self):
         """
         Load the version of the pipline
         """
         raise NotImplementedError
    
    

class TrainTestSplit(ExpressionDataset):
    """
     A base train_test_split defining the API for train-test splitting
    """
    @abstractmethod
    def train_test_split(self,
                         train_fraction,
                         test_fraction,
                         seed):
        """
        Split the dataset into two portion, 
        as seen in scikit-learn's `train_test_split` function
        """
         
        raise NotImplementedError
    
    @abstractmethod
    def shuffle_data(self, X, y, seed):
        """
        Random shuffle of the samples in X and y
        """
        raise NotImplementedError
    
    
class RF_OOB_Dataset(TrainTestSplit):
    """
    A class containing logic used by all the types of gwas datasets for computing out of bag score
    The RF_OOB_Dataset inheritance pattern from class ExpressionDataset and TrainTestSplit
    """
    
    def __init__(self,
                 gwas_gen_dir,
                 label_df_dir,
                 env_df_dir):
        
        """
        An initializer for the class
        """
        
        self.all_gen_df = pd.read_csv(gwas_gen_dir, sep=",")
        self.all_gen_df = self.all_gen_df.drop(['FID','IID'], axis=1)
        self.env_df = pd.read_csv(env_df_dir, sep="\t")
        
        self.all_gwas_df = pd.concat([self.all_gen_df, self.env_df], axis=1)
        self.label_df = pd.read_csv(label_df_dir, sep="\t")
    
     
    @classmethod 
    def from_config(class_object,
                    config_file,
                    weight_tissue):
        """
        A function to create a new object from paths to its data
        """
        
            
        data_dir = Path(config_file['dataset']['data_dir'])
        gwas_df_dir = data_dir / weight_tissue / ("predict_expression_" + weight_tissue + "_output.csv")
        
        return class_object(gwas_df_dir, config_file['dataset']['phentoype_dir'], config_file['dataset']['env_dir'])	    
    
    def get_samples(self):
        """
        Return the list of sample accessions for all samples currently available in the dataset
        """
        return list(self.all_gwas_df.index)
    
    def get_features(self):
        """
        Return the list of the ids of all the features in the currently available in the dataset 
        """
        return list(self.all_gwas_df.columns)
    
    def shuffle_data(self, X, y, seed):
        """
        Random shuffle of the samples in X and y
        """
        
        np.random.seed(seed)
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        
        return X[idx], y[idx]
    
    def train_test_split(self, X, y, seed, test_size=0.2):
        """
        Split the data into train and test sets
        """
        
        X, y = self.shuffle_data(X, y, seed)
        split_i = len(y) - int(len(y)// (1 / test_size))
        X_train, X_test = X[:split_i], X[split_i:]
        y_train , y_test = y[:split_i], y[split_i:]
        
        return X_train, X_test, y_train, y_test
    
    def generate_labels(self, phen_trait):
        """
        Random shuffle of the samples in X and y
        """
        y_given_phen = self.label_df.loc[:, [phen_trait]]
        
        return y_given_phen
    
    @staticmethod
    def save_pipeline(pipeline_to_save, save_file_name):
         """
         Save the version of the pipline
         """
         joblib.dump(pipeline_to_save, save_file_name)
         
    @staticmethod
    def load_pipeline(pipline_file_path):
         """
         Load the version of the pipline
         """		
         pipline_file_path  = joblib.load(filename=pipline_file_path)
         return pipline_file_path

