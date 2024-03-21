from abc import ABC, abstractmethod
from  pathlib import Path

import numpy as np
import pandas as pd
import gzip
import pickle



#repo_root = Path(__file__).resolve().parent.parent

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
    def save(self):
         """
         Save the preprocessed file
         """
         raise NotImplementedError
     
    @abstractmethod
    def load(self):
         """
         Load the preprocessed file
         """
         raise NotImplementedError
     
class TrainTestSplitMixin():
    """
    A mixin class providing train-test splitting functionality
    """
    
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
        split_i = len(y) - int(len(y) // (1 / test_size))
        X_train, X_test = X[:split_i], X[split_i:]
        y_train, y_test = y[:split_i], y[split_i:]
        
        return X_train, X_test, y_train, y_test

class GroupShuffleSplitMixin():
    """
    A mixin class providing group shuffle splitting functionality
    """
    def group_shuffle_split(self, X, y, groups, seed, n_splits=1 , test_size=0.2):
        """
        Split the data into train and test sets
        """
        
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)  
        split = gss.split(X, y, groups=groups)
        train_ids, test_ids = next(split)
        
        X_train, X_test = X[train_ids], X[test_ids]
        y_train, y_test = y[train_ids], y[test_ids]
        
        return X_train, X_test, y_train, y_test, train_ids, test_ids
    
class Repeated_GroupKFoldShuffleMixin():
    """
    A mixin class providing repeated group shuffle splitting functionality
    """
    def repeated_groupKFoldshuffle_split(self, X, y, n_splits=3, groups=None, random_state=None):
        # Find the unique groups in the dataset.
        unique_groups = np.unique(groups)

        # Shuffle the unique groups if shuffle is true.
        np.random.RandomState(random_state).shuffle(unique_groups)
        # Split the shuffled groups into n_splits.
        split_groups = np.array_split(unique_groups, n_splits)

        # For each split, determine the train and test indices.
        for test_group_ids in split_groups:
            test_mask = np.isin(groups, test_group_ids)
            train_mask = ~test_mask

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            yield train_idx, test_idx

class GTEX_raw_Dataset(ExpressionDataset, GroupShuffleSplitMixin, Repeated_GroupKFoldShuffleMixin):
    """
    A class containing logic used by all the types of gwas datasets for computing out of bag score
    The GTEX_raw_Dataset inheritance pattern from class ExpressionDataset, GroupShuffleMixin and Repeated_GroupKFoldShuffleMixin
    """

    def __init__(self, gwas_gen_dir, cov_df_dir):
        """
        An initializer for the class
        """
        self.all_gen_df = pd.read_csv(gwas_gen_dir, sep=",")
        self.all_gen_df = self.all_gen_df.drop(['FID', 'IID'], axis=1)
        self.cov_df = pd.read_csv(cov_df_dir, sep="\t")
        self.all_gwas_df = pd.concat([self.all_gen_df, self.env_df, self.cov_df], axis=1)
    

    @classmethod
    def from_config(cls, config_file, weight_tissue):
        """
        A function to create a new object from paths to its data
        """
      
        data_dir = Path(config_file['dataset']['data_dir'])
        gwas_df_dir = data_dir / weight_tissue / (weight_tissue + "_imputed.txt")
        gene_cor_dir = data_dir / "genetic_correlation.pkl.gz"
        
        return cls(gwas_df_dir, config_file['dataset']['cov_dir'])

    @staticmethod
    def save(save_df, save_file_name):
        """
        Save the preprocessed file
        """
        save_df.to_csv(save_file_name, index=0)

    @staticmethod
    def load(save_file_path):
        """
        Load the preprocessed file
        """
        preprocessed_df = pd.read_csv(save_file_path, sep='\t')
        return preprocessed_df