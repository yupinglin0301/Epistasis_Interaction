""" A module containing the models to be trained on genomic data"""

import pickle
import pandas as pd
from abc import ABC, abstractmethod
from irf.ensemble import RandomForestClassifier
from irf.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


class Regression_Eval(object):
    
    def get_metrics(self,
                    y_train_df,
                    y_test_df,
                    y_cv_df,
                    y_pred_train,
                    y_pred_test,
                    **kwargs):
            
            """
            Get regression metric values for given model predictions
            """

            train_metrics = self.get_continuous_metrics(y_train_df.status, y_pred_train)
            cv_metrics = self.get_continuous_metrics(y_train_df.status, y_cv_df)
            test_metrics = self.get_continuous_metrics(y_test_df.status, y_pred_test)

            columns = list(train_metrics.keys()) + ['data_type'] + list(kwargs.keys())
            train_metrics = list(train_metrics.values()) + ['train'] + list(kwargs.values())
            cv_metrics = list(cv_metrics.values()) + ['cv'] + list(kwargs.values())
            test_metrics = list(test_metrics.values()) + ['test'] + list(kwargs.values())

            return pd.DataFrame([train_metrics, cv_metrics, test_metrics], columns=columns)


    def get_continuous_metrics(self, y_true, y_pred):
    
            r2 = r2_score(y_true, y_pred)
            return {'r2': r2}


class GWAS_Model(ABC):
    """ 
    A model API similar to the scikit-learn API that will specifiy the
    base acceptable functions for modles in this module's benchmarking code
    """
    def __init__(self):
        """
        Standard model init function.
        """
        pass

    @abstractmethod
    def fit(self, X_train, Y_train):
        """
        Create and fit the iterative random forest classifier
        
        :param X_train: training data
        :param Y_train: training label
        :return: The fitted model 
        """

        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def load_model(model_path):
        """
        Read a pickled model from a file and return it
        
        :param model_path: The location where the model is stored
        :return: The model saved at `model_path`
        """

        raise NotImplementedError

    @abstractmethod
    def save_model(self, outputpath):
        """
        Write a model to a file
        
        :param outputpath: the path to the file to write the model to
        :return: None
        """

        raise NotImplementedError

class IterativeRFClassifier(GWAS_Model):
    """
    A Iterative Random Forest Classifier implementation designed to hew closely to the skl defaults
    """
    def __init__(self, rseed, **model_kwargs):
        """
        Model initialization function
        
        :param seed: The random seed to use in training
        :param model_kwargs: kwargs arguments to pass to iterative random forest classifier
        :return: None
        """

        self.rseed = rseed
        self.model_kwargs = model_kwargs
        self.model = None

    def fit(self, X_train, Y_train, feature_weight=None):
        """
        Create and fit the iterative random forest classifier
        
        :param X_train: training data
        :param Y_train: training label
        :param feature_weight: weights for each feature
        :return: The fitted model 
        """

        self.model = RandomForestClassifier(random_state=self.rseed, **self.model_kwargs)
        self.model = self.model.fit(X_train, Y_train, feature_weight=feature_weight)
        return self

    def save_model(self, out_path):
        """
        Write the classifier to a file
        
        :param out_path: The path to the file to write the classifier to
        :return: None
        """

        with open(out_path, "wb") as out_file:
            pickle.dump(self, out_file)
    
    def predict(self, X_train):
        """
        Predict class probability from data
        
        :param X_train: training data
        """
        assert self.model is not None, "Need to fit model first"
        return self.model.predict_proba(X_train)
    
    def predict_proba(self, X_train):
        """
        Predict class labels from data
        
        :param X_train: training data
        """
        assert self.model is not None, "Need to fit model first"
        return self.mdoel.predict(X_train)
    

    @staticmethod
    def load_model(model_path):
        """
        Read a pickled model from a file and return it
        
        :param model_path The location where the model is 
        :return: The model saved at `model_path`
        """

        with open(model_path, "rb") as model_file:
            return pickle.load(model_file)
    
  
class IterativeRFRegression(GWAS_Model, Regression_Eval):
    """
    A Iterative Random Forest Regression implementation designed to hew closely to the skl defaults
    """
    def __init__(self, rseed, **model_kwargs):
        """
        Model initialization function
        
        :param seed: The random seed to use in training
        :param model_kwargs: kwargs arguments to pass to iterative random forest classifier
        :return: None
        """

        self.rseed = rseed
        self.model_kwargs = model_kwargs
        self.model = None

    def fit(self, X_train, Y_train, feature_weight=None):
        """
        Create and fit the iterative random forest regression
        
        :param X_train: training data
        :param Y_train: training label
        :param feature_weight: weights for each feature
        :return: The fitted model 
        """

        self.model = RandomForestRegressor(random_state=self.rseed, **self.model_kwargs)
        self.model = self.model.fit(X_train, Y_train, feature_weight=feature_weight)
        return self


    def save_model(self, out_path):
        """
        Write the regressor to a file
        
        :param out_path: The path to the file to write the classifier to
        :return: None
        """

        with open(out_path, "wb") as out_file:
            pickle.dump(self, out_file)
    
    def predict(self, X_train):
        """
        Predict target value based on X_train
        """
        
        assert self.model is not None, "Need to fit model first"
        return self.model.predict(X_train)
        
    @staticmethod
    def load_model(model_path):
        """
        Read a pickled model from a file and return it
        
        :param model_path The location where the model is 
        :return: The model saved at `model_path`
        """

        with open(model_path, "rb") as model_file:
            return pickle.load(model_file)