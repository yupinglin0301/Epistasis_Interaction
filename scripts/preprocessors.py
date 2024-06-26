from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

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

class CategoricalEncoder_Income(BaseEstimator, TransformerMixin):
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

class NormalizeDataTransformer(BaseEstimator, TransformerMixin):
    """
    Min-max normalize a DataFrame column
    """
    
    def __init__(self):
        self.column_max = None
        self.column_min = None

    def fit(self, X, y=None):
        self.column_max = X.max()
        self.column_min = X.min()
        return self

    def transform(self, X):
        standardized_X = X.copy()
        for col in X.columns:
            standardized_X[col] = self.standardize_column(X[col], col)
        return standardized_X

    def standardize_column(self, col, col_name):
        """Zero-one standardize a dataframe column"""
        max_val = self.column_max[col_name]
        min_val = self.column_min[col_name]
        col_range = max_val - min_val

        if col_range == 0:
            standardized_column = np.zeros(len(col))
        else:
            standardized_column = (col - min_val) / col_range

        return standardized_column