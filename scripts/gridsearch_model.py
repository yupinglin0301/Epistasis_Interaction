from sklearn.model_selection import ParameterGrid
from copy import deepcopy
import numpy as np
import pandas as pd
import joblib
import eval_metrics

class OOB_ParamGridSearch:
    def __init__(self, 
                 estimator, 
                 param_grid,
                 seed,
                 n_jobs=-1, 
                 refit=True, 
                 task="regression", 
                 metric="mse"):
        """
        Initializes the OOB_ParamGridSearch class.

       
        :param estimator (object): The base estimator to be used.
        :param param_grid (dict or list of dicts): The parameter grid to search over.
        :param seed (int): The random 
        :param n_jobs (int, optional): The number of jobs to run in parallel. Defaults to -1.
        :param refit (bool, optional): Indicates whether to refit the model with the best hyperparameters. Defaults to True.
        :param task (str, optional): The task type, either "classification" or "regression". Defaults to "classification".
        :param metric (str, optional): The evaluation metric to use. Defaults to "mse".
        """
        self.n_jobs = n_jobs
        self.seed = seed 
        self.estimator = estimator
        self.param_grid = param_grid
        self.refit = refit
        self.task = task
        self.metric = metric

    def fit(self, 
            X_train, 
            y_train):
        """
        Fits the model with the given training data using the parameter grid search.

        :param X_train (array-like): The input features for training.
        :param y_train (array-like): The target values for training.

        :return self (object): Returns self.
        """
        params_iterable = list(ParameterGrid(self.param_grid))
        parallel = joblib.Parallel(self.n_jobs)

        output = parallel(
            joblib.delayed(self.fit_and_score)(deepcopy(self.estimator), X_train, y_train, parameters)
            for parameters in params_iterable)

        output_array = np.array(output, dtype=np.float64)

        best_index = np.argmin(output_array)
        self.best_score_ = output_array[best_index]
        self.best_param_ = params_iterable[best_index]

        cv_results = pd.DataFrame(output, columns=['OOB_Error_Score'])
        df_params = pd.DataFrame(params_iterable)
        cv_results = pd.concat([cv_results, df_params], axis=1)
        cv_results["params"] = params_iterable
        self.cv_results = (cv_results.
                           sort_values(['OOB_Error_Score'], ascending=True).
                           reset_index(drop=True))

        if self.refit:
            # Final fit with best hyperparameters
            self.cv_model = deepcopy(self.estimator)(rseed=self.seed, **self.best_param_)
            self.cv_model.fit(X_train, y_train, feature_weight=None)
            self.cv_model.save_model("/exeh_4/yuping/123.pkl")

        return self

    def fit_and_score(self, 
                      estimator, 
                      X_train, 
                      y_train, 
                      parameters):
        """
        Fits the model and calculates the out-of-bag (OOB) error score.

        :param estimator (object): The estimator object.
        :param X_train (array-like): The input features for training.
        :param y_train (array-like): The target values for training.
        :param parameters (dict): The hyperparameters to use for fitting the model.

        :return oob_error (float): The calculated out-of-bag error score.
        """
        train_model = estimator(rseed=self.seed, **parameters)
        train_model.fit(X_train, y_train, feature_weight=None)
        oob_error = 1 - self.oob_score_accuracy(train_model, X_train, y_train, task=self.task, metric=self.metric)

        return oob_error

    def oob_score_accuracy(self, 
                           rf, 
                           X_train, 
                           y_train, 
                           task, 
                           metric):
        """
        Calculates the out-of-bag (OOB) score accuracy.

       
        :param rf (object): The random forest model.
        :param X_train (array-like): The input features for training.
        :param y_train (array-like): The target values for training.
        :param task (str): The task type, either "classification" or "regression".
        :param metric (str): The evaluation metric to use.

        :return oob_score (float): The calculated out-of-bag score accuracy.
        """
        from sklearn.ensemble._forest import _generate_unsampled_indices, _get_n_samples_bootstrap

        X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y = y_train.values if isinstance(y_train, pd.Series) else y_train

        if task == "classification":
            n_samples = len(X)
            n_classes = len(np.unique(y))
            predictions = np.zeros((n_samples, n_classes))
            for tree in getattr(rf, "model").estimators_:
                n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, n_samples)
                unsampled_indices = _generate_unsampled_indices(tree.random_state, n_samples, n_samples_bootstrap)

                tree_preds = tree.predict_proba(X[unsampled_indices, :])
                predictions[unsampled_indices] += tree_preds

            oob_score = eval_metrics.get_evaluation_report(predictions, y, task, metric)

            return oob_score

        else:
            n_samples = len(X)
            predictions = np.zeros(n_samples)
            n_predictions = np.zeros(n_samples)
            for tree in getattr(rf, "model").estimators_:
                n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, n_samples)
                unsampled_indices = _generate_unsampled_indices(tree.random_state, n_samples, n_samples_bootstrap)

                tree_preds = tree.predict(X[unsampled_indices, :])
                predictions[unsampled_indices] += tree_preds
                n_predictions[unsampled_indices] += 1

            predictions /= n_predictions

            oob_score = eval_metrics.get_evaluation_report(predictions, y, task, metric)

            return oob_score

