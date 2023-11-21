"""
Functions for running iterative ranodom forest analysis.
"""

import numpy as np
import argparse
import utils 
import model
import yaml


from math import ceil
from RF_dataset_model import RF_DataModel, RIT_DataModel
from sklearn.utils import resample


#python run_irf.py \
#   --supervised_config "/exeh_4/yuping/Epistasis_Interaction/01_Discover_Interaction/supervised_configure.yaml" 


def run_iRF(X_train,
            X_test,
            y_train,
            y_test,
            rf_bootstrap=None,
            initial_weights=None,
            K=1,
            B=10,
            random_state_classifier=2018,
            propn_n_samples=0.2,
            bin_class_type=1,
            M=4,
            max_depth=2,
            noisy_split=False,
            num_splits=2,
            **supervised_params):
    
    predictor = supervised_params['predictor']

    # Set the random state for reproducibility
    np.random.seed(random_state_classifier)

    # Convert the bootstrap resampling proportion to the number
    # of rows to resample from the training data
    n_samples = ceil(propn_n_samples * X_train.shape[0])

    # I Random Forest data
    all_K_iter_rf_data = {}

    # Initialize dictionary of rf weights
    all_rf_weights = {}

    # Initialize dictionary of bootstrap rf output
    all_rf_bootstrap_output = {}

    # Initialize dictionary of bootstrap RIT output
    all_rit_bootstrap_output = {}

    model_list = {
        'classify': model.IterativeRFClassifier,
        'regress': model.IterativeRFRegression
    }

    # hyperparameter to tune
    model_params = {
        'n_estimators': supervised_config['n_estimators'],
        'max_features': supervised_config['max_features']        
    }

    # Create the model
    train_model = model_list[predictor](rseed=random_state_classifier, **model_params)

    # Loop through number of iteration
    for k in range(K):

        if k == 0:

            # Initially feature weights are None
            feature_importances = initial_weights

            # Update the dictionary of all our RF weights
            all_rf_weights["rf_weight{}".format(k)] = feature_importances

            # fit the model
            train_model.fit(X_train=X_train,
                            Y_train=y_train,
                            feature_weight=None)

            # Update feature weights using the
            # new feature importance score
            feature_importances = getattr(train_model,"model").feature_importances_

            # Load the weights for the next iteration
            all_rf_weights["rf_weight{}".format(k + 1)] = feature_importances

        else:

            # fit weighted RF
            # Use the weights from the previous iteration
            train_model.fit(
                X_train=X_train,
                Y_train=y_train,
                feature_weight=all_rf_weights["rf_weight{}".format(k)])

            # Update feature weights using the
            # new feature importance score
            feature_importances = getattr(train_model, "model").feature_importances_

            # Load the weights for the next iteration
            all_rf_weights["rf_weight{}".format(k + 1)] = feature_importances

        all_K_iter_rf_data["rf_iter{}".format(k)] = RF_DataModel().get_rf_tree_data(rf=train_model.model,
                                                                                    X_train=X_train,
                                                                                    X_test=X_test,
                                                                                    y_test=y_test)

    # Run the RITs
    if rf_bootstrap is None:
        rf_bootstrap = train_model

    # Loop through number of bootstrap sample
    for b in range(B):
        X_train_rsmpl, y_rsmpl = resample(X_train,

                                          y_train,
                                          n_samples=n_samples)

        # Set up the weighted random forest
        # Using the weight from the (K-1)th iteration
        rf_bootstrap.fit(
            X_train=X_train_rsmpl,
            Y_train=y_rsmpl,
            feature_weight=all_rf_weights["rf_weight{}".format(K)]
        )

        # All RF tree data
        all_rf_tree_data = RF_DataModel().get_rf_tree_data(
            rf=rf_bootstrap.model,
            X_train=X_train_rsmpl,
            X_test=X_test,
            y_test=y_test
        )

        # Run RIT on the interaction rule set
        all_rit_tree_data = RIT_DataModel().get_rit_tree_data(
            all_rf_tree_data=all_rf_tree_data,
            bin_class_type=bin_class_type,
            M=M,  # number of RIT 
            max_depth=max_depth,  # Tree depth for RIT
            noisy_split=noisy_split,
            num_splits=num_splits  # number of children to add
        )

        # Updata the rf bootstap output dictionary for rit object
        all_rit_bootstrap_output['rf_bootstrap{}'.format(b)] = all_rit_tree_data

    stability_score = RIT_DataModel().get_stability_score(
        all_rit_bootstrap_output=all_rit_bootstrap_output)

    return stability_score


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--supervised_configure', 
        action="store", 
        help="The yaml formatted dataset configuration file."
    )

    
    args = parser.parse_args()
   
    logger = utils.logging_config("gene")
    
    with open(args.supervised_configure) as infile:
        supervised_config = yaml.safe_load(infile)
    
    supervised_params = {
         'predictor': supervised_config['predictor'],
         'n_estimators' :  supervised_config['n_estimators'],
         'max_features' : supervised_config['max_features']
    }
    
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    raw_data = load_boston()

    X_train, X_test, y_train, y_test = train_test_split(raw_data.data, raw_data.target, train_size=0.5, random_state=2018)

    stability= run_iRF(
         X_train=X_train,
         X_test=X_test,
         y_train=y_train,
         y_test=y_test,
         K=5,
         B=30,
         random_state_classifier=supervised_config['seed'],
         propn_n_samples=.2,
         bin_class_type=None,
         M=20,
         max_depth=5,
         noisy_split=False,
         num_splits=2,
         **supervised_params
    )

    print(stability)
