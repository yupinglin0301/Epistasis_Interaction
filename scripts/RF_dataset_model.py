""" The file contain the data objective for model prediction experiment """

import numpy as np
from sklearn.tree import _tree


class RF_DataModel(object):

    def __init__(self):
        """
        Initialize  prediction model/data
        """
        pass


    def all_tree_paths(self, dtree, node_id=0, depth=0):
        """
        Get all the individual tree paths from root node to the leaves
        for a decision tree classifier object
        """

        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

        left_child = dtree.tree_.children_left[node_id]
        right_child = dtree.tree_.children_right[node_id]

        if left_child != _tree.TREE_LEAF:

            left_paths = [
                np.append(node_id, path)
                for path in self.all_tree_paths(dtree, left_child, depth=depth + 1)
            ]

            right_paths = [
                np.append(node_id, path)
                for path in self.all_tree_paths(dtree, right_child, depth=depth + 1)
            ]

            paths = left_paths + right_paths

        else:
            paths = [[node_id]]

        return paths


    def get_rf_tree_data(self, rf, X_train, X_test, y_test, predictor="regress"):
        """
        Get the entire fitted random forest and its decision tree data
        as a convenient dictionary format
        """


        # random forest feature importances i.e. next iRF iteration weights
        feature_importances = rf.feature_importances_

        # standard deviation of the feature importances
        feature_importances_std = np.std(
            [dtree.feature_importances_ for dtree in rf.estimators_], axis=0)
        feature_importances_rank_idx = np.argsort(feature_importances)[::-1]


        # Create a dictionary with all random forest metrics
        # This currently includes the entire random forest fitted object
        all_rf_tree_outputs = {
            "rf_obj": rf,
            "get_params": rf.get_params,
            "feature_importances": feature_importances,
            "feature_importances_std": feature_importances_std,
            "feature_importances_rank_idx": feature_importances_rank_idx
        }


        for idx, dtree in enumerate(rf.estimators_):
            dtree_out = self.get_tree_data(
                X_train=X_train,
                X_test=X_test,
                y_test=y_test,
                dtree=dtree,
                node_id=0,
                predictor=predictor
            )

            # Append output to our combined random forest outputs dict
            all_rf_tree_outputs["dtree{}".format(idx)] = dtree_out

        return all_rf_tree_outputs

    def get_tree_data(self, X_train, X_test, y_test, dtree, predictor, node_id=0):

        """
        This returns all of the required summary results from an
        individual decision tree
        """

        value = dtree.tree_.value
        
        n_node_samples = dtree.tree_.n_node_samples

        # Get the total number of features in the training data
        tot_num_features = X_train.shape[1]

        # Get indices for all the features used - 0 indexed and ranging
        # to the total number of possible features in the training data
        all_features_idx = np.array(range(tot_num_features), dtype='int64')

        # Get the raw node feature indices from the decision tree classifier
        node_features_raw_idx = dtree.tree_.feature

        # Get the refined non-negative feature indices for each node
        node_features_idx = all_features_idx[np.array(node_features_raw_idx)]


        # Get all of the paths used in the tree
        all_leaf_node_paths = self.all_tree_paths(
                dtree=dtree, node_id=node_id
        )

        # Get all of the features used along the leaf node paths
        all_leaf_paths_features = [
            node_features_idx[path[:-1]]
            for path in all_leaf_node_paths
        ]

        # Get the unique list of features along a path
        all_uniq_leaf_paths_features = [
                np.unique(feature_path)
                for feature_path in all_leaf_paths_features
        ]

        # Get list of leaf nodes
        # In all paths it is the final node value
        all_leaf_nodes = [
            path[-1]
            for path in all_leaf_node_paths
        ]

        # Final predicted values in each class at each leaf node
        all_leaf_node_values = [
            value[node_id].astype(int)
            for node_id in all_leaf_nodes
        ]


        # Get the total number of training samples used in each leaf node
        tot_leaf_node_values = [n_node_samples[node_id].astype(int)
                                for node_id in all_leaf_nodes]

        # Get all feature index
        all_features_idx = np.array(range(tot_num_features), dtype='int64')
        
        if predictor == "regress":
            # Predicted Classes
            all_leaf_node_classes = [
                np.unique(y_test)[np.argmax(value)] 
                for value in all_leaf_node_values
            ]
        else:
            all_leaf_node_classes = [all_features_idx[np.argmax(value)] for value in all_leaf_node_values]
            

        # Dictionary of all tree values
        tree_data = {
            "node_features_idx": node_features_idx,
            "all_leaf_paths_features": all_leaf_paths_features,
            "all_uniq_leaf_paths_features": all_uniq_leaf_paths_features,
            "tot_leaf_node_values": tot_leaf_node_values,
            "all_leaf_node_classes": all_leaf_node_classes,
            "all_leaf_node_values": all_leaf_node_values
        }

        return tree_data