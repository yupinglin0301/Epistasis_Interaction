""" The file contain the data objective for model prediction experiment """

import numpy as np
from functools import partial
from sklearn.tree import _tree
from scipy import stats


class RITNode(object):

    def __init__(self, val):
        self._val = val
        self._children = []

    @property
    def children(self):
        return self._children

    def add_child(self, val):
        val_intersect = np.intersect1d(self._val, val)
        self._children.append(RITNode(val_intersect))

    def is_empty(self):
        return len(self._val) == 0

    def _traverse_depth_first(self, _idx):
        yield _idx[0], self
        for child in self.children:
            _idx[0] += 1
            yield from child._traverse_depth_first(_idx=_idx)


class RITTree(RITNode):

    def __init__(self, val):
        super().__init__(val)

    def traverse_depth_first(self):
        yield from self._traverse_depth_first(_idx=[0])


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


    def get_rf_tree_data(self, rf, X_train, X_test, y_test):
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


        # get all the validation rf_metrics
        #rf_validation_metrics = get_validation_metrics(inp_class_reg_obj=rf,
        #                                           y_true=y_test,
        #                                           X_test=X_test)


        # Create a dictionary with all random forest metrics
        # This currently includes the entire random forest fitted object
        all_rf_tree_outputs = {
            "rf_obj": rf,
            "get_params": rf.get_params,
            #"rf_validation_metrics": rf_validation_metrics,
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
                node_id=0
            )

            # Append output to our combined random forest outputs dict
            all_rf_tree_outputs["dtree{}".format(idx)] = dtree_out

        return all_rf_tree_outputs

    def get_tree_data(self, X_train, X_test, y_test, dtree, node_id=0):

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


        # Total number of training samples used in the prediction of
        # each class at each leaf node
        #tot_leaf_node_values = [
        #        np.sum(leaf_node_values)
        #        for leaf_node_values in all_leaf_node_values
        #]
        
        # Get the total number of training samples used in each leaf node
        tot_leaf_node_values = [n_node_samples[node_id].astype(int)
                                for node_id in all_leaf_nodes]

        # Get all feature index
        #all_features_idx = np.array(range(tot_num_features), dtype='int64')
        
        # Predicted Classes
        all_leaf_node_classes = [
            np.unique(y_test)[np.argmax(value)] 
            for value in all_leaf_node_values
        ]
        
        # Dictionary of all tree values
        tree_data = {
            "node_features_idx": node_features_idx,
            "all_leaf_paths_features": all_leaf_paths_features,
            "all_uniq_leaf_paths_features": all_uniq_leaf_paths_features,
            "tot_leaf_node_values": tot_leaf_node_values,
            "all_leaf_node_classes": all_leaf_node_classes
        }

        return tree_data



class RIT_DataModel(object):

    def __init__(self):
        """
        Initialize  prediction model/data
        """
        pass

    def build_tree(self,
                   feature_paths,
                   max_depth=3,
                   num_splits=5,
                   noisy_split=False,
                   _parent=None,
                   _depth=0):

        expand_tree = partial(
            self.build_tree,
            feature_paths,
            max_depth=max_depth,
            num_splits=num_splits,
            noisy_split=noisy_split
        )

        if _parent is None:
            tree = RITTree(next(feature_paths))
            expand_tree(_parent=tree, _depth=0)
            return tree
        else:
            _depth += 1
            if _depth >= max_depth:
                return
            for i in range(num_splits):
                _parent.add_child(next(feature_paths))
                added_node = _parent.children[-1]
                if not added_node.is_empty():
                    expand_tree(_parent=added_node, _depth=_depth)


    def feature_paths_generator(self,
                                all_paths,
                                all_weights):

        weights = np.array(all_weights)
        # normalize the weights
        weights = weights / weights.sum()
        dist = stats.rv_discrete(values=(range(len(weights)), weights))
        while True:
            yield all_paths[dist.rvs()]

    def filter_leaves_classifier(self,
                                 dtree_data,
                                 bin_class_type):

        # Filter based on the specific value of the leaf node classes
        leaf_node_classes = dtree_data['all_leaf_node_classes']

        if bin_class_type is not None:

            # unique feature paths from root to leaf node
            unique_feature_paths = [
                i for i, j in zip(dtree_data['all_uniq_leaf_paths_features'],
                              leaf_node_classes) if j == bin_class_type
             ]

            # total number of training samples ending up at each node
            tot_leaf_node_values = [
                i for i, j in zip(dtree_data['tot_leaf_node_values'],
                              leaf_node_classes) if j == bin_class_type
            ]
        else:
            unique_feature_paths = list(dtree_data['all_uniq_leaf_paths_features'])
            tot_leaf_node_values = list(dtree_data['tot_leaf_node_values'])
            
        all_filtered_output = {
            "Unique_feature_paths": unique_feature_paths,
            "tot_leaf_node_values": tot_leaf_node_values
        }

        return all_filtered_output

    def generate_rit_samples(self,
                             all_rf_tree_data,
                             bin_class_type):

        # Number of decsion tree
        n_estimators = all_rf_tree_data['rf_obj'].n_estimators
        all_weights = []
        all_paths = []

        for dtree in range(n_estimators):
            fitered = self.filter_leaves_classifier(
                dtree_data=all_rf_tree_data['dtree{}'.format(dtree)],
                bin_class_type=bin_class_type
            )
            all_paths.extend(fitered["Unique_feature_paths"])
            all_weights.extend(fitered["tot_leaf_node_values"])

        return self.feature_paths_generator(all_paths, all_weights)

    def get_rit_tree_data(self,
                          all_rf_tree_data,
                          bin_class_type=1,
                          M=10,
                          max_depth=3,
                          noisy_split=False,
                          num_splits=2):
        """
        Random Intersection Trees (RIT) algorithm
        """

        all_rit_tree_outputs = {}
        for idx, rit_tree in enumerate(range(M)):

            # Create the weighted randomly sample path as a generator
            gen_random_leaf_paths = self.generate_rit_samples(
                all_rf_tree_data=all_rf_tree_data,
                bin_class_type=bin_class_type
            )

            # Create the RIT object
            rit = self.build_tree(
                feature_paths=gen_random_leaf_paths,
                max_depth=max_depth,
                noisy_split=noisy_split,
                num_splits=num_splits
            )

            # Get the intersected node values
            rit_intersected_values = [
                node[1]._val for node in rit.traverse_depth_first()
            ]

            rit_output = {
                "rit": rit,
                "rit_intersected_values": rit_intersected_values
            }

            # Append output to our combined random forest outputs dict
            all_rit_tree_outputs["rit{}".format(idx)] = rit_output

        return all_rit_tree_outputs


    def rit_interactions(self, all_rit_tree_data):

        interactions = []
        # loop through all the random intersection tree
        for tree_data in all_rit_tree_data.values():
            intersected_values = tree_data['rit_intersected_values']
            # loop through all found interactions
            for value in intersected_values:
                if len(value) != 0:
                    intersection_str = "_".join(map(str, value))
                    interactions.append(intersection_str)

        return(set(interactions))

    def get_stability_score(self, all_rit_bootstrap_output):


        bootstrap_interact = []
        B = len(all_rit_bootstrap_output)

        for b in range(B):
            rit_counts = self.rit_interactions(all_rit_bootstrap_output['rf_bootstrap{}'.format(b)])
            bootstrap_interact.append(list(rit_counts))

        all_rit_interactions = [item for sublist in bootstrap_interact for item in sublist]
        stability = {m: all_rit_interactions.count(m) / B for m in all_rit_interactions}

        return stability
