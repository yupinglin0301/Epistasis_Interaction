


def write_configure_run_IRF(logger, 
                            configure_file, 
                            oob_rst_file, 
                            interaction_rst_file,
                            phen_file,
                            predictor_file):
    """
    Save model hyperparameters/metadata to output directory.
    """

   
    logger.info('+++++++++++ File INFORMATION +++++++++++')
    logger.info("Parameter file dir: {}".format(configure_file))
    logger.info("Out of bag error result dir: {}".format(oob_rst_file))
    logger.info("Interaction result dir: {}".format(interaction_rst_file))
    logger.info("Phenotype file dir: {}".format(phen_file))
    logger.info("Predictor file dir: {}".format(predictor_file))
            

def group_shuffle_split(X, y, groups, seed, n_splits=1 , test_size=0.2):
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

def get_rit_counts(b, all_rit_bootstrap_output, column_name):
    """
    Get each bootstrapping  interaction term
    """  
    from RF_dataset_model import RF_DataModel, RIT_DataModel 
    rit_counts = RIT_DataModel().rit_interactions(all_rit_bootstrap_output['rf_bootstrap{}'.format(b)], column_name)
    
    
    return list(rit_counts)