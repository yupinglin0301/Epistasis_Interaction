import sklearn
import numpy as np
def get_evaluation_report(y_pred, y_true, task, metric):
    """
    Get values for common evaluation metrics

    :param y_pred: predicted values
    :param y_true: true values
    :param task: ML task to solve
    :param metic: choose specificed metric to assess the performance

    :return: dictionary with specificed metrics
    """
   
    if task == 'classification':
        average = 'micro' if len(np.unique(y_true)) > 2 else 'binary'
        eval_report_dict = {
            'auroc': sklearn.metrics.roc_auc_score(y_true=y_true, y_pred=y_pred, average=average),
            'aupr': sklearn.metrics.average_precision_score(y_true=y_true, y_pred=y_pred, average=average)
        }
        eval_report_dict = eval_report_dict[metric]
    else:
        eval_report_dict = {
            'mse': sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred),
            'rmse': sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False),
            'r2_score': sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred),
        }
        eval_report_dict = eval_report_dict[metric]
        
    return eval_report_dict
