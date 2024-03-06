# Determine the optimal parameter combination for a model and identify interaction terms



**LIN Yu-Ping 2024**

In this module, we utilized an iterative random forest framework to detect stable interaction terms, which were subsequently incorporated into a simple linear regression model for predicting reading ability.

Our objective is to compare the aforementioned proposed method with other penalty linear models, namely Lasso, Ridge, and Elastic Net, in order to assess whether incorporating interactions within the model enhances its explanatory power in explaining the variance in reading ability.


Each model was trained and tested using the following three-step strategy.  

+ Data splitting. We randomly **group shuffle split** the data set into a separate training set and test set (80% train, 20% test). 

+ Model training. We used **Group Kfold cross-validation** on the training set to train and optimize the model via validation. 

+ Model testing and comparison. We applied the final model to the independent test set to obtain an unbiased estimate of model performance. 


To account for any variability that may have been introduced by the random state of the train-internal test split, we conducted bootstrap resampling by generating fifty bootstrap samples, re-fitting the models on each bootstrap train set, and evaluating their performance on each bootstrap test set. All confidence intervals generated represent the 95% confidence intervals derived from bootstrap resampling. We also validated each model’s performance on its respective external test set, which we set aside during data splitting.







## Reproducible Analysis

To reproduce the results of the  analysis perform the following:

```bash

``````

## Reference
- https://www.medrxiv.org/content/10.1101/2022.09.14.22279940v1.full
- https://www.nature.com/articles/s43856-024-00437-7


## Note
