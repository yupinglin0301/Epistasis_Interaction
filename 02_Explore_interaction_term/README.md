# Determine the optimal parameter combination for a model and identify interaction terms

**LIN Yu-Ping 2024**

In this module, we utilized an iterative random forest framework to detect stable interaction terms, which were subsequently incorporated into a simple linear regression model for predicting reading ability.

Our objective is to compare the aforementioned proposed method with other penalty linear models, namely Lasso, Ridge, and Elastic Net, in order to assess whether incorporating interactions within the model enhances its explanatory power in explaining the variance in reading ability.


Each model was trained and tested using the following three-step strategy.  

+ Data splitting. We randomly **group shuffle split** the data set into a separate training set and test set (80% train, 20% test). 

+ Model training. We used **Group Kfold cross-validation** on the training set to train and optimize the model via validation. 

+ Model testing and comparison. We applied the final model to the independent test set to obtain an unbiased estimate of model performance. 









## Reproducible Analysis

To reproduce the results of the  analysis perform the following:

```bash

``````

## Reference
- https://www.medrxiv.org/content/10.1101/2022.09.14.22279940v1.full
- https://www.nature.com/articles/s43856-024-00437-7


## Notes

### 2024.04.16
1. Identidy the interaction, mainly focus on 2 or 3 combination.
   
   - Select Top feature with smallest  out of bag error in [run_IRF_TopK.py](run_IRF_TopK.py)
     - Top feature [100, 500, 1000, 2366]
     - parameter used in the script in [IRF_RF_TopK.yaml](model_configure/IRF_RF_TopK.yaml)
  
     - **Initial run** 
       - checking Log file in [logfile_2024-04-09T16:56:39.092087_Brain_AmygdalaCWR_Totalrun_IRF_TopK.log](Log/logfile_2024-04-09T16:56:39.092087_Brain_AmygdalaCWR_Totalrun_IRF_TopK.log).
       We can see most of time are spent in computing stability_score
       -  Result in [2024-04-15T22:28:07.741631_Brain_Amygdala_CWR_Total_interaction_run_IRF_TopK_result.csv](results/Brain_Amygdala/2024-04-15T22:28:07.741631_Brain_Amygdala_CWR_Total_interaction_run_IRF_TopK_result.csv)
  
     - Second run 
       - checking Log file in [logfile_2024-04-15T22:28:07.741631_Brain_AmygdalaCWR_Totalrun_IRF_TopK.log](Log/logfile_2024-04-15T22:28:07.741631_Brain_AmygdalaCWR_Totalrun_IRF_TopK.log).
       We can see most of time are spent in computing stability_score
       -  Result in [2024-04-15T22:28:07.741631_Brain_Amygdala_CWR_Total_interaction_run_IRF_TopK_result.csv](results/Brain_Amygdala/2024-04-15T22:28:07.741631_Brain_Amygdala_CWR_Total_interaction_run_IRF_TopK_result.csv)

   - Using all the feature in [run_IRF.py](run_IRF.py)  
     - parameter used in the script in [IRF_RF_test.yaml](model_configure/IRF_RF_test.yaml)
     - **Intial run** 
       - checking Log file in [logfile_2024-04-09T16:56:39.092087_Brain_AmygdalaCWR_Totalrun_IRF_TopK.log](Log/logfile_2024-04-16T15:47:28.802255_Brain_AmygdalaCWR_Totalrun_IRF.log).
       We can see most of time are spent in computing stability_score
       -  Result in [2024-04-16T15:47:28.802255_Brain_Amygdala_CWR_Total_interaction_run_IRF_result.csv](results/Brain_Amygdala/2024-04-16T15:47:28.802255_Brain_Amygdala_CWR_Total_interaction_run_IRF_result.csv)
   - ![OpenAI Logo](pnas.1711236115fig01.jpeg)

2. Permutation p-value for IRF
    
   - Permutation Procedure
     + step1 : permute (ie shuffle) the outcome label (eg reading score)
     + step2:  repeat the entire algorithm
     + step3: record the stability score in each permutation.
       + Focus on the interaction term exist in observed dataset.
       + if the interaction term is not found, simply set stability score = 0   
      ```
      Observed_data = {'a':4, 'b':5}
      Perm_data = [{'a': 4, 'b': 5, 'd': 6}, {'a': 7, 'c': 8}]  # Example list of dictionaries
      final_perm_data = [{'a': 4, 'b': 5}, {'a': 7, 'b': 0}]
      ```
   - Permutation p-value for IRF script in [permutation_IRF.py](permutation_IRF.py)
   - Group shuffle y or not


### 2024.05.06
   - Conduct permuation test to assign p-value for each interaction term
     + 1000 round
     + Using interaction term Result in [2024-04-16T15:47:28.802255_Brain_Amygdala_CWR_Total_interaction_run_IRF_result.csv](results/Brain_Amygdala/2024-04-16T15:47:28.802255_Brain_Amygdala_CWR_Total_interaction_run_IRF_result.csv)
     +   checking Log file in [logfile_2024-04-25T18:21:20.319339_Brain_AmygdalaCWR_TotalPermutation_Test.log]( logfile_2024-04-25T18:21:20.319339_Brain_AmygdalaCWR_TotalPermutation_Test.log)
   - Compute heritability using predicted gene expression [03_Compute_heritability](../03_Compute_heritability/READ.md) 
### Future plan
![OpenAI Logo](123.png)
  - compare the aforementioned proposed method with other penalty linear models, namely Lasso, Ridge, and Elastic Net, in order to assess whether incorporating interactions within the model enhances its explanatory power in explaining the variance in reading ability.
  -  Apply model with selected optimal parameter on all train data to obtain SNPs and weights -> GWAS Summary Staticis.
  - UK Biobak
       - Self-reported race (Data-Field 21000) is coded first by the larger race group  
         + EUR = European ancestry, CSA = Central/South Asian ancestry, AFR = African ancestry,EAS = East Asian ancestry,MID = Middle Eastern ancestry, AMR = Admixed American ancestry
       - Covariate 
         - first 10 genetic principal components (Data-Field 22009)
         - genetic relatedness factor (Data-Field 22012) => BLUP
         - age, sex, Age * Sex, Age2
       - Phenotype
         - Fluid Intelligence score from UK Biobank [Data-Field 20016](https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=20016)
  