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
     + Using interaction term in [2024-04-16T15:47:28.802255_Brain_Amygdala_CWR_Total_interaction_run_IRF_result.csv](results/Brain_Amygdala/2024-04-16T15:47:28.802255_Brain_Amygdala_CWR_Total_interaction_run_IRF_result.csv)
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



### 2024.05.20
Workflow Overview


1. Input a numeric feature matrix x (`Agyer`, `Gender`, `Income`, `Total_Education`, `Gene_expression`)and a response vector y (`CWR_Total`)

2. Iteratively train n.iter random forests
   - Populate the weight vector rep(1, ncol(x)), which indicating the probabilty each feature would be chosen when training the random forests.
   - Train a random forest with x and y, and save it for later use.
   - Update select.prob with the Gini importance of each feature, so that the more prediction accuracy a certain feature provides, the more likely it will be selected in the next iteration.
   - Repeat this routine n.iter times.
   -  n_estimators: [500] max_features: [700] max_depth: [2] 
  
3. Run Random Intersection Tree 
    - We apply the generalized RIT to the last feature-weighted RF grown in iteration K. 
    - decision rules generated in the process of fitting RF(W(K)) provide the mapping from continuous or categorical to binary features required for the RIT.
    -  n_bootstrapped: [1000] propn_n_samples: [0.5] n_intersection_tree: [500] max_depth_rit: [5] num_splits: [2]

4. Calculate stability score for the interactions



5.  We consider a parametric model, assuming additive effects, for both SNP-SNP and SNP-environment interaction effects for linear regression, and construct a hypothesis test to infer the presence of interactions. For the test of SNP-SNP interactions between two SNPs a and b, the null model will be:

$\begin{aligned}  \mathbf{x }_{i,c}^{T}\gamma + \alpha g_{i,a} + \beta g_{i,b}, \end{aligned}
$

The corresponding alternative model incorporating an additive interaction effect will be (`SNP-SNP interaction`):


$\begin{aligned}  \mathbf{x }_{i,c}^{T}\gamma + \alpha g_{i,a} + \beta g_{i,b} + \nu g_{i,a} g_{i,b}. \end{aligned}$

The corresponding SNP-SBPOalternative model incorporating an additive interaction effect will be(`SNP-Env interaction`):

$\begin{aligned}  \mathbf{x }_{i,c}^{T}\gamma + \alpha g_{i,a} + \beta _e x_{i,e} + \phi g_{i,a} x_{i,e}, \end{aligned}
$

- For the testing of the interactions we apply the linear F-statistic to test the null hypothesis that 
   - Null hypothesis: for SNP-SNP interactions $\begin{aligned}  \nu \end{aligned}$ or for SNP–environment interactions  $\begin{aligned}   \phi \end{aligned}$ is 0
   - Alternate  hypothesis: $\begin{aligned}  \nu \end{aligned}$ or  $\begin{aligned}   \phi \end{aligned}$ is greater than 0


$F^*=\left(\frac{S S E(R)-S S E(F)}{d f_R-d f_F}\right) \div\left(\frac{S S E(F)}{d f_F}\right)$

6. For detail [F-test_interaction.ipynb](F-test_interaction.ipynb) and [run_IRF.py](run_IRF.py)