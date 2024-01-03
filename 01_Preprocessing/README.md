#  Data Preprocessing and Feature Engineering Pipeline for genetic and environment data

In this module, we have two goal : 
- Apply feature engineering techniques, which encompass tasks such as imputing missing values, generating new features, and encoding features from raw data.(Step 0)
- Regress out potential genetic relationship structure and normalized gene expression.(Step 1)


## Scripts
### Step 0 : Feature engineering on environmental variables.
```
Functions for making pipline for feature engineering

Usage:
    
    python run_making_pipline.py \
      --work_dir "/exeh_4/yuping/Epistasis_Interaction/01_Preprocessing" \
      --weight_tissue "Brain_Amygdala" \
      --normalized  
      
Output:
Feature engineering pipeline for specific imputed brain tissue and environmental factors
```



### Step 1 : Regress out potential genetic relationship structure
The script is designed to utilize the following equation in order to construct Genomic Best Linear Unbiased Prediction (GBLUP), taking into consideration the influence of genetic correlation structure and subtracting the genetic relationship structure from the phenotype.

$\left[\begin{array}{l}\hat{\mu} \\ \hat{g}\end{array}\right]=\left[\begin{array}{cc}1_n^{\prime} 1_n & \mathbf{1}_n^{\prime} Z \\ Z^{\prime} 1_n & Z^{\prime} Z+G^{-1} \frac{\sigma_e^2}{\sigma_g^2}\end{array}\right]^{-1}\left[\begin{array}{l}1_n^{\prime} y \\ Z^{\prime} y\end{array}\right]$


```
Usage:
    
    python regress_out_grm.py \
      --work_dir "/exeh_4/yuping/Epistasis_Interaction/01_Preprocessing" \
      --phen_name "CCR_Total" \
      --weight_tissue "Brain_Amygdala" 
      
Output:
residual phenotype
```
## Reproducible Analysis

To reproduce the results of the coverage analysis perform the following:

```

```





