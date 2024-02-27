#  Data Preprocessing and Feature Engineering Pipeline for genetic and environment data

In this module, we have two goal : 
- Apply feature engineering techniques, which encompass tasks such as imputing missing values, generating new features, encoding features from raw data and normalized gene expression data.(Step 0)
- Regress out potential genetic relationship structure.(Step 1)


## Scripts
### Step 0 : Feature engineering on gene expression and environmental variables
```
Functions for making pipline for feature engineering

Usage:
    
    python run_making_pipline.py \
      --work_dir "/exeh_4/yuping/Epistasis_Interaction/01_Preprocessing" \
      --weight_tissue "Brain_Amygdala" \
      --normalized  
      
Output:
Feature engineering pipeline for specific imputed brain tissue and environmental factors(e.g. feature_eng_Brain_Amygdala_output.pkl)
```
**Notes:** 
+ If the "normalized" parameter is included in the arguments, the script will perform normalization on the gene expression features.
+ if the "normalized" parameter is not provided, the script will proceed with imputing missing values and encoding categorical variables in the environmental features.


### Step 1 : Regress out potential genetic relationship structure
The script is designed to utilize the following equation in order to construct Genomic Best Linear Unbiased Prediction (GBLUP), taking into consideration the influence of genetic correlation structure and subtracting the genetic relationship structure from the phenotype.




$y=1_n \mu+\sum_i W q_i+e$
+ q is the random effect for each SNPs.
+ W is a genotype matrix.

$y=1_n \mu+Z g+e$
+ Z is a design matrix.
+ g is a vector of additive genetic effects for an individual.
  
$\sum_i W q_i$ is equal to $g$

GBLUP solutions:

$\left[\begin{array}{l}\hat{\mu} \\ \hat{g}\end{array}\right]=\left[\begin{array}{cc}1_n^{\prime} 1_n & \mathbf{1}_n^{\prime} Z \\ Z^{\prime} 1_n & Z^{\prime} Z+G^{-1} \frac{\sigma_e^2}{\sigma_g^2}\end{array}\right]^{-1}\left[\begin{array}{l}1_n^{\prime} y \\ Z^{\prime} y\end{array}\right]$


```
Usage:
    
    python regress_out_grm.py \
      --phen_name "CWR_Total" \
      --weight_tissue "Brain_Amygdala" 
      
Output:
residual phenotype(e.g. CCR_Total_imputed.csv)

```

```
R script for BLUP



y <- fread("/mnt/data/share/yuping/data/phenotype_info.csv")
y <- ifelse(is.na(y$CWR_Total), mean(y$CWR_Total, na.rm=TRUE), y$CWR_Total)
y <- data.matrix(y)
G <- data.matrix(fread("test.csv"))
G <- G + diag(1046)*0.001


lambda <- 1
Ginv <- solve(G)
ones <- matrix(1, ncol=1, nrow=1046)
Z <- diag(1046)


LHS1 <- cbind(crossprod(ones), crossprod(ones, Z)) 
LHS2 <- cbind(crossprod(Z, ones), crossprod(Z) +  Ginv*lambda)
LHS <- rbind(LHS1, LHS2)
RHS <- rbind( crossprod(ones, y), crossprod(Z,y) )
sol <- solve(LHS, RHS)

          [,1]
[1,] 80.313953
[2,]  7.910075
[3,] -4.813314
[4,] 13.364069
[5,] 13.367066
[6,] 31.096525

predicted_y = Z %*%sol[-1] + sol[1]

         [,1]
[1,]  88.22403
[2,]  75.50064
[3,]  93.67802
[4,]  93.68102
[5,] 111.41048
[6,] 108.78295

==========================================================================
library("rrBLUP")
fit <- mixed.solve(y = y, K=G)
random_effect = fit$u
random_effect_matrix = as.matrix(random_effect)

      [,1]
[1,] 80.313953
[2,]  6.729155
[3,] -3.692628
[4,] 12.098731
[5,] 12.100952
[6,] 27.191630

```

## Reproducible Analysis

To reproduce the results of the coverage analysis perform the following:

```

```

## Reference 
+ https://cnsgenomics.com/data/teaching/GNGWS23/module5/Lecture3_BLUP.pdf
+ https://jyanglab.com/AGRO-932/chapters/a2.1-qg/rex11_gs2.html#16



