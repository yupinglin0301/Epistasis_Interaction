#  Preprocessing 

In this module, we construct ETL pipelie for both phentype and predictor and we later use the processed file in `Select_Paramter_Model`


## Scripts
### Step 0 : ETL pipeline for phenotype

```
Usage:
    
    python etl_for_phenotype.py \
      --phen_name CWR_Total 
Output:
residual phenotype for specifice phenotype store in sqlite format
```

**Notes:** 
+ The ETL pipeline in the transform function has two steps: NumericalImputer and compute_expected_value. Here are the details of each step:

  + NumericalImputer: This transformer is responsible for imputing missing values in the dataset. It utilizes the median imputation method, where the median value of each numerical feature is used to fill in the missing data. By using the median, we aim to impute missing values in a way that minimally affects the overall distribution of the data.

  + compute_expected_value: This step involves regressing out the family structure from the dataset. The precise details of this process can be found in the ETL.ipynb notebook and the "Additional Note" section. This step helps remove any potential confounding effects related to family structure, enabling a more accurate analysis of brain tissue and environmental factors.

+ For detail can refer to [ETL.ipynb](ETL.ipynb) and `Additional Note`.

### Step 1 : ETL pipeline for predictor
```
Usage:
    
    python etl_for_predictor.py \
      --weight_tissue "Brain_Amygdala" 
      
Output:
normalized specific imputed brain tissue and environmental factors
```
**Notes:** 

+ The ETL pipeline in the transform function includes several steps to normalize and process the data. Here are the details of each step:

  + `NormalizeDataTransformer``: This transformer applies Min-Max normalization to the gene expression feature. Min-Max normalization scales the values of the gene expression feature to a specific range (e.g., between 0 and 1) to ensure consistency and facilitate further analysis.

  + `CategoricalEncoder_Education``: This transformer handles the education-related categorical variables. It employs the mode imputation technique to fill in missing values. Additionally, it creates a new variable called Total_Education by summing up the FatherEducation and MotherEducation variables. The Total_Education variable is then categorized based on whether it is above or below the median value of the average of FatherEducation and MotherEducation. Values above the median are encoded as 1, while values below the median are encoded as 0.

  + `CategoricalEncoder_Income``: This transformer encodes the income variable, which is initially provided as a string. It assigns numerical values to different income categories using a predefined mapping. For example, the income category "<1000" is encoded as 0, "10001~50000" as 1, and ">50001" as 2.


+ For detail can refer to [ETL.ipynb](ETL.ipynb).


  

## Additional Note for remove out family structure
 Genomic Best Linear Unbiased Prediction (GBLUP), taking into consideration the influence of genetic correlation structure and subtracting the genetic relationship structure from the phenotype.


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



