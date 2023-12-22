# Feature engineering on imputed gene expression and environmental factors

In this module, we have two goal : 
- Regress out potential genetic corrleation structure (Step 0)
- Apply feature engineering techniques, which encompass tasks such as imputing missing values, generating new features, and encoding features from raw data.(Step 1)

## Scripts
### Step 0 : Regress out potential genetic relationship structure
The script is designed to utilize the following equation in order to construct Genomic Best Linear Unbiased Prediction (GBLUP), taking into consideration the influence of genetic correlation structure and subtracting the genetic relationship structure from the phenotype.

$\left[\begin{array}{l}\hat{\mu} \\ \hat{g}\end{array}\right]=\left[\begin{array}{cc}1_n^{\prime} 1_n & \mathbf{1}_n^{\prime} Z \\ Z^{\prime} 1_n & Z^{\prime} Z+G^{-1} \frac{\sigma_e^2}{\sigma_g^2}\end{array}\right]^{-1}\left[\begin{array}{l}1_n^{\prime} y \\ Z^{\prime} y\end{array}\right]$

--------------------------
The construction of the Genetic Relationship Matrix (GRM) was based on the following description:

W is a matrix for standardized genotype matrix, where its element for i-th SNP of j-th individual wij=(xij−2pi)/2pi(1−pi) with xij being the genotype value coded as the number of copies of the reference alleles {0, 1, 2}; then A = WW'/m (genetic relationship matrix (GRM) between individuals)

```
Usage:
    
   
      
Output:

```

### Step 1 :

## Reproducible Analysis

To reproduce the results of the coverage analysis perform the following:

```

```


## Reference
+  Hayes BJ, Visscher PM, Goddard ME. Increased accuracy of artificial selection by using the realized relationship matrix. Genet Res (Camb). 2009;91:47–60.
+  https://cnsgenomics.com/data/teaching/GNGWS22/module4/Practical3.pdf
+  https://zjuwhw.github.io/2021/08/20/GRM.html
+  


  

  



