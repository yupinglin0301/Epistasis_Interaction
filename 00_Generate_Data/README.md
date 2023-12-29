# Data Preprocessing and Impute gene expression

In this module, we aim to impute gene expression for different tissue using gtex version 7 for reference.
- Regress out potential genetic corrleation structure. (Step 0)
- Impute gene expression for different tissue using gtex version 7 for reference. (Step 1)

## Scripts
### Step 0: Regress out potential genetic corrleation structure
The script is designed to utilize the following equation in order to construct Genomic Best Linear Unbiased Prediction (GBLUP), taking into consideration the influence of genetic correlation structure and subtracting the genetic relationship structure from the phenotype.

$y=1_n \mu+\sum_i W q_i+e$

$\left[\begin{array}{l}\hat{\mu} \\ \hat{g}\end{array}\right]=\left[\begin{array}{cc}1_n^{\prime} 1_n & \mathbf{1}_n^{\prime} Z \\ Z^{\prime} 1_n & Z^{\prime} Z+G^{-1} \frac{\sigma_e^2}{\sigma_g^2}\end{array}\right]^{-1}\left[\begin{array}{l}1_n^{\prime} y \\ Z^{\prime} y\end{array}\right]$

--------------------------
The construction of the Genetic Relationship Matrix (GRM) was based on the following description:

W is a matrix for standardized genotype matrix, where its element for i-th SNP of j-th individual wij=(xij−2pi)/2pi(1−pi) with xij being the genotype value coded as the number of copies of the reference alleles {0, 1, 2}; then A = WW'/m (genetic relationship matrix (GRM) between individuals).

```
Usage:

    python compute_grm.py \
      --data_config  \
      --work_dir  \
      --dosage_prefix \
      --dosage_end_prefix 

Output:
genetic relationship matrix (GRM)
```

### Step 1: Impute gene expression for different tissue using gtex version 7 for reference
The script is designed is estimated tissue-specific gene expression on GTExV7 data.

```
Usage:

    python run_impute_gene_expression.py \
      --data_config  \
      --data_dir  \
      --work_dir  \
      --weight_tissue  \
      --weight_end_prefix  \
      --weight_prefix  \
      --dosage_prefix  \
      --dosage_end_prefix  

Output:
gene expression for specifice tissue
```

#### **Gene set collections and data sets used**


| Dataset | Tissue|
| :------ | :--------- |
| gtexV7 | Brain_Amygdala |
| gtexV7 | Brain_Anterior_cingulate_cortex_BA24 |
| gtexV7 | Brain_Caudate_basal_ganglia |
| gtexV7 | Brain_Cerebellar_Hemisphere |
| gtexV7 | Brain_Cortex |
| gtexV7 | Brain_Frontal_Cortex_BA9 |
| gtexV7 | Brain_Hippocampus |
| gtexV7 | Brain_Hypothalamus |
| gtexV7 | Brain_Putamen_basal_ganglia_ |
| gtexV7 | Brain_Nucleus_accumbens_basal_ganglia |
| gtexV7 | Brain_Spinal_cord_cervical_c-1 |
| gtexV7 | Brain_Substantia_nigra |  
| gtexV7 | Brain_Cerebellum |        


## Reproducible Analysis

To reproduce the results of the coverage analysis perform the following:

```

```


## Reference
+  Hayes BJ, Visscher PM, Goddard ME. Increased accuracy of artificial selection by using the realized relationship matrix. Genet Res (Camb). 2009;91:47–60.
+  https://cnsgenomics.com/data/teaching/GNGWS22/module4/Practical3.pdf
+  https://zjuwhw.github.io/2021/08/20/GRM.html
+  https://www.nature.com/articles/ncomms8432


  

  


