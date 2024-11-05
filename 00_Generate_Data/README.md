# Construct genetic relationship matrix and Impute gene expression

In this module, we aim to impute gene expression for different tissue using gtex version 7 for reference and create genetic relationship matrix
- Create genetic relationship matrix. (Step 0)
- Impute gene expression for different tissue using gtex version 7 for reference. (Step 1)

  

## Scripts
### Step 0: Create genetic relationship matrix

The construction of the Genetic Relationship Matrix (GRM) was based on the following description:

+ W is a matrix for standardized genotype matrix, where its element for i-th SNP of j-th individual wij=(xij−2pi)/2pi(1−pi) with xij being the genotype value coded as the number of copies of the reference alleles {0, 1, 2}; 
+ Then A = WW'/m (genetic relationship matrix (GRM) between individuals).
+ The final GRM is the weighted averaged matrix combined multiple GRMS. Weights proportional to the numbers of SNPs

```
Usage:

    python compute_grm.py \
      --data_config  \
      --work_dir  \
      --dosage_prefix \
      --dosage_end_prefix 

Output:
genetic relationship matrix (GRM) e.g. genetic_correlation.pkl.gz
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
gene expression for specifice tissue e.g. /exeh_4/yuping/Epistasis_Interaction/00_Generate_Data/results/Brain_Amygdala/Brain_Amygdala_imputed.txt
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





  


