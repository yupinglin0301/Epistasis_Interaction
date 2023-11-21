# Impute gene expression

In this module, we aim to impute gene expression for different tissue using gtex version 7 for reference.

## Scripts

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

| Flag | Description |
|-|-|
| data_config | The yaml formatted dataset configuration file |
| data_dir | Data directory where the phenotype and genotype matrix are stored. |
| work_dir | Working directory where the experiment will be conducted |
| weight_tissue | Specify the tissue SQLite database |
| weight_end_prefix | Specify the end prefix of the tissue SQLite database |
| weight_prefix | Specify the prefix of the tissue SQLite database |
| dosage_prefix | Specify the prefix of filenames of dosage files |
| dosage_end_prefix  | Specify the end prefix of filenames of dosage files |



## Gene set collections and data sets used

We calculated coverage for the following combinations:

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

## Notes

#### 2023.11.01
**Finish Task**

+ Imputed gene expression in the Brain_Amygdala using our dosage genotype.
  - Since the rsid in our dosage file is represented as "chr:pos".
  - In the script we used ` snp151_GRCh37.txt ` to extract SNPID - see analysis.ipynb (1.GenotypeDataset function).
  
**To Do List**
- [ ] Imputed gene expression on other brain tissue using our dosage genotype.
  




