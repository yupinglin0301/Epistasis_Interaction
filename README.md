# Epistasis Interaction: Discover Gene-Gene / SNP-SNP interactions

**Yu-Ping LIN**

**The Chinese University of Hong Kong**

The repository stores both data and data processing modules for conducting the analysis

Problem:GWAS studies have been extensively used SNP to understand the genetic architecture by conducting association test between a SNP and a complex trait. However, the single-marker association methods are not sufficient to fully explain the heritable component of phenotype variability, as they ignore the type of non-additive interactions between multiple alleles. Hence, it is essential to investigate the SNP- SNP and gene-gene interactions, as they are a likely source of the undelaying heritability and can better explain full extent of the genetic architecture in complex quantitative traits.

Approach:To overcome such issue and find interacting genetic factors in GWAS studies, we plan to perform interaction analysis with a machine learning tree-based method called iterative random forest algorithm (IRF).

## Analysis Modules

| Name | Description |
| :--- | :---------- |
| [00_Generate_Data](00_Generate_Data/) | Imputed gene expression data on different tissue |
| [01_Preprocess_Data](01_Preprocess_Data/) | Feature engineering on imputed gene expression data for different tissue |
| [02_Select_Parameter_Model](02_Select_Parameter_Model/) | Find optimal set of parameters for each model |
| [03_Discover_Interaction](03_Discover_Interaction/) | Inpute imputed gene expression data / snp-level genotype data into supervised machine learing - iterative random forest framework to identify interaction terms |