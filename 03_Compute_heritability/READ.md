## Estimating PVE by predicted transcriptome of a single tissue



In this module, we estimate the proportion of varaince explained (PVE) by the predicted transcriptome of single tissue.


After calcauted the predicted transcriptome [00_Generate_Data](../00_Generate_Data/results) using GTEx V7 brain tissue weights using dosage genotype data. Using these predicted expression levels, we calculated the "predicted expression relatedness matrix" (analgous to genetic relationship matrix)
and applied the maxium likehood estimation to calcuate the porption of variance explained by the predcited transcriptome. The approach is analogous to standard SNP-heritability estimation 


### Step 1: Estimate Heritability
```
Usage:
    
    python estimate_hsq.py \
         --weight_tissue "Brain_Amygdala" \
         --phen_name CWR_Total \
         --work_dir 03_Compute_heritability > /exeh_4/yuping/Epistasis_Interaction/03_Compute_heritability/Log/nohup.txt &
     
Output:
heritability score
```
**Notes:** 

The estimation procedure is as follows:
+ Given an m × n expression matrix with m samples and n genes, for each gene, we standardized the expression values to mean 0 and variance 1 by subtracting the mean and dividing by the standard deviation of that gene’s expression values.
+ If we let Z be the m × n standardized expression matrix, we defined K as the m × m covariance matrix of Z, which we calculated as $\frac{{\boldsymbol{Z}}{{\boldsymbol{Z}}}^{T}}{n}
$
+   
${\boldsymbol{y}}={\beta }_{0}1+{\beta }_{1}{{\boldsymbol{x}}}_{k}+{\boldsymbol{u}}+\varepsilon
$

where
${\boldsymbol{u}} \sim {\mathscr{N}}(0,{\sigma }_{g}^{2}{\boldsymbol{K}})$ and 
$\varepsilon \sim {\mathscr{N}}(0,{\sigma }_{e}^{2}{\boldsymbol{I}})$

${\boldsymbol{Y}}={\sigma }_{g}^{2}{\boldsymbol{K}}+{\sigma }_{e}^{2}{\boldsymbol{I}}$

The two parameter  $\sigma _{g}^{2}$ and $\sigma _{e}^{2}$ are optimized from our sample data by maximizing the log-likelihood function 

${\hat{\sigma }}_{g}^{2},{\hat{\sigma }}_{e}^{2}={{\rm{argmax}}}_{{\sigma }_{g}^{2},{\sigma }_{e}^{2}}-\frac{1}{2}[n\,\mathrm{log}(2\pi )+\,\mathrm{log}\,|{\boldsymbol{V}}|+{({\boldsymbol{y}}-{\beta }_{0}1)}^{T}{{\boldsymbol{V}}}^{-1}({\boldsymbol{y}}-{\beta }_{0}1)]
$


The parameters of the model were estimated using [glimix-core](https://glimix-core.readthedocs.io/en/latest/lmm.html#association-scan) to estimate the heritability = $\frac {\hat \sigma _{g}^{2}}{\hat \sigma _{e}^{2} + \hat \sigma _{g}^{2}}$


+ For detail can refer to [estimate_hsq](estimate_hsq.py)