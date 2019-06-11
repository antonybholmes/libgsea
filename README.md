# libgsea

A library providing functions for making better GSEA plots and a
reference implementation of Extended GSEA from Lim et al.

[https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2740937/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2740937/)


## Installation

```
pip install libgsea
```

## Extended GSEA

Extended GSEA can be run my modifying extended_gsea_example.py and
suppying your own input files.

1. Modify **expression_matrix.txt** to your input expression matrix
where rows are genes/probes and columns are samples.
2. Modify **gene_set_1** and **gene_set_2** to represent your positive
and negative gene sets respectively. Each should be a set of unique
gene symbols.
3. **ix_phen1** and **ix_phen2** are the column indices of the two 
phenotypes you want to test in your expression matrix.
