#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:09:25 2018

@author: antony
"""

import matplotlib
matplotlib.use('agg')
import numpy as np
import pandas as pd
import sys
import matplotlib
import libgsea

df = pd.read_csv('expression_matrix.txt', sep='\t', header=0)


ix_phen1 = np.where(df.columns.str.contains('phen1'))[0]


df_genes = pd.read_csv('gene_set_1.txt', sep='\t', header=0)
gene_set_1 = set(df_genes.iloc[:, 0].values)

df_genes = pd.read_csv('gene_set_2.txt', sep='\t', header=0)
gene_set_2 = set(df_genes.iloc[:, 0].values)


ix_phen2 = np.where(df.columns.str.contains('phen2'))[0]

# Remove zero rows

df2 = df.iloc[:, np.concatenate((ix_phen2, ix_phen1), axis=0)]
rix = np.where(df2.sum(axis=1) != 0)[0]
df_non_zero = df.iloc[rix, :]

m1 = df_non_zero.iloc[:, ix_phen2].mean(axis=1).values
sd1 = df_non_zero.iloc[:, ix_phen2].std(axis=1).values
m2 = df_non_zero.iloc[:, ix_phen1].mean(axis=1).values
sd2 = df_non_zero.iloc[:, ix_phen1].std(axis=1).values


snr = (m1 - m2) / (sd1 + sd2)
ix_snr = np.argsort(snr)[::-1]

sorted_snr = snr[ix_snr]
sorted_genes = df_non_zero['Gene'].values[ix_snr]

gsea = libgsea.ExtGSEA(sorted_genes, sorted_snr)
es, nes, pv, ledge = gsea.ext_gsea(gene_set_1, gene_set_2, name1='gene_set_1', name2='gene_set_2')
gsea.plot(title='{} vs {}'.format(phen2, phen1), out='plot.pdf')

