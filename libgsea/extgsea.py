#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 14:13:10 2018

@author: antony
"""

import numpy as np
import pandas as pd
import sys
import matplotlib
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import libplot
import matplotlib.gridspec as gridspec


# http://arep.med.harvard.edu/N-Regulation/Tolonen2006/GSEA/index.html


class ExtGSEA:
    def __init__(self, ranked_gene_list, ranked_scores, permutations=1000, w=1):
        self._w = w
        self._np = permutations

        l = len(ranked_gene_list)

        # the negative versions are for the second gene set
        rk = np.concatenate((ranked_gene_list, ranked_gene_list), axis=0)
        rsc = np.concatenate((ranked_scores, -ranked_scores), axis=0)
        # descending order
        ix = np.argsort(rsc)[::-1]

        print(np.sort(rsc)[::-1])

        pn = np.concatenate((np.ones(l), -np.ones(l)), axis=0)

        self._ranked_gene_list = ranked_gene_list
        self._ranked_scores = ranked_scores

        self._rkc = rk[ix]
        self._rsc = rsc[ix]
        self._pn = pn[ix]

        # Defaults if nothing found
        self._es = -1
        self._nes = -1
        self._pv = -1
        self._ledge = []
        self._bg = {}

        self._gsn1 = "n1"
        self._gsn2 = "n2"

        self._run = False

    def enrichment_score(self, gs1):
        l = len(self._ranked_gene_list)

        hits = np.zeros(l)

        for i in range(0, l):
            if self._ranked_gene_list[i] in gs1:
                hits[i] = 1

        # Compute ES
        score_hit = np.cumsum(np.abs(self._ranked_scores * hits) ** self._w)
        score_hit = score_hit / score_hit[-1]
        score_miss = np.cumsum(1 - hits)
        score_miss = score_miss / score_miss[-1]

        es_all = score_hit - score_miss
        es = np.max(es_all) + np.min(es_all)

        isen = np.zeros(l)

        if es < 0:
            ixpk = np.where(es_all == np.min(es_all))[0][0]
            isen[ixpk:] = 1
            ledge = self._ranked_gene_list[(isen == 1) & (hits == 1)]
            ledge = ledge[::-1]
        else:
            ixpk = np.where(es_all == np.max(es_all))[0][0]
            print(ixpk)
            isen[0 : (ixpk + 1)] = 1
            ledge = self._ranked_gene_list[(isen == 1) & (hits == 1)]

        return es, es_all, hits, ledge

    def ext_gsea(self, gs1, gs2, name1="Gene set 1", name2="Gene set 2"):
        self._gs1 = gs1
        self._gs2 = gs2
        self._gsn1 = name1
        self._gsn2 = name2

        l = len(self._ranked_gene_list)

        self._hits1 = np.zeros(l)
        self._hits2 = np.zeros(l)

        for i in range(0, l):
            if self._ranked_gene_list[i] in gs1:
                self._hits1[i] = 1

            if self._ranked_gene_list[i] in gs2:
                self._hits2[i] = 1

        l = len(self._rkc)

        self._isgs = np.zeros(l)

        for i in range(l):
            if (self._pn[i] > 0 and self._rkc[i] in gs1) or (
                self._pn[i] < 0 and self._rkc[i] in gs2
            ):
                self._isgs[i] = 1

        # Compute ES
        self._score_hit = np.cumsum(np.abs(self._rsc * self._isgs) ** self._w)
        self._score_hit = self._score_hit / self._score_hit[-1]

        self._score_miss = np.cumsum(1 - self._isgs)
        self._score_miss = self._score_miss / self._score_miss[-1]

        self._es_all = self._score_hit - self._score_miss
        self._es = np.max(self._es_all) + np.min(self._es_all)

        # identify leading edge
        isen = np.zeros(l)

        if self._es < 0:
            ixpk = np.where(self._es_all == np.min(self._es_all))[0][0]
            isen[ixpk:] = 1
            self._ledge = self._rkc[(isen == 1) & (self._isgs == 1)]
            self._ledge = self._ledge[::-1]
        else:
            ixpk = np.where(self._es_all == np.max(self._es_all))[0][0]
            isen[0 : (ixpk + 1)] = 1
            self._ledge = self._rkc[(isen == 1) & (self._isgs == 1)]

        if self._np > 0:
            self._bg["es"] = np.zeros(self._np)
            n = self._isgs.size
            for i in range(0, self._np):
                self._bg["isgs"] = self._isgs[np.random.permutation(n)]

                self._bg["hit"] = np.cumsum(
                    (np.abs(self._rsc * self._bg["isgs"])) ** self._w
                )

                self._bg["hit"] = self._bg["hit"] / self._bg["hit"][-1]

                self._bg["miss"] = np.cumsum(1 - self._bg["isgs"])
                self._bg["miss"] = self._bg["miss"] / self._bg["miss"][-1]
                
                self._bg["all"] = self._bg["hit"] - self._bg["miss"]
                self._bg["es"][i] = max(self._bg["all"]) + min(self._bg["all"])

            if self._es < 0:
                self._pv = np.sum(self._bg["es"] <= self._es) / self._np
                self._nes = self._es / np.abs(
                    np.mean(self._bg["es"][self._bg["es"] < 0])
                )
            else:
                self._pv = np.sum(self._bg["es"] >= self._es) / self._np
                self._nes = self._es / np.abs(
                    np.mean(self._bg["es"][self._bg["es"] > 0])
                )
        else:
            self._pv = -1
            self._nes = -1

        self._run = True

        return self._es, self._nes, self._pv, self._ledge

    @property
    def bg(self):
        return self._bg

    @property
    def score_hit(self):
        return self._score_hit

    @property
    def isgs(self):
        return self._isgs

    @property
    def es(self):
        return self._es

    @property
    def es_all(self):
        return self._es_all

    @property
    def score_miss(self):
        return self._score_miss

    def plot(self, title=None, out=None):
        """
        Replot existing GSEA plot to make it better for publications
        """

        if not self._run:
            return

        libplot.setup()

        # output truetype
        # plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
        # in most case, we will have mangy plots, so do not display plots
        # It's also convinient to run this script on command line.

        fig = libplot.new_base_fig(w=10, h=7)

        # GSEA Plots
        gs = gridspec.GridSpec(16, 1)

        x = np.array(list(range(0, len(self._ranked_gene_list))))

        es1, es_all1, hits1, ledge1 = self.enrichment_score(self._gs1)
        es2, es_all2, hits2, ledge2 = self.enrichment_score(self._gs2)

        # Ranked Metric Scores Plot

        # subsample so we don't draw every point
        ix = list(range(0, len(x), 100))

        print(ix)

        x1 = x[ix]
        y1 = self._ranked_scores[ix]

        print(hits1)

        ax1 = fig.add_subplot(gs[10:])
        ax1.fill_between(x1, y1=y1, y2=0, color="#2c5aa0")
        ax1.set_ylabel("Ranked list metric", fontsize=14)

        ax1.text(
            0.05,
            0.9,
            self._gsn1,
            color="black",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax1.transAxes,
        )
        ax1.text(
            0.95,
            0.05,
            self._gsn2,
            color="red",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax1.transAxes,
        )
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.set_xlim((0, len(x)))

        #
        # Hits
        #

        # gene hits
        ax2 = fig.add_subplot(gs[8:9], sharex=ax1)

        # the x coords of this transformation are data, and the y coord are axes
        trans2 = transforms.blended_transform_factory(ax2.transData, ax2.transAxes)
        ax2.vlines(
            np.where(hits1 == 1)[0],
            0,
            1,
            linewidth=0.5,
            transform=trans2,
            color="black",
        )
        libplot.invisible_axes(ax2)

        ax3 = fig.add_subplot(gs[9:10], sharex=ax1)

        # the x coords of this transformation are data, and the y coord are axes
        trans3 = transforms.blended_transform_factory(ax3.transData, ax3.transAxes)
        ax3.vlines(
            np.where(hits2 == 1)[0], 0, 1, linewidth=0.5, transform=trans3, color="red"
        )
        libplot.invisible_axes(ax3)

        #
        # Enrichment score plot
        #

        ax4 = fig.add_subplot(gs[:8], sharex=ax1)

        # max es
        y2 = np.max(es_all1)
        x1 = np.where(es_all1 == y2)[0]
        print(x1, y2)
        ax4.vlines(x1, 0, y2, linewidth=0.5, color="grey")

        y2 = np.min(es_all2)
        x1 = np.where(es_all2 == y2)[0]
        print(x1, y2)
        ax4.vlines(x1, 0, y2, linewidth=0.5, color="grey")

        y1 = es_all1
        y2 = es_all2

        ax4.plot(x, y1, linewidth=3, color="black")
        ax4.plot(x, y2, linewidth=3, color="red")

        ax4.tick_params(axis="both", which="both", color="dimgray")
        # ax4.spines['left'].set_color('dimgray')
        ax4.spines["bottom"].set_visible(False)  # set_color('dimgray')

        # the y coords of this transformation are data, and the x coord are axes
        trans4 = transforms.blended_transform_factory(ax4.transAxes, ax4.transData)
        ax4.hlines(0, 0, 1, linewidth=0.5, transform=trans4, color="grey")

        ax4.set_ylabel("Enrichment score (ES)", fontsize=14)
        ax4.set_xlim(min(x), max(x))
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
        ax4.tick_params(
            axis="both",
            which="both",
            bottom="off",
            top="off",
            labelbottom="off",
            right="off",
        )
        ax4.locator_params(axis="y", nbins=5)
        # FuncFormatter need two argment, I don't know why. this lambda function used to format yaxis tick labels.
        ax4.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda tick_loc, tick_num: "{:.1f}".format(tick_loc))
        )

        if title is not None:
            fig.suptitle(title)

        fig.tight_layout(pad=2)  # rect=[o, o, w, w])

        if out is not None:
            plt.savefig(out, dpi=600)
