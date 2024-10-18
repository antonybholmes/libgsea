#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 14:14:10 2018

@author: antony
"""

import math
from typing import Optional
import numpy as np
import pandas as pd
import sys
import matplotlib
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import libplot
import matplotlib.gridspec as gridspec
import svgplot
from svgplot.axis import Axis
from svgplot.svgfigure import SVGFigure

# http://arep.med.harvard.edu/N-Regulation/Tolonen2006/GSEA/index.html

LINE_GREEN = "#00b359"


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

        #print(np.sort(rsc)[::-1])

        pn = np.concatenate((np.ones(l), -np.ones(l)), axis=0)

        self._ranked_gene_list = ranked_gene_list
        self._ranked_scores = ranked_scores

        self._rkc = rk[ix]
        self._rsc = rsc[ix]
        self._pn = pn[ix]

        # Defaults if nothing found
        self._es = -1
        self._nes = -1
        self._pvalue = -1
        self._ledge = []
        self._bg = {}

        self._gsn1 = "n1"
        self._gsn2 = "n2"

        self._run = False

    def enrichment_score(self, gs1: list[str]):
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

        is_leading_edge = np.zeros(l)

        if es < 0:
            # where does the leading edge start
            ixpk = np.where(es_all == np.min(es_all))[0][0]
            is_leading_edge[ixpk:] = 1
            ledge = self._ranked_gene_list[(is_leading_edge == 1) & (hits == 1)]
            ledge = ledge[::-1]
        else:
            ixpk = np.where(es_all == np.max(es_all))[0][0]
            #print(ixpk)
            is_leading_edge[0 : (ixpk + 1)] = 1
            ledge = self._ranked_gene_list[(is_leading_edge == 1) & (hits == 1)]

        # just the indices of the leading edge
        is_leading_edge = np.array(sorted(np.where(is_leading_edge == 1)[0]))

        return {
            "es": es,
            "es_all": es_all,
            "hits": hits,
            "is_leading_edge": is_leading_edge,
            "ledge": ledge,
        }

    def ext_gsea(
        self,
        gs1: list[str],
        gs2: list[str],
        name1: str = "Gene set 1",
        name2: str = "Gene set 2",
    ):
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
                self._pvalue = np.sum(self._bg["es"] <= self._es) / self._np
                self._nes = self._es / np.abs(
                    np.mean(self._bg["es"][self._bg["es"] < 0])
                )
            else:
                self._pvalue = np.sum(self._bg["es"] >= self._es) / self._np
                self._nes = self._es / np.abs(
                    np.mean(self._bg["es"][self._bg["es"] > 0])
                )
        else:
            self._pvalue = -1
            self._nes = -1

        self._run = True

        return {
            "es": self._es,
            "nes": self._nes,
            "pvalue": self._pvalue,
            "ledge": self._ledge,
        }

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

    def plot(self, title: Optional[str] = None, out: Optional[str] = None):
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

        x1 = x[ix]
        y1 = self._ranked_scores[ix]
 
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

    def svg_plot(
        self,
        svg: SVGFigure,
        title: Optional[str] = None,
        w: int = 500,
        ylabel: Optional[str] = "ES",
        show_leading_edge: bool = True,
        stroke: int = 4,
        line_color: list[str] = ["red", "royalblue"],
        le_fill_opacity: float = 0.3,
        hit_height: int = 25,
        showsnr: bool = True,
        aspect_ratio: float = 0.6,
    ):
        if not self._run:
            return
        
        h = w * aspect_ratio

        # plot 1
        es1 = self.enrichment_score(self._gs1)
        is_leading_edge1 = es1["is_leading_edge"]
        es2 = self.enrichment_score(self._gs2)
        is_leading_edge2 = es2["is_leading_edge"]

        y = es1["es_all"]  # self._ranked_scores
        x = np.array(range(y.size))

        # subsample so we don't draw every point
        ix = list(range(0, len(x), 100))

        x1 = x[ix]
        y1 = y[ix]
        xmax = max(x)
        ymax = max(abs(np.concatenate([es1["es_all"], es2["es_all"]])))
        ymax = np.round((ymax * 10) / 10, 1)
        ymin = -ymax
 
        xaxis = Axis(lim=[0, xmax], w=w)
        yaxis = Axis(lim=[ymin, ymax], w=h, label=ylabel if ylabel is not None else "")
        # leading edge 1

        xlead = x[is_leading_edge1]
        ylead = y[is_leading_edge1]

        xlead = np.insert(xlead, 0, xlead[0])
        ylead = np.insert(ylead, 0, 0)

        xlead = np.append(xlead, xlead[-1])
        ylead = np.append(ylead, 0)

        if show_leading_edge:
            points = [
                [xaxis.scale(px), h - yaxis.scale(py)] for px, py in zip(xlead, ylead)
            ]
            svg.add_polyline(
                points,
                color="none",
                fill=line_color[0],
                stroke=0,
                fill_opacity=le_fill_opacity,
            )

        # fill in points
        y1[0] = 0
        y1[-1] = 0

        # scale points
        points = [[xaxis.scale(px), h - yaxis.scale(py)] for px, py in zip(x1, y1)]

        # python light green as html
        svg.add_polyline(points, color=line_color[0], stroke=stroke)

        # plot 2

        y = es2["es_all"]  # self._ranked_scores
        x = np.array(range(y.size))

        y1 = y[ix]
 
        xaxis = Axis(lim=[0, xmax], w=w)
        yaxis = Axis(lim=[ymin, ymax], w=h, label=ylabel if ylabel is not None else "")
        # leading edge 1

        xlead = x[is_leading_edge2]
        ylead = y[is_leading_edge2]

        # if xlead[0] != 0:
        xlead = np.insert(xlead, 0, xlead[0])
        ylead = np.insert(ylead, 0, 0)

        xlead = np.append(xlead, xlead[-1])
        ylead = np.append(ylead, 0)

        if show_leading_edge:
            points = [
                [xaxis.scale(px), h - yaxis.scale(py)] for px, py in zip(xlead, ylead)
            ]
            svg.add_polyline(
                points,
                color="none",
                fill=line_color[1],
                stroke=0,
                fill_opacity=le_fill_opacity,
            )

        if es2["es"] >= 0:
            svg.add_line(x1=xaxis.scale(xlead[-1 if es2["es"] >= 0 else 9]), y2=yaxis.scale(max(ylead)), dashed=True, color=line_color[1])
        else:
            svg.add_line(x1=xaxis.scale(xlead[0]), y2=yaxis.scale(max(ylead)), dashed=True, color=line_color[1])
        # fill in points
        y1[0] = 0
        y1[-1] = 0

        # scale points
        points = [[xaxis.scale(px), h - yaxis.scale(py)] for px, py in zip(x1, y1)]

        # python light green as html
        svg.add_polyline(points, color=line_color[1], stroke=stroke)

        ticks = [ymin, 0, ymax]
        ticks = [np.round(t, 1) for t in ticks]

        # if ymin == 0:
        #     ticks = [ymin, ymax]
        # elif ymax == 0:
        #     ticks = [ymin, ymax]
        # else:
        #     ticks = [ymin, 0, ymax]

        svgplot.add_y_axis(
            svg,
            axis=yaxis,
            ticks=ticks,
            padding=svgplot.TICK_SIZE,
            showticks=True,
            stroke=stroke,
            title_offset=120,
        )

        # draw line at y =0
        y1 = h - yaxis.scale(0)  # (0 - ymin) / (ymax - ymin) * scaleh
        svg.add_line(y1=y1, x2=w, stroke=stroke)
        # add label for max gene count
        svg.add_text_bb(f"{x.size:,}", x=w + 10, y=y1 + 20)

        # draw hits
        pos = (0, h + 20)

        for hit in np.where(es1["hits"] > 0)[0]:
            x1 = xaxis.scale(hit)  # hit / xmax * w
            svg.add_line(x1=x1, y1=pos[1], y2=pos[1] + hit_height, color=line_color[0])
        svg.add_text_bb(
            self._gsn1, color=line_color[0], x=w + 20, y=pos[1] + hit_height / 2 + 2
        )

        pos = (0, pos[1] + hit_height * 1.5)
        for hit in np.where(es2["hits"] > 0)[0]:
            x1 = xaxis.scale(hit)  # hit / xmax * w
            svg.add_line(x1=x1, y1=pos[1], y2=pos[1] + hit_height, color=line_color[1])
        svg.add_text_bb(
            self._gsn2, color=line_color[1], x=w + 20, y=pos[1] + hit_height / 2 + 2
        )

        if showsnr:
            pos = (0, pos[1] + 50)
            snr = self._ranked_scores
            zero_cross = snr[snr > 0].shape[0]
            m = round(int(max(abs(snr)) * 10) / 10, 1)
            ymin = -m
            ymax = m
            h = w * aspect_ratio * 0.5
            yaxis = Axis(lim=[ymin, ymax], w=h, label="SNR")

            svgplot.add_y_axis(
                svg,
                pos=pos,
                axis=yaxis,
                ticks=[-m, 0, m],
                padding=svgplot.TICK_SIZE,
                showticks=True,
                stroke=stroke,
                title_offset=120,
            )

            # gray
            points = [[xaxis.scale(0), pos[1] + h - yaxis.scale(0)]]
            points.extend(
                [
                    [xaxis.scale(px), pos[1] + h - yaxis.scale(py)]
                    for px, py in zip(range(0, xmax), snr)
                ]
            )
            points.extend([[xaxis.scale(xmax), pos[1] + h - yaxis.scale(0)]])
            svg.add_polyline(
                points, color="none", fill="#4d4d4d", stroke=stroke, fill_opacity=0.3
            )

            x1 = xaxis.scale(zero_cross)
            svg.add_line(
                x1=x1,
                y2=pos[1] + h - yaxis.scale(ymax),
                x2=x1,
                y1=pos[1] + h - yaxis.scale(ymin),
                stroke=stroke,
                dashed=True,
            )

            svg.add_text_bb(
                f"Zero cross at {zero_cross:,}", x=x1, y=pos[1] + h + 30, align="c"
            )

            # draw line at y = 0
            y1 = pos[1] + h - yaxis.scale(0)
            # svg.add_line(x1=xoffset, y1=y1, x2=xoffset+w, y2=y1, stroke=core.AXIS_STROKE)
