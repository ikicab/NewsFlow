# -*- coding: utf-8 -*-
"""
@author: Barbara Ikica
"""

from NewsFlow import newsFlow

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import igraph as ig
import seaborn as sns
sns.set()
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

import pandas as pd
import numpy as np
import sys, os

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def sub_plot(k, time_range, data, msg):
    """Adds a subplot to a given plot (a helper function to plot_newsFlow)."""

    _ = plt.subplot(2, 2, k)
    _ = plt.plot(time_range, data)
    _ = plt.xlabel(r'Simulation time $t$')
    _ = plt.ylabel(msg)
    _ = plt.tight_layout()

def plot_newsFlow(N, F0, eta, lambda_F, outDeg=20, power=-2.5, M=10, tol=0.0005, window_size=0, t_max=0):
    """Visualises the simulation data."""

    g = ig.Graph.Barabasi(N, outDeg, power=power) # generate the BA graph
    g.simplify()
    g.delete_vertices(g.vs.select(lambda vertex: vertex.degree() == 0))

    N = len(g.vs)
    E = len(g.es)

    time_range, list_polarised, list_rhoF, list_FT_edges = newsFlow(g=g, F0=F0, eta=eta, lambda_F=lambda_F, M=M,
             t_max=t_max, window_size=window_size, tol=tol, sfreq=0)

    _ = plt.tight_layout()
    _ = plt.figure(figsize=(8, 8))
    _ = plt.suptitle(r'$N =$ %s, $F_0 =$ %s, $M =$ %s, $\eta =$ %s, $\lambda_F =$ %s' %
                     (N, F0, 2*M, eta, lambda_F), y=1.01)
    _ = plt.tight_layout()

    _ = sub_plot(1, time_range, [i/N for i in list_polarised], r'$\rho^P(t)$')
    _ = plt.ylim((0, 1))
    _ = sub_plot(2, time_range, [i/N for i in list_rhoF], r'$\rho^F(t)$')
    _ = plt.ylim((0, 1))
    _ = sub_plot(3, time_range, [i/E for i in list_FT_edges], r'$\rho^{FT}(t)$')
    _ = plt.ylim((0, 1))
    _ = sub_plot(4, time_range, range(len(time_range)), 'Iterations')
    _ = plt.show()

def rho_heatmaps(N, F0, lambdaF_list, eta_list, rep, outDeg=20, power=-2.5, M=10, tol=0.0005, window_size=0, t_max=0):
    """Generates data for the density heatmap."""

    blockPrint()
    g = ig.Graph.Barabasi(N, outDeg, power=power) # generate the BA graph
    g.simplify()
    g.delete_vertices(g.vs.select(lambda vertex: vertex.degree() == 0))

    N = len(g.vs)
    E = len(g.es)

    final_polarised = {}
    final_rhoF = {}
    final_FT_edges = {}

    for lambda_F in lambdaF_list:
        for eta in eta_list:
            avg_pol = 0
            avg_rhoF = 0
            avg_FT = 0
            for i in range(rep):
                v_polarised, rhoF, nFT_edges = newsFlow(g=g, F0=F0, eta=eta, lambda_F=lambda_F, M=M,
                                                        t_max=t_max, window_size=window_size, tol=tol, sfreq=-1)

                avg_pol += (v_polarised/N)
                avg_rhoF += (rhoF/N)
                avg_FT += (nFT_edges/E)
            avg_pol /= rep
            avg_rhoF /= rep
            avg_FT /= rep
            final_polarised[(lambda_F, eta)] = avg_pol
            final_rhoF[(lambda_F, eta)] = avg_rhoF
            final_FT_edges[(lambda_F, eta)] = avg_FT

    return final_polarised, final_rhoF, final_FT_edges

def draw_heatmap(data, title, cmap, x_name='$\lambda^F$', y_name='$\eta$', markCells=[], **kwargs):
    """Draws the density heatmap."""

    ser = pd.Series(list(data.values()), index=pd.MultiIndex.from_tuples(data.keys(), names=(x_name, y_name)))
    df = ser.unstack(level=0).fillna(0)
    df.sort_index(axis=0, ascending=False, inplace=True)

    fig, ax = plt.subplots(figsize=(8,6))
    _ = ax.xaxis.label.set_size(16)
    _ = ax.yaxis.label.set_size(16)
    _ = ax.set_title(title, fontsize=18)
    _ = sns.heatmap(df, cmap=cmap, square=True, linewidths=.5, annot=True, ax=ax, **kwargs)

    lambdaF_vals = set()
    eta_vals = set()

    for k in data.keys():
        lambdaF_vals.add(k[0])
        eta_vals.add(k[1])

    w = len(lambdaF_vals)
    h = len(eta_vals)

    mask = np.zeros((h,w))
    mask[0:h,0:w] = False
    for pair in markCells:
        mask[pair[0], pair[1]] = True
        _ = ax.add_patch(Rectangle((pair[1], pair[0]), 1, 1, fill=False, edgecolor='k', lw=2))

    _ = sns.heatmap(df, cmap=cmap, square=True, linewidths=.5, cbar=False, annot=False, mask=mask, **kwargs)

    for text, show_annot in zip(ax.texts, (element for row in mask for element in row)):
        text.set_visible(show_annot)

    _ = plt.tight_layout()

def time_evolution_F0(N, list_F0, lambda_F, eta, outDeg=20, power=-2.5, M=10, tol=0.0005, window_size=0, t_max=0):
    """Generates data for the time evolution in dependence on F0."""

    blockPrint()
    g = ig.Graph.Barabasi(N, outDeg, power=power) # generate the BA graph
    g.simplify()
    g.delete_vertices(g.vs.select(lambda vertex: vertex.degree() == 0))

    N = len(g.vs)
    E = len(g.es)

    results_times = {}
    results_polarised = {}
    results_rhoF = {}
    results_FT_edges = {}

    for F0 in list_F0:
        time_range, list_polarised, list_rhoF, list_FT_edges = newsFlow(g=g, F0=F0, eta=eta, lambda_F=lambda_F, M=M,
             t_max=t_max, window_size=window_size, tol=tol, sfreq=0)

        results_times[F0] = time_range
        results_polarised[F0] = [i/N for i in list_polarised]
        results_rhoF[F0] = [i/N for i in list_rhoF]
        results_FT_edges[F0] = [i/E for i in list_FT_edges]

    return results_times, results_rhoF, results_FT_edges, results_polarised, lambda_F, eta, N

def draw_time_evolution_F0(data, colours, inset_lims=[], inset_width=0):
    """Draws the time evolution in dependence on F0."""

    fig, ax = plt.subplots(1, 3, figsize=(16,6), sharex=True, sharey=True)
    titles = [r'$\rho^F(t)$', r'$\rho^{FT}(t)$', r'$\rho^P(t)$']

    list_F0 = list(data[0].keys())
    print(list_F0)

    N = data[6]

    for i in range(3):
        j = 0
        for F0 in list_F0:
            _ = ax[i].plot(data[0][F0], data[i+1][F0],
                     label=r'$\rho^F(0) = $' + '{0:.2f}'.format(F0/N),
                     linewidth=1.9, alpha=0.9, color=colours[j])
            _ = ax[i].set_xscale('log')
            _ = ax[i].tick_params(labelsize=14)
            _ = ax[i].set_title(titles[i], fontsize=16)

            j += 1

    ax[0].set_ylim(0, 1)

    if inset_lims != []:
        axins = zoomed_inset_axes(ax[2], inset_width, loc=10)
        patch, pp1, pp2 = mark_inset(ax[2], axins, loc1=1, loc2=1, fc="none", ec='0.5')
        pp1.loc1 = 3
        pp1.loc2 = 2
        pp2.loc1 = 4
        pp2.loc2 = 1

        j = 0
        for F0 in list_F0:
            _ = axins.plot(data[0][F0], data[3][F0],
                       linewidth=1.9, alpha=0.9, color=colours[j])#color=palette(j))
            _ = axins.set_xscale('log')
            _ = axins.set_xlim(inset_lims[0], inset_lims[1])
            _ = axins.set_ylim(inset_lims[2], inset_lims[3])
            _ = axins.tick_params(labelbottom=False,labeltop=True)

            j += 1

    plt.suptitle(r'Time evolution of the densities $\rho^F(t)$, $\rho^{FT}(t)$, and $\rho^P(t)$', fontsize=18,
                 fontweight=0, color='black', y=1.02)
    layer = fig.add_subplot(111, frameon=False)
    _ = plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    _ = plt.xlabel(r'Simulation time $t$', fontsize=16)
    _ = plt.ylabel(r'$\lambda^F = {0}, \eta = {0}$'.format(data[4], data[5]), fontsize=16)
    _ = plt.grid(False)
    handles, labels = ax[0].get_legend_handles_labels()
    _ = layer.legend(handles=handles, labels=labels, loc=9,
                     bbox_to_anchor=(0.5, -0.12),fancybox=True, shadow=False, ncol=4, fontsize=16)
    _ = plt.tight_layout()

def time_evolution_lambda_F_eta(N, F0, list_lambdaF, list_eta, outDeg=20, power=-2.5, M=10, tol=0.0005, window_size=0, t_max=0):
    """Generates data for the time evolution in dependence on lambda_F and eta."""

    blockPrint()
    g = ig.Graph.Barabasi(N, outDeg, power=power) # generate the BA graph
    g.simplify()
    g.delete_vertices(g.vs.select(lambda vertex: vertex.degree() == 0))

    N = len(g.vs)
    E = len(g.es)

    results_times = {}
    results_polarised = {}
    results_rhoF = {}
    results_FT_edges = {}

    for lambda_F in list_lambdaF:
        for eta in list_eta:
            time_range, list_polarised, list_rhoF, list_FT_edges = newsFlow(g=g, F0=F0, eta=eta, lambda_F=lambda_F, M=M,
             t_max=t_max, window_size=window_size, tol=tol, sfreq=0)

            results_times[(lambda_F, eta)] = time_range
            results_polarised[(lambda_F, eta)] = [i/N for i in list_polarised]
            results_rhoF[(lambda_F, eta)] = [i/N for i in list_rhoF]
            results_FT_edges[(lambda_F, eta)] = [i/E for i in list_FT_edges]

    return results_times, results_rhoF, results_FT_edges, results_polarised

def draw_time_evolution_lambda_F_eta(data, colours, linestyles, alphas, legend_elements):
    """Draws the time evolution in dependence on lambda_F and eta."""

    fig, ax = plt.subplots(1, 3, figsize=(13,5), sharex=True, sharey=True)
    titles = [r'$\rho^F(t)$', r'$\rho^{FT}(t)$', r'$\rho^P(t)$']

    for col in range(3):
        for key in data[0].keys():
            if col == 0:
                _ = ax[col].plot(data[0][key], data[col+1][key], label='$\lambda^F$ =' + str(key[0])\
                         + ', $\eta =$' + str(key[1]), alpha=alphas[key[1]], color=colours[key[0]],
                             linewidth=1.9, linestyle=linestyles[key[1]])
            else:
                 _ = ax[col].plot(data[0][key], data[col+1][key], alpha=alphas[key[1]], color=colours[key[0]],
                             linewidth=1.9, linestyle=linestyles[key[1]])
            _ = ax[col].set_xscale('log')
            _ = ax[col].set_title(titles[col], size=16)

        _ = ax[col].tick_params(axis='both', which='major', labelsize=14)
        _ = ax[col].tick_params(axis='both', which='minor', labelsize=14)

    _ = plt.suptitle(r'Time evolution of $\rho^F(t)$, $\rho^{FT}(t)$, and $\rho^P(t)$ in dependence on $\lambda^F$ and $\eta$', fontsize=18,
                 fontweight=0, color='black', y=1.03)

    _ = fig.add_subplot(111, frameon=False)
    _ = plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    _ = plt.xlabel(r'Simulation time $t$', fontsize=16)
    _ = plt.grid(False)

    _ = plt.tight_layout()
    _ = plt.legend(handles=legend_elements, fancybox=True, loc='best',
                       fontsize=16, ncol=2, framealpha=0.5)

def rho_lambdaF(list_N, list_rhoF0, rep, list_lambdaF, eta=1, outDeg=20, power=-2.5, M=10, tol=0.0005, window_size=0, t_max=0):
    """Generates data for the evolution of the densities in dependence on lambda_F."""

    blockPrint()
    final_rhoF_lambdaF = {}
    final_rhoFT_lambdaF = {}
    final_rhoP_lambdaF = {}

    for N in list_N:
        g = ig.Graph.Barabasi(N, outDeg, power=power) # generate the BA graph
        g.simplify()
        g.delete_vertices(g.vs.select(lambda vertex: vertex.degree() == 0))

        N = len(g.vs)
        E = len(g.es)

        for rhoF0 in list_rhoF0:
            F0 = int(N*rhoF0)

            final_rhoF_lambdaF[(N, rhoF0)] = {}
            final_rhoFT_lambdaF[(N, rhoF0)] = {}
            final_rhoP_lambdaF[(N, rhoF0)] = {}

            for lambda_F in list_lambdaF:
                avg_rhoF = 0
                avg_FT = 0
                avg_pol = 0

                for i in range(rep):
                    v_polarised, rhoF, nFT_edges = newsFlow(g=g, F0=F0, eta=eta, lambda_F=lambda_F,
                                        M=M, t_max=t_max, window_size=window_size, tol=tol, sfreq=-1)

                    avg_rhoF += rhoF/N
                    avg_FT += nFT_edges/E
                    avg_pol += v_polarised/N

                avg_rhoF /= rep
                avg_FT /= rep
                avg_pol /= rep
                final_rhoF_lambdaF[(N, rhoF0)][lambda_F] = avg_rhoF
                final_rhoFT_lambdaF[(N, rhoF0)][lambda_F] = avg_FT
                final_rhoP_lambdaF[(N, rhoF0)][lambda_F] = avg_pol

    return final_rhoF_lambdaF, final_rhoFT_lambdaF, final_rhoP_lambdaF, eta

def plot_densities_lambdaF(data, colours, markers, legend_elements):
    """Draws the evolution of the densities in dependence on lambda_F."""

    fig, ax = plt.subplots(1, 3, figsize=(13,5), sharex=True, sharey=True)
    titles = [r'$\rho^F$', r'$\rho^{FT}$', r'$\rho^P$']

    for N, rhoF0 in data[0].keys():
        colour = colours[rhoF0]
        marker = markers[N]

        for col in range(3):
            _ = ax[col].plot(list(data[col][(N, rhoF0)].keys()), list(data[col][(N, rhoF0)].values()), marker=marker, color=colour)
            _ = ax[col].set_title(titles[col], size=16)
            _ = ax[col].tick_params(axis='both', which='major', labelsize=14)
            _ = ax[col].tick_params(axis='both', which='minor', labelsize=14)
            _ = ax[col].set(adjustable='box')

    plt.suptitle(r'Dependence of $\rho^F$, $\rho^{FT}$, and $\rho^P$ on $\lambda^F$ ' + '($\eta = {0}$)'.format(data[3]), fontsize=18,
             fontweight=0, color='black', y=1.03)

    _ = ax[1].set_xlabel('$\lambda^F$', fontsize=16)

    _ = plt.tight_layout()
    _ = plt.legend(handles=legend_elements, loc='best', fancybox=True,
                   fontsize=16, ncol=2, framealpha=0.5)

def rho_eta(list_N, list_rhoF0, rep, list_eta, lambda_F=1, outDeg=20, power=-2.5, M=10, tol=0.0005, window_size=0, t_max=0):
    """Generates data for the evolution of the densities in dependence on eta."""

    blockPrint()
    final_rhoF_eta = {}
    final_rhoFT_eta = {}
    final_rhoP_eta = {}

    for N in list_N:
        g = ig.Graph.Barabasi(N, outDeg, power=power) # generate the BA graph
        g.simplify()
        g.delete_vertices(g.vs.select(lambda vertex: vertex.degree() == 0))

        N = len(g.vs)
        E = len(g.es)

        for rhoF0 in list_rhoF0:
            F0 = int(N*rhoF0)

            final_rhoF_eta[(N, rhoF0)] = {}
            final_rhoFT_eta[(N, rhoF0)] = {}
            final_rhoP_eta[(N, rhoF0)] = {}

            for eta in list_eta:
                avg_rhoF = 0
                avg_FT = 0
                avg_pol = 0

                for i in range(rep):
                    v_polarised, rhoF, nFT_edges = newsFlow(g=g, F0=F0, eta=eta, lambda_F=lambda_F,
                                        M=M, t_max=t_max, window_size=window_size, tol=tol, sfreq=-1)

                    avg_rhoF += rhoF/N
                    avg_FT += nFT_edges/E
                    avg_pol += v_polarised/N

                avg_rhoF /= rep
                avg_FT /= rep
                avg_pol /= rep
                final_rhoF_eta[(N, rhoF0)][eta] = avg_rhoF
                final_rhoFT_eta[(N, rhoF0)][eta] = avg_FT
                final_rhoP_eta[(N, rhoF0)][eta] = avg_pol

    return final_rhoF_eta, final_rhoFT_eta, final_rhoP_eta, lambda_F

def plot_densities_eta(data, colours, markers, legend_elements):
    """Draws the evolution of the densities in dependence on eta."""

    fig, ax = plt.subplots(1, 3, figsize=(13,5), sharex=True, sharey=True)
    titles = [r'$\rho^F$', r'$\rho^{FT}$', r'$\rho^P$']

    for N, rhoF0 in data[0].keys():
        colour = colours[rhoF0]
        marker = markers[N]

        for col in range(3):
            _ = ax[col].plot(list(data[col][(N, rhoF0)].keys()), list(data[col][(N, rhoF0)].values()), marker=marker, color=colour)
            _ = ax[col].set_title(titles[col], size=16)
            _ = ax[col].tick_params(axis='both', which='major', labelsize=14)
            _ = ax[col].tick_params(axis='both', which='minor', labelsize=14)
            _ = ax[col].set(adjustable='box')

    plt.suptitle(r'Dependence of $\rho^F$, $\rho^{FT}$, and $\rho^P$ on $\eta$ ' + '($\lambda^F = {0}$)'.format(data[3]), fontsize=18,
             fontweight=0, color='black', y=1.03)

    _ = ax[1].set_xlabel('$\eta$', fontsize=16)

    _ = plt.tight_layout()
    _ = plt.legend(handles=legend_elements, loc='best', fancybox=True,
                   fontsize=16, ncol=2, framealpha=0.5)

def get_media_repertoire(N, lambda_F, eta, sfreq, rhoF0=0.5, outDeg=20, power=-2.5, M=10, tol=0.0005, window_size=0, t_max=0):
    """Generates news-consumption statistics."""

    blockPrint()
    g = ig.Graph.Barabasi(N, outDeg, power=power) # generate the BA graph
    g.simplify()
    g.delete_vertices(g.vs.select(lambda vertex: vertex.degree() == 0))

    N = len(g.vs)
    F0 = int(N*rhoF0)

    list_rhoF, T_Fmedia, F_Fmedia, T_Tmedia, F_Tmedia, Tmedia_T, Tmedia_F, Fmedia_T, Fmedia_F = newsFlow(g=g,
        F0=F0, eta=eta, lambda_F=lambda_F, M=M, t_max=t_max, window_size=window_size, tol=tol, sfreq=sfreq)

    df = pd.DataFrame(columns=['t_step', 'type', '#F_media']) # store the results as a dataframe

    for t_step in T_Fmedia.keys():
        TF = T_Fmedia[t_step]
        TT = T_Tmedia[t_step]

        temp = pd.DataFrame({'t_step': [round(t_step) for i in range(len(TF))],
                             'type': [True for i in range(len(TF))],
                             '#F_media': [TF[i]/(TF[i]+TT[i]) for i in range(len(TF))]})
        df = pd.concat([df, temp])

    for t_step in F_Fmedia.keys():
        FF = F_Fmedia[t_step]
        FT = F_Tmedia[t_step]

        temp = pd.DataFrame({'t_step': [round(t_step) for i in range(len(FF))],
                             'type': [False for i in range(len(FF))],
                             '#F_media': [FF[i]/(FF[i]+FT[i]) for i in range(len(FF))]})
        df = pd.concat([df, temp])

    df['#F_media'] = df['#F_media'].astype(float)

    return df, lambda_F, eta, T_Fmedia, F_Fmedia, T_Tmedia, F_Tmedia, M

def media_repertoire(data):
    """Draws the time evolution of the distribution of per capita relative false-news consumption."""

    df = data[0]
    lambda_F = data[1]
    eta = data[2]

    fig, ax = plt.subplots(figsize=(15,3))
    _ = ax.set_title('Individual false-news consumption ($\lambda^F = {%s}$, $\eta = {%s})$' %(str(lambda_F), str(eta)),
                     size=18, y=1.07)
    ax = sns.boxplot(x='t_step', y='#F_media', hue='type', data=df,
                 palette={True: 'limegreen', False: 'crimson'}, width=0.5)

    for i,artist in enumerate(ax.artists):
        artist.set_edgecolor('white')

    _ = ax.set_xlabel('Simulation time $t$', size=16)
    _ = ax.set_ylabel(r'$\rho_i^F(t)$', size=16)
    _ = ax.tick_params(axis='both', which='major', labelsize=14)
    _ = ax.tick_params(axis='both', which='minor', labelsize=14)
    handles, _ = ax.get_legend_handles_labels()
    _ = ax.legend(handles, ['$\mathcal{R}^F(t)$', '$\mathcal{R}^T(t)$'], bbox_to_anchor=(1,1.15),
                  loc='center right', fancybox=True, framealpha=0.5, fontsize=16, ncol=2)
    _ = plt.tight_layout()

def public_scatter(data):
    """Draws the time evolutions of individual-level distributions of news sources across the entire media spectrum."""

    lambda_F = data[1]
    eta = data[2]
    T_Fmedia = data[3]
    F_Fmedia = data[4]
    T_Tmedia = data[5]
    F_Tmedia = data[6]
    M = data[7]

    fig, ax = plt.subplots(1, 3, figsize=(15,6), sharex=True, sharey=True)
    _ = plt.suptitle('Individual-level distributions of news sources ($\lambda^F = {%s}$, $\eta = {%s})$' %(str(lambda_F),
                     str(eta)), size=18, y=0.88)

    sorted_time = sorted(T_Tmedia.keys())
    indices = [0, int(len(sorted_time) / 2), -1]

    xcomb = []
    ycomb = []

    for ind in range(3):
        time = sorted_time[indices[ind]]
        xcomb += T_Tmedia[time]
        ycomb += T_Fmedia[time]
        xcomb += F_Tmedia[time]
        ycomb += F_Fmedia[time]

    xminel = min(xcomb)
    xmaxel = max(xcomb)
    yminel = min(ycomb)
    ymaxel = max(ycomb)

    for ind in range(3):
        time = sorted_time[indices[ind]]

        T_Tmedia_fin = T_Tmedia[time]
        T_Fmedia_fin = T_Fmedia[time]
        F_Tmedia_fin = F_Tmedia[time]
        F_Fmedia_fin = F_Fmedia[time]

        Tscale = []
        Fscale = []

        for i in range(len(T_Tmedia_fin)):
            Tscale.append((T_Tmedia_fin[i] + T_Fmedia_fin[i])*12)
        for i in range(len(F_Tmedia_fin)):
            Fscale.append((F_Tmedia_fin[i] + F_Fmedia_fin[i])*12)

        _ = ax[ind].scatter([i/M+0.012 for i in T_Tmedia_fin], [i/M for i in T_Fmedia_fin], c='limegreen',
                            s=Tscale, label=True, alpha=0.7, edgecolors='white')

        _ = ax[ind].scatter([i/M-0.012 for i in F_Tmedia_fin], [i/M for i in F_Fmedia_fin], c='crimson',
                            s=Fscale, label=False, alpha=0.7, edgecolors='white')

        _ = ax[ind].set_title('$t = {%d}$' %(time), size=16)

        if ind == 0:
            _ = ax[ind].set_ylabel('$r_i^F(t)/M^F$', size=16)

        _ = ax[ind].tick_params(axis='both', which='major', labelsize=14)
        _ = ax[ind].tick_params(axis='both', which='minor', labelsize=14)

        _ = ax[ind].set_xlim(xminel-0.08, xmaxel/M+0.08)
        _ = ax[ind].set_ylim(yminel-0.08, ymaxel/M+0.08)

    _ = fig.add_subplot(111, frameon=False)
    _ = plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    _ = plt.xlabel('$r_i^T(t)/M^T$', fontsize=16, labelpad=10)
    _ = plt.grid(False)

    handles, _ = ax[0].get_legend_handles_labels()
    _ = plt.legend(handles, ['$\mathcal{R}^T(t)$', '$\mathcal{R}^F(t)$'], bbox_to_anchor=(1,1.22),
                  loc='upper right', fancybox=True, framealpha=0.5, fontsize=16, ncol=2, markerscale=1.4)
    _ = plt.tight_layout()

def bins_labels(bins, **kwargs):
    """Arranges histogram bin labels."""

    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins) + bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])

def subscription(data):
    """Draws the time evolution of the distribution of the media repertoires at the level of news consumers."""

    lambda_F = data[1]
    eta = data[2]
    T_Fmedia = data[3]
    F_Fmedia = data[4]
    T_Tmedia = data[5]
    F_Tmedia = data[6]

    fig, ax = plt.subplots(2, 3, figsize=(9,6), sharex=True, sharey=True, dpi= 80)
    sorted_time = sorted(T_Tmedia.keys())
    indices = [0, int(len(sorted_time) / 2), -1]

    for ind in range(3):
        time = sorted_time[indices[ind]]
        bins = range(12)
        _ = ax[0,ind].hist([T_Tmedia[time],T_Fmedia[time]],bins=bins,
                 density=True, histtype='bar', color=['limegreen','crimson'], label=['t','f'], stacked=True)
        _ = ax[1,ind].hist([F_Tmedia[time],F_Fmedia[time]],bins=bins,
                 density=True, histtype='bar', color=['limegreen','crimson'], label=['t','f'], stacked=True)
        bins_labels(bins, fontsize=14)
        _ = ax[0,ind].set_title('$t = {%d}$' %(time), size=16)

        for i in range(2):
            _ = ax[i,ind].tick_params(axis='both', which='major', labelsize=14)
            _ = ax[i,ind].tick_params(axis='both', which='minor', labelsize=14)
            _ = ax[i,ind].grid(False, axis='x')

    _ = ax[0,0].set_ylabel('$\mathcal{R}^T(t)$', fontsize=16)
    _ = ax[1,0].set_ylabel('$\mathcal{R}^F(t)$', fontsize=16)
    _ = plt.tight_layout()
    _ = plt.suptitle('Public distribution of news sources ($\lambda^F = {%s}$, $\eta = {%s})$' %(str(lambda_F), str(eta)),
                      size=18, y=1.02)
    handles, _ = ax[0,2].get_legend_handles_labels()
    _ = ax[0,0].legend(handles, ['$\mathcal{M}^T$', '$\mathcal{M}^F$'],
                   loc='upper left', fancybox=True, framealpha=0.5, fontsize=16)
    _ = ax[1,1].set_xlabel('$(r_i^T(t), r_i^F(t))$', fontsize=16)
    _ = plt.tight_layout()

def get_media_subscriptions(N, lambda_F, eta, sfreq, rhoF0=0.5, outDeg=20, power=-2.5, M=10, tol=0.0005, window_size=0, t_max=0):
    """Generates media-reach statistics."""

    blockPrint()
    g = ig.Graph.Barabasi(N, outDeg, power=power) # generate the BA graph
    g.simplify()
    g.delete_vertices(g.vs.select(lambda vertex: vertex.degree() == 0))

    N = len(g.vs)
    F0 = int(N*rhoF0)

    list_rhoF, T_Fmedia, F_Fmedia, T_Tmedia, F_Tmedia, Tmedia_T, Tmedia_F, Fmedia_T, Fmedia_F = newsFlow(g=g,
        F0=F0, eta=eta, lambda_F=lambda_F, M=M, t_max=t_max, window_size=window_size, tol=tol, sfreq=sfreq)

    df = pd.DataFrame(columns=['t_step', 'type', '#F_subscribers']) # store the results as a dataframe

    for t_step in Tmedia_T.keys():
        TF = Tmedia_F[t_step]
        TT = Tmedia_T[t_step]
        temp = pd.DataFrame({'t_step': [round(t_step) for i in range(len(TF))],
                             'type': [True for i in range(len(TF))],
                             '#F_subscribers': [TF[i]/(TF[i]+TT[i]) for i in range(len(TF))]})
        df = pd.concat([df, temp])

    for t_step in Fmedia_F.keys():
        FF = Fmedia_F[t_step]
        FT = Fmedia_T[t_step]
        temp = pd.DataFrame({'t_step': [round(t_step) for i in range(len(FF))],
                             'type': [False for i in range(len(FF))],
                             '#F_subscribers': [FF[i]/(FF[i]+FT[i]) for i in range(len(FF))]})
        df = pd.concat([df, temp])

    df['#F_subscribers'] = df['#F_subscribers'].astype(float)

    return df, lambda_F, eta, Tmedia_T, Tmedia_F, Fmedia_T, Fmedia_F, M, N, list_rhoF

def media_subscriptions(data):
    """Draws the time evolution of the distribution of per-media relative reach to false-news consumers."""

    df = data[0]
    lambda_F = data[1]
    eta = data[2]

    fig = plt.figure(figsize=(13,5), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax2.tick_params(labelbottom=False, labelleft=False)
    ax3.tick_params(labelleft=False)

    _ = ax1.set_title('Media reach to $\mathcal{R}^F(t)$ ($\lambda^F = {%s}$, $\eta = {%s})$' %(str(lambda_F), str(eta)),
                     size=18, y=1.025)
    df = df.rename(columns={'t_step': 'tstep'})
    _ = sns.boxplot(x=r'tstep', y='#F_subscribers', hue='type', data=df,
                 palette={True: 'limegreen', False: 'crimson'}, width=0.5, ax=ax1)
    for i,artist in enumerate(ax1.artists):
        artist.set_edgecolor('white')

    ymin_top = min(df[(df[r'tstep'] > 0) & (df['type'] == False)]['#F_subscribers'])
    ymax_top = max(df[(df[r'tstep'] > 0) & (df['type'] == False)]['#F_subscribers'])

    ymin_bot = min(df[(df[r'tstep'] > 0) & (df['type'] == True)]['#F_subscribers'])
    ymax_bot = max(df[(df[r'tstep'] > 0) & (df['type'] == True)]['#F_subscribers'])

    _ = ax1.tick_params(axis='both', which='major', labelsize=14)
    _ = ax1.tick_params(axis='both', which='minor', labelsize=14)
    handles, _ = ax1.get_legend_handles_labels()
    _ = ax1.legend(handles, ['$\mathcal{M}^F$', '$\mathcal{M}^T$'],
                  loc='best', fancybox=True, framealpha=0.5, fontsize=16)
    _ = ax1.set_xlabel('Simulation time $t$', size=16)
    _ = ax1.set_ylabel(r'$\mu_j^F(t)$', size=16)
    _ = ax1.set_ylim(0,1)

    _ = ax1.fill_between([0.5, 10.5], ymin_top-0.01, ymax_top+0.01, facecolor='crimson', alpha=0.2)
    _ = ax1.fill_between([0.5, 10.5], ymin_bot-0.01, ymax_bot+0.01, facecolor='limegreen', alpha=0.2)

    for ax in [ax2, ax3]:
        _ = sns.boxplot(x=r'tstep', y='#F_subscribers', hue='type', data=df[df['tstep'] > 0],
                        palette={True: 'limegreen', False: 'crimson'}, width=0.5, ax=ax)
        for i,artist in enumerate(ax.artists):
            artist.set_edgecolor('white')
            _ = ax.tick_params(axis='both', which='major', labelsize=14)
            _ = ax.tick_params(axis='both', which='minor', labelsize=14)
            _ = ax.set_xlabel('')
            _ = ax.set_ylabel('')
    _ = ax2.legend_.remove()
    _ = ax3.legend_.remove()

    ax2.set_ylim(ymin_top-0.02, ymax_top+0.02)
    ax3.set_ylim(ymin_bot-0.02, ymax_bot+0.02)
    ax2.tick_params(labelright='on')
    ax3.tick_params(labelright='on')
    _ = ax2.fill_between([-0.5, 9.5], ymin_top-0.01, ymax_top+0.01, facecolor='crimson', alpha=0.2)
    _ = ax3.fill_between([-0.5, 9.5], ymin_bot-0.01, ymax_bot+0.01, facecolor='limegreen', alpha=0.2)

def media_scatter(data):
    """Draws the time evolutions of individual-level distributions of media reach across the entire public sphere."""

    lambda_F = data[1]
    eta = data[2]
    Tmedia_T = data[3]
    Tmedia_F = data[4]
    Fmedia_T = data[5]
    Fmedia_F = data[6]
    N = data[8]
    list_rhoF = data[9]

    fig, ax = plt.subplots(1, 4, figsize=(14,4), sharex=True, sharey=True)
    _ = plt.suptitle('Media reach ($\lambda^F = {%s}$, $\eta = {%s})$' %(str(lambda_F), str(eta)), size=18,
                     y=1.02)

    sorted_time = sorted(Tmedia_T.keys())
    indices = [0, int(len(sorted_time) / 4), int(len(sorted_time) / 2), -1]

    for ind in range(4):
        time = sorted_time[indices[ind]]

        Tmedia_T_fin = Tmedia_T[time]
        Tmedia_F_fin = Tmedia_F[time]
        Fmedia_T_fin = Fmedia_T[time]
        Fmedia_F_fin = Fmedia_F[time]

        Tscale = []
        Fscale = []

        RF = list_rhoF[indices[ind]]
        RT = N - RF

        for i in range(len(Tmedia_T_fin)):
            Tscale.append((Tmedia_T_fin[i] + Tmedia_F_fin[i]))
            Fscale.append((Fmedia_T_fin[i] + Fmedia_F_fin[i]))

        _ = ax[ind].scatter([i/RT for i in Tmedia_T_fin], [i/RF for i in Tmedia_F_fin],
                            c='limegreen', s=Tscale, label=True, alpha=0.7, edgecolors='black')
        _ = ax[ind].scatter([i/RT for i in Fmedia_T_fin], [i/RF for i in Fmedia_F_fin],
                            c='crimson', s=Fscale, label=False, alpha=0.7, edgecolors='black')
        _ = ax[ind].set_title(r'$t = {%d}$' %(time), size=16)

        if ind == 0:
            _ = ax[ind].set_ylabel('$m_j^F(t)/R^F(t)$', size=16)

        _ = ax[ind].tick_params(axis='both', which='major', labelsize=14)
        _ = ax[ind].tick_params(axis='both', which='minor', labelsize=14)
        _ = ax[ind].set_xlim(0, 1)
        _ = ax[ind].set_ylim(0, 1)

    _ = fig.add_subplot(111, frameon=False)
    _ = plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    _ = plt.xlabel('$m_j^T(t)/R^T(t)$', fontsize=16, labelpad=10)
    _ = plt.grid(False)

    handles, _ = ax[0].get_legend_handles_labels()
    _ = plt.legend(handles, ['$\mathcal{M}^T$', '$\mathcal{M}^F$'], #bbox_to_anchor=(0.97,0.77),
                  loc='upper right', fancybox=True, framealpha=0.5, fontsize=16, markerscale=0.6)
    _ = plt.tight_layout()

def readership(data):
    """Draws the time evolutions of individual-level compositions of readership."""

    lambda_F = data[1]
    eta = data[2]
    Tmedia_T = data[3]
    Tmedia_F = data[4]
    Fmedia_T = data[5]
    Fmedia_F = data[6]
    N = data[8]

    fig, ax = plt.subplots(1, 3, figsize=(12,3.5), sharex=True, sharey=True, dpi= 80)

    for d in [Tmedia_T, Tmedia_F, Fmedia_T, Fmedia_F]:
        d[sorted(d.keys())[-1]] = d[sorted(d.keys())[-1]][0:10]

    sorted_time = sorted(Tmedia_T.keys())
    indices = [0, int(len(sorted_time) / 2), -1]

    for ind in range(3):
        time = sorted_time[indices[ind]]

        Tmedia_T_fin = Tmedia_T[time]
        Tmedia_F_fin = Tmedia_F[time]
        Fmedia_T_fin = Fmedia_T[time]
        Fmedia_F_fin = Fmedia_F[time]

        xticks = []

        if sum(Tmedia_T_fin) + sum(Fmedia_T_fin) == 0:
            colors = ['crimson']
        elif sum(Tmedia_F_fin) + sum(Fmedia_F_fin)== 0:
            colors = ['limegreen']
        else:
            colors = ['crimson', 'limegreen']

        rows_list = []
        for row in range(len(Tmedia_T_fin)):

            for i in range(Tmedia_T_fin[row]):
                dict1 = {}
                dict1.update({'id': row, 'type': True, 'type_subscribers': 'T subscribers', 'amount_subscribers': 1})
                rows_list.append(dict1)

            for i in range(Tmedia_F_fin[row]):
                dict1 = {}
                dict1.update({'id': row, 'type': True, 'type_subscribers': 'F subscribers', 'amount_subscribers': 1})
                rows_list.append(dict1)

            for i in range(Fmedia_T_fin[row]):
                dict1 = {}
                dict1.update({'id': row+12, 'type': False, 'type_subscribers': 'T subscribers', 'amount_subscribers': 1})
                rows_list.append(dict1)

            for i in range(Fmedia_F_fin[row]):
                dict1 = {}
                dict1.update({'id': row+12, 'type': False, 'type_subscribers': 'F subscribers', 'amount_subscribers': 1})
                rows_list.append(dict1)

            xticks.append('')
            if row == 4:
                xticks.append('$\mathcal{M}^T$')

        for row in range(len(Fmedia_T_fin)):
            xticks.append('')
            if row == 5:
                xticks.append('$\mathcal{M}^F$')

        df = pd.DataFrame(rows_list)

        x_var = 'id'
        groupby_var = 'type_subscribers'
        df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
        vals = [df[x_var].values.tolist() for i, df in df_agg]

        n, bins, patches = ax[ind].hist(vals, df[x_var].unique().__len__()+2, stacked=True, density=False,
                                    color=colors)
        _ = plt.legend({group:col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})
        _ = plt.legend(labels=['$\mathcal{R}^F(t)$', '$\mathcal{R}^T(t)$'], loc='upper right', fontsize=16,
                       fancybox=True, framealpha=0.5)
        _ = ax[ind].set_title(r'$t = {%d}$' %(time), size=16)
        _ = plt.suptitle(r'Subscriber polarity ($\lambda^F =$ {%s}, $\eta =$ {%s})' %(str(lambda_F), str(eta)),
                         size=18, y=1.04)

        _ = ax[ind].set_ylim(0,N)
        _ = ax[ind].set_xticks(ticks=bins)
        _ = ax[ind].set_xticklabels(xticks, size=16)
        if ind == 0:
            _ = ax[ind].set_ylabel(r'$m_j^F(t)/R \qquad m_j^T(t)/R$', size=16)
        yticks = ax[ind].get_yticks()
        _ = ax[ind].set_yticklabels([y/N for y in yticks], size=14)
        _ = ax[ind].grid(False, axis='x')
    _ = plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    _ = plt.tight_layout()