# -*- coding: utf-8 -*-
"""
@author: Barbara Ikica
"""

import random
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import time


def stopaj(f):
    def stopana_f(*args, **kwargs):
        zacetek = time.time()
        rezultat = f(*args, **kwargs)
        konec = time.time()
        print(f.__name__, konec - zacetek)
        return rezultat
    return stopana_f

random.seed(12345)
np.random.seed(12345)

n = 100
k = 3
M = 100
F0 = 50
t_max = 100
Lambda_T = 1
Lambda_F = 1.5
eta = 0.8

CHANGE_TT = 0
CHANGE_FF = 1
UPDATE_VERTEX = 2


global sumFx_T
global sumFx_F


def rand_distr(items, p):
    """items - items to randomly choose from
    p - corresponding probabilities"""

    r = random.random()
    s = 0
    l = len(items)
    i = 0
    for i in range(l):
        s += p[i]
        if s >= r:
            return items[i]
    return items[i]


global G
G = ig.Graph.Barabasi(n, k, power=-3)
G.simplify()
G.delete_vertices(G.vs.select(lambda vertex: vertex.degree() == 0))

N = len(G.vs)
E = len(G.es)


def is_change(action):
    return action == CHANGE_TT or action == CHANGE_FF


def update_neighbours(v, action):
    """v - vertex
    action - current action"""

    global sumFx_T
    global sumFx_F

    # active & inactive incident edges exchange their roles
    v_incident = G.es[G.incident(v)]

    if is_change(action):
        v["state"] = action
        v_incident["FT"] = [not x for x in v_incident["FT"]]

    #? if action == UPDATE_VERTEX:
    #?     NFv = v["F_news"]
    #?     NTv = v["T_news"]
    #?     v["x"] = (NFv - NTv)/(NFv + NTv)

    # endpoints of incident edges update their F(x) values
    for incident_edge in v_incident:
        #? t1 = G.vs[incident_edge.source]
        #? t2 = G.vs[incident_edge.target]
        #? s = t1 if t2 == v else t1

        s = G.vs[incident_edge.source]
        t = G.vs[incident_edge.target]

        if action == UPDATE_VERTEX:
            NFs = s["F_news"]
            NTs = s["T_news"]
            s["x"] = (NFs - NTs)/(NFs + NTs)

            NFt = t["F_news"]
            NTt = t["T_news"]
            t["x"] = (NFt - NTt)/(NFt + NTt)

        xs = s["x"]
        #? xt = v["x"]
        xt = t["x"]

        if incident_edge["FT"]:
            if action == UPDATE_VERTEX:
                sumFx_T -= incident_edge["Fx_T"]
                sumFx_F -= incident_edge["Fx_F"]

            if s["state"] == 0:
                incident_edge["Fx_F"] = 1/2 * (xs + 1)  # probability of transitioning to FF
                incident_edge["Fx_T"] = 1/2 * (-xt + 1)  # probability of transitioning to TT

            else:
                incident_edge["Fx_F"] = 1/2 * (xt + 1)  # probability of transitioning to FF
                incident_edge["Fx_T"] = 1/2 * (-xs + 1)  # probability of transitioning to TT

            sumFx_T += incident_edge["Fx_T"]
            sumFx_F += incident_edge["Fx_F"]

        elif is_change(action):
            sumFx_T -= incident_edge["Fx_T"]
            sumFx_F -= incident_edge["Fx_F"]


@stopaj
def newsFlow(plot=0):
    """plot - if set to 1, output a plot"""

    global sumFx_T
    global sumFx_F

    list_rhoF = []  # density of false news followers
    list_one_media = []  # number of people following only either true or false media
    list_FT_edges = []  # density of FT (active) edges
    list_iter = []  # iteration step

    # initialise the amount of true/false media outlets an individual follows
    T_news = np.random.randint(1, M+1, size=N)
    F_news = np.random.randint(1, M+1, size=N)

    G.vs["T_news"] = np.ndarray.tolist(T_news)
    G.vs["F_news"] = np.ndarray.tolist(F_news)
    G.vs["x"] = [(v["F_news"] - v["T_news"])/(v["F_news"] + v["T_news"]) for v in G.vs]

    # states of nodes: T (0), F (1)
    G.vs["state"] = 0  # start with an all-true state

    # introduce a few false states
    rF0 = np.ndarray.tolist(np.random.choice(range(N), size=F0, replace=0))
    G.vs[rF0]["state"] = 1

    # in action 3, each vertex is chosen uniformly at random
    prob_v = np.empty(N)
    prob_v.fill(1/N)

    # characterise edges according to their endpoints
    for edge in G.es:
        i = G.vs[edge.source]
        j = G.vs[edge.target]

        if i["state"] == j["state"]:
            edge["FT"] = False  # the endpoints of the edge correspond to same states

        else:
            edge["FT"] = True  # the endpoints of the edge correspond to different states

            xi = i["x"]
            xj = j["x"]

            if i["state"] == 0:
                edge["Fx_F"] = 1/2 * (xi + 1)  # probability of transitioning to FF
                edge["Fx_T"] = 1/2 * (-xj + 1)  # probability of transitioning to TT

            else:
                edge["Fx_F"] = 1/2 * (xj + 1)  # probability of transitioning to FF
                edge["Fx_T"] = 1/2 * (-xi + 1)  # probability of transitioning to TT

    # number of FT (imbalanced/active) edges
    FT_edges = G.es.select(FT=True)
    nFT_edges = len(FT_edges)

    sumFx_T = sum(FT_edges["Fx_T"])
    sumFx_F = sum(FT_edges["Fx_F"])

    Fx_F_avg = sumFx_F/nFT_edges
    Fx_T_avg = sumFx_T/nFT_edges

    # compute the number of actions
    N_Act_TT = nFT_edges * Lambda_T * Fx_T_avg  # an active edge (FT) updates to a TT edge
    N_Act_FF = nFT_edges * Lambda_F * Fx_F_avg  # an active edge (FT) updates to a FF edge
    N_Act_node = N * eta  # a node updates the amount of true/false media it follows
    N_Act = N_Act_TT + N_Act_FF + N_Act_node  # total number of actions

    time_range = []  # time span

    t_step = 0  # time elapsed at current step
    step = 0  # counting the number of iterations realised

    if plot == 1:
        time_range.append(t_step)
        list_one_media.append(len([v for v in G.vs if v["T_news"] == 0 or v["F_news"] == 0]))
        list_rhoF.append(len([v for v in G.vs if v["state"] == 1])/N)
        list_FT_edges.append(nFT_edges/E)
        list_iter.append(step)

    # display current number of FT (active) edges
    #print("nFT_edges:", nFT_edges)

    while t_step <= t_max:
        # choose a random action:
        # 0: edge becomes TT,
        # 1: edge becomes FF,
        # 2: node updates its true to false news ratio (T/F += 1 & F/T -=1)

        action = rand_distr([CHANGE_TT, CHANGE_FF, UPDATE_VERTEX], [
                            N_Act_TT/N_Act, N_Act_FF/N_Act, N_Act_node/N_Act])

        if action == CHANGE_TT:
            # a randomly chosen active edge becomes TT

            # select an edge at random, with probability proportional to its F_T(x) value
            p = [f/sumFx_T for f in FT_edges["Fx_T"]]
            e_ind = rand_distr(range(nFT_edges), p)
            edge = FT_edges[e_ind]

            i = G.vs[edge.source]
            j = G.vs[edge.target]

            if i["state"] == 1:  # i updates to T
                update_neighbours(i, CHANGE_TT)

            else:  # j updates to T
                update_neighbours(j, CHANGE_TT)

        elif action == CHANGE_FF:
            # a randomly chosen active edge becomes FF

            # select an edge at random, with probability proportional to its F_F(x) value
            p = [f/sumFx_F for f in FT_edges["Fx_F"]]
            e_ind = rand_distr(range(nFT_edges), p)
            edge = FT_edges[e_ind]

            i = G.vs[edge.source]
            j = G.vs[edge.target]

            if i["state"] == 0:  # i updates to F
                update_neighbours(i, CHANGE_FF)

            else:  # j updates to F
                update_neighbours(j, CHANGE_FF)

        else:
            # a node updates its true to false media outlet ratio (T/F += 1 & F/T -=1)

            # select a vertex at random
            v_ind = rand_distr(range(N), prob_v)
            v = G.vs[v_ind]

            # if v is in T state: T_news += 1, F_news -= 1
            if v["state"] == 0:
                if v["T_news"] < M:
                    v["T_news"] += 1
                if v["F_news"] > 0:
                    v["F_news"] -= 1

            # if v is in F state: F_news += 1, T_news -= 1
            else:
                if v["T_news"] > 0:
                    v["T_news"] -= 1
                if v["F_news"] < M:
                    v["F_news"] += 1

            # endpoints of incident edges update their F(x) values
            update_neighbours(v, UPDATE_VERTEX)

        FT_edges = G.es.select(FT=1)
        nFT_edges = len(FT_edges)

        if nFT_edges == 0:
            step -= 1
            print("There are no more FT edges.")
            break

        Fx_F_avg = sumFx_F/nFT_edges
        Fx_T_avg = sumFx_T/nFT_edges

        if Fx_F_avg < 1e-9 or Fx_T_avg < 1e-9:
            step -= 1
            print("Either Fx_F_avg or Fx_T_avg equals 0.")
            break

        step += 1

        # update the number of actions
        N_Act_TT = nFT_edges * Lambda_T * Fx_T_avg
        N_Act_FF = nFT_edges * Lambda_F * Fx_F_avg
        N_Act_node = N * eta
        N_Act = N_Act_TT + N_Act_FF + N_Act_node  # number of actions

        if plot == 1:
            # update the data
            t_step += 1/N_Act
            time_range.append(t_step)
            list_one_media.append(len([v for v in G.vs if v["T_news"] == 0 or v["F_news"] == 0]))
            list_rhoF.append(len([v for v in G.vs if v["state"] == 1])/N)
            list_FT_edges.append(nFT_edges/E)
            list_iter.append(step)

    if t_step >= t_max:
        print("Max time step exceeded.")

    if plot == 1:
        # visualising the simulation data
        plt.tight_layout()
        plt.figure(figsize=(8, 8))
        plt.suptitle(r'$N =$ %s, $F_0 =$ %s, $M =$ %s, $\eta =$ %s, $\lambda_F =$ %s' %
                     (N, F0, 2*M, eta, Lambda_F), y=1.01)
        plt.tight_layout()

        # plotting the number of people who only listen to one type of media over time
        plt.subplot(2, 2, 1)
        plt.plot(time_range, list_one_media)
        plt.xlabel(r'$t$')
        plt.ylabel('# people only following T/F news')
        plt.tight_layout()

        # plotting the density of false news followers over time
        plt.subplot(2, 2, 2)
        plt.plot(time_range, list_rhoF)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\rho_F$')
        plt.tight_layout()

        # plotting the density of FT (active) edges over time
        plt.subplot(2, 2, 3)
        plt.plot(time_range, list_FT_edges)
        plt.ylim((0, 1))
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\rho_{FT}$')
        plt.tight_layout()

        # plotting the number of iterations over time
        plt.subplot(2, 2, 4)
        plt.plot(time_range, list_iter)
        plt.xlabel(r'$t$')
        plt.ylabel('# iter')
        plt.tight_layout()

    return len([v for v in G.vs if v["state"] == 1])/N

#list_lambda_F = [x * 0.5 for x in range(0,10)]
# for lambda_F in list_lambda_F:
#    newsFlow(n,k,M,F0,t_max,lambda_T,lambda_F,eta,rep)
