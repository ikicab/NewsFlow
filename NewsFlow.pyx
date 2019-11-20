# cython: infer_types=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: language = c++
# cython -c=-std=c++11 --cplus
# -*- coding: utf-8 -*-

### to enable profiling: # cython: profile=True

"""
@author: Barbara Ikica
"""

import igraph as ig
import matplotlib.pyplot as plt

cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, exp
from libc.time cimport clock_t, CLOCKS_PER_SEC, clock
from libcpp cimport bool

from cpython.array cimport array

from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from cyrandom import * # pip install git+https://github.com/Noctem/cyrandom.git

cdef double randnum():
    """Generates a random float in [0, 1]."""

    cdef double r = rand()/(RAND_MAX*1.0)
    return r

cdef int randint(int k):
    """Generates a random integer in [1, k+1]."""

    cdef int i = rand() / (RAND_MAX / (k + 1) + 1) + 1
    return i

cdef vector[int] randlist(int sample_size, int population_size):
    """Generates a list of random integers in [0, population_size) of length
    sample_size (without repetition)."""

    cdef int t = 0
    cdef int m = 0
    cdef int i = 0
    cdef vector[int] samples
    cdef double u

    while m < sample_size:

        u = randnum()

        if  (population_size - t)*u >= sample_size - m :
            t += 1
        else:
            samples.push_back(t)
            t += 1
            m += 1

    return samples

cdef int bisect_right(double [::1] a, double x, int hi):
    """Assuming that the list a is sorted, return the index at which the item x
    should be inserted in this list. The returned value lo is such that e <= x
    holds for all e in a[:lo] and e > x holds for all e in a[lo:]. Thus, if x
    already appears in the list, the new item would be inserted right after the
    rightmost x already there. Optional argument hi bounds the slice of a to be
    searched."""

    cdef int lo = 0

    while lo < hi:
        mid = (lo+hi)//2
        if x < a[mid]: hi = mid
        else: lo = mid+1

    return lo

cdef double balance(int kT, int kF):
    """Computes the balance of the readership of a media outlet.
    ===========================================================================
    Parameters
    ---------------------------------------------------------------------------
    kT : number of T subscribers to the given news agency
    kF : number of F subscribers to the given news agency
    ---------------------------------------------------------------------------
    Returns
    ---------------------------------------------------------------------------
    B : balance based on the normalised Shannon entropy
    """

    cdef double B = 0.0
    cdef double r = 0.0

    if (kT == 0) or (kF == 0):
        pass
    else:
        r = kF/((kT + kF)*1.0)
        B = -(1 - r)*log(1 - r) - r*log(r)
        B /= log(2)

    return B

cdef bool isinvector(int val, vector[int] vec):
    """Returns TRUE if val is present in vec and FALSE otherwise."""

    cdef int i = 0

    for i in vec:
        if i == val:
            return True

    return False

ctypedef vector[int] int_vec
ctypedef vector[int_vec] int_dict

def newsFlow(g, int F0, double eta, double lambda_F, double lambda_T=1, int M=10,
             int t_max=0, Py_ssize_t window_size=0, double tol=0.02, int sfreq=0):
    """
    Simulates the flow of the news across a co-evolving graph with a media layer and a population layer.
    ===========================================================================
    Parameters
    ---------------------------------------------------------------------------
    g : an unweighted undirected graph
    F0 : initial number of F news consumers
    eta : rate of rewiring to the media
    lambda_F : spreading rate of F news
    lambda_T : spreading rate of T news (default value: 1)
    M : number of T (= F) media outlets; in total, there are 2*M outlets (default value: 10)
    t_max : (optional) upper limit on the simulation time (default value: 0)
    window_size : (optional) length of the sliding window (default value: 0)
    tol : tolerance on the sliding-window variances (default value: 0.02)
    sfreq : (optional) state saving frequency (default value: 0)
    ---------------------------------------------------------------------------
    Returns
    ---------------------------------------------------------------------------
    if sfreq > 0:
        list_rhoF : detailed time evolution (with a sampling frequency of sfreq) of the number of F news consumers
        X_Ymedia : detailed time evolution (with a sampling frequency of sfreq) of the number of Y media outlets subscribed to by X news consumers (X and Y standing for T or F)
        Xmedia_Y : detailed time evolution (with a sampling frequency of sfreq) of the number of Y news consumers subscribed to X media outlets (X and Y standing for T or F)
    elif sfreq == 0:
        time_range : time-step evolution
        list_polarised : detailed time evolution of the number of polarised individuals
        list_rhoF : detailed time evolution of the number of F news consumers
        list_FT_edges : detailed time evolution of the number of FT (unbalanced/active) edges
    else:
        v_polarised : final number of polarised individuals
        rhoF : final number of F news consumers in the population
        nFT_edges : final number of FT (unbalanced/active) edges
    """

    g.simplify()
    g.delete_vertices(g.vs.select(lambda vertex: vertex.degree() == 0))

    if (t_max == 0) & (window_size == 0):
        window_size = len(g.vs)

    cdef int N = int(len(g.vs)) # realised number of vertices
    cdef Py_ssize_t E = len(g.es) # realised number of edges

    assert F0 >= 1, "Less than one F news consumer, F0 < 1."
    assert F0 < N, "The initial number of F news consumers F0 exceeds the size of the population N."

    cdef int e, i, j, k, s, t, flag
    cdef double total, x

    cdef double sumFx_T = 0
    cdef double sumFx_F = 0

    cdef array[double] Fx_T = array('d', [0 for i in range(E)])
    cdef double [::1] Fx_T_view = Fx_T
    cdef array[double] Fx_F = array('d', [0 for i in range(E)])
    cdef double [::1] Fx_F_view = Fx_F

    cdef int nFT_edges = 0

    cdef clock_t time_start = clock()

    cdef vector[int] list_rhoF # number of F news consumers
    cdef vector[int] list_polarised  # number of polarised individuals
    cdef vector[int] list_FT_edges # number of FT edges

    # initialise the number of T/F media outlets an individual subscribes to
    cdef array[int] T_news = array('i', [0 for i in range(N)])
    cdef int [::1] T_news_view = T_news
    cdef array[int] F_news = array('i', [0 for i in range(N)])
    cdef int [::1] F_news_view = F_news

    cdef list states_list = [0 for i in range(N)] # states of individuals: T (0), F (1)

    # introduce F0 false states
    for i in range(F0):
        states_list[i] = 1
    cyrandom.shuffle(states_list)
    cdef array[int] states = array('i', states_list)
    cdef int [::1] states_view = states

    cdef int rhoF = F0 # number of F news consumers (density: rhoF/N)

    # store the indices of the media that individuals subscribe to
    cdef int_dict map_Tmedia
    cdef int_dict map_Fmedia
    map_Tmedia.reserve(N)
    map_Fmedia.reserve(N)

    # store the numbers of T/F news consumers subscribed to T/F media outlets
    cdef array[int] Tmedia_kT = array('i', [0 for i in range(M)])
    cdef int [::1] Tmedia_kT_view = Tmedia_kT
    cdef array[int] Tmedia_kF = array('i', [0 for i in range(M)])
    cdef int [::1] Tmedia_kF_view = Tmedia_kF
    cdef array[int] Fmedia_kT = array('i', [0 for i in range(M)])
    cdef int [::1] Fmedia_kT_view = Fmedia_kT
    cdef array[int] Fmedia_kF = array('i', [0 for i in range(M)])
    cdef int [::1] Fmedia_kF_view = Fmedia_kF

    for i in range(N):
        T_news_view[i] = randint(M-1) # assign 1 <= T media <= M to individual i
        F_news_view[i] = randint(M-1) # assign 1 <= F media <= M to individual i

        map_Tmedia.push_back(randlist(T_news_view[i], M)) # indices of T media that individual i subscribes to
        map_Fmedia.push_back(randlist(F_news_view[i], M)) # indices of F media that individual i subscribes to

        if states_view[i] == 0:
            for j in map_Tmedia[i]:
                Tmedia_kT_view[j] += 1
            for j in map_Fmedia[i]:
                Fmedia_kT_view[j] += 1
        else:
            for j in map_Tmedia[i]:
                Tmedia_kF_view[j] += 1
            for j in map_Fmedia[i]:
                Fmedia_kF_view[j] += 1

    # store neighbourhood information
    cdef array[int] edges_s = array('i', [edge.source for edge in g.es])
    cdef int [::1] edges_s_view = edges_s
    cdef array[int] edges_t = array('i', [edge.target for edge in g.es])
    cdef int [::1] edges_t_view = edges_t
    cdef array[int] edges_FT = array('i', [0 for i in range(E)])
    cdef int [::1] edges_FT_view = edges_FT

    cdef int_dict map_incidentEdges
    map_incidentEdges.reserve(N)

    for i in range(N):
        map_incidentEdges.push_back(g.incident(i))

        for e in map_incidentEdges[i]:
            # characterise edges according to their endpoints
            s = edges_s_view[e]
            t = edges_t_view[e]

            if states_view[s] != states_view[t]: # the endpoints of the edge correspond to different states
                nFT_edges += 1
                edges_FT_view[e] = 1

                if states_view[s] == 1:
                    Fx_F_view[e] = F_news_view[t]/((F_news_view[t] + T_news_view[t])*1.0)
                    Fx_T_view[e] = T_news_view[s]/((F_news_view[s] + T_news_view[s])*1.0)

                else:
                    Fx_F_view[e] = F_news_view[s]/((F_news_view[s] + T_news_view[s])*1.0)
                    Fx_T_view[e] = T_news_view[t]/((F_news_view[t] + T_news_view[t])*1.0)

                sumFx_T += Fx_T_view[e]
                sumFx_F += Fx_F_view[e]

    # avoid overcounting (e was counted twice; with i = s and i = t)
    nFT_edges /= 2
    sumFx_T /= 2.0
    sumFx_F /= 2.0

    cdef double Fx_T_avg = sumFx_T/nFT_edges
    cdef double Fx_F_avg = sumFx_F/nFT_edges

    # compute the number of actions
    cdef double N_Act_TT = nFT_edges * lambda_T * Fx_T_avg  # FT --> TT
    cdef double N_Act_FF = nFT_edges * lambda_F * Fx_F_avg  # FT --> FF
    cdef double N_Act_vert = N * eta  # an individual updates its T/F media ratio
    cdef double N_Act = N_Act_TT + N_Act_FF + N_Act_vert  # total number of actions

    cdef vector[double] time_range # time span

    cdef double t_step = 0  # track the total time elapsed
    cdef int step = 0  # count the number of iterations realised

    cdef int v_polarised = 0 # number of polarised individuals

    time_range.push_back(t_step)
    list_polarised.push_back(v_polarised)
    list_rhoF.push_back(rhoF) # number of F news consumers
    list_FT_edges.push_back(nFT_edges) # number of FT edges

    cdef int action

    cdef array[double] cum_weights = array('d', [0 for i in range(E)])
    cdef double [::1] cum_weights_view = cum_weights

    cdef array[int] temp = array('i', [0 for i in range(E)])
    cdef int [::1] temp_view = temp

    cdef int T_news_old
    cdef int F_news_old

    # all possible actions
    cdef int CHANGE_TT = 0
    cdef int CHANGE_FF = 1
    cdef int UPDATE_VERTEX = 2

    cdef double mu_e = nFT_edges
    cdef double mu_e_old = 0.0
    cdef double var_e = nFT_edges**2

    cdef double mu_v = v_polarised
    cdef double mu_v_old = 0.0
    cdef double var_v = v_polarised**2

    cdef double E_squared = E*E * tol*tol
    cdef double N_squared = N*N * tol*tol

    cdef int counter = 0

    # X_Ymedia : detailed time evolution of the number of Y media outlets subscribed to by X news consumers (X and Y standing for T or F)
    cdef unordered_map[double, vector[int]] T_Fmedia
    cdef unordered_map[double, vector[int]] F_Fmedia
    cdef unordered_map[double, vector[int]] T_Tmedia
    cdef unordered_map[double, vector[int]] F_Tmedia

    # Xmedia_Y : detailed time evolution of the number of Y news consumers subscribed to X media outlets (X and Y standing for T or F)
    cdef unordered_map[double, vector[int]] Tmedia_T
    cdef unordered_map[double, vector[int]] Tmedia_F
    cdef unordered_map[double, vector[int]] Fmedia_T
    cdef unordered_map[double, vector[int]] Fmedia_F

    for i in range(N):
        if states_view[i] == 0:
            T_Fmedia[0].push_back(F_news_view[i])
            T_Tmedia[0].push_back(T_news_view[i])
        else:
            F_Fmedia[0].push_back(F_news_view[i])
            F_Tmedia[0].push_back(T_news_view[i])

    for i in range(M):
        Tmedia_T[0].push_back(Tmedia_kT_view[i])
        Tmedia_F[0].push_back(Tmedia_kF_view[i])
        Fmedia_T[0].push_back(Fmedia_kT_view[i])
        Fmedia_F[0].push_back(Fmedia_kF_view[i])

    while 1:
        # choose a random action:
        # CHANGE_TT: a randomly picked FT edge becomes TT,
        # CHANGE_FF: a randomly picked FT edge becomes FF,
        # UPDATE_VERTEX: a randomly picked vertex updates its T to F news ratio (T += 1 and F -=1, or vice-versa)

        total = N_Act_TT
        cum_weights_view[0] = total
        total += N_Act_FF
        cum_weights_view[1] = total
        total += N_Act_vert
        cum_weights_view[2] = total

        x = randnum() * total
        action = bisect_right(cum_weights_view, x, 2)

        if action == CHANGE_TT:
            rhoF -= 1

            # select an edge at random with probability proportional to its F_T(x) value
            j = 0
            total = 0

            for e in range(E):
                if edges_FT_view[e] == 1:
                    total += Fx_T_view[e]
                    cum_weights_view[j] = total
                    temp_view[j] = e
                    j += 1

            x = randnum() * total
            i = bisect_right(cum_weights_view, x, j-1)
            e = temp_view[i]

            i = edges_s_view[e]
            j = edges_t_view[e]

            if states_view[i] != 0:
                states_view[i] = 0 # update the state of individual i from F to T

                for e in map_incidentEdges[i]:
                    edges_FT_view[e] = 1 - edges_FT_view[e] # exchange the roles of active and inactive edges

                    s = edges_s_view[e]
                    t = edges_t_view[e]

                    # update the F(x) values of the incident edges
                    if edges_FT_view[e] == 1:
                        nFT_edges += 1

                        if states_view[s] == 1:
                            Fx_F_view[e] = F_news_view[t]/((F_news_view[t] + T_news_view[t])*1.0)
                            Fx_T_view[e] = T_news_view[s]/((F_news_view[s] + T_news_view[s])*1.0)
                        else:
                            Fx_F_view[e] = F_news_view[s]/((F_news_view[s] + T_news_view[s])*1.0)
                            Fx_T_view[e] = T_news_view[t]/((F_news_view[t] + T_news_view[t])*1.0)

                        sumFx_T += Fx_T_view[e]
                        sumFx_F += Fx_F_view[e]
                    else:
                        nFT_edges -= 1

                        sumFx_T -= Fx_T_view[e]
                        sumFx_F -= Fx_F_view[e]

                for e in map_Tmedia[i]:
                    Tmedia_kT_view[e] += 1
                    Tmedia_kF_view[e] -= 1

                for e in map_Fmedia[i]:
                    Fmedia_kT_view[e] += 1
                    Fmedia_kF_view[e] -= 1

            else:
                states_view[j] = 0 # update the state of individual j from F to T

                for e in map_incidentEdges[j]:
                    edges_FT_view[e] = 1 - edges_FT_view[e] # exchange the roles of active and inactive edges

                    s = edges_s_view[e]
                    t = edges_t_view[e]

                    # update the F(x) values of the incident edges
                    if edges_FT_view[e] == 1:
                        nFT_edges += 1

                        if states_view[s] == 1:
                            Fx_F_view[e] = F_news_view[t]/((F_news_view[t] + T_news_view[t])*1.0)
                            Fx_T_view[e] = T_news_view[s]/((F_news_view[s] + T_news_view[s])*1.0)
                        else:
                            Fx_F_view[e] = F_news_view[s]/((F_news_view[s] + T_news_view[s])*1.0)
                            Fx_T_view[e] = T_news_view[t]/((F_news_view[t] + T_news_view[t])*1.0)

                        sumFx_T += Fx_T_view[e]
                        sumFx_F += Fx_F_view[e]
                    else:
                        nFT_edges -= 1

                        sumFx_T -= Fx_T_view[e]
                        sumFx_F -= Fx_F_view[e]

                for e in map_Tmedia[j]:
                    Tmedia_kT_view[e] += 1
                    Tmedia_kF_view[e] -= 1

                for e in map_Fmedia[j]:
                    Fmedia_kT_view[e] += 1
                    Fmedia_kF_view[e] -= 1

        elif action == CHANGE_FF:
            rhoF += 1

            # select an edge at random with probability proportional to its F_F(x) value
            j = 0
            total = 0

            for e in range(E):
                if edges_FT_view[e] == 1:
                    total += Fx_F_view[e]
                    cum_weights_view[j] = total
                    temp_view[j] = e
                    j += 1

            x = randnum() * total
            i = bisect_right(cum_weights_view, x, j-1)
            e = temp_view[i]

            i = edges_s_view[e]
            j = edges_t_view[e]

            if states_view[i] != 1:
                states_view[i] = 1 # update the state of individual i from T to F

                for e in map_incidentEdges[i]:
                    edges_FT_view[e] = 1 - edges_FT_view[e] # exchange the roles of active and inactive edges

                    s = edges_s_view[e]
                    t = edges_t_view[e]

                    # update the F(x) values of the incident edges
                    if edges_FT_view[e] == 1:
                        nFT_edges += 1

                        if states_view[s] == 1:
                            Fx_F_view[e] = F_news_view[t]/((F_news_view[t] + T_news_view[t])*1.0)
                            Fx_T_view[e] = T_news_view[s]/((F_news_view[s] + T_news_view[s])*1.0)
                        else:
                            Fx_F_view[e] = F_news_view[s]/((F_news_view[s] + T_news_view[s])*1.0)
                            Fx_T_view[e] = T_news_view[t]/((F_news_view[t] + T_news_view[t])*1.0)

                        sumFx_T += Fx_T_view[e]
                        sumFx_F += Fx_F_view[e]
                    else:
                        nFT_edges -= 1

                        sumFx_T -= Fx_T_view[e]
                        sumFx_F -= Fx_F_view[e]

                for e in map_Tmedia[i]:
                    Tmedia_kT_view[e] -= 1
                    Tmedia_kF_view[e] += 1

                for e in map_Fmedia[i]:
                    Fmedia_kT_view[e] -= 1
                    Fmedia_kF_view[e] += 1

            else:
                states_view[j] = 1 # update the state of individual j from T to F

                for e in map_incidentEdges[j]:
                    edges_FT_view[e] = 1 - edges_FT_view[e] # exchange the roles of active and inactive edges

                    s = edges_s_view[e]
                    t = edges_t_view[e]

                    # update the F(x) values of the incident edges
                    if edges_FT_view[e] == 1:
                        nFT_edges += 1

                        if states_view[s] == 1:
                            Fx_F_view[e] = F_news_view[t]/((F_news_view[t] + T_news_view[t])*1.0)
                            Fx_T_view[e] = T_news_view[s]/((F_news_view[s] + T_news_view[s])*1.0)
                        else:
                            Fx_F_view[e] = F_news_view[s]/((F_news_view[s] + T_news_view[s])*1.0)
                            Fx_T_view[e] = T_news_view[t]/((F_news_view[t] + T_news_view[t])*1.0)

                        sumFx_T += Fx_T_view[e]
                        sumFx_F += Fx_F_view[e]
                    else:
                        nFT_edges -= 1

                        sumFx_T -= Fx_T_view[e]
                        sumFx_F -= Fx_F_view[e]

                for e in map_Tmedia[j]:
                    Tmedia_kT_view[e] -= 1
                    Tmedia_kF_view[e] += 1

                for e in map_Fmedia[j]:
                    Fmedia_kT_view[e] -= 1
                    Fmedia_kF_view[e] += 1

        else:
            # choose an individual uniformly at random
            i = randint(N-1) - 1

            T_news_old = T_news_view[i]
            F_news_old = F_news_view[i]

            flag = 0

            if states_view[i] == 1: # if the individual is a F news consumer: F_news += 1, T_news -= 1
                if T_news_old > 0:
                    flag = 1
                    T_news_view[i] -= 1

                    # unsubscribe from a T media outlet chosen uniformly at random
                    j = 0
                    total = 0

                    for e in map_Tmedia[i]:
                        total += 1
                        cum_weights_view[j] = total
                        j += 1

                    x = randnum() * total
                    s = bisect_right(cum_weights_view, x, j-1)
                    t = map_Tmedia[i][s]

                    Tmedia_kF_view[t] -= 1

                    map_Tmedia[i].erase(map_Tmedia[i].begin() + s)

                    # subscribe to a F or T media outlet with probability proportional to media_Fprob
                    j = 0
                    total = 0

                    for e in range(M):
                        if not isinvector(e, map_Tmedia[i]):
                            total += balance(Tmedia_kT_view[e], Tmedia_kF_view[e])/(1 + exp(-Tmedia_kF_view[e]))
                            cum_weights_view[j] = total
                            temp_view[j] = e
                            j += 1

                    k = j

                    if F_news_old < M:
                        for e in range(M):
                            if not isinvector(e, map_Fmedia[i]):
                                total += balance(Fmedia_kT_view[e], Fmedia_kF_view[e])/(1 + exp(-Fmedia_kF_view[e]))
                                cum_weights_view[j] = total
                                temp_view[j] = e
                                j += 1

                    x = randnum() * total
                    s = bisect_right(cum_weights_view, x, j-1)
                    t = temp_view[s]

                    if s < k: # T media was chosen
                        T_news_view[i] += 1
                        Tmedia_kF_view[t] += 1
                        map_Tmedia[i].push_back(t)
                    else:
                        # F media was chosen
                        F_news_view[i] += 1
                        Fmedia_kF_view[t] += 1
                        map_Fmedia[i].push_back(t)

            else: # if the individual is a T news consumer: T_news += 1, F_news -= 1
                if F_news_old > 0:
                    flag = 1
                    F_news_view[i] -= 1

                    # unsubscribe from a F media outlet chosen uniformly at random
                    j = 0
                    total = 0

                    for e in map_Fmedia[i]:
                        total += 1
                        cum_weights_view[j] = total
                        j += 1

                    x = randnum() * total
                    s = bisect_right(cum_weights_view, x, j-1)
                    t = map_Fmedia[i][s]

                    Fmedia_kT_view[t] -= 1

                    map_Fmedia[i].erase(map_Fmedia[i].begin() + s)

                    # subscribe to a F or T media outlet with probability proportional to media_Tprob
                    j = 0
                    total = 0

                    for e in range(M):
                        if not isinvector(e, map_Fmedia[i]):
                            total += balance(Fmedia_kT_view[e], Fmedia_kF_view[e])/(1 + exp(-Fmedia_kT_view[e]))
                            cum_weights_view[j] = total
                            temp_view[j] = e
                            j += 1

                    k = j

                    if T_news_old < M:
                        for e in range(M):
                            if not isinvector(e, map_Tmedia[i]):
                                total += balance(Tmedia_kT_view[e], Tmedia_kF_view[e])/(1 + exp(-Tmedia_kT_view[e]))
                                cum_weights_view[j] = total
                                temp_view[j] = e
                                j += 1

                    x = randnum() * total
                    s = bisect_right(cum_weights_view, x, j-1)
                    t = temp_view[s]

                    if s < k:
                        # F media was chosen
                        F_news_view[i] += 1
                        Fmedia_kT_view[t] += 1
                        map_Fmedia[i].push_back(t)
                    else:
                        # T media was chosen
                        T_news_view[i] += 1
                        Tmedia_kT_view[t] += 1
                        map_Tmedia[i].push_back(t)

            if flag:
                if ((T_news_old > 0) & (T_news_view[i] == 0)) or ((F_news_old > 0) & (F_news_view[i] == 0)):
                    v_polarised += 1
                elif ((T_news_old == 0) & (T_news_view[i] > 0)) or ((F_news_old == 0) & (F_news_view[i] > 0)):
                    v_polarised -= 1

                # update the F(x) values of the incident edges
                for e in map_incidentEdges[i]:
                    s = edges_s_view[e]
                    t = edges_t_view[e]

                    if edges_FT_view[e] == 1:
                        sumFx_T -= Fx_T_view[e]
                        sumFx_F -= Fx_F_view[e]

                        if states_view[s] == 1:
                            Fx_F_view[e] = F_news_view[t]/((F_news_view[t] + T_news_view[t])*1.0)
                            Fx_T_view[e] = T_news_view[s]/((F_news_view[s] + T_news_view[s])*1.0)
                        else:
                            Fx_F_view[e] = F_news_view[s]/((F_news_view[s] + T_news_view[s])*1.0)
                            Fx_T_view[e] = T_news_view[t]/((F_news_view[t] + T_news_view[t])*1.0)

                        sumFx_T += Fx_T_view[e]
                        sumFx_F += Fx_F_view[e]

        # update the number of actions
        N_Act_TT = nFT_edges * lambda_T * Fx_T_avg
        N_Act_FF = nFT_edges * lambda_F * Fx_F_avg
        N_Act_vert = N * eta
        N_Act = N_Act_TT + N_Act_FF + N_Act_vert

        t_step += 1/N_Act

        # update the data
        time_range.push_back(t_step)
        list_polarised.push_back(v_polarised)
        list_rhoF.push_back(rhoF)
        list_FT_edges.push_back(nFT_edges)

        if sfreq > 0:
            if t_step >= counter:
                for i in range(N):
                    if states_view[i] == 0:
                        T_Fmedia[t_step].push_back(F_news_view[i])
                        T_Tmedia[t_step].push_back(T_news_view[i])
                    else:
                        F_Fmedia[t_step].push_back(F_news_view[i])
                        F_Tmedia[t_step].push_back(T_news_view[i])

                for i in range(M):
                    Tmedia_T[t_step].push_back(Tmedia_kT_view[i])
                    Tmedia_F[t_step].push_back(Tmedia_kF_view[i])
                    Fmedia_T[t_step].push_back(Fmedia_kT_view[i])
                    Fmedia_F[t_step].push_back(Fmedia_kF_view[i])

                counter += sfreq

        if nFT_edges == 0:
            print("There are no more FT edges to choose from.")
            break

        Fx_F_avg = sumFx_F/nFT_edges
        Fx_T_avg = sumFx_T/nFT_edges

        if Fx_F_avg < 1e-9 or Fx_T_avg < 1e-9:
            print("Either Fx_F_avg or Fx_T_avg equals 0.")
            break

        if t_max != 0:
            if t_step >= t_max:
                print("Max time step exceeded.")
                break

        step += 1

        if window_size != 0:
            if step < window_size-1:
                mu_e += nFT_edges
                var_e += nFT_edges**2

                mu_v += v_polarised
                var_v += v_polarised**2
            elif step == window_size-1:
                mu_e += nFT_edges
                mu_e /= (window_size*1.0)
                var_e += nFT_edges**2
                var_e = 1.0/((window_size - 1)*1.0) * (var_e - window_size * mu_e**2)

                mu_v += v_polarised
                mu_v /= (window_size*1.0)
                var_v += v_polarised**2
                var_v = 1.0/((window_size - 1)*1.0) * (var_v - window_size * mu_v**2)
            else:
                mu_e_old = mu_e
                mu_e += 1.0/(window_size*1.0) * (nFT_edges - list_FT_edges[step-window_size])
                var_e += 1.0/((window_size-1)*1.0) * ((nFT_edges - list_FT_edges[step-window_size]) *
                              (nFT_edges + list_FT_edges[step-window_size] - mu_e - mu_e_old))

                mu_v_old = mu_v
                mu_v += 1.0/(window_size*1.0) * (v_polarised - list_polarised[step-window_size])
                var_v += 1.0/((window_size-1)*1.0) * ((v_polarised - list_polarised[step-window_size]) *
                              (v_polarised + list_polarised[step-window_size] - mu_v - mu_v_old))

                if (var_e < E_squared) or (var_v < N_squared):
                    print('Variance in the number of FT edges/polarised individuals is too low.')
                    break

    cdef clock_t time_end = clock()

    map_incidentEdges.clear()
    map_incidentEdges.shrink_to_fit()
    map_Tmedia.clear()
    map_Tmedia.shrink_to_fit()
    map_Fmedia.clear()
    map_Fmedia.shrink_to_fit()

    for i in range(N):
        if states_view[i] == 0:
            T_Fmedia[t_step].push_back(F_news_view[i])
            T_Tmedia[t_step].push_back(T_news_view[i])
        else:
            F_Fmedia[t_step].push_back(F_news_view[i])
            F_Tmedia[t_step].push_back(T_news_view[i])

    for i in range(M):
        Tmedia_T[t_step].push_back(Tmedia_kT_view[i])
        Tmedia_F[t_step].push_back(Tmedia_kF_view[i])
        Fmedia_T[t_step].push_back(Fmedia_kT_view[i])
        Fmedia_F[t_step].push_back(Fmedia_kF_view[i])

    print('Elapsed time: %s s'%((time_end - time_start)*1.0 / CLOCKS_PER_SEC))

    if sfreq > 0:
        time_range.clear()
        time_range.shrink_to_fit()
        list_polarised.clear()
        list_polarised.shrink_to_fit()
        list_FT_edges.clear()
        list_FT_edges.shrink_to_fit()
        return (list_rhoF, T_Fmedia, F_Fmedia, T_Tmedia, F_Tmedia,
                Tmedia_T, Tmedia_F, Fmedia_T, Fmedia_F)
    elif sfreq == 0:
        return (time_range, list_polarised, list_rhoF, list_FT_edges)
    else:
        time_range.clear()
        time_range.shrink_to_fit()
        list_polarised.clear()
        list_polarised.shrink_to_fit()
        list_rhoF.clear()
        list_rhoF.shrink_to_fit()
        list_FT_edges.clear()
        list_FT_edges.shrink_to_fit()
        return (v_polarised, rhoF, nFT_edges)