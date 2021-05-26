"""

Main library of custom functions for network analysis

"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import joblib
import numpy as np
import pandas as pd
from itertools import permutations, combinations, product
from numba import njit, prange
from tqdm import tqdm
import networkx as nx
import multiprocessing as mp
from math import factorial
from copy import copy
from matplotlib import pyplot as plt
import seaborn as sns

import networkx as nx
from networkx.algorithms.distance_measures import diameter
from networkx.algorithms.components import is_weakly_connected, is_strongly_connected, strongly_connected_components
from networkx.algorithms.centrality import degree_centrality, betweenness_centrality
from networkx.convert_matrix import to_numpy_array
from networkx.algorithms.swap import double_edge_swap
from collections import namedtuple

# combination number combinatorial formula
n_combs = lambda n, k: int(factorial(n)/factorial(n-k)/factorial(k))


def get_actual_parametrization(source, check_input=True, verbose=True):
    """
    Check the common config file and output study parameters
    Attributes:
    source, str - path to config
    check_input, bool - validate config params if True
    verbose - output logs if True
    Return: config, dict
    """ 
    cfg = source if type(source) is dict else json.load(open(source, "r"))
    
    if check_input:
        assert cfg["NETWORK_TO_SEARCH_IN"] in ["ecoli", "test", "yeast", "ecoli", "gs0.01", "gs0.1", "gs1"]
    
    if verbose:
        for param, value in cfg.items():
            print(f"{param}: {value}")
    
    return cfg

def update_cfg(path, param, value, verbose=True):
    """
    Update params in config
    Attributes:
    path, str - path to config
    param, str - parameter to update
    value - new value for parameter
    verbose, bool - output logs if True
    Return: updated config
    """ 
    cfg = get_actual_parametrization(path, check_input=False, verbose=False)
    cfg[param] = value
    cfg = get_actual_parametrization(cfg, verbose=verbose)
    json.dump(cfg, open(path, "w"))
    
    return cfg


def get_interaction_matrix(cfg):
    
    cwd = os.getcwd()    
    network = cfg["NETWORK_TO_SEARCH_IN"]
    interaction_matrix = joblib.load(
        os.path.join(cwd, "networks", network, f"interaction_matrix.gz")
    )
    
    return interaction_matrix


def build_motif_from_string(string):
    """
    Create numpy array from flattened string representation
    Attributes:
    string, str - string of 1/0 separated by spaces
    Return: 3x3 numpy array
    """ 
    return np.array(list(map(int, string.split()))).reshape(3, 3)


def get_equivalents(core_pattern):
    """
    Generate all equivalent forms for motif interaction matrix
    Attributes:
    core_pattern, numpy  - string of 1/0 separated by spaces
    Return: list of equivalent motif matricies
    """
    pattern_variants = []
    for permutation in permutations(range(3)):
        variant = core_pattern[permutation, :]
        variant = variant[:, permutation]
        for prev_variant in pattern_variants:
            if (variant - prev_variant == np.zeros((3, 3))).all():
                break
        else:
            pattern_variants.append(variant)
    return pattern_variants


def print_equivalents(cfg):
    """
    Print all equivalent forms for motif interaction matrix
    Attributes:
    cfg, dict - string of 1/0 separated by spaces
    """
    m = build_motif_from_string(json.load(open("./motifs_collection.json", "r"))[cfg["MOTIF_TO_SEARCH_FOR"]])
    if cfg["SELFLOOPS_INCLUDED"]: m += np.diag([1]*3)
    equivalents = get_equivalents(m)
    print(f"""Equivalent forms for {cfg["MOTIF_TO_SEARCH_FOR"]}{" with selfloops" if cfg["SELFLOOPS_INCLUDED"] else ""}\
    ({len(equivalents)} total):""")
    for x in equivalents:
        print(x)
        print()

        
def get_triad_codes():
    """
    Encode all possible triads using bitmask
    Return: resulting codes
    """
    motifs = json.load(open("./motifs_collection.json", "r"))
    salt = np.array([2**i for i in range(6)])
    mapping = {x: i for i, x in enumerate(motifs.keys())}
    codes = {}
    for motif in motifs.keys():
        form = build_motif_from_string(motifs[motif])
        isoforms = get_equivalents(form)
        for isoform in isoforms:
            mask = np.concatenate([np.diag(isoform, k=i) for i in [-2, -1, 1, 2]])
            code =  mask @ np.array([2**i for i in range(6)])
            codes[code] = mapping[motif]
    xcodes = [-1 for _ in range(sum(salt)+1)]
    for code, motif in codes.items():
        xcodes[code] = motif
    xcodes
    return xcodes, {i: x for x, i in mapping.items()}


@njit(cache=True)
def get_motifs(interaction_matrix, combs, codes, n):
    """
    Count and save all triads by the given interaction matrix
    Return: resulting triads
    """
    triads = [[(-1, -1, -1)] for _ in range(n)]
    salt = np.array([2**i for i in range(6)]).astype(np.float64)
    n_combinations = len(combs)
    for i in prange(n_combinations):
        c = combs[i]
        cl = np.array(c)
        triad = interaction_matrix[cl, :][:, cl]
        mask = [0]
        for k in [-2, -1, 1, 2]:
            mask += list(np.diag(triad, k=k))
        mask = np.array(mask[1:]).astype(np.float64)
        code = int(mask @ salt)
        idx = codes[code]
        if idx == -1:
            pass
        else:
            triads[idx] += [c]
    return triads


def motif_search(cfg, interaction_matrix, batch_size, dump=False, verbose=False):
    """
    The main function for motif search in the given interaction matrix
    Attributes:
    cfg, dict - config file dictionary
    interaction_matrix, numpy array - adjecency matrix of analysed netwotk
    batch_size, int - number of triads analyzed in the one step of parallelized algorithm
    dump, bool - save motifs if True
    verbose, bool - output logs if True
    Return: resulting motifs, their indecies and matricies, dict
    """
    network_name = cfg["NETWORK_TO_SEARCH_IN"]
    codes, mapping = get_triad_codes()
    N_CORES = mp.cpu_count() if cfg["N_CORES_TO_USE"] == -1 else cfg["N_CORES_TO_USE"]
    
    def connected_triads_generator(interaction_matrix):

        interaction_matrix_adj = interaction_matrix - np.diag(np.diag(interaction_matrix))
        tg_idxs, tf_idxs = np.where(interaction_matrix_adj != 0)
        links = pd.DataFrame(index=range(len(tf_idxs)), columns=["tf", "tg"])
        links.tf = tf_idxs
        links.tg = tg_idxs
        links_tf = links.set_index("tf", drop=False)[["tg"]]

        cascades = links.join(links_tf[["tg"]], on="tg", how="inner", rsuffix="_final")
        cascades = cascades[cascades.tf != cascades.tg_final]

        for cascade in cascades.values:
            yield tuple(cascade)

        grouper = links.groupby("tg")
        counter = grouper["tf"].count()
        for tg in counter[counter > 1].index:
            tf_pairs = combinations(links[links.tg == tg].tf.values, 2)
            for tf_1, tf_2 in tf_pairs:
                yield tf_1, tf_2, tg

        grouper = links.groupby("tf")
        counter = grouper["tg"].count()
        for tf in counter[counter > 1].index:
            tg_pairs = combinations(links[links.tf == tf].tg.values, 2)
            for tg_1, tg_2 in tg_pairs:
                yield tf, tg_1, tg_2
    
    triads = connected_triads_generator(interaction_matrix)
    
    def batch_generator(triads):
        batch = []
        counter = 0
        for triad in triads:
            batch.append(triad)
            counter += 1
            if counter == batch_size:
                yield batch
                batch = []
                counter = 0
        yield batch    
    
    def processor(splitted_triads):
        
        def gen_to_queue(input_q, splitted_triads):
            for batch in splitted_triads:
                input_q.put(batch)
            for _ in range(N_CORES):
                input_q.put(None)

        def process(input_q, output_q):
            while True:
                batch = input_q.get()
                if batch is None:
                    output_q.put(None)
                    break
                output_q.put(get_motifs(interaction_matrix, batch, codes, len(mapping)))

        input_q = mp.Queue(maxsize = N_CORES * 2)
        output_q = mp.Queue(maxsize = N_CORES * 2)

        gen_pool = mp.Pool(1, initializer=gen_to_queue, initargs=(input_q, splitted_triads))
        pool = mp.Pool(N_CORES, initializer=process, initargs=(input_q, output_q))

        finished_workers = 0
        while True:
            result = output_q.get()
            if result is None:
                finished_workers += 1
                if finished_workers == N_CORES:
                    break
            else:
                yield result
        
        input_q = None
        output_q = None
        gen_pool.close()
        gen_pool.join()
        pool.close()
        pool.join()
    
    
    splitted_triads = batch_generator(triads)
    motifs_generator = processor(splitted_triads)

    motifs = [[] for _ in range(len(mapping))]
    for batch in tqdm(motifs_generator) if verbose else motifs_generator:
        for i in range(len(mapping)):
            if batch[i][1:] != []:
                for triad in batch[i][1:]:
                    motifs[i].append("_".join(map(str, sorted(triad))))
    motifs = {mapping[i]: list(set(motifs[i])) for i in range(len(mapping))}
    counter = {x: len(y) for x, y in motifs.items()}
    
    if dump:
        joblib.dump(motifs, f"./networks/{network_name}/motifs.gz")
        json.dump(counter, open(f"./networks/{network_name}/counter.json", "w"))
    
    return motifs, counter


def count_triads_nx(interaction_matrix):
    """
    Triads counting using NetworkX module
    """
    G = nx.DiGraph(interaction_matrix.T)
    return nx.algorithms.triads.triadic_census(G)


def get_metrics_report(interaction_matrix):
    """
    Buid topological metrics report
    Attributes:
    interaction_matrix, numpy array - adjecency matrix of analysed netwotk
    Return: results as named tuple class: degree sequence, average degree, diameter, fraction of the largest connected conponent,
    degree and betweenness centrality
    """
    Report = namedtuple(
        "report",
        ["degree_seq", "avg_degree", "diameter_strong", "diameter_weak",
         "largest_component_frac", "degree_centrality", "betweenness_centrality"]
    )
    G = nx.DiGraph(interaction_matrix.T)
    degree_seq = pd.Series(np.array([x[1] for x in G.degree]))
    avg_degree = degree_seq.mean()
    diameter_weak = diameter(G.to_undirected()) if is_weakly_connected(G) else np.inf
    if is_strongly_connected(G):
        diameter_strong = diameter(G)
        largest_component_frac = 1
    else:
        diameter_strong = np.inf
        strong_components = [(c, len(c)) for c in strongly_connected_components(G)]
        strong_components = sorted(strong_components, key=lambda x: x[1], reverse=True)
        largest_component_frac = strong_components[0][1]/interaction_matrix.shape[0]
    dc = pd.Series(degree_centrality(G))
    bc = pd.Series(betweenness_centrality(G))
    report = Report(*[degree_seq, avg_degree, diameter_strong, diameter_weak, largest_component_frac, dc, bc])
    return report


def get_loops(matrix):
    m = matrix + matrix.T
    x = sorted([sorted([x, y]) for x, y in zip(*np.where(m == 2))])
    y = [x[k] for k in range(len(x)) if k % 2 == 0]
    return y


@njit
def get_shuffled_matrix(interaction_matrix, nswaps):
    """
    Shuffle incoming matrix preserving in/out degree for each vertex
    Attributes:
    interaction_matrix, numpy array - adjecency matrix of analysed netwotk
    nswaps, int - number of swaps in shuffling process
    Return: shuffled matrix, numpy array
    """
    shuffled = interaction_matrix.copy()
    tf_nodes = np.where(shuffled.sum(axis=0) != 0)[0]
    for i in range(nswaps):
        tf_1, tf_2 = np.random.choice(tf_nodes, size=2, replace=True)
        tg = shuffled[:, np.array([tf_1, tf_2])]
        x = np.where((tg[:, 0] == 1) & (tg[:, 1] == 0))[0]
        if x.shape[0] > 0:
            tg_1 = np.random.choice(x)
        else:
            continue
        y = np.where((tg[:, 1] == 1) & (tg[:, 0] == 0))[0]
        if y.shape[0] > 0:
            tg_2 = np.random.choice(y)
        else:
            continue
        s = shuffled[np.array([tg_1, tg_2]), :][:, np.array([tf_1, tf_2])]
        e1 = np.diag(np.array([1, 1]))
        e2 = e1[::-1]
        if (s == e1).all():
            shuffled[tg_1, tf_1] = 0
            shuffled[tg_1, tf_2] = 1
            shuffled[tg_2, tf_1] = 1
            shuffled[tg_2, tf_2] = 0
        else:
            shuffled[tg_1, tf_1] = 1
            shuffled[tg_1, tf_2] = 0
            shuffled[tg_2, tf_1] = 0
            shuffled[tg_2, tf_2] = 1
    return shuffled  


def corruption_score(shuffled_matrix, interaction_matrix):
    """
    Compute fraction of shifted links in shuffled matrix comparing with original one
    Attributes:
    shuffled_matrix, numpy array - adjecency matrix of shuffled netwotk
    interaction_matrix, numpy array - adjecency matrix of original netwotk
    Return: corruption score, float
    """
    i, j = np.where(interaction_matrix == 1)
    return shuffled_matrix[i, j].sum()/interaction_matrix[i, j].sum()


def plot_distr(counters_shuffled, counter_orig, label, highlight):
    """
    Plot motif number distribution (original versus shuffled)
    """
    df = pd.DataFrame(columns=["motif", "abundance", "network"])
    df.motif = counter_orig.keys(); df.abundance = counter_orig.values(); df.network = "original"
    for counter_shuffled in tqdm(counters_shuffled):
        df2 = pd.DataFrame(columns=["motif", "abundance", "network"])
        df2.motif = counter_shuffled.keys(); df2.abundance = counter_shuffled.values(); df2.network = "shuffled"
        df = pd.concat([df, df2], axis=0)
    df.abundance = df.abundance/1000
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
    fig.suptitle(label, fontsize=30)
    for i in range(len(counter_orig.keys())):
        motif = list(counter_orig.keys())[i]
        b = sns.barplot(data=df[df["motif"]==motif], x="motif", y="abundance", hue="network", ax=ax[i], 
                        palette="Blues_r")
        if highlight and motif == highlight:
            b.set_facecolor('xkcd:wheat')
        b.legend_.remove()
#         else:
#             plt.setp(b.get_legend().get_texts(), fontsize='13')
#             plt.setp(b.get_legend().get_title(), fontsize='13')
        b.tick_params("x", labelsize=20)
        b.set_xlabel("",fontsize=0)
        b.set_ylabel("",fontsize=0);
    return df, fig


def get_shuffled_mp(params):
    """Auxiliary parallel functioin"""
    matrix = params["matrix"]
    nswaps = params["nswaps"]
    return get_shuffled_matrix(matrix, nswaps)

def generate_random_networks(cfg, interaction_matrix, nsims, nsteps, nswaps):
    """
    Make the number of shuffled matricies and count motifs in them
    Attributes:
    interaction_matrix, numpy array - adjecency matrix of original netwotk
    nsims, int - number of shuffled matrix to generate
    nsteps, int - number of steps
    nswaps, int - number of swaps in every shuffling process
    Return: list of counted motifs for every generated shuffled matrix
    """
    counters = []
    for _ in range(nsteps):
        pool = mp.Pool(mp.cpu_count())
        params = {"matrix": interaction_matrix, "nswaps": nswaps}
        shuffled_arrays = pool.map(get_shuffled_mp, (params for _ in range(int(nsims/nsteps))))
        pool.close()
        pool.join()
        for arr in tqdm(shuffled_arrays):
            motifs, counter = motif_search(cfg, arr, batch_size=10000)
            counters.append(counter)
    return counters


def plot_distr_2(counters, counter_orig, ticks):
    """
    Plot distributioins
    """
    distr = {triad: [] for triad in counters[0].keys()}
    for counter in counters:
        for triad, n in counter.items():
            distr[triad].append(n)
    distr = {x: np.array(y) for x, y in distr.items()}
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
    for i, motif in enumerate(counters[0].keys()):
        ax = axes[i//3, i%3]
        ax.set_title(motif, fontsize=25)
        pd.Series(distr[motif]).hist(bins=50, ax=ax)
        ax.plot([counter_orig[motif]]*100, np.linspace(0, ticks[i], 100), "r")
        
        
def build_zscores_report(counters, counter_orig):
    """
    Compute z-scores for every motif based on the set of shuffled matricies
    Attributes:
    counters, list - list of counter dicitionaries for every shuffled matrix
    counter_orig, dict - counted motifs for original network
    Return: report with z-score and p-value for every motif type
    """
    distr = {triad: [] for triad in counters[0].keys()}
    for counter in counters:
        for triad, n in counter.items():
            distr[triad].append(n)
    distr = {x: np.array(y) for x, y in distr.items()}
    zscores_report = pd.DataFrame(
        index=["N_real", "mean(N_rand)", "sd(N_rand)", "Z-score", "P-value", "Result"]
    )
    for motif in counters[0].keys():
        n_hypothesis = len(counters[0].keys())
        d = distr[motif]
        zscore = (counter_orig[motif]-np.mean(distr[motif]))/np.std(distr[motif])
        pvalue = len(d[d <= counter_orig[motif]])/len(d)
        if pvalue > 0.5:
            pvalue = len(d[d >= counter_orig[motif]])/len(d)
        if pvalue < 0.01/n_hypothesis:
            result = " < 0.01"
        elif pvalue < 0.05/n_hypothesis:
            result = " < 0.05"
        else:
            result = "non-significant"
        result_list = [
            counter_orig[motif],
            np.mean(distr[motif]),
            np.std(distr[motif]),
            zscore,
            pvalue,
            result
        ]
        zscores_report[motif] = result_list
    return zscores_report.T


split_motif = lambda x: list(map(int, x.split("_")))


def build_vmn(motifs, verbose=False):
    """
    Build vertex-based motif network (by shared nodes)
    Attributes:
    motifs, list - triads of motifs as string of indecies sepated by "_"
    Return: adjecenct matrix for VMN
    """
    motifs_network = np.zeros((len(motifs), len(motifs)))
    iterator = combinations(range(len(motifs)), 2)
    if verbose:
        iterator = tqdm(iterator, total=int(len(motifs)*(len(motifs)-1)/2))
    for i, j in iterator:
        m1, m2 = map(lambda x: set(map(int, x.split("_"))), [motifs[i], motifs[j]])
        motifs_network[i, j] = len(m1 & m2)
        motifs_network[j, i] = motifs_network[i, j]
    return motifs_network


def get_sparcity(matrix):
    """Compute netwotk sparcity by adjacency matrix"""
    return matrix.sum()/matrix.shape[0]


def get_tf_content(matrix):
    """Compute netwotk TF/TG content by adjacency matrix"""
    return len(np.where(matrix.sum(axis=0)!=0)[0])/matrix.shape[0]
