"""

Auxiliary functions

"""

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

import warnings
warnings.filterwarnings("ignore")


def read_ecoli_network(path):
    f = open(path)
    line = f.readline()
    while line.startswith('#'):
        line = f.readline()
    df = pd.read_csv(f, sep="\t", header=None)
    df.loc[-1] = line.split("\t")
    df.index = df.index + 1
    df = df.sort_index()
    f.close()
    return df


def get_actual_parametrization(source, check_input=True, verbose=True):
        
    cfg = source if type(source) is dict else json.load(open(source, "r"))
    
    if check_input:
        assert cfg["NETWORK_TO_SEARCH_IN"] in ["ecoli", "test", "yeast", "ecoli", "gs0.01", "gs0.1", "gs1"]
    
    if verbose:
        for param, value in cfg.items():
            print(f"{param}: {value}")
    
    return cfg

def update_cfg(path, param, value, verbose=True):
    
    cfg = get_actual_parametrization(path, check_input=False, verbose=False)
    cfg[param] = value
    cfg = get_actual_parametrization(cfg, verbose=verbose)
    json.dump(cfg, open(path, "w"))
    
    return cfg


def get_interacion_matrix(cfg):
    
    cwd = os.getcwd()    
    network = cfg["NETWORK_TO_SEARCH_IN"]
    interaction_matrix = joblib.load(
        os.path.join(cwd, "networks", network, f"interaction_matrix.gz")
    )
    
    return interaction_matrix


def build_motif_from_string(string):
    return np.array(list(map(int, string.split()))).reshape(3, 3)


def get_equivalents(core_pattern):
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
    m = build_motif_from_string(json.load(open("./motifs_collection.json", "r"))[cfg["MOTIF_TO_SEARCH_FOR"]])
    if cfg["SELFLOOPS_INCLUDED"]: m += np.diag([1]*3)
    equivalents = get_equivalents(m)
    print(f"""Equivalent forms for {cfg["MOTIF_TO_SEARCH_FOR"]}{" with selfloops" if cfg["SELFLOOPS_INCLUDED"] else ""}\
    ({len(equivalents)} total):""")
    for x in equivalents:
        print(x)
        print()

        
def get_triad_codes():
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


def motif_search(cfg, interaction_matrix, batch_size, verbose=False):
    
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
    
    joblib.dump(motifs, f"./networks/{network_name}/motifs.gz")
    json.dump(counter, open(f"./networks/{network_name}/counter.json", "w"))
    
    return motifs, counter


def count_triads_nx(interaction_matrix):    
    G = nx.DiGraph(interaction_matrix.T)
    return nx.algorithms.triads.triadic_census(G)