#!/opt/anaconda3/bin/python

"""

Script for motif enrichment analysis

"""




"""Requirements"""

import json
import joblib
import numpy as np
import pandas as pd
import multiprocessing as mp
from numba import njit, prange
from tqdm import tqdm
from itertools import combinations, permutations
import warnings
warnings.filterwarnings("ignore")




"""Functions"""

def get_actual_parametrization(source, check_input=True, verbose=True): 
    cfg = source if type(source) is dict else json.load(open(source, "r"))
    print(cfg)
    if check_input:
        assert cfg["GRN_TO_ANALYSE"] in ["test", "yeast", "mouse", "human"]
    if verbose:
        print("Input parameters:")
        for param, value in cfg.items():
            print(f"{param}: {value}") 
        print()
    return cfg

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
        
def get_triad_codes():
    motifs = json.load(open("motifs_collection.json", "r"))
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
    return xcodes, {i: x for x, i in mapping.items()}

@njit
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
    return motifs, counter

@njit
def get_shuffled_matrix(interaction_matrix, nswaps):
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

def get_shuffled_mp(params):
    matrix = params["matrix"]
    nswaps = params["nswaps"]
    return get_shuffled_matrix(matrix, nswaps)

def generate_random_networks(cfg, interaction_matrix):
    nsims = cfg["N_SIMULATIONS"]
    nsteps = cfg["N_STEPS"]
    nswaps = cfg["N_SWAPS"]
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




"""Procedure"""

# if __name__ == "main":
cfg = get_actual_parametrization("config.json")
path_to_input = f"""./{cfg["GRN_TO_ANALYSE"]}_matrix.gz"""
path_to_output = f"""./results/{cfg["GRN_TO_ANALYSE"]}_counters.gz"""
interaction_matrix = joblib.load(path_to_input)
print("Interactions are loaded\nMotif search is in progress...")
motifs, counter = motif_search(cfg, interaction_matrix, batch_size=10000)
print("Motif search is completed:")
print(counter)
del motifs
print("Randomization is started")
shuffled_counters = generate_random_networks(cfg, interaction_matrix)
print("That's all finally! Saving results...")
result = counter, shuffled_counters
joblib.dump(result, path_to_output)
print("All computations are completed. Ciao!")
    
    