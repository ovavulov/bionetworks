"""

Auxiliary functions

"""

import os
import json
import joblib
import numpy as np
from itertools import permutations
from numba import njit, prange
from tqdm import tqdm

from subprocess import Popen, PIPE
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


def get_actual_parametrization(path, verbose=True):
        
    cfg = json.load(open(path, "r"))
    assert cfg["NETWORK_TO_SEARCH_IN"] in ["test", "yeast", "ecoli", "gs0.01", "gs0.1", "gs1"]
    assert cfg["MOTIF_TO_SEARCH_FOR"] in ["feedforward", "fanout", "fanin", "cascade"]
    if verbose:
        for param, value in cfg.items():
            print(f"{param}: {value}")
    
    return cfg

def update_cfg(path, param, value, verbose=True):
    
    cfg = get_actual_parametrization(path, verbose=False)
    cfg[param] = value
    json.dump(cfg, open(path, "w"))
    cfg = get_actual_parametrization(path, verbose=verbose)
    
    return cfg


def add_self_loops(interaction_matrix, indexes):
    
    selfloops = np.zeros(interaction_matrix.shape[0])
    selfloops[indexes] = 1
    selfloops = np.diag(selfloops)
    
    return interaction_matrix + selfloops


def get_interacion_matrix(cfg):
    
    cwd = os.getcwd()    
    network = cfg["NETWORK_TO_SEARCH_IN"]
    interaction_matrix = joblib.load(
        os.path.join(cwd, "networks", network, f"interaction_matrix.pkl")
    )
    
    return interaction_matrix


def split_search_space(cfg, combs):
    
    idxs = np.linspace(0, len(combs), cfg["PARALLEL_THREADS_NUMBER"] + 1).astype(int)
    combs_splitted = [combs[idxs[i]:idxs[i+1]] for i in range(cfg["PARALLEL_THREADS_NUMBER"])]
    
    splits_path = f"""./networks/{cfg["NETWORK_TO_SEARCH_IN"]}/splits"""

    if not os.path.exists(splits_path):
        os.mkdir(splits_path)
        counter = 1
        for combs_split in tqdm(combs_splitted):
            joblib.dump(combs_split, os.path.join(splits_path, f"combs_{counter}.pkl"))
            counter += 1
    else:
        print(f"Search space have been splitted into {len(os.listdir(splits_path))}")

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
    

@njit(cache=True)
def get_motifs(pattern_variants, interaction_matrix, combs):
    
    genes = []
    motifs = []
    n_combinations = len(combs)
    n_patterns = len(pattern_variants)
    for i in prange(n_combinations):
        cl = np.array(combs[i])
        triad = interaction_matrix[cl, :][:, cl]
        for j in prange(n_patterns):
            pattern = pattern_variants[j]
            if (triad - pattern == np.zeros(3)).all():
                genes.append(cl)
                motifs.append(triad)
                break
    
    return genes, motifs


def make_parallel_search(cfg):
    
    network = cfg["NETWORK_TO_SEARCH_IN"]
    motif = cfg["MOTIF_TO_SEARCH_FOR"]
    selfloops_included = cfg["SELFLOOPS_INCLUDED"]
    results_path = f"./motif_search_results"
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    results_path = os.path.join(results_path, network)
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    results_path = os.path.join(results_path, motif + selfloops_included*"_sl")
    if not os.path.exists(results_path):
        os.mkdir(results_path)
        
    start = datetime.now()

    N_PROCESSES = cfg["PARALLEL_THREADS_NUMBER"]
    SCRIPT_PATH = "./scripts/find_motifs.py"

    cmds_list = [["python", SCRIPT_PATH, str(i)] for i in range(1, N_PROCESSES + 1)]
    procs_list = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmds_list]
    for proc in procs_list:
        proc.wait()
    
    file_names = os.listdir(results_path)
    genes_names = sorted([name for name in file_names if name.startswith("genes")])
    motifs_names = sorted([name for name in file_names if name.startswith("motifs")])

    genes = []
    for genes_name in genes_names:
        genes += joblib.load(os.path.join(results_path, genes_name))
        os.remove(os.path.join(results_path, genes_name))
    joblib.dump(genes, os.path.join(results_path, "genes.pkl"))

    motifs = []
    for motifs_name in motifs_names:
        motifs += joblib.load(os.path.join(results_path, motifs_name))
        os.remove(os.path.join(results_path, motifs_name))
    joblib.dump(motifs, os.path.join(results_path, "motifs.pkl"))
    print(f"""{len(motifs)} {motif}{" with selfloops" if selfloops_included else ""} motifs have been found""")

    print(f"Total time spent: {datetime.now() - start}")

