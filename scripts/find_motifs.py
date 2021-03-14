import os
import sys
import numpy as np
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

import functions as f

cfg = json.load(open("./config.json", "r"))
motifs = json.load(open("./motifs_collection.json", "r"))
network_name =  cfg["NETWORK_TO_SEARCH_IN"]
motif_name = cfg["MOTIF_TO_SEARCH_FOR"]
selfloops_included = cfg["SELFLOOPS_INCLUDED"]
matrix_name = cfg["MATRIX_NAME"]


JOB_ID = sys.argv[1]
MOTIF = f.build_motif_from_string(motifs[motif_name])
if selfloops_included:
    MOTIF +=np.diag([1]*3)
STUDY_ID = np.random.randint(int(1e6))

results_path = f"./motif_search_results/{network_name}/{motif_name}" + selfloops_included*"_sl"
matrix_path = f"./networks/{network_name}/{matrix_name}.pkl"
combs_path = f"./networks/{network_name}/splits/combs_{JOB_ID}.pkl"

interaction_matrix = joblib.load(matrix_path)
combs = joblib.load(combs_path)

variants = f.get_equivalents(MOTIF)
genes, motifs = f.get_motifs(variants, interaction_matrix, combs)

joblib.dump(genes, os.path.join(results_path, f"genes_{STUDY_ID}.pkl"))
joblib.dump(motifs, os.path.join(results_path, f"motifs_{STUDY_ID}.pkl"))