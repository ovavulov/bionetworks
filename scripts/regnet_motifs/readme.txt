Motif enrichment analysis

Set the environment:
conda grn-env create -f grn_env.yml

Configurate pipeline via config.json:
RANDOM_SEED:	19
N_CORES_TO_USE:	-1	# number of cores, -1 for all accessible ones
GRN_TO_ANALYSE:	mouse	# which network to use for analysis {"mouse", "human", "test", "yeast"}
N_SWAPS:	100000	# number of swaps during shuffling
N_STEPS:	10	# number of steps the procedure is devided on
N_SIMULATIONS:	1000	# number of shuffled networks

Launch by: 
1) /path/to/python ./main.py
2) ./main.py (is the case, enshure that shebang string is appropriate)

