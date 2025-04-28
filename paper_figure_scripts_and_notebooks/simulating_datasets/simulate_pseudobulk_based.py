import numpy as np
import os
import sys
from pathlib import Path
import pandas as pd
from scipy.special import logsumexp
from argparse import ArgumentParser
import h5py
import csv

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
os.chdir(parent_dir)

parser = ArgumentParser(
    description='Simulates a binary tree.')

# Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
parser.add_argument('--input_dataset', type=str, default='examples/example_data/baron.hdf',
                    help='Path to file with information from the gene table from the Baron-dataset. In fact, only '
                         'number of rows, columns and row-sums, column-sums are stored..')
parser.add_argument('--input_pseudobulk', type=str, default='data/additional_data/zeisel_pseudobulk',
                    help='Path to folder with Sanity-output calculated on pseudobulk based on Zeisel-dataset. In '
                         'addition, there should be a file type_abudance.txt that states the abundance of the '
                         'different celltypes in the original data.')
parser.add_argument('--results_folder', type=str, default='data/simulated_datasets',
                    help="Relative path from bonsai_development to base-folder where simulated trees need to be stored.")
parser.add_argument('--n_cells', type=int, default=1024,
                    help="Number of cells in our simulated dataset.")

args = parser.parse_args()
print(args)

baron_hdf = h5py.File(args.input_dataset, 'r')
N_c = baron_hdf['N_c'][:]
# nGenes = baron_hdf.attrs['nGenes']
nCells = baron_hdf.attrs['nCells']
# N_g = baron_hdf['N_g'][:]
baron_hdf.close()

pseudobulk_ltqs_df = pd.read_csv(os.path.join(args.input_pseudobulk, 'Sanity', 'log_transcription_quotients.txt'),
                              header=0, index_col=0, sep='\t')
pseudobulk_ltqs = pseudobulk_ltqs_df.values.astype(dtype=float)
ct_cats = list(pseudobulk_ltqs_df.columns)
gene_ids = list(pseudobulk_ltqs_df.index)

abundances = []
with open(os.path.join(args.input_pseudobulk, 'type_abundance.txt'), 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        abundances.append(int(row[0]))
abundances = np.array(abundances)
nGenes = pseudobulk_ltqs.shape[0]

seed = 2462
np.random.seed(seed)

# Randomly sample which cell gets what total umi count (based on Baron dataset)
np.random.shuffle(N_c)

nDraws = args.n_cells
if nDraws > nCells:
    N_c = np.tile(N_c, int(np.ceil(nDraws / nCells)))

ltqs_gc = np.zeros((nGenes, nDraws))

cumsum_abundance = np.cumsum(abundances)
cumsum_abundance = cumsum_abundance * args.n_cells / cumsum_abundance[-1]
allocated_cells = 0
celltypes = []
intra_cluster_var = 0.05
# Sample cells from cluster-centers while adding a little bit of variation, to not have all cells exactly at the same
# place
for cat_ind, new_cells in enumerate(cumsum_abundance):
    new_cells = int(np.ceil(new_cells))
    ltqs_gc[:, allocated_cells: new_cells] = pseudobulk_ltqs[:, cat_ind][:, None] + np.random.normal(scale=np.sqrt(intra_cluster_var), size=(nGenes, new_cells-allocated_cells))
    celltypes += [ct_cats[cat_ind]] * (new_cells - allocated_cells)
    allocated_cells = new_cells

# Normalise such that each cell's TQs add up to 1
log_tqs = logsumexp(ltqs_gc, axis=0)
ltqs_gc = ltqs_gc - log_tqs

# Get true means and vars
true_means_g = np.mean(ltqs_gc, axis=1)
true_vars_g = np.var(ltqs_gc, axis=1)
delta_gc = ltqs_gc - true_means_g[:, None]

"""---------Now store the simulated dataset somewhere---------"""

datadir = "simulated_pseudobulk_based_ncells_" + str(args.n_cells)
datadir += '_seed_{}'.format(seed)

data_path = os.path.abspath(os.path.join(args.results_folder, datadir))
Path(data_path).mkdir(parents=True, exist_ok=True)

cellIDs = ['Cell_' + str(ind) for ind in range(nDraws)]


print("Writing true deltas to file.")
delta_df = pd.DataFrame(delta_gc, columns=cellIDs, index=gene_ids)
delta_df.to_csv(os.path.join(data_path, 'delta_true.txt'), sep='\t', header=False, index=False)

print("Writing true variances to file:")
with open(os.path.join(data_path, 'variance_true.txt'), 'w') as f:
    for var in true_vars_g:
        f.write("%s\n" % var)

print("Writing cell IDs to file:")
with open(os.path.join(data_path, 'cellID.txt'), 'w') as f:
    for ID in cellIDs:
        f.write("%s\n" % ID)
print("Writing gene IDs to file:")
with open(os.path.join(data_path, 'geneID.txt'), 'w') as f:
    for ID in gene_ids:
        f.write("%s\n" % ID)

print("Writing celltypes to file:")
annotation_dict = {'pseudobulk': celltypes}
annotation_df = pd.DataFrame(annotation_dict, index=cellIDs)

Path(os.path.join(data_path, 'annotation')).mkdir(parents=True, exist_ok=True)
annotation_df.to_csv(os.path.join(data_path, 'annotation', 'pseudobulk_annotation.csv'))

print("Sampling UMI counts:")
umi_counts = np.zeros((nGenes, nDraws))

for cell_ind in range(nDraws):
    if cell_ind % 100 == 0:
        print("Sampling counts for cell %d." % cell_ind)
    umi_counts[:, cell_ind] = np.random.multinomial(N_c[cell_ind], np.exp(ltqs_gc[:, cell_ind]))

print("Writing UMI counts to file:")
umi_df = pd.DataFrame(umi_counts, columns=cellIDs, index=gene_ids)
umi_df.to_csv(os.path.join(data_path, 'Gene_table.txt'), sep='\t', index_label="GeneID")
