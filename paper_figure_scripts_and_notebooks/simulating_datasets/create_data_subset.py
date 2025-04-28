import numpy as np
import os
import sys
from argparse import ArgumentParser
import pandas as pd
import csv
from pathlib import Path

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
os.chdir(parent_dir)

parser = ArgumentParser(description='Takes a dataset, and creates a new dataset with a random subset of the cells.')

# Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
parser.add_argument('--input_dataset', type=str, default='data/Zeisel_genes/Gene_table.txt',
                    help='Path to original UMI-count file. ')
parser.add_argument('--input_annotation', type=str, default='data/Zeisel_genes/Celltype.txt',
                    help='If not empty, this celltype information will also be used for the subset')
parser.add_argument('--n_cells', type=int, default=1024,
                    help='Number of cells in new subset')
parser.add_argument('--results_folder', type=str, default='data/Zeisel_subset_1024cells',
                    help="Relative path from bonsai_development to base-folder where new dataset needs to be stored.")

args = parser.parse_args()
print(args)
np.random.seed(1231)

original_umis_df = pd.read_csv(args.input_dataset, sep='\t', header=0, index_col=0)
umis_orig = original_umis_df.values
gene_ids_orig = list(original_umis_df.index)
cell_ids_orig = list(original_umis_df.columns)
n_genes_orig, n_cells_orig = umis_orig.shape

ct_annotation_orig = []
with open(args.input_annotation) as f:
    reader = csv.reader(f)
    for row in reader:
        ct_annotation_orig.append(row[0])

subset_cells = np.sort(np.random.choice(n_cells_orig, args.n_cells, replace=False))
umis_subset = umis_orig[:, subset_cells]
cell_ids = [cell_ids_orig[ind] for ind in subset_cells]
ct_annotation = [ct_annotation_orig[ind] for ind in subset_cells]

print("Writing UMI counts to file:")
umi_df = pd.DataFrame(umis_subset, columns=cell_ids, index=gene_ids_orig)
Path(args.results_folder).mkdir(parents=True, exist_ok=True)
umi_df.to_csv(os.path.join(args.results_folder, 'Gene_table.txt'), sep='\t', index_label="GeneID")

print("Writing cell IDs to file:")
with open(os.path.join(args.results_folder, 'cellID.txt'), 'w') as f:
    for ID in cell_ids:
        f.write("%s\n" % ID)

print("Writing gene IDs to file:")
with open(os.path.join(args.results_folder, 'geneID.txt'), 'w') as f:
    for ID in gene_ids_orig:
        f.write("%s\n" % ID)

print("Writing celltypes to file:")
annotation_dict = {}
annotation_dict['celltype_original'] = ct_annotation
annotation_df = pd.DataFrame(annotation_dict, index=cell_ids)
Path(os.path.join(args.results_folder, 'annotation')).mkdir(parents=True, exist_ok=True)
annotation_df.to_csv(os.path.join(args.results_folder, 'annotation', 'orig_annotation.csv'))

