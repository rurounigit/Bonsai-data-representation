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

parser = ArgumentParser(
    description='Simulates a binary tree.')

# Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
parser.add_argument('--input_dataset', type=str, default='data/Zeisel_genes/Gene_table.txt',
                    help='Path to original UMI-count file. ')
parser.add_argument('--input_annotation', type=str, default='data/Zeisel_genes/Celltype.txt',
                    help='Path to annotation file on which we will base the pseudobulk')
parser.add_argument('--results_folder', type=str, default='data/additional_data/zeisel_pseudobulk',
                    help="Relative path from bonsai_development to base-folder where simulated trees need to be stored.")

args = parser.parse_args()
print(args)

original_umis_df = pd.read_csv(args.input_dataset, sep='\t', header=0, index_col=0)
original_umis = original_umis_df.values
gene_ids = list(original_umis_df.index)
ct_annotation = []
with open(args.input_annotation) as f:
    reader = csv.reader(f)
    for row in reader:
        ct_annotation.append(row[0])

ct_cats = list(set(ct_annotation))
cat_abundance = np.zeros(len(ct_cats))
annot_to_ind = {ct_cat: ind for ind, ct_cat in enumerate(ct_cats)}
pseudobulk_umis = np.zeros((original_umis.shape[0], len(ct_cats)))
for cell_ind in range(original_umis.shape[1]):
    ct_cat = ct_annotation[cell_ind]
    cat_ind = annot_to_ind[ct_cat]
    cat_abundance[cat_ind] += 1
    pseudobulk_umis[:, cat_ind] += original_umis[:, cell_ind]

print("Writing UMI counts to file:")
umi_df = pd.DataFrame(pseudobulk_umis, columns=ct_cats, index=gene_ids)
Path(args.results_folder).mkdir(parents=True, exist_ok=True)
umi_df.to_csv(os.path.join(args.results_folder, 'Gene_table.txt'), sep='\t', index_label="GeneID")

print("Writing cell IDs to file:")
with open(os.path.join(args.results_folder, 'type_abundance.txt'), 'w') as f:
    for abundance in cat_abundance:
        f.write("%d\n" % int(abundance))
print("Writing cell IDs to file:")
with open(os.path.join(args.results_folder, 'cellID.txt'), 'w') as f:
    for ID in ct_cats:
        f.write("%s\n" % ID)
print("Writing gene IDs to file:")
with open(os.path.join(args.results_folder, 'geneID.txt'), 'w') as f:
    for ID in gene_ids:
        f.write("%s\n" % ID)
