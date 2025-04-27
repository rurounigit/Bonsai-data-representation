from scipy.io import mmread
from argparse import ArgumentParser
import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
# from icecream import ic

SKIP_COUNTFILE = False

if __name__ == '__main__':
    """Parse input arguments"""
    parser = ArgumentParser(
        description='Selects subset of rows and columns in mtx folder.')
    parser.add_argument('--folder_raw_umi_counts', type=str, default='.',
                        help='Path to the folder where the raw-umis output can be found')
    parser.add_argument('--file_raw_umi_counts', type=str, default='gene_count.run_4.mtx',
                        help='Path to the folder where the raw-umis output can be found')
    parser.add_argument('--file_cell_annotation', type=str, default='cell_annotation.run_4.csv')
    parser.add_argument('--file_gene_annotation', type=str, default='gene_annotation.csv')
    parser.add_argument('--desired_cell_annotation', type=str, default='embryo_6,embryo_3')
    parser.add_argument('--desired_gene_annotation', type=str, default='protein_coding')
    parser.add_argument('--file_celltype_annotation', type=str, default='GSE186069_cell_annotate.csv')

    args = parser.parse_args()

    print("Reading cell IDs:")
    selected_cell_inds = []
    selected_cell_ids = []
    desired_cell_annotations = args.desired_cell_annotation.split(',')
    cell_annotation_celltype = []
    with open(os.path.join(args.folder_raw_umi_counts, args.file_cell_annotation), 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        for ind, row in enumerate(reader):
            if row[2] in desired_cell_annotations:
                selected_cell_inds.append(ind - 1)  # Correct for header by subtracting 1 from index
                selected_cell_ids.append(row[0])
                cell_annotation_celltype.append(row[2])
    selected_cell_inds = np.array(selected_cell_inds)

    cropped_set_cell_ids = {cell_id.split('_')[-1] for cell_id in selected_cell_ids}
    prefix = '_'.join(selected_cell_ids[0].split('_')[:2]) + '_'
    print("Reading cell annotations:")
    cell_ids_to_annot = {}
    with open(os.path.join(args.folder_raw_umi_counts, args.file_celltype_annotation), 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        for ind, row in enumerate(reader):
            if ind % 10000 == 0:
                print(ind)
            # Get matches to cell-IDs
            if row[0] in cropped_set_cell_ids:
                cell_ids_to_annot[prefix+row[0]] = row[8]

    print("Reading gene annotation:")
    selected_gene_inds = []
    selected_gene_ids = []
    with open(os.path.join(args.folder_raw_umi_counts, args.file_gene_annotation), 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        for ind, row in enumerate(reader):
            if row[6] == args.desired_gene_annotation:
                selected_gene_inds.append(ind - 1)
                selected_gene_ids.append(row[0])
    selected_gene_inds = np.array(selected_gene_inds)

    if not SKIP_COUNTFILE:
        # Read in count matrix of all cells
        print("Reading in count file:")
        M = mmread(os.path.join(args.folder_raw_umi_counts, args.file_raw_umi_counts))
        M = M.tocsc()

        umi_counts = M[selected_gene_inds,:][:, selected_cell_inds]
        umi_counts = umi_counts.toarray().astype(dtype=int)

        new_umi_counts_df = pd.DataFrame(umi_counts, columns=selected_cell_ids)
        new_umi_counts_df.insert(0, 'GeneID', selected_gene_ids)

    Path(os.path.join(args.folder_raw_umi_counts, args.desired_cell_annotation)).mkdir(parents=True, exist_ok=True)
    if not SKIP_COUNTFILE:
        new_umi_counts_df.to_csv(
            os.path.join(args.folder_raw_umi_counts, args.desired_cell_annotation, 'Gene_table.txt'), index=False,
            sep='\t')
    with open(os.path.join(args.folder_raw_umi_counts, args.desired_cell_annotation, 'cellID.txt'), 'w') as f:
        for ID in selected_cell_ids:
            f.write("%s\n" % ID)
    with open(os.path.join(args.folder_raw_umi_counts, args.desired_cell_annotation, 'Celltype.txt'), 'w') as f:
        for ID in selected_cell_ids:
            f.write("%s\n" % cell_ids_to_annot[ID])
    with open(os.path.join(args.folder_raw_umi_counts, args.desired_cell_annotation, 'Celltype_sample.txt'), 'w') as f:
        for ct in cell_annotation_celltype:
            f.write("%s\n" % ct)
