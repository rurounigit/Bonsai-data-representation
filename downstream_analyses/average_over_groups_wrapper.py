from argparse import ArgumentParser
import numpy as np
import pandas as pd
import os

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# Add the parent directory to sys.path
print(parent_dir)
sys.path.append(parent_dir)
from downstream_analyses.average_over_groups import run_averaging


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise exit('Boolean value expected.')


if __name__ == '__main__':
    """Parse input arguments"""
    """
    Example
    python /scicore/home/nimwegen/morill0000/software/waddington-code-github/useful_scripts_not_bonsai/sarah_scripts/run_averaging_wrapper.py \
    -x /scicore/home/nimwegen/huijne0001/data_peykonmb/OUTPUT/sanity_w_genes/delta.txt \
    -ue /scicore/home/nimwegen/huijne0001/data_peykonmb/OUTPUT/sanity_w_genes/d_delta.txt \
    -names /scicore/home/nimwegen/huijne0001/data_peykonmb/OUTPUT/sanity_w_genes/geneID.txt \
    -group_file /scicore/home/nimwegen/huijne0001/Annemiek/BarCodesSamplesAnnemiek.tsv \
    -output_folder /scicore/home/nimwegen/huijne0001/Annemiek/averaging_results \
    -col_idx 1 \
    """
    parser = ArgumentParser(
        description='get group expression and uncertainty estimate')
    parser.add_argument('-x', type=str, default='./delta.txt',
                        help='Path to the file with expression values')
    parser.add_argument('-ue', type=str, default='./d_delta.txt',
                        help='Path to the file with uncertainty estimates')
    parser.add_argument('-names', type=str, default='./geneID.txt',
                        help='Path to the file with names (row/genes)')
    parser.add_argument('-group_file', type=str, default='.',
                        help='Path to the folder where cell group mapping is stored, csv, where first row is the cb '
                             'in cellID.txt order, and then the next columns are annotation. tsv, needs to be '
                             'tabseparated')
    parser.add_argument('-col_idx', type=str, default='.',
                        help='Path to the folder where cell group mapping is stored')
    parser.add_argument('-output_folder', type=str, default='.',
                        help='Path to the folder where output should be stored')
    parser.add_argument('-read_ids_from_file', type=str2bool, default=False,
                        help='If this is true, the script assumes that the -x and -ue contain header and index')

    args = parser.parse_args()

    if not args.read_ids_from_file:
        # read in data
        print("load data")
        deltas = np.loadtxt(args.x)
        d_deltas = np.loadtxt(args.ue)
        geneIDs = list(pd.read_csv(args.names, header=None)[0])
    else:
        deltas_df = pd.read_csv(args.x, sep='\t', index_col=0)
        geneIDs = list(deltas_df.index)
        deltas = deltas_df.values.astype(dtype=float)
        d_deltas = pd.read_csv(args.ue, sep='\t', index_col=0).values.astype(dtype=float)

    print("get clusters")
    clusters = {}
    with open(args.group_file) as fin:
        idx = 0
        for line in fin:
            data = line.strip().split("\t")
            # cluster = data[0]
            cluster = data[int(args.col_idx)]
            if cluster == "NULL":
                idx += 1
                continue
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(idx)
            idx += 1

    # run averaging
    print("run averaging")
    avg_activities, avg_deltas, significance = run_averaging(activities=deltas.T,
                                                             deltas=d_deltas.T,
                                                             clusters=clusters,
                                                             wms=geneIDs)

    # save results
    avg_activities_file = os.path.join(args.output_folder, "avg_activities.csv")
    avg_deltas_file = os.path.join(args.output_folder, "avg_deltas.csv")
    significance_file = os.path.join(args.output_folder, "significance.csv")

    avg_activities.to_csv(avg_activities_file)
    avg_deltas.to_csv(avg_deltas_file)
    significance.to_csv(significance_file)
