from argparse import ArgumentParser
import os
import sys
import numpy as np
import csv
from pathlib import Path

# Get the parent directory
# parent_dir = Path(os.path.realpath(__file__)).parents[1]
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# Add the parent directory to sys.path
print(parent_dir)
sys.path.append(parent_dir)
from bonsai.bonsai_helpers import mp_print, str2bool, Run_Configs
from bonsai.bonsai_dataprocessing import initializeSCData
from bonsai.bonsai_treeHelpers import TreeNode
import bonsai.bonsai_globals as bs_glob

parser = ArgumentParser(
    description='Creates a tree with all common cells of a specific celltype hanging from one internal node. These'
                'internal nodes are then connected to the root.')

# Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
parser.add_argument('--config_filepath', type=str, default=None,
                    help='Absolute (or relative to "bonsai-development") path to YAML-file that contains all arguments'
                         'needed to run Bonsai.')
# parser.add_argument('--dataset', type=str, default='Zeisel',
#                     help='Name of dataset. This will determine name of results-folder that is created.')
# parser.add_argument('--data_folder', type=str, default='data',
#                     help='path to folder where input data can be found. This folder should contain a file with means'
#                          'and standard-deviations in files "delta.txt" and "d_delta.txt" unless argument '
#                          'filenames_data changes this behaviour.')
# parser.add_argument('--filenames_data', type=str, default='delta.txt,d_delta.txt',
#                     help='Filenames of input-files for means and standard deviations separated by a comma. '
#                          'These files should have different cells in the columns, and '
#                          'features (like gene expression quotients) as rows.')
parser.add_argument('--premerged_folder', type=str, default='Zeisel_genes/data/startpoints',
                    help='absolute path pointing towards folder, '
                         'where created tree should be stored.')
parser.add_argument('--cellstates_file', type=str, default='/Users/Daan/Documents/postdoc/waddington-code-github/python_waddington_code/data/Zeisel_genes/optimised_clusters.txt',
                    help='Absolute path to file that contains clustering found by cellstates. ')
# parser.add_argument('--zscore_cutoff', type=float, default=-1.,
#                     help='Genes with a variability under this cutoff will be dropped. Negative means: keep all.'
#                          'zscore_cutoff.')
parser.add_argument("--opt_times", type=str2bool, default=True,
                    help="Decides whether all times are optimised after tree reconstruction.")
# Arguments that determine running configurations of bonsai. How much is printed, which steps are run?
parser.add_argument('--verbose', type=str2bool, default=True,
                    help='--verbose False only shows essential print messages (default: True)')

args = parser.parse_args()
mp_print(args)

premerged_folder = args.premerged_folder
cellstates_file = args.cellstates_file
opt_times = args.opt_times
verbose = args.verbose

args = Run_Configs(args.config_filepath)

args.premerged_folder = premerged_folder
args.cellstates_file = cellstates_file
args.opt_times = opt_times
args.verbose = verbose

bs_glob.nwk_counter = None
scData = initializeSCData(args, createStarTree=True, getOrigData=False, otherRanksMinimalInfo=True, noDataNeeded=True,
                          optTimes=args.opt_times)

# scData.tree.root.mergeZeroTimeChilds()
# scData.mergers = scData.tree.getMergeList()
# scData.makeIgraphTree()
# scData.get_celltype_annotations(hierarchy=False)

cs_clustering = []
with open(args.cellstates_file, 'r') as file:
    reader = csv.reader(file, delimiter="\t")
    for row in reader:
        cs_clustering.append(row[0])

# Get all cellstates with more than one cell
cell_categories, cat_counts = np.unique(cs_clustering, return_counts=True)
cs_lt_one = [cell_cat for ind_cat, cell_cat in enumerate(cell_categories) if cat_counts[ind_cat] > 1]

# Add an ancestor for all cellstates with more than one cell
ctAncestors = []
for ct in cs_lt_one:
    ctAncestors.append(TreeNode(nodeInd=bs_glob.nNodes - 1, childNodes=[], isLeaf=False, tParent=1.0))
    bs_glob.nNodes += 1

for child in scData.tree.root.childNodes:
    childId = child.nodeId
    cellInd = scData.metadata.cellIds.index(childId)
    cs = cs_clustering[cellInd]
    if cs in cs_lt_one:
        csAncInd = cs_lt_one.index(cs)
        ctAncestors[csAncInd].childNodes.append(child)
    else:
        ctAncestors.append(child)

scData.tree.root.childNodes = ctAncestors

# Do a first optimisation of all times
if args.opt_times:
    scData.tree.optTimes(verbose=True)
else:
    scData.tree.calcLogLComplete()

scData.storeTreeInFolder(args.premerged_folder, with_coords=True, verbose=args.verbose)
