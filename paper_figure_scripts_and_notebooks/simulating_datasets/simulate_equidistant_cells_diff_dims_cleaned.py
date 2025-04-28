import numpy as np
import os
import sys
import h5py
from pathlib import Path
import pandas as pd
from scipy.special import logsumexp
from argparse import ArgumentParser

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
os.chdir(parent_dir)

from bonsai.bonsai_helpers import str2bool

parser = ArgumentParser(
    description='Simulates a binary tree.')

# Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
parser.add_argument('--input_dataset', type=str, default='baron.npy',
                    help='Path to file with information from the gene table from the Baron-dataset. In fact, only number'
                         'of rows, columns and row-sums, column-sums are stored..')
parser.add_argument('--results_folder', type=str, default='data/simulated_datasets',
                    help="Relative path from bonsai_development to base-folder where simulated trees need to be stored.")
parser.add_argument('--num_dims', type=str, default="100",
                    help="Number of dimensions in which we sample the cells.")
parser.add_argument('--n_sampled_clsts', type=str, default="20",
                    help="Number of clusters in star-tree.")
parser.add_argument('--n_cells_per_clst', type=str, default="1,2,5,10,20,50",
                    help="Number of cells per cluster.")
parser.add_argument('--random_times', type=str2bool, default=True,
                    help="Determine if branch lengths of true tree should be sampled uniform in a logspace between"
                         "0.1 and 10, instead of keeping them all at 1.")
parser.add_argument('--sample_umi_counts', type=str2bool, default=False,
                    help="Determines if we want to ensure the tqs add up to 1. Doesn't have to when we don't sample "
                         "counts but rather give the true data to Bonsai.")
parser.add_argument('--add_noise', type=str2bool, default=True,
                    help="Determines if we want to ensure the tqs add up to 1. Doesn't have to when we don't sample "
                         "counts but rather give the true data to Bonsai.")
parser.add_argument('--seed', type=int, default=1231,
                    help="Sets the random seed.")
parser.add_argument('--noise_var', type=float, default=5.0,
                    help="Determines the variance of the Gaussian noise from which we sample cells around cluster"
                         "Typical variance from cluster-centers to mean is 1.0.")

args = parser.parse_args()
print(args)

seed = args.seed
np.random.seed(seed)

baron_hdf = h5py.File(args.input_dataset, 'r')
N_c = baron_hdf['N_c'][:]
nGenes = baron_hdf.attrs['nGenes']
nCells = baron_hdf.attrs['nCells']
all_N_g = baron_hdf['N_g'][:]
baron_hdf.close()

num_dims_list = [int(num_dim) for num_dim in args.num_dims.split(',')]
n_cells_per_clst_list = [int(ncpc) for ncpc in args.n_cells_per_clst.split(',')]
n_clsts = int(args.n_sampled_clsts)
NOISE_VAR = args.noise_var

# For Figure where Bonsai gets better at higher dimensions, use this:
# if not args.add_noise:
#     num_dims_list = [2, 10, 100, 1000, 10000]
#     n_cells_per_clst_list = [1]

# For Figure where Bonsai gets better at more cells use this:
# if args.add_noise:
#     num_dims_list = [100]
#     n_cells_per_clst_list = [1, 2, 5, 10, 20, 50, 100]

# Draw random seeds here, making sure that the random noise is the same in all datasets (having either 1,2,5,10,20 cells
# per clust. So, the first random sample is the same in all datasets, the 2nd is the same in all but the first dataset.
random_seeds_noise = np.random.randint(1e6, size=(n_clsts, np.max(n_cells_per_clst_list)))
random_seeds_clsts = np.random.randint(1e6, size=n_clsts)

if args.random_times:
    lbTime = 0.1
    ubTime = 10
    diff_times = np.exp(np.random.uniform(np.log(lbTime), np.log(ubTime), size=n_clsts))
    diff_times /= np.mean(diff_times)
else:
    diff_times = np.ones(n_clsts)

for n_cells_ind, n_cells_per_clst in enumerate(n_cells_per_clst_list):
    for num_dims in num_dims_list:
        N_g = all_N_g[:num_dims]

        # Rescale N_c such that we have around the same count per gene as in the Baron dataset
        N_c = np.ceil(N_c * (num_dims / nGenes))

        # Randomly sample which cell gets what total umi count
        np.random.shuffle(N_c)

        if not args.add_noise:
            # Draw gene variances from an exponential with mean 2
            gene_variances = np.random.exponential(2, num_dims)
        else:
            gene_variances = np.ones(num_dims)

        # Take the mean ltqs such that they are sampled from the real data, and that the sum of tqs will in expectation be 1
        mean_ltq = np.log(N_g / np.sum(N_g)) - .5 * gene_variances

        nDraws = n_clsts * n_cells_per_clst
        if nDraws > nCells:
            N_c = np.tile(N_c, int(np.ceil(nDraws / nCells)))

        # Sample ltqs for each cluster
        true_coords_gc = np.zeros((num_dims, nDraws))
        delta_gc = np.zeros((num_dims, nDraws))
        for ind in range(n_clsts):
            diff_time = diff_times[ind]
            np.random.seed(random_seeds_clsts[ind])
            true_coords_gc[:, n_cells_per_clst * ind: n_cells_per_clst * (ind + 1)] = np.random.normal(0, np.sqrt(
                diff_time * gene_variances), num_dims)[:, None]
            if args.add_noise:
                for cell_ind in range(n_cells_per_clst):
                    np.random.seed(random_seeds_noise[ind, cell_ind])
                    delta_gc[:, n_cells_per_clst * ind + cell_ind] = true_coords_gc[:, n_cells_per_clst * ind + cell_ind] + np.random.normal(0, np.sqrt(NOISE_VAR), num_dims)
            else:
                delta_gc = true_coords_gc

        # Now center the deltas of the cells such that the average is 0
        delta_mean = np.mean(delta_gc, axis=1)
        delta_gc -= delta_mean[:, None]

        # Normalise such that each gene has the prescribed variance
        if not args.add_noise:
            factor = np.sqrt(gene_variances / np.var(delta_gc, axis=1))
            delta_gc *= factor[:, None]

        # Add average LTQ for each gene
        ltqs_gc = mean_ltq[:, None] + delta_gc

        """We only need to make sure that the TQs add up to 1 when we really sample counts. If we just run Bonsai on the
        true data, we don't have to do it. Then it's better to not do it, because normalizing this basically makes a 2-dim
        dataset into a 1D-one. This makes capturing the distances in a tree too easy."""
        if args.sample_umi_counts:
            # Normalise such that each cell's TQs add up to 1
            log_tqs = logsumexp(ltqs_gc, axis=0)
            ltqs_gc = ltqs_gc - log_tqs

        # Get true means and vars
        true_means_g = np.mean(ltqs_gc, axis=1)
        if args.add_noise:
            true_vars_g = gene_variances
        else:
            true_vars_g = np.var(ltqs_gc, axis=1)
        delta_gc = ltqs_gc - true_means_g[:, None]

        """---------Now store the simulated dataset somewhere---------"""

        datadir = "simulate_equidistant_{}_clsts_{}_cells_{}_dims".format(n_clsts, nDraws, num_dims)
        if args.random_times:
            datadir += '_random_times'
        if not args.sample_umi_counts:
            datadir += '_no_umi_counts'
        if args.add_noise:
            datadir += '_add_noise_{}'.format(int(NOISE_VAR))


        datadir += '_seed_{}'.format(seed)

        geneID = ['Gene_' + str(ind) for ind in range(num_dims)]
        cellIds = ['Cell_' + str(ind) for ind in range(nDraws)]

        data_path = os.path.abspath(os.path.join(args.results_folder, datadir))
        Path(data_path).mkdir(parents=True, exist_ok=True)

        print("Writing to path: {}".format(data_path))
        # print("Writing celltypes to file:")
        # annotation_dict = {}
        # most_generations = np.max([len(ID.split('Cell')[1]) for ID in cellIds])
        # celltype_list = []
        # for clst_ind in range(n_clsts):
        #     celltype_list += ['Clst_{}'.format(clst_ind)] * n_cells_per_clst
        #     annotation_dict['Clusters'] = celltype_list
        # annotation_df = pd.DataFrame(annotation_dict, index=cellIds)
        # Path(os.path.join(data_path, 'annotation')).mkdir(parents=True, exist_ok=True)
        # annotation_df.to_csv(os.path.join(data_path, 'annotation', 'cell_annotation.csv'))

        print("Writing true deltas to file.")
        if args.add_noise:
            delta_true_df = pd.DataFrame(true_coords_gc, columns=cellIds, index=geneID)
            delta_true_df.to_csv(os.path.join(data_path, 'delta_true.txt'), sep='\t', header=False, index=False)
            delta_df = pd.DataFrame(delta_gc, columns=cellIds, index=geneID)
            delta_df.to_csv(os.path.join(data_path, 'delta.txt'), sep='\t', header=False, index=False)
        else:
            delta_df = pd.DataFrame(delta_gc, columns=cellIds, index=geneID)
            delta_df.to_csv(os.path.join(data_path, 'delta_true.txt'), sep='\t', header=False, index=False)
            delta_df.to_csv(os.path.join(data_path, 'delta.txt'), sep='\t', header=False, index=False)

        if args.add_noise:
            d_delta_gc = np.ones_like(delta_gc) * np.sqrt(NOISE_VAR)
        else:
            d_delta_gc = np.ones_like(delta_gc) * 1e-6
        print("Writing true d_deltas to file.")
        d_delta_df = pd.DataFrame(d_delta_gc, columns=cellIds, index=geneID)
        d_delta_df.to_csv(os.path.join(data_path, 'd_delta_true.txt'), sep='\t', header=False, index=False)
        d_delta_df.to_csv(os.path.join(data_path, 'd_delta.txt'), sep='\t', header=False, index=False)

        # print("Writing true variances to file:")
        # with open(os.path.join(data_path, 'variance_true.txt'), 'w') as f:
        #     for var in true_vars_g:
        #         f.write("%s\n" % var)

        print("Writing cell IDs to file:")
        with open(os.path.join(data_path, 'cellID.txt'), 'w') as f:
            for ID in cellIds:
                f.write("%s\n" % ID)
        print("Writing gene IDs to file:")
        with open(os.path.join(data_path, 'geneID.txt'), 'w') as f:
            for ID in geneID:
                f.write("%s\n" % ID)

        if args.sample_umi_counts:
            print("Sampling UMI counts:")
            umi_counts = np.zeros((num_dims, nDraws))

            for cell_ind in range(nDraws):
                if cell_ind % 100 == 0:
                    print("Sampling counts for cell %d." % cell_ind)
                umi_counts[:, cell_ind] = np.random.multinomial(N_c[cell_ind], np.exp(ltqs_gc[:, cell_ind]))

            print("Writing UMI counts to file:")
            umi_df = pd.DataFrame(umi_counts, columns=cellIds, index=geneID)
            umi_df.to_csv(os.path.join(data_path, 'Gene_table.txt'), sep='\t', index_label="GeneID")
