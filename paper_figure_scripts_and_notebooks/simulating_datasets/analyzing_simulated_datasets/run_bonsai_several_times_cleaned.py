import subprocess
import os, sys
from argparse import ArgumentParser

LOG_FILE = 'bonsai_output_10k.log'

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
os.chdir(parent_dir)

from bonsai.bonsai_helpers import str2bool

parser = ArgumentParser(
    description='Runs Bonsai on several simulated datasets.')

# Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
parser.add_argument('--input_folder', type=str, default='data/simulated_datasets',
                    help="Relative path from bonsai_development to base-folder where simulated trees can be found.")
parser.add_argument('--results_folder', type=str, default='results/simulated_datasets',
                    help="Relative path from bonsai_development to base-folder where reconstructed trees can be found.")
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

num_dims_list = [int(num_dim) for num_dim in args.num_dims.split(',')]
n_cells_per_clst_list = [int(ncpc) for ncpc in args.n_cells_per_clst.split(',')]
n_clsts = int(args.n_sampled_clsts)
noise_var = args.noise_var

# For Figure where Bonsai gets better at higher dimensions, use this:
# gene_nums = [2, 10, 100, 1000, 10000]
# ns_cells_per_clst = [1]

# For Figure where Bonsai gets better at more cells use this:
# gene_nums = [20]
# ADD_NOISE = True
# noise_var = 1
# ns_cells_per_clst = [1, 2, 5, 10, 20]  # , 50, 100]
# ns_cells_per_clst = [10, 20]  # , 50, 100]

# CELL_DEPENDENT = False
#
# if args.add_noise:
#     if not CELL_DEPENDENT:
#         add_noise = '_add_noise_{}'.format(int(noise_var))
#     else:
#         add_noise = '_add_noise_{}_celldependent'.format(int(noise_var))
# else:
#     add_noise = ''

for gene_num in num_dims_list:
    for n_cells_per_clst in n_cells_per_clst_list:
        n_cells = n_clsts * n_cells_per_clst

        datadir = "simulate_equidistant_{}_clsts_{}_cells_{}_dims".format(n_clsts, n_cells, gene_num)
        if args.random_times:
            datadir += '_random_times'
        if not args.sample_umi_counts:
            datadir += '_no_umi_counts'
        if args.add_noise:
            datadir += '_add_noise_{}'.format(int(noise_var))
        datadir += '_seed_{}'.format(seed)
        dataset = os.path.join(args.input_folder, datadir)
        data_path = os.path.abspath(os.path.join(args.input_folder, datadir))
        results_path = os.path.abspath(os.path.join(args.results_folder, datadir))

        command1 = [sys.executable,
                    'bonsai/create_config_file.py',
                   '--new_yaml_path',
                   os.path.join(data_path, 'new_yaml.yaml'),
                   '--dataset',
                   dataset,
                   '--verbose',
                   'True',
                   '--data_folder',
                   data_path,
                   '--results_folder',
                   results_path,
                   '--input_is_sanity_output',
                   'False',
                   '--zscore_cutoff',
                   '1.0',
                   '--UB_ellipsoid_size',
                   '1.0',
                   '--skip_greedy_merging',
                   'False',
                   '--skip_redo_starry',
                   'False',
                   '--skip_opt_times',
                   'False',
                   '--skip_nnn_reordering',
                   'False',
                   '--nnn_n_randommoves',
                   '100',
                   '--nnn_n_randomtrees',
                   '2',
                   '--pickup_intermediate',
                   'False',
                   '--use_knn',
                   '10',
                   '--filenames_data',
                   'delta.txt,d_delta.txt',
                   '--rescale_by_var',
                   'False']

        output1 = subprocess.run(command1, stdout=subprocess.PIPE, text=True)
        print(output1.stdout)
        print(output1.stderr)

        command2 = [sys.executable,
            'bonsai/bonsai_main.py',
            '--config_filepath',
            os.path.join(data_path, 'new_yaml.yaml'),
            '--step',
            'all']

        with open(LOG_FILE, "a") as file:
            output2 = subprocess.run(command2, stdout=file, text=True)

        print(output2.stdout)
        print("\n\n\n\n\n STDERR \n\n")
        print(output2.stderr)

print(output1)

