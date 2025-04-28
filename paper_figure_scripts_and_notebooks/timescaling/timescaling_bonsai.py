import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

scaling_df_all = pd.read_csv(
    '/Users/Daan/Documents/postdoc/bonsai-development/data/additional_data/timescaling.csv')
# n_cores_per_node_list = [10, 20]
n_cores_per_node_list = [20]
n_cores_opts = len(n_cores_per_node_list)

fig_core, ax_core = plt.subplots()
fig_total, ax_total = plt.subplots()

for n_cores_ind, n_cores_per_node in enumerate(n_cores_per_node_list):
    scaling_df = scaling_df_all[scaling_df_all['n_cores_per_node'] == n_cores_per_node]

    # Plot timescaling n_cells vs time, but normalized to n_genes
    n_cells = scaling_df['n_cells'].values
    date_format = "%B %d, %H:%M:%S"
    n_secs_total = []
    for row_ind in range(scaling_df.shape[0]):
        final_datetime = datetime.strptime(scaling_df['end_time_metadata'].iloc[row_ind], date_format)
        first_datetime = datetime.strptime(scaling_df['start_time_preprocess'].iloc[row_ind], date_format)
        n_secs_total.append((final_datetime - first_datetime).total_seconds())
    n_secs_total = np.array(n_secs_total)

    # n_secs = np.array([(datetime.strptime(time_str, '%H:%M:%S') - datetime(1900, 1, 1)).total_seconds() for time_str in
    #                    scaling_df['compute_time_core']])
    n_secs_core = scaling_df['compute_time_core'].values.astype(dtype=float)
    n_secs_total_normalized = n_secs_total / scaling_df['n_genes'].values * 2000
    n_secs_core_normalized = n_secs_core / scaling_df['n_genes'].values * 2000
    max_memories = scaling_df['max_memory_GB']

    """Make timescaling plot for core-computation time only."""
    figure_folder = '/Users/Daan/Documents/postdoc/bonsai-development/useful_scripts_not_bonsai/timescaling_checks/figures'

    skip_first_n = 3
    n_cells_corr = np.log(n_cells)[skip_first_n:]
    n_secs_corr = np.log(n_secs_core_normalized)[skip_first_n:]

    Clog = np.cov(np.vstack((n_cells_corr, n_secs_corr)))
    corrlog = Clog[0, 1] / np.sqrt(Clog[0, 0] * Clog[1, 1])
    eigVals, eigVecs = np.linalg.eig(Clog)
    max_eigval = np.argmax(eigVals)
    slopepca1log = eigVecs[1, max_eigval] / eigVecs[0, max_eigval]
    regLineXlog = np.linspace(n_cells_corr.min(), n_cells_corr.max(), 20)
    regLineYlog = slopepca1log * (regLineXlog - n_cells_corr.mean()) + n_secs_corr.mean()

    ax = ax_core
    if n_cores_per_node == 20:
        ax.plot(n_cells, n_secs_core_normalized, '*', c='gray', lw=2, markersize=10, label='Number of cores: {}'.format(n_cores_per_node))
    else:
        ax.plot(n_cells, n_secs_core_normalized, 'o', c='gray', lw=2, markersize=10, label='Number of cores: {}'.format(n_cores_per_node))
    ax.plot(np.exp(regLineXlog), np.exp(regLineYlog), '--', lw=2, c='black')
    if n_cores_ind == (n_cores_opts - 1):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
        ax.grid(True, which='minor', lw=0.25, linestyle='--')
        ax.set_xlabel("Number of cells")
        ax.set_ylabel("Compute time (seconds) \n normalized per 2000 genes.")
        ax.set_title(r'Fitted scaling: $T \propto |C|^{%.2f}$' % slopepca1log)
        plt.savefig(os.path.join(figure_folder, 'core_compute_scaling.png'), dpi=300)
        plt.savefig(os.path.join(figure_folder, 'core_compute_scaling.svg'))

    """Make same timescaling plot for total computation time as well"""
    n_cells_corr = np.log(n_cells)[skip_first_n:]
    n_secs_corr = np.log(n_secs_total_normalized)[skip_first_n:]

    Clog = np.cov(np.vstack((n_cells_corr, n_secs_corr)))
    corrlog = Clog[0, 1] / np.sqrt(Clog[0, 0] * Clog[1, 1])
    eigVals, eigVecs = np.linalg.eig(Clog)
    max_eigval = np.argmax(eigVals)
    slopepca1log = eigVecs[1, max_eigval] / eigVecs[0, max_eigval]
    regLineXlog = np.linspace(n_cells_corr.min(), n_cells_corr.max(), 20)
    regLineYlog = slopepca1log * (regLineXlog - n_cells_corr.mean()) + n_secs_corr.mean()
    prefactor = np.exp(slopepca1log * (0 - n_cells_corr.mean()) + n_secs_corr.mean())

    ax = ax_total
    if n_cores_per_node == 20:
        ax.plot(n_cells, n_secs_total_normalized, '*', c='gray', lw=2, markersize=10, label='Number of cores: {}'.format(n_cores_per_node))
    else:
        ax.plot(n_cells, n_secs_total_normalized, 'o', c='gray', lw=2, markersize=10, label='Number of cores: {}'.format(n_cores_per_node))
    ax.plot(np.exp(regLineXlog), np.exp(regLineYlog), '--', lw=2, c='black')
    # ax.plot(np.exp(regLineXlog), np.exp(prefactor) * np.exp(regLineXlog) ** slopepca1log, '--', lw=2, c='red')
    if n_cores_ind == (n_cores_opts - 1):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
        ax.grid(True, which='minor', lw=0.25, linestyle='--')
        ax.set_xlabel("Number of cells")
        ax.set_ylabel("Compute time (seconds) \n normalized per 2000 genes.")
        ax.set_title(r'Fitted scaling: $T \sim {%.4f}|C|^{%.2f}$' % (prefactor, slopepca1log))
        plt.savefig(os.path.join(figure_folder, 'total_compute_scaling.png'), dpi=300)
        plt.savefig(os.path.join(figure_folder, 'total_compute_scaling.svg'))

    """Make plot for max memory usage as well"""
    n_cells_corr = np.log(n_cells)[skip_first_n:]
    n_secs_corr = np.log(max_memories)[skip_first_n:]

    Clog = np.cov(np.vstack((n_cells_corr, n_secs_corr)))
    corrlog = Clog[0, 1] / np.sqrt(Clog[0, 0] * Clog[1, 1])
    eigVals, eigVecs = np.linalg.eig(Clog)
    max_eigval = np.argmax(eigVals)
    slopepca1log = eigVecs[1, max_eigval] / eigVecs[0, max_eigval]
    regLineXlog = np.linspace(n_cells_corr.min(), n_cells_corr.max(), 20)
    regLineYlog = slopepca1log * (regLineXlog - n_cells_corr.mean()) + n_secs_corr.mean()

    if n_cores_per_node == 10:
        fig, ax = plt.subplots()
        ax.plot(n_cells, max_memories, '*', c='gray', lw=2, markersize=10)
        ax.plot(np.exp(regLineXlog), np.exp(regLineYlog), '--', lw=2, c='black')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        ax.grid(True, which='minor', lw=0.25, linestyle='--')
        ax.set_xlabel("Number of cells")
        ax.set_ylabel("Maximal memory used by a single CPU")
        ax.set_title(r'Fitted scaling: $M \propto |C|^{%.2f}$' % slopepca1log)
        plt.savefig(os.path.join(figure_folder, 'memory_scaling.png'), dpi=300)
        plt.savefig(os.path.join(figure_folder, 'memory_scaling.svg'))

plt.show()