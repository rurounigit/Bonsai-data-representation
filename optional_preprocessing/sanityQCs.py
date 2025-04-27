from pathlib import Path
import os
from argparse import ArgumentParser
import csv
import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser(
    description='')
# Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
parser.add_argument('--sanity_folder', type=str,
                    default='/Users/Daan/Documents/postdoc/waddington-code-github/python_waddington_code/data/Zeisel',
                    help='Path to folder where Sanity results can be found.')

args = parser.parse_args()
figsPath = os.path.join(args.sanity_folder, 'QCfigs')
Path(figsPath).mkdir(parents=True, exist_ok=True)

# Get mean LTQ for each gene
mu_g = []
with open(os.path.join(args.sanity_folder, 'mu.txt'), 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        mu_g.append(float(row[0]))
mu_g = np.array(mu_g)

# Plot histogram of mean LTQs
fig, ax = plt.subplots()
ax.hist(mu_g, bins=100)
ax.set_xlabel("Mean LTQ per gene")
ax.set_ylabel("Frequency")
ax.set_title("Histogram of inferred mean log-transcription-quotients (LTQs)")
fig.savefig(os.path.join(figsPath, 'mean_LTQ_histogram.png'))

# Get variance for each gene
var_g = []
with open(os.path.join(args.sanity_folder, 'variance.txt'), 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        var_g.append(float(row[0]))
var_g = np.array(var_g)

# Plot histogram for variances
fig, ax = plt.subplots()
ax.hist(var_g, bins=100)
ax.set_xlabel("Variance per gene")
ax.set_ylabel("Frequency")
ax.set_title("Histogram of inferred variances per gene (v_g)")
ax.set_yscale('log')
fig.savefig(os.path.join(figsPath, 'var_histogram.png'))

# Plot scatter of mean against variance
fig, ax = plt.subplots()
ax.scatter(mu_g, var_g, s=0.1, alpha=0.5)
ax.set_xlabel("Mean LTQ per gene")
ax.set_ylabel("Variance in LTQs per gene")
ax.set_title("Scatter of mean log-transcription-quotient (LTQ) against variance in LTQ")
ax.set_yscale('log')
fig.savefig(os.path.join(figsPath, 'scatter_mean_var.png'))

# Get LTQ per cell and gene
deltas = []
with open(os.path.join(args.sanity_folder, 'delta.txt'), 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        deltas.append(list(map(float, row[0].split('\t'))))
deltas = np.array(deltas)

# Get error on LTQ per cell and gene
d_deltas = []
with open(os.path.join(args.sanity_folder, 'd_delta.txt'), 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        d_deltas.append(list(map(float, row[0].split('\t'))))
d_deltas = np.array(d_deltas)

# Plot reverse cumulative of z-scores
zscores = np.sqrt(np.mean(((deltas - np.mean(deltas, axis=1)[:, np.newaxis]) ** 2) / (d_deltas ** 2), axis=1))
# Plot histogram for variances
fig, ax = plt.subplots()
ax.hist(zscores, bins=200, cumulative=-1, histtype='step')
ax.set_xlabel("Zscore per gene: <((ltq - <ltq>)/std)^2> ")
ax.set_ylabel("Frequency")
ax.set_title("Histogram of zscores per gene, {} genes with zscore > 1".format(np.sum(zscores > 1)))
ax.set_yscale('log')
# ax.set_xscale('log')
plt.show()
fig.savefig(os.path.join(figsPath, 'zscores_histogram.png'))
