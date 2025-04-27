import pandas as pd
import numpy as np
from pathlib import Path
import os

LOG_TRANSFORM = False
PCA = 50

if LOG_TRANSFORM:
    data_path = '/Users/Daan/Documents/postdoc/bonsai-development/data/football_stats_logged'
elif PCA is None:
    data_path = '/Users/Daan/Documents/postdoc/bonsai-development/data/football_stats_no_pca'
elif PCA < 0:
    data_path = '/Users/Daan/Documents/postdoc/bonsai-development/data/football_stats_pca_rotation'
else:
    data_path = '/Users/Daan/Documents/postdoc/bonsai-development/data/football_stats_all'

Path(data_path).mkdir(parents=True, exist_ok=True)

acronym_to_descr = {}
with open('/Users/Daan/Documents/postdoc/bonsai-development/data/football_stats/acronyms.txt', 'r') as f:
    for line in f:
        acr, descr = line.strip().split(' : ')
        acronym_to_descr[acr] = descr

football_stats_file = '/Users/Daan/Documents/postdoc/bonsai-development/data/football_stats/2021-2022 Football Player Stats.csv'
football_stats = pd.read_csv(football_stats_file, encoding='ISO-8859-1', delimiter=';')

# Drop one player with NaN-values
football_stats = football_stats.dropna()

# Take only the 50% of players that played most
football_stats = football_stats[football_stats['Min'] > football_stats['Min'].median()]

# TODO: Remove this
# Take subset
# stats = stats[stats['Squad'].isin(['Manchester City', 'Manchester Utd', 'Barcelona'])]
# stats = stats[stats['Comp'].isin(['Premier League'])]

# Drop players that occur twice (they switched club, complicated)
player_counts = football_stats["Player"].value_counts()
players_to_drop = player_counts[player_counts > 1].index
football_stats = football_stats[~football_stats["Player"].isin(players_to_drop)]

# Get numerical columns
football_stats_nums = football_stats.select_dtypes(include=["number"]).drop(columns='Rk')
football_stats_text = football_stats.select_dtypes(include=["object", "string"])

player_names = football_stats_text['Player']

player_market_values = pd.read_csv("/Users/Daan/Documents/postdoc/bonsai-development/data/football_stats_all/player_market_value.csv")
market_vals_names = list(player_market_values.player_name)
market_vals = []
for ind_fs, player in enumerate(player_names):
    if player in market_vals_names:
        market_vals.append(player_market_values.player_market_value_euro[market_vals_names.index(player)])
    else:
        market_vals.append(np.nan)

player_names = [name.replace(' ', '_') for name in player_names]

#
# GK_stats = football_stats_nums[football_stats_text.Pos == 'GK']
# GK_stats.index = football_stats_text[football_stats_text.Pos == 'GK'].Player
# GK_ranks = pd.DataFrame(np.argsort(np.argsort(GK_stats, axis=0), axis=0), columns=GK_stats.columns, index=GK_stats.index)

# Drop some redundant columns
# redundant_cols = ["Born", '90s']
# football_stats_nums = football_stats_nums.drop(columns=redundant_cols)
feature_names = football_stats_nums.columns

annotation = football_stats_text.drop(columns='Player')
annotation['market_value'] = market_vals
annotation['log10_market_value'] = np.log10(market_vals)


def do_pca(data, n_comps=50, return_svd=False):
    """

    :param data: should be a numpy array with features (genes) as rows, observations (cells) as columns
    :param n_comps: Should be an integer indicating for what number of components, PCA should be done
    :return:
    """
    data_T = data.T
    n, m = data_T.shape
    pca_centers = data_T.mean(axis=0)
    data_cd = data_T - pca_centers

    U, S, Vh = np.linalg.svd(data_cd, full_matrices=False)
    transformed_data = np.matmul(U, np.diag(S))

    if n_comps > 0:
        proj_data = transformed_data[:, :n_comps].T
    else:
        proj_data = transformed_data.T
    if return_svd:
        return proj_data, U, S, Vh
    return proj_data

orig_stats_annot = football_stats_nums.values.copy()
football_features = football_stats_nums.values.T

if LOG_TRANSFORM:
    football_features_norm = np.log(1 + football_features * np.median(football_stats_nums.Min/90))
else:
    # Center and normalize variance on all features
    football_features_norm = football_features - np.mean(football_features, axis=1, keepdims=True)
    football_features_norm /= np.sqrt(football_features.var(axis=1, keepdims=True))

if PCA is not None:
    football_features_pca, U, S, Vh = do_pca(football_features_norm, PCA, return_svd=True)
    football_features_pca /= np.sqrt(np.sum((S ** 2) / football_features.shape[1]))
    dominant_features_first_pcas = [[acronym_to_descr[feature_names[feat_ind]] for feat_ind in row[:10]] for row in
                                    np.argsort(-np.abs(Vh.T), axis=0).T]
    np.save(os.path.join(data_path, 'svd_U'), U)
    np.save(os.path.join(data_path, 'svd_S'), S)
    np.save(os.path.join(data_path, 'svd_Vh'), Vh)
else:
    football_features_pca = football_features_norm

football_features_std = np.ones_like(football_features_pca) * 1e-3

print("Writing annotation to file:")
annotation['Pos_coarse'] = [pos[:2] for pos in annotation.Pos]
annotation.index = player_names
Path(os.path.join(data_path, 'annotation')).mkdir(parents=True, exist_ok=True)
annotation.to_csv(os.path.join(data_path, 'annotation', 'player_annotation.csv'))

print("Writing original data to annotation file")
orig_data_annot = pd.DataFrame(orig_stats_annot, index=player_names, columns=feature_names)
orig_data_annot.to_csv(os.path.join(data_path, 'annotation', 'mat_orig_features.csv'))

orig_data_annot = pd.DataFrame(football_features_norm.T, index=player_names, columns=feature_names)
orig_data_annot.to_csv(os.path.join(data_path, 'annotation', 'mat_orig_features_norm.csv'))

print("Writing deltas to file.")
geneID = ['PCA_{}'.format(ind) for ind in range(football_features_pca.shape[0])]
delta_df = pd.DataFrame(football_features_pca, columns=player_names, index=geneID)
delta_df.to_csv(os.path.join(data_path, 'delta_vmax.txt'), sep='\t', header=False, index=False)

print("Writing d_deltas to file.")
d_delta_df = pd.DataFrame(football_features_std, columns=player_names, index=geneID)
d_delta_df.to_csv(os.path.join(data_path, 'd_delta_vmax.txt'), sep='\t', header=False, index=False)

print("Writing true variances to file:")
with open(os.path.join(data_path, 'variance_vmax.txt'), 'w') as f:
    for var in np.ones(football_features_pca.shape[0]):
        f.write("%s\n" % var)

print("Writing cell IDs to file:")
with open(os.path.join(data_path, 'cellID.txt'), 'w') as f:
    for ID in player_names:
        f.write("%s\n" % ID)
print("Writing gene IDs to file:")
with open(os.path.join(data_path, 'geneID.txt'), 'w') as f:
    for ID in geneID:
        f.write("%s\n" % ID)
