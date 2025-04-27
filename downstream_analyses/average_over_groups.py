
import numpy as np
import pandas as pd
import scipy.optimize as so
import sys



def find_tau(activities, variances):
    """ finds tau """
    (tau, fval, ierr, numfunc) = so.fminbound(log_likelihood, 0, 10000,
                                              args=(activities, variances),
                                              full_output=1, xtol=1.e-06)
    if ierr:
        sys.exit("log_activities has not converged after %d iterations.\n"
                 % numfunc)
    alpha = 1. / (variances + np.power(tau, 2))
    m0 = np.sum(alpha)
    m1 = np.sum(alpha * activities)
    mu = m1 / m0
    new_sig = np.sqrt(1 / m0)
    return (tau, mu, new_sig)


def log_likelihood(t, activities, variances):
    """ Calculate likelihood """
    alpha = 1. / (variances + t * t)
    #mSqrt = np.sum(np.sqrt(alpha))
    m0 = np.sum(alpha)
    m1 = np.sum(alpha * activities)
    m2 = np.sum(alpha * np.power(activities, 2))
    mu = m1 / m0
    #return (0.5 * (m0 * mu * mu - 2 * mu * m1 + m2) - np.log(mSqrt))
    return (0.5 * (m0 * mu * mu - 2 * mu * m1 + m2) - 0.5*np.sum(np.log(alpha)))



def run_averaging(activities, deltas, clusters, wms):
    variances = np.power(deltas, 2)

    # read cluster info and make groups
    # clusters is really a dict with list of cells assigned to 
        
    # init empty arrays for averaged data
    avg_activities = np.zeros((len(clusters), np.shape(activities)[1]))
    avg_deltas = np.zeros((len(clusters), np.shape(activities)[1]))
    taus = np.zeros((len(clusters), np.shape(activities)[1]))

    # calculate average activities and deltas
    # for clust_idx, cluster in tqdm(enumerate(sorted(clusters.keys()))):
    for clust_idx, cluster in enumerate(sorted(clusters.keys())):
        for mat_idx in range(np.shape(activities)[1]):
            # get sample subsets
            activities_subset = activities[clusters[cluster], mat_idx]
            deltas_subset = variances[clusters[cluster], mat_idx]
            (tau, avg_actvt, avg_dlt) = find_tau(activities_subset,
                                                 deltas_subset)
            # save data to tables
            taus[clust_idx, mat_idx] = tau
            avg_activities[clust_idx, mat_idx] = avg_actvt
            avg_deltas[clust_idx, mat_idx] = avg_dlt

    # write data
    avg_activities = pd.DataFrame(avg_activities,
                              index=sorted(clusters.keys()),
                              columns=wms)
    
    avg_deltas = pd.DataFrame(avg_deltas,
                              index=sorted(clusters.keys()),
                              columns=wms)
    
    # np.savetxt("averaged_taus",
    #            taus, delimiter="\t", fmt="%.6f")

    # recalculate Z-values
    zvals = avg_activities / avg_deltas
    significance = np.sqrt((zvals**2).sum(axis=0) / len(clusters.keys()))
    
    
    
    return avg_activities, avg_deltas, significance.sort_values(ascending=False)
