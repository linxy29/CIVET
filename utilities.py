import numpy as np
from scipy.stats import dirichlet, multinomial

def create_simMat(K, confuse_rate):
    """
    Generate similarity matrix with uniform confusion rate to non-self clusters.
    
    Parameters:
    K (int): Number of clusters.
    confuse_rate (float): Confusion rate, uniformly distributed to non-self clusters.
    
    Returns:
    numpy.ndarray: A similarity matrix with uniform confusion with other clusters.
    """
    return np.eye(K) * (1 - confuse_rate) + confuse_rate * (1 - np.eye(K)) / (K - 1)

# Example usage
# K = 4
# confuse_rate = 0.1za
# simMM = create_simMat(K, confuse_rate)
# print(simMM)


def multinom_EM(X, simMM, min_iter=10, max_iter=1000, logLik_threshold=1e-2):
    K = simMM.shape[1]

    mu = np.random.dirichlet(np.ones(K))
    Z = np.zeros((K, K))
    logLik_old = logLik_new = np.sum(np.log(np.dot(mu, simMM)) * X)

    for it in range(max_iter):
        for i in range(K):
            for j in range(K):
                Z[i, j] = simMM[i, j] * mu[i] / np.sum(mu * simMM[:, j])
        
        mu = np.dot(Z, X)
        mu = mu / np.sum(mu)

        logLik_new = np.sum(np.log(np.dot(mu, simMM)) * X)
        if it > min_iter and logLik_new - logLik_old < logLik_threshold:
            break
        else:
            logLik_old = logLik_new
    
    return {
        "mu": mu,
        "logLik": logLik_new,
        "simMM": simMM,
        "X": X,
        "X_prop": X / np.sum(X),
        "predict_X_prop": np.dot(mu, simMM)
    }

# # Example usage
# X = np.array([100, 300, 1500, 500, 1000])
# simMM = create_simMat(5, confuse_rate=0.2)
# print(simMM)
# result = multinom_EM(X, simMM)

# print(result)

def simulator_base(totals_cond1, totals_cond2, dirichlet_s1, dirichlet_s2, similarity_mat=None):
    """
    Simulates data based on the input totals, Dirichlet parameters, and similarity matrix.
    """
    K = len(dirichlet_s1)  # number of clusters
    n_rep1 = len(totals_cond1)  # number of replicates in condition 1
    n_rep2 = len(totals_cond2)  # number of replicates in condition 2

    prop_cond1 = np.zeros((n_rep1, K))
    prop_cond2 = np.zeros((n_rep2, K))
    numb_cond1 = np.zeros((n_rep1, K), dtype=int)
    numb_cond2 = np.zeros((n_rep2, K), dtype=int)

    # Sampling for condition 1
    for i in range(n_rep1):
        prop_cond1[i, :] = dirichlet.rvs(dirichlet_s1)[0]
        numb_cond1[i, :] = multinomial.rvs(totals_cond1[i], prop_cond1[i, :])

    # Sampling for condition 2
    for i in range(n_rep2):
        prop_cond2[i, :] = dirichlet.rvs(dirichlet_s2)[0]
        numb_cond2[i, :] = multinomial.rvs(totals_cond2[i], prop_cond2[i, :])

    if similarity_mat is None:
        similarity_mat = np.eye(K)

    # Adjusting counts based on similarity matrix
    numb_out_cond1 = np.zeros((n_rep1, K), dtype=int)
    numb_out_cond2 = np.zeros((n_rep2, K), dtype=int)

    for i in range(n_rep1):
        for j in range(K):
            numb_out_cond1[i, :] += multinomial.rvs(numb_cond1[i, j], similarity_mat[j, :])

    for i in range(n_rep2):
        for j in range(K):
            numb_out_cond2[i, :] += multinomial.rvs(numb_cond2[i, j], similarity_mat[j, :])

    return {
        "numb_cond1": numb_out_cond1,
        "numb_cond2": numb_out_cond2,
        "true_cond1": numb_cond1,
        "true_cond2": numb_cond2
    }


# Example usage:
# K = 2
# totals1 = np.array([100, 800, 1300, 600])
# totals2 = np.array([250, 700, 1100])
# diri_s1 = np.ones(K) * 20
# diri_s2 = np.ones(K) * 20
# confuse_rate = 0.2
# simil_mat = create_simMat(K, confuse_rate)
# print(simil_mat)
# sim_dat = simulator_base(totals1, totals2, diri_s1, diri_s2, simil_mat)
# print(sim_dat)
