import numpy as np
from utilities import create_simMat

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

# Example usage
X = np.array([100, 300, 1500, 500, 1000])
simMM = create_simMat(5, confuse_rate=0.2)
print(simMM)
result = multinom_EM(X, simMM)

print(result)
