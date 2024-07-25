import numpy as np

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
# confuse_rate = 0.1
# simMM = create_simMat(K, confuse_rate)
# print(simMM)
