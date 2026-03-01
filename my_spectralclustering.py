import numpy as np
from scipy.spatial.distance import pdist, squareform
from my_kmeans import my_kmeans

def my_spectralclustering(data, K, sigma):
    """
    Spectral clustering algorithm.
    :param data: N x d numpy array
    :param K: number of clusters
    :param sigma: bandwidth for Gaussian kernel
    :return: N-dimensional clustering result label
    """
    N = data.shape[0]
    
    sq_dists = squareform(pdist(data, 'sqeuclidean'))
    
    W = np.exp(-sq_dists / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)
    
    D_vec = np.sum(W, axis=1)
    
    D_vec_inv_sqrt = 1.0 / np.sqrt(D_vec + 1e-10)
    
    L_sym = np.eye(N) - (D_vec_inv_sqrt[:, np.newaxis] * W * D_vec_inv_sqrt[np.newaxis, :])
    
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
    
    U = eigenvectors[:, :K]
    
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    U = U / (norms + 1e-10)
    
    labels = my_kmeans(U, K)
    
    return labels
