import numpy as np

def my_kmeans(data, K):
    """
    Lloyd's K-means clustering algorithm.
    :param data: N x d numpy array
    :param K: number of clusters
    :return: N-dimensional clustering result label
    """
    N, d = data.shape
    
    indices = np.random.choice(N, K, replace=False)
    centroids = data[indices].copy()
    
    labels = np.zeros(N, dtype=int) - 1
    
    num_iterations = 0
    max_iterations = 1000
    
    while num_iterations < max_iterations:
        num_iterations += 1
        
        diffs = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        
        new_labels = np.argmin(distances, axis=1)
        
        if np.array_equal(labels, new_labels):
            break
            
        labels = new_labels
        
        for k in range(K):
            cluster_data = data[labels == k]
            if len(cluster_data) > 0:
                centroids[k] = np.mean(cluster_data, axis=0)
            else:
                centroids[k] = data[np.random.choice(N)].copy()
                
    return labels
