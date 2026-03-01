import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from my_kmeans import my_kmeans
from my_spectralclustering import my_spectralclustering

def run():
    mat_files = glob.glob('*.mat')
    if not mat_files:
        mat_files = glob.glob('toydata/cluster/*.mat')
        
    if not mat_files:
        print("No .mat files found. Please extract toydata.zip to the current directory.")
        return

    config = {
        'Aggregation': {'K': 7, 'sigma': 2.0},
        'Bridge': {'K': 2, 'sigma': 1.0},
        'Compound': {'K': 6, 'sigma': 1.5},
        'Flame': {'K': 2, 'sigma': 1.0},
        'Jain': {'K': 2, 'sigma': 1.5},
        'Spiral': {'K': 3, 'sigma': 2.0},
        'TwoDiamond': {'K': 2, 'sigma': 0.5},
    }

    for f in mat_files:
        print(f"Processing {f}...")
        try:
            mat = sio.loadmat(f)
        except Exception as e:
            print(f" Could not load {f}: {e}")
            continue
            
        if 'D' not in mat or 'L' not in mat:
            print(f" Skipping {f}: 'D' or 'L' not found.")
            continue
            
        data = mat['D']
        ground_truth = mat['L'].flatten()
        
        filename = os.path.basename(f).split('.')[0]
        
        matched_key = None
        for k in config:
            if k.lower() in filename.lower():
                matched_key = k
                break
                
        if matched_key:
            K = config[matched_key]['K']
            sigma = config[matched_key]['sigma']
        else:
            K = len(np.unique(ground_truth))
            sigma = 1.0
            print(f" Dataset name not recognized in config. Deduced K={K} from ground truth, using default sigma={sigma}")
            
        print(f" Dataset {filename}: K={K}, sigma={sigma}")
        
        np.random.seed(42)
        labels_kmeans = my_kmeans(data, K)
        
        np.random.seed(42)
        labels_spectral = my_spectralclustering(data, K, sigma)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.scatter(data[:, 0], data[:, 1], c=ground_truth, s=10, cmap='tab10')
        plt.title('Ground Truth')
        
        plt.subplot(1, 3, 2)
        plt.scatter(data[:, 0], data[:, 1], c=labels_kmeans, s=10, cmap='tab10')
        plt.title('K-Means')
        
        plt.subplot(1, 3, 3)
        plt.scatter(data[:, 0], data[:, 1], c=labels_spectral, s=10, cmap='tab10')
        plt.title(f'Spectral Clustering (sigma={sigma})')
        
        plt.suptitle(filename)
        output_name = f"{filename}_results.png"
        plt.savefig(output_name)
        plt.close()
        print(f" Saved plot to {output_name}\n")

if __name__ == '__main__':
    run()
