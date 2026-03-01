use nalgebra::{DMatrix, SymmetricEigen};
use crate::my_kmeans::my_kmeans;

pub fn my_spectralclustering(data: &DMatrix<f64>, k: usize, sigma: f64) -> Vec<usize> {
    let n = data.nrows();
    let mut w = DMatrix::zeros(n, n);
    
    // 1. Compute the similarity graph (Gaussian kernel)
    for i in 0..n {
        for j in (i+1)..n {
            let dist_sq = (&data.row(i) - &data.row(j)).norm_squared();
            let sim = (-dist_sq / (2.0 * sigma * sigma)).exp();
            w[(i, j)] = sim;
            w[(j, i)] = sim;
        }
    }
    
    // 2. Compute the Normalized Laplacian matrix L_sym
    let mut d_inv_sqrt = DMatrix::zeros(n, n);
    for i in 0..n {
        let row_sum: f64 = w.row(i).sum();
        if row_sum > 0.0 {
            d_inv_sqrt[(i, i)] = 1.0 / (row_sum + 1e-10).sqrt();
        }
    }
    
    let l_sym = DMatrix::identity(n, n) - &d_inv_sqrt * &w * &d_inv_sqrt;
    
    // 3. Compute the first K eigenvectors of L_sym
    let eig = SymmetricEigen::new(l_sym);
    let vals = eig.eigenvalues;
    let vecs = eig.eigenvectors;
    
    let mut val_idx: Vec<(f64, usize)> = vals.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    // Sort ascending
    val_idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    let mut u = DMatrix::zeros(n, k);
    for j in 0..k {
        let idx = val_idx[j].1;
        for i in 0..n {
            u[(i, j)] = vecs[(i, idx)];
        }
    }
    
    // Normalize rows of U to unit length
    for i in 0..n {
        let mut row_mut = u.row_mut(i);
        let norm = row_mut.norm();
        if norm > 1e-10 {
            row_mut /= norm;
        }
    }
    
    // 4. Cluster the rows of U with K-means
    my_kmeans(&u, k)
}
