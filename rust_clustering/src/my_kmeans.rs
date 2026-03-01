use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom;
use rand::thread_rng;

pub fn my_kmeans(data: &DMatrix<f64>, k: usize) -> Vec<usize> {
    let n = data.nrows();
    let d = data.ncols();
    
    let mut rng = thread_rng();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    
    let mut centroids = DMatrix::zeros(k, d);
    for i in 0..k {
        for j in 0..d {
            centroids[(i, j)] = data[(indices[i], j)];
        }
    }
    
    let mut labels = vec![0; n];
    let max_iterations = 1000;
    
    for _iteration in 0..max_iterations {
        let mut new_labels = vec![0; n];
        for i in 0..n {
            let row_i = data.row(i);
            let mut min_dist = f64::MAX;
            let mut best_k = 0;
            
            for c in 0..k {
                let centroid_c = centroids.row(c);
                let dist = (&row_i - &centroid_c).norm_squared();
                if dist < min_dist {
                    min_dist = dist;
                    best_k = c;
                }
            }
            new_labels[i] = best_k;
        }
        
        if labels == new_labels {
            break;
        }
        labels = new_labels;
        
        let mut counts = vec![0.0; k];
        centroids.fill(0.0);
        
        for i in 0..n {
            let cluster = labels[i];
            counts[cluster] += 1.0;
            for j in 0..d {
                centroids[(cluster, j)] += data[(i, j)];
            }
        }
        
        for c in 0..k {
            if counts[c] > 0.0 {
                for j in 0..d {
                    centroids[(c, j)] /= counts[c];
                }
            } else {
                let idx = *indices.choose(&mut rng).unwrap();
                for j in 0..d {
                    centroids[(c, j)] = data[(idx, j)];
                }
            }
        }
    }
    
    labels
}
