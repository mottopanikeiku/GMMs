mod my_kmeans;
mod my_spectralclustering;

use nalgebra::DMatrix;

fn main() {
    println!("--- Clustering Implementations in Rust ---");
    println!("Since loading `.mat` files directly in Rust often requires specialized crates or HDF5 bindings, we demonstrate the algorithms with a toy dataset.");
    println!("To run tests on `.mat` files, you can use the Python version: `Run_clustering.py`.");
    
    // 1. Create a toy dataset
    // Four points: two in corner (0,0),(0.1,0.1) and two in corner (1,1),(1.1,1.1)
    let n = 4;
    let d = 2;
    let data = DMatrix::from_row_slice(n, d, &[
        0.0, 0.0,
        0.1, 0.1,
        1.0, 1.0,
        1.1, 1.1,
    ]);
    
    let k = 2;
    let sigma = 1.0;
    
    println!("\nDataset:\n{}", data);
    
    // 2. Run K-Means
    let labels_kmeans = my_kmeans::my_kmeans(&data, k);
    println!("K-Means Labels: {:?}", labels_kmeans);
    
    // 3. Run Spectral Clustering
    let labels_spectral = my_spectralclustering::my_spectralclustering(&data, k, sigma);
    println!("Spectral Clustering Labels: {:?}", labels_spectral);
}
