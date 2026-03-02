import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from my_kmeans import my_kmeans
from scipy.optimize import linear_sum_assignment

def align_labels(true_labels, pred_labels):
    K = max(len(np.unique(true_labels)), len(np.unique(pred_labels)))
    cost_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost_matrix[i, j] = -np.sum((true_labels == i) & (pred_labels == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    aligned = np.zeros_like(pred_labels)
    for i, j in zip(row_ind, col_ind):
        aligned[pred_labels == j] = i
    return aligned

def acc(true, pred):
    return np.mean(align_labels(true, pred) == true)

def best_kmeans(data, K, labels, tries=30):
    best_lab, best_acc = None, -1
    for s in range(tries):
        np.random.seed(s)
        kl = my_kmeans(data, K)
        a = acc(labels, kl)
        if a > best_acc:
            best_acc = a
            best_lab = kl
    return align_labels(labels, best_lab), best_acc

def worst_kmeans(data, K, labels, tries=30):
    worst_lab, worst_acc = None, 2.0
    for s in range(tries):
        np.random.seed(s)
        kl = my_kmeans(data, K)
        a = acc(labels, kl)
        if a < worst_acc:
            worst_acc = a
            worst_lab = kl
    return align_labels(labels, worst_lab), worst_acc


def q1_spherical():
    np.random.seed(10)

    mu1 = np.array([0.0, 0.0])
    cov1 = 12.0 * np.eye(2)
    N1 = 400

    mu2 = np.array([6.0, 0.0])
    cov2 = 0.2 * np.eye(2)
    N2 = 60

    mu3 = np.array([-6.0, 0.0])
    cov3 = 0.2 * np.eye(2)
    N3 = 60

    data = np.vstack([
        np.random.multivariate_normal(mu1, cov1, N1),
        np.random.multivariate_normal(mu2, cov2, N2),
        np.random.multivariate_normal(mu3, cov3, N3),
    ])
    labels = np.concatenate([np.zeros(N1), np.ones(N2), np.full(N3, 2)]).astype(int)

    print("=== Problem 4.1: Spherical Gaussians ===")
    print(f"Gaussian 1: Mean = {mu1}, Cov = {cov1[0,0]}*I")
    print(f"Gaussian 2: Mean = {mu2}, Cov = {cov2[0,0]}*I")
    print(f"Gaussian 3: Mean = {mu3}, Cov = {cov3[0,0]}*I")

    km_labels, km_acc = worst_kmeans(data, 3, labels)

    gmm = GaussianMixture(n_components=3, covariance_type='spherical',
                          random_state=42, n_init=10)
    gmm_labels = align_labels(labels, gmm.fit_predict(data))
    gmm_acc = np.mean(gmm_labels == labels)

    print(f"K-Means acc: {km_acc:.1%} | Spherical GMM acc: {gmm_acc:.1%}\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(data[:,0], data[:,1], c=labels, s=10, cmap='tab10')
    axes[0].set_title('Ground Truth')
    axes[1].scatter(data[:,0], data[:,1], c=km_labels, s=10, cmap='tab10')
    axes[1].set_title(f'K-Means ({km_acc:.0%})')
    axes[2].scatter(data[:,0], data[:,1], c=gmm_labels, s=10, cmap='tab10')
    axes[2].set_title(f'Spherical GMM ({gmm_acc:.0%})')
    plt.suptitle("Q1: Spherical GMM works, K-Means fails")
    plt.tight_layout(); plt.savefig("q1_spherical.png", dpi=150); plt.close()
    print("Saved q1_spherical.png\n")


def q2_diagonal():
    np.random.seed(15)
    N = 200

    mu1 = np.array([0.0, 0.0])
    cov1 = np.diag([30.0, 0.5])

    mu2 = np.array([8.0, 0.0])
    cov2 = np.diag([0.5, 12.0])

    mu3 = np.array([-8.0, 0.0])
    cov3 = np.diag([0.5, 12.0])

    data = np.vstack([
        np.random.multivariate_normal(mu1, cov1, N),
        np.random.multivariate_normal(mu2, cov2, N),
        np.random.multivariate_normal(mu3, cov3, N),
    ])
    labels = np.concatenate([np.zeros(N), np.ones(N), np.full(N, 2)]).astype(int)

    print("=== Problem 4.2: Diagonal Gaussians ===")
    print(f"Gaussian 1: Mean = {mu1}, Cov = diag{list(np.diag(cov1))}")
    print(f"Gaussian 2: Mean = {mu2}, Cov = diag{list(np.diag(cov2))}")
    print(f"Gaussian 3: Mean = {mu3}, Cov = diag{list(np.diag(cov3))}")

    km_labels, km_acc = worst_kmeans(data, 3, labels)

    gmm_sph = GaussianMixture(n_components=3, covariance_type='spherical',
                               random_state=42, n_init=10)
    sph_labels = align_labels(labels, gmm_sph.fit_predict(data))
    sph_acc = np.mean(sph_labels == labels)

    gmm_diag = GaussianMixture(n_components=3, covariance_type='diag',
                                random_state=42, n_init=10)
    diag_labels = align_labels(labels, gmm_diag.fit_predict(data))
    diag_acc = np.mean(diag_labels == labels)

    print(f"K-Means acc: {km_acc:.1%} | Spherical GMM acc: {sph_acc:.1%} | Diagonal GMM acc: {diag_acc:.1%}\n")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].scatter(data[:,0], data[:,1], c=labels, s=10, cmap='tab10')
    axes[0].set_title('Ground Truth')
    axes[1].scatter(data[:,0], data[:,1], c=km_labels, s=10, cmap='tab10')
    axes[1].set_title(f'K-Means ({km_acc:.0%})')
    axes[2].scatter(data[:,0], data[:,1], c=sph_labels, s=10, cmap='tab10')
    axes[2].set_title(f'Spherical GMM ({sph_acc:.0%})')
    axes[3].scatter(data[:,0], data[:,1], c=diag_labels, s=10, cmap='tab10')
    axes[3].set_title(f'Diagonal GMM ({diag_acc:.0%})')
    plt.suptitle("Q2: Diagonal GMM works, Spherical GMM & K-Means fail")
    plt.tight_layout(); plt.savefig("q2_diagonal.png", dpi=150); plt.close()
    print("Saved q2_diagonal.png\n")


def q3_full():
    np.random.seed(20)
    N = 200

    mu1 = np.array([0.0, 0.0])
    cov1 = np.array([[10.0, 8.0], [8.0, 10.0]])

    mu2 = np.array([-6.0, 6.0])
    cov2 = np.array([[10.0, 8.0], [8.0, 10.0]])

    mu3 = np.array([6.0, -6.0])
    cov3 = np.array([[10.0, 8.0], [8.0, 10.0]])

    data = np.vstack([
        np.random.multivariate_normal(mu1, cov1, N),
        np.random.multivariate_normal(mu2, cov2, N),
        np.random.multivariate_normal(mu3, cov3, N),
    ])
    labels = np.concatenate([np.zeros(N), np.ones(N), np.full(N, 2)]).astype(int)

    print("=== Problem 4.3: Full (Unrestricted) Gaussians ===")
    print(f"Gaussian 1: Mean = {mu1}, Cov =\n{cov1}")
    print(f"Gaussian 2: Mean = {mu2}, Cov =\n{cov2}")
    print(f"Gaussian 3: Mean = {mu3}, Cov =\n{cov3}")

    km_labels, km_acc = worst_kmeans(data, 3, labels)

    gmm_diag = GaussianMixture(n_components=3, covariance_type='diag',
                                random_state=42, n_init=10)
    diag_labels = align_labels(labels, gmm_diag.fit_predict(data))
    diag_acc = np.mean(diag_labels == labels)

    gmm_full = GaussianMixture(n_components=3, covariance_type='full',
                                random_state=42, n_init=10)
    full_labels = align_labels(labels, gmm_full.fit_predict(data))
    full_acc = np.mean(full_labels == labels)

    print(f"K-Means acc: {km_acc:.1%} | Diagonal GMM acc: {diag_acc:.1%} | Full GMM acc: {full_acc:.1%}\n")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].scatter(data[:,0], data[:,1], c=labels, s=10, cmap='tab10')
    axes[0].set_title('Ground Truth')
    axes[1].scatter(data[:,0], data[:,1], c=km_labels, s=10, cmap='tab10')
    axes[1].set_title(f'K-Means ({km_acc:.0%})')
    axes[2].scatter(data[:,0], data[:,1], c=diag_labels, s=10, cmap='tab10')
    axes[2].set_title(f'Diagonal GMM ({diag_acc:.0%})')
    axes[3].scatter(data[:,0], data[:,1], c=full_labels, s=10, cmap='tab10')
    axes[3].set_title(f'Full GMM ({full_acc:.0%})')
    plt.suptitle("Q3: Full GMM works, Diagonal GMM & K-Means fail")
    plt.tight_layout(); plt.savefig("q3_full.png", dpi=150); plt.close()
    print("Saved q3_full.png\n")


if __name__ == "__main__":
    q1_spherical()
    q2_diagonal()
    q3_full()
