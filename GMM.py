# import imageio
# import matplotlib.animation as ani
# import matplotlib.cm as cmx
# import matplotlib.colors as colors
# import matplotlib.pyplot as plt
import numpy as np

# from matplotlib.patches import Ellipse
# from PIL import Image
# from sklearn import datasets
from sklearn.cluster import KMeans

def normal_dist_3D(mean):
    mu = np.mean(mean)
    cov_matrix = np.zeros((len(mean), len(mean)))
    cov_matrix[0][0] = np.sum((mean[:,0] - mu)**2) / len(mean)
    cov_matrix[1][1] = np.sum((mean[:,1] - mu)**2) / len(mean)
    cov_matrix[2][2] = np.sum((mean[:,2] - mu)**2) / len(mean)
    
    cov_matrix[0][1] = np.sum((mean[:,0] - mu)*(mean[:,1] - mu)) / len(mean)
    cov_matrix[0][2] = np.sum((mean[:,0] - mu)*(mean[:,2] - mu)) / len(mean)
    cov_matrix[1][0] = np.sum((mean[:,1] - mu)*(mean[:,0] - mu)) / len(mean)
    cov_matrix[1][2] = np.sum((mean[:,1] - mu)*(mean[:,2] - mu)) / len(mean)
    cov_matrix[2][0] = np.sum((mean[:,2] - mu)*(mean[:,0] - mu)) / len(mean)
    cov_matrix[2][1] = np.sum((mean[:,2] - mu)*(mean[:,1] - mu)) / len(mean)
    
    return mu, cov_matrix
    
def gaussian(X, mu, cov):
    n = X.shape[1]
    diff = (X - mu).T
    return np.diagonal(1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5) * np.exp(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))).reshape(-1, 1)

def initialize_clusters(X, n_clusters):
    clusters = []
    idx = np.arange(X.shape[0])
    
    # We use the KMeans centroids to initialise the GMM
    
    kmeans = KMeans().fit(X)
    mu_k = kmeans.cluster_centers_
    
    for i in range(n_clusters):
        clusters.append({
            'pi_k': 1.0 / n_clusters,
            'mu_k': mu_k[i],
            'cov_k': np.identity(X.shape[1], dtype=np.float64)
        })
    
    return clusters


def expectation_step(X, clusters):
    totals = np.zeros((X.shape[0], 1), dtype=np.float64)
    
    for cluster in clusters:
        pi_k = cluster['pi_k']
        mu_k = cluster['mu_k']
        cov_k = cluster['cov_k']
        
        gamma_nk = (pi_k * gaussian(X, mu_k, cov_k)).astype(np.float64)
        
        for i in range(X.shape[0]):
            totals[i] += gamma_nk[i]
        
        cluster['gamma_nk'] = gamma_nk
        cluster['totals'] = totals
    
    for cluster in clusters:
        cluster['gamma_nk'] /= cluster['totals']


def maximization_step(X, clusters):
    N = float(X.shape[0])
    
    for cluster in clusters:
        gamma_nk = cluster['gamma_nk']
        cov_k = np.zeros((X.shape[1], X.shape[1]))
        
        N_k = np.sum(gamma_nk, axis=0)
        
        pi_k = N_k / N
        mu_k = np.sum(gamma_nk * X, axis=0) / N_k
        
        for j in range(X.shape[0]):
            diff = (X[j] - mu_k).reshape(-1, 1)
            cov_k += gamma_nk[j] * np.dot(diff, diff.T)
        
        cov_k /= N_k
        
        cluster['pi_k'] = pi_k
        cluster['mu_k'] = mu_k
        cluster['cov_k'] = cov_k
        


def get_likelihood(X, clusters):
    likelihood = []
    sample_likelihoods = np.log(np.array([cluster['totals'] for cluster in clusters]))
    return np.sum(sample_likelihoods), sample_likelihoods


def train_gmm(X, n_clusters, n_epochs):
    clusters = initialize_clusters(X, n_clusters)
    likelihoods = np.zeros((n_epochs,))
    scores = np.zeros((X.shape[0], n_clusters))
    history = []
    
    for i in range(n_epochs):
        clusters_snapshot = []
        
        # This is just for our later use in the graphs
        for cluster in clusters:
            clusters_snapshot.append({
                'mu_k': cluster['mu_k'].copy(),
                'cov_k': cluster['cov_k'].copy()
            })
        
        history.append(clusters_snapshot)
        
        expectation_step(X, clusters)
        maximization_step(X, clusters)
        
        likelihood, sample_likelihoods = get_likelihood(X, clusters)
        likelihoods[i] = likelihood
        
        print('Epoch: ', i + 1, 'Likelihood: ', likelihood)
    
    for i, cluster in enumerate(clusters):
        scores[:, i] = np.log(cluster['gamma_nk']).reshape(-1)
    
    return clusters, likelihoods, scores, sample_likelihoods, history

