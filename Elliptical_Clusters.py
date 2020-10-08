# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 11:45:10 2018

@author: brussell
"""
# Example using three clusters
import numpy as np
import matplotlib.pyplot as plt
X = np.loadtxt("Three_clusters.txt")
y_clusters = np.loadtxt("Three_Clusters_Labels.txt")
plt.scatter(X[:,0],X[:,1],c=y_clusters,marker='o',s=50, cmap='rainbow')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Correct ordering of elliptical clusters')
plt.colorbar()
plt.grid()
plt.show()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='rainbow')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-means clustering of elliptical clusters')
plt.colorbar()
k_centers = kmeans.cluster_centers_
plt.scatter(k_centers[:, 0], k_centers[:, 1], c='black', s=100, alpha=1.0)
plt.grid()
plt.show()
# Import and run the Gaussian Mixture Model algorithm from scikit learn   
from sklearn.mixture import GaussianMixture
# covariance type = full for Mahalanobis, spherical for Euclidean
gmm = GaussianMixture(n_components=3, covariance_type='full').fit(X)
y_gmm = gmm.predict(X)
g_centers = gmm.means_
g_cov = gmm.covariances_
g_weights = gmm.weights_
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, s=50, cmap='rainbow')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('GMM clustering of elliptical clusters')
plt.colorbar()
plt.scatter(g_centers[:, 0], g_centers[:, 1], c='black', s=100, alpha=1.0);
plt.grid()
plt.show()
print('GMM Covariances:')
print(g_cov)
print('GMM Weights:')
print(g_weights)
np.savetxt('Elliptical_K_means_labels.txt',y_kmeans, fmt ='%i')
np.savetxt('Elliptical_GMM_labels.txt',y_gmm, fmt ='%i') 