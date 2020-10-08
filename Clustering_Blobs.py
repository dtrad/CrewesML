# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 11:45:10 2018

@author: brussell
"""
# Example taken from Data Science Handbook
from sklearn.datasets import make_blobs
X,y=make_blobs(n_samples=150,n_features=2,centers=4,
               cluster_std=0.5,shuffle=True,random_state=0)
import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1],c=y,marker='o',s=50,cmap='rainbow')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Iniitial clusters with color labels')
plt.colorbar()
plt.grid()
plt.show()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, marker = 'o', s=50, cmap='rainbow')
k_centers = kmeans.cluster_centers_
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-means clusters with color labels and means')
plt.colorbar()
plt.scatter(k_centers[:, 0], k_centers[:, 1], c='black', s=100, alpha=1.0);
plt.grid()
plt.show()
# Import and run the Gaussian Mixture Model algorithm from scikit learn   
from sklearn.mixture import GaussianMixture
# covariance type = full for Mahalanobis, spherical for Euclidean
gmm = GaussianMixture(n_components=4, covariance_type='full').fit(X)
y_gmm = gmm.predict(X)
g_centers = gmm.means_
g_cov = gmm.covariances_
g_weights = gmm.weights_
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, marker = 'o', s=50, cmap='rainbow')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('GMM clusters with color labels and means')
plt.colorbar()
plt.scatter(g_centers[:, 0], g_centers[:, 1], c='black', s=100, alpha=1.0);
plt.grid()
plt.show()
print('GMM Covariances:')
print(g_cov)
print('GMM Weights:')
print(g_weights)