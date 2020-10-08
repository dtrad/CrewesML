# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:24:11 2020

@author: brussell
"""
import numpy as np
import matplotlib.pyplot as plt
# REad the data, consisting of a 9639 x 5 dimenstional array,
# where each column consists of a single 119 x 81 dimensional structure slice
# from the Blackfoot seismic volume
M = np.loadtxt("Blackfoot_data.txt")
# Extract and reshape each map slice, then plot the slice
# First slice
M1 = M[:,0]; M1 = M1.reshape(119,81)
plt.pcolor(np.transpose(M1),cmap = 'rainbow')
plt.xlabel('Inline Number')
plt.ylabel('Crossline Number')
plt.title('Blackfoot Seismic 10 ms below Lower Mannville')
plt.colorbar()
plt.show()
# SEcond slice
M2 = M[:,1]; M2 = M2.reshape(119,81)
plt.pcolor(np.transpose(M2),cmap = 'rainbow')
plt.xlabel('Inline Number')
plt.ylabel('Crossline Number')
plt.title('Blackfoot Seismic 12 ms below Lower Mannville')
plt.colorbar()
plt.show()
# Third slice
M3 = M[:,2]; M3 = M3.reshape(119,81)
plt.pcolor(np.transpose(M3),cmap = 'rainbow')
plt.xlabel('Inline Number')
plt.ylabel('Crossline Number')
plt.title('Blackfoot Seismic 14 ms below Lower Mannville')
plt.colorbar()
plt.show()
# Fourth slice
M4 = M[:,3]; M4 = M4.reshape(119,81)
plt.pcolor(np.transpose(M4),cmap = 'rainbow')
plt.xlabel('Inline Number')
plt.ylabel('Crossline Number')
plt.title('Blackfoot Seismic 16 ms below Lower Mannville')
plt.colorbar()
plt.show()
# Fifth slice
M5 = M[:,4]; M5 = M5.reshape(119,81)
plt.pcolor(np.transpose(M5),cmap = 'rainbow')
plt.xlabel('Inline Number')
plt.ylabel('Crossline Number')
plt.title('Blackfoot Seismic 18 ms below Lower Mannville')
plt.colorbar()
plt.show()
# Import and run the Gaussian Mixture Model algorithm from scikit learn   
from sklearn.mixture import GaussianMixture
# covariance type = full for Mahalanobis, spherical for Euclidean
gmm = GaussianMixture(n_components=10, covariance_type='full').fit(M)
labels_gmm = gmm.predict(M)
g_centers = gmm.means_
g_cov = gmm.covariances_
g_weights = gmm.weights_
c  = labels_gmm.reshape(119,81)
plt.pcolor(np.transpose(c),cmap = 'rainbow')
plt.xlabel('Inline Number')
plt.ylabel('Crossline Number')
plt.title('Gaussian Mixture Modeling with 10 clusters')
plt.colorbar()
plt.show()
# Plot the GMM result
#hrsplt.plot_slice(c, volinfo, "Gaussian Mixture Model with 10 clusters")
# Import and run the K-means algorithm from scikit learn 
from sklearn.cluster import KMeans
kmeans = KMeans(10, random_state=0)
labels_km = kmeans.fit(M).predict(M)
d  = labels_km.reshape(119,81)
plt.pcolor(np.transpose(d),cmap = 'rainbow')
plt.xlabel('Inline Number')
plt.ylabel('Crossline Number')
plt.title('K-means clustering with 10 clusters')
plt.colorbar()
plt.show()
#print('GMM Covariances:')
#print(g_cov)
print('GMM Weights:')
print(g_weights)
np.savetxt('Blackfoot_K_means_labels.txt',labels_km, fmt ='%i')
np.savetxt('Blackfoot_GMM_labels.txt',labels_gmm, fmt ='%i') 