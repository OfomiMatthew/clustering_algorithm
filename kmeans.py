# -*- coding: utf-8 -*-
"""
Created on Sat May  9 18:00:59 2020

@author: Ofomi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('utilities.csv')
X = data.iloc[:,[3,6]].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++",n_init=10,max_iter=300,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Determining n_clusters')
plt.xlabel('cluster_number')
plt.ylabel("wcss")
plt.show()

#Fitting the KMeans algorithm to dataset
kmeans = KMeans(n_clusters=3,init='k-means++',n_init=10,max_iter=300,random_state=0)
Y_kmeans=kmeans.fit_predict(X)

#Visualising result
plt.scatter(X[Y_kmeans==0,0],X[Y_kmeans==0,1],s=100, c='red',label='c1')

plt.scatter(X[Y_kmeans==1,0],X[Y_kmeans==1,1],s=100, c='blue',label='c2')
plt.scatter(X[Y_kmeans==2,0],X[Y_kmeans==2,1],s=100, c='green',label='c3')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='black',label='centroid')

plt.title('cluster group of customers(cost vs sales)')
plt.xlabel('cost')
plt.ylabel('sales')
plt.legend(loc='best')
plt.show()



















