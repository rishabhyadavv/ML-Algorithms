#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 17:03:02 2018

@author: risyadav
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans

x = [1, 1.5, 3.0, 5.0, 3.5, 4.5,3.5]
y = [1.0, 2.0, 4.0, 7.0, 5.0, 5.0, 4.5]

plt.scatter(x,y)
plt.show()

X = np.array([[1, 1.0],
              [1.5, 2.0],
              [3.0, 4.0],
              [5.0, 7.0],
              [3.5, 5.0],
              [4.5, 5.0],[3.5, 4.5]])
		

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)


colors = ["g.","r.","c.","y."]

for i in range(len(X)):
    print("coordinate:",X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)


plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)

plt.show()
		