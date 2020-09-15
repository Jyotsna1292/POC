# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:48:56 2020

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
Wine = pd.read_csv("C:/Users/USER/Downloads/wine.csv")
wine= Wine.iloc[:,1:]

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

wine_norm=scale(wine) # normalizing the data
pca=PCA()

pca_values=pca.fit_transform(wine_norm)
pca_values.shape

var=pca.explained_variance_ratio_
var
var1=np.cumsum(np.round(var, decimals=4)*100)
var1

#variance plot for PCA component obtained
plt.plot(var1, color="red")
df_new=pd.DataFrame(pca_values[:,0:3])

# kmeans clustering
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i).fit(df_new)

    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_new.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_new.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

TWSS
# scree plot
plt.plot(k,TWSS,'ro-');plt.xlabel("no. of clusters");plt.ylabel("TWSS");plt.xticks(k)

# getting little elbow at 6 and 9, so will take k=9

model=KMeans(n_clusters=9).fit(df_new)
model.labels_
clusters=model.labels_
df_new["Cluster"]=clusters























