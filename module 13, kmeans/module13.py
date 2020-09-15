# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:28:34 2020

@author: USER
"""

import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np

insurance = pd.read_csv("C:/Users/USER/Downloads/Insurance Dataset.csv")


def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)
df_norm = norm_func(insurance.iloc[:,:])
df_norm.head(5)

k = list(range(2,12))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i).fit(df_norm)

    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

TWSS
#scree plot
plt.plot(k,TWSS,'ro-');plt.xlabel("no. of clusters");plt.ylabel("TWSS");plt.xticks(k)

model= KMeans(n_clusters=6).fit(df_norm)# getting elbow at 4 and 6
model
model.cluster_centers_[1].reshape(1,df_norm.shape[1])
model.labels_
cluster= model.labels_
final_df = df_norm.insert(2,cluster)
df_norm['cluster']= cluster
























