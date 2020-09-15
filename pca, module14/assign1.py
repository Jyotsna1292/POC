# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 13:38:13 2020

@author: USER
"""

# importing libraries
import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# importing dataset
wine = pd.read_csv("C:/Users/USER/Downloads/wine.csv")

# ignoring the Type column of dataset which categories the entire data into three classes
Wine = wine.iloc[:,1:]

# Normalizing the numerical data 
wine_normal = scale(Wine)
wine_normal

pca = PCA(n_components = 3)
pca_values = pca.fit_transform(wine_normal)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

pca.components_
pca.components_[0]
# Cumulative variance 

var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1

# our first three principal components contain 66.53% of information

# Variance plot for PCA components obtained 
plt.plot(var1,color="red")

# plot between PCA1 and PCA2 
x = pca_values[:,0]
y = pca_values[:,1]
pca_values
# z = pca_values[:2:3]
plt.scatter(x,y)

# kmeans clustering without PCA

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)
df_norm = norm_func(Wine.iloc[:,:])
df_norm.head(5)

k = list(range(2,11))
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

# getting elbow at 3 in scree plot , so will make three clusters of our dataset
model= KMeans(n_clusters=3).fit(df_norm)
model
model.cluster_centers_[1].reshape(1,df_norm.shape[1])
model.labels_
cluster= model.labels_


Wine['cluster']= cluster

count = Wine['cluster'].value_counts() 
print(count) 

# kmeans clustering using PCA
# we have already done the PCA of wine data above, now we have to do clustering using using only first three principal components

pca_values
type(pca_values)

# converting it into dataframe
Wine_pca = pd.DataFrame(pca_values)

k = list(range(2,11))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i).fit(Wine_pca)

    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(Wine_pca.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,Wine_pca.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

TWSS
#scree plot
plt.plot(k,TWSS,'ro-');plt.xlabel("no. of clusters");plt.ylabel("TWSS");plt.xticks(k)

# with pca also we are getting elbow at 3 in our scree plot, so will make model using no. of cluster=3
# getting elbow at 3 in scree plot , so will make three clusters of our dataset
model1= KMeans(n_clusters=3).fit(Wine_pca)
model1

model1.labels_
cluster= model1.labels_

from collections import Counter
count_1 = Counter(cluster)

# with pca also i am getting the same result, same number of clusters in kmeans



























