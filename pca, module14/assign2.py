# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 18:35:24 2020

@author: USER
"""

# importing libraries

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.cluster.hierarchy import linkage  
import scipy.cluster.hierarchy as sch # for dendrogram

# importing dataset
wine = pd.read_csv("C:/Users/USER/Downloads/wine.csv")

# removing the Type column
Wine = wine.iloc[:,1:]

# now doing hierarchical clustering without pca

# normalizing dataset
def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return(x)
    
wine_data = norm_func(Wine.iloc[:,:])    

x = linkage(wine_data, method='complete',metric='euclidean')
plt.figure(figsize=(15,5));plt.title('hierarchical clustering dendrogram');plt.xlabel('index'),plt.ylabel("distance")


sch.dendrogram(x,
               leaf_rotation=0.,#rotates the x axis lable
               leaf_font_size=8.,#font size for the x axis lable
               
               
               )
# now using agglomerative clustering choosing three as cluster from dendrogram

from sklearn.cluster import AgglomerativeClustering

groups= AgglomerativeClustering(n_clusters=3, linkage="complete", affinity="euclidean").fit(wine_data)
groups.labels_
Wine['clusters']=groups.labels_
final_wine = Wine.iloc[:,[13,0,1,2,3,4,5,6,7,8,9,10,11,12]]
final_wine.iloc[:,1:].groupby(final_wine.clusters).mean() #getting aggregate mean of each clusters

count = final_wine["clusters"].value_counts()

# now doing hierarchical clustering with PCA

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 

# Normalizing the numerical data 
Wine = Wine.drop(["clusters"], axis=1)
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

# performing hierarchical clustering 

wine_pca = pd.DataFrame(pca_values)

from scipy.cluster.hierarchy import linkage  
import scipy.cluster.hierarchy as sch # for dendrogram

x = linkage(wine_pca, method='complete',metric='euclidean')
plt.figure(figsize=(15,5));plt.title('hierarchical clustering dendrogram');plt.xlabel('index'),plt.ylabel("distance")


sch.dendrogram(x,
               leaf_rotation=0.,#rotates the x axis lable
               leaf_font_size=8.,#font size for the x axis lable
               
               
               )

# now seeing the dendrogram we will choose clusters as 4
# now using agglomerative clustering choosing 4 as cluster from dendrogram
from sklearn.cluster import AgglomerativeClustering
groups= AgglomerativeClustering(n_clusters=4, linkage="complete", affinity="euclidean").fit(wine_pca)
cluster=groups.labels_

from collections import Counter
count_1 = Counter(cluster)

# getting different dendrograms for both cases
# with pca we are choosing 4 clusters  whereas without pca choosing only 3 clusters by seeing the dendrogram


















