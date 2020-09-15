# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:05:23 2020

@author: USER
"""

import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np

crime = pd.read_csv("C:/Users/USER/Downloads/crime_data.csv")

def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)
df_norm = norm_func(crime.iloc[:,1:])
df_norm.head(5)
?KMeans
k= list(range(1,12))
k
#scree plot
TWSS = []
for i in k :
    km = KMeans(n_clusters= i).fit(df_norm)
    for j in range(i):
        WSS=[]
        WSS.append(sum(cdist(df_norm.iloc[km.labels_==j,:],km.cluster_centers_[j].reshape(1:df_norm.shape[1],"euclidean")))
    TWSS.append(sum(WSS))
 