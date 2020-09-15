# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:00:36 2020

@author: USER
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

crime = pd.read_csv("C:/Users/USER/Downloads/crime_data.csv")
def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return(x)
    
crime_data = norm_func(crime.iloc[:,1:])    
from scipy.cluster.hierarchy import linkage  
import scipy.cluster.hierarchy as sch # for dendrogram
x = linkage(crime_data, method='complete',metric='euclidean')
plt.figure(figsize=(15,5));plt.title('hierarchical clustering dendrogram');plt.xlabel('index'),plt.ylabel("distance")


sch.dendrogram(x,
               leaf_rotation=0.,#rotates the x axis lable
               leaf_font_size=8.,#font size for the x axis lable
               
               
               )
# now using agglomerative clustering choosing six as cluster from dendrogram
from sklearn.cluster import AgglomerativeClustering
groups= AgglomerativeClustering(n_clusters=6, linkage="complete", affinity="euclidean").fit(crime_data)
groups.labels_
crime['clusters']=groups.labels_
final_crime = crime.iloc[:,[5,0,1,2,3,4]]
final_crime.iloc[:,2:].groupby(final_crime.clusters).mean() #getting aggregate mean of each clusters
#creating a csv file
final_crime.to_csv("crime_data.csv")
import os
os.getcwd()












