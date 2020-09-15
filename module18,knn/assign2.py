# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:52:31 2020

@author: USER
"""

# Importing Libraries 
import pandas as pd
import numpy as np

zoo = pd.read_csv("C:/Users/USER/Downloads/Zoo.csv")

# removing first column

zoo_1= zoo.iloc[:, 1:]

# Training and Test data using 
from sklearn.model_selection import train_test_split
train,test = train_test_split(zoo_1,test_size = 0.2) # 0.2 => 20 percent of entire data 

# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

# for 3 nearest neighbours 
neigh = KNC(n_neighbors= 3)

# Fitting with training data 
neigh.fit(train.iloc[:,0:16],train.iloc[:,16])

# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,0:16])==train.iloc[:,16]) # 100%

# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,0:16])==test.iloc[:,16]) # 95%   

# for 5 nearest neighbours
neigh = KNC(n_neighbors=5)

# fitting with training data
neigh.fit(train.iloc[:,0:16],train.iloc[:,16])

# train accuracy 
train_acc1 = np.mean(neigh.predict(train.iloc[:,0:16])==train.iloc[:,16]) # 95%

# test accuracy
test_acc1 = np.mean(neigh.predict(test.iloc[:,0:16])==test.iloc[:,16]) # 95.23%

# creating empty list variable 
acc=[]

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values 
 
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:16],train.iloc[:,16])
    train_acc = np.mean(neigh.predict(train.iloc[:,0:16])==train.iloc[:,16])
    test_acc = np.mean(neigh.predict(test.iloc[:,0:16])==test.iloc[:,16])
    acc.append([train_acc,test_acc])





import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")


plt.legend(["train","test"])

# getting max accuracy for k = 3,5,7




















