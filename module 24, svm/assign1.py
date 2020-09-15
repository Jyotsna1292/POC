# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 22:54:33 2020

@author: USER
"""

import pandas as pd 
import numpy as np 
import seaborn as sns

forestfire = pd.read_csv("C:/Users/USER/Downloads/forestfires.csv")

# removing columns which is not required
forest = forestfire.iloc[:,2:]

from sklearn import preprocessing
le= preprocessing.LabelEncoder()
forest['size_category']= le.fit_transform(forest['size_category'])

forest_in = forest.iloc[:,:28]
forest_out = forest.iloc[:,28]

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(forest_in,forest_out)

from sklearn.svm import SVC

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'


# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_x,train_y)
pred_test_linear = model_linear.predict(test_x)

np.mean(pred_test_linear==test_y) # Accuracy = 99.23%

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_x,train_y)
pred_test_poly = model_poly.predict(test_x)

np.mean(pred_test_poly==test_y)# accuracy = 99.23%

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_x,train_y)
pred_test_rbf = model_rbf.predict(test_x)
 
np.mean(pred_test_rbf==test_y) # 74.61%





























