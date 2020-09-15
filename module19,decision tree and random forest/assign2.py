# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:35:48 2020

@author: USER
"""

import pandas as pd
import numpy as np

diabetes = pd.read_csv("C:/Users/USER/Downloads/Diabetes.csv")

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

diabetes_norm = norm_func(diabetes.iloc[:,0:8])

diabetes_norm["class"]=diabetes.iloc[:,8]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test = train_test_split(diabetes_norm,test_size = 0.2)

X_train = np.array(train.ix[:,:8]) # Input
X_test = np.array(test.ix[:,:8]) # Input
Y_train = np.array(train['class']) # Output
Y_test = np.array(test['class']) # Output

from sklearn.ensemble import RandomForestClassifier

rfmodel = RandomForestClassifier(n_estimators=15)

rfmodel.fit(X_train,Y_train)

# Train Data Accuracy
train["rf_pred"] = rfmodel.predict(X_train)
train_acc = np.mean(train["class"]==train["rf_pred"])
train_acc # 100%

# Test Data Accuracy
test["rf_pred"] = rfmodel.predict(X_test)
test_acc = np.mean(test["class"]==test["rf_pred"])
test_acc  # 72%

























