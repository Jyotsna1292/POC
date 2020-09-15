# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:29:30 2020

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

computer = pd.read_csv("C:/Users/USER/Downloads/Computer_Data.csv")

# removing first column
computer = computer.iloc[:,1:]

# converting categorical columns into numerical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

dummy = computer[['cd', 'multi','premium']].apply(le.fit_transform)
x=computer.drop(['cd', 'multi','premium'], axis = 1) 

Computer = pd.concat([x, dummy], axis=1)

#correlation matrix
Computer.corr()

import seaborn as sns
sns.pairplot(Computer)

# splitting data into train and test
x = Computer.iloc[:,1:].values
y = Computer.iloc[:,0].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
t = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)

train_pred = regressor.predict(x_train)
test_pred = regressor.predict(x_test)

train_resid = train_pred - y_train

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid)) #275.228
train_rmse

test_resid = test_pred - y_test
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
test_rmse # 274.87

















