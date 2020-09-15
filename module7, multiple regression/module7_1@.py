# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:37:19 2020

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing dataset
startup = pd.read_csv("C:/Users/USER/Downloads/50_Startups.csv")

x = startup.iloc[:, :-1].values
y = startup.iloc[:, -1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
x

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

train_pred = regressor.predict(x_train)
test_pred = regressor.predict(x_test)

train_resid = train_pred - y_train

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid)) #9031.666
train_rmse

test_resid = test_pred - y_test
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
test_rmse # 9137.99


















