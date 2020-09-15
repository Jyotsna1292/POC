# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 19:29:28 2020

@author: USER
"""
import pandas as pd 
import numpy as np 
import seaborn as sns

salary_train = pd.read_csv("C:/Users/USER/Downloads/SalaryData_Train.csv")
salary_test = pd.read_csv("C:/Users/USER/Downloads/SalaryData_Test.csv")
salary_train.head()
salary_train.describe()
salary_train.columns


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
Salary_train = salary_train.apply(le.fit_transform)
Salary_test = salary_test.apply(le.fit_transform)

from sklearn.svm import SVC

salary_train_x = Salary_train.iloc[:,:13]
salary_train_y = Salary_train.iloc[:,13]
salary_test_x = Salary_test.iloc[:,:13]
salary_test_y = Salary_test.iloc[:,13]

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'


# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(salary_train_x,salary_train_y)
pred_test_linear = model_linear.predict(salary_test_x)

np.mean(pred_test_linear==salary_test_y) # Accuracy = 80.41%

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(salary_train_x,salary_train_y)
pred_test_poly = model_poly.predict(salary_test_x)

np.mean(pred_test_poly==salary_test_y) 

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(salary_train_x,salary_train_y)
pred_test_rbf = model_rbf.predict(salary_test_x)

np.mean(pred_test_rbf==salary_test_y) 


















