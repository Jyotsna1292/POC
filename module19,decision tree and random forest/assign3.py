# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:28:23 2020

@author: USER
"""

import pandas as pd
import numpy as np

Fraud = pd.read_csv("C:/Users/USER/Downloads/Fraud_check.csv")

# bringing target variable in front
cols = list(Fraud)
cols.insert(0, cols.pop(cols.index('Taxable.Income')))
cols
Fraud = Fraud.ix[:,cols]

# converting taxable.income into catgorical variable
x=pd.cut(Fraud["Taxable.Income"],bins=[0,30000,100000],labels=['Risky','Good'])
fraud=pd.concat([Fraud,x], axis=1)

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
fraud["Undergrad"]=lb.fit_transform(fraud["Undergrad"])
fraud["Urban"]=lb.fit_transform(fraud["Urban"])
fraud["Marital.Status"]=lb.fit_transform(fraud["Marital.Status"])

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test = train_test_split(fraud,test_size = 0.2)

X_train = np.array(train.ix[:,1:6]) # Input
X_test = np.array(test.ix[:,1:6]) # Input
Y_train = np.array(train.ix[:,6]) # Output
Y_test = np.array(test.ix[:,6]) # Output

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(X_train,Y_train)

# Prediction on Train Data
preds = model.predict(X_train)
pd.crosstab(X_train,preds,rownames=['Actual'],colnames=['Predictions'])

np.mean(preds==Y_train) # 100% accuracy


# Prediction on Test Data
preds = model.predict(X_test)
pd.crosstab(Y_test,preds,rownames=['Actual'],colnames=['Predictions'])

np.mean(preds==Y_test) # 61.6% accuracy


















