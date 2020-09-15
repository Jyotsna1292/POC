# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 22:17:44 2020

@author: USER
"""

import pandas as pd
import numpy as np

company = pd.read_csv("C:/Users/USER/Downloads/Company_Data.csv")
company.head(15)

# we will convert target variable Sales into categorical form

sales_cat = pd.cut(company.Sales,bins=[0,5,10,15,20],labels=['low','medium','high','very high'])
company["sales"]= sales_cat

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
company["ShelveLoc"]=lb.fit_transform(company["ShelveLoc"])
company["Urban"]=lb.fit_transform(company["Urban"])
company["US"]=lb.fit_transform(company["US"])

company['sales'].unique()
company['sales'].value_counts()
colnames = list(company.columns)
type(company.columns)
predictors = colnames[1:11]
target = colnames[11]
target

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test = train_test_split(company,test_size = 0.2)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors],train[target])

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target],preds,rownames=['Actual'],colnames=['Predictions'])

np.mean(preds==train[target]) # 100%


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target],preds,rownames=['Actual'],colnames=['Predictions'])

np.mean(preds==test[target]) # 60%













