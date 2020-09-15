# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:23:15 2020

@author: USER
"""

import pandas as pd
import numpy as np

data = pd.read_csv("C:/Users/USER/Downloads/Company_Data.csv")
data.head(15)

# we will convert target variable Sales into categorical form
np.median(data["Sales"]) # 7.49
data["sales"]="<=7.49"
data.loc[data["Sales"]>=7.49,"sales"]=">=7.49"

data['sales'].unique()
data['sales'].value_counts()

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
data["ShelveLoc"]=lb.fit_transform(data["ShelveLoc"])
data["Urban"]=lb.fit_transform(data["Urban"])
data["US"]=lb.fit_transform(data["US"])

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size = 0.2)


colnames = list(data.columns)
type(data.columns)
predictors = colnames[1:11]
target = colnames[11]


from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors],train[target])

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target],preds,rownames=['Actual'],colnames=['Predictions'])

np.mean(preds==train[target]) # 100% accuracy


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target],preds,rownames=['Actual'],colnames=['Predictions'])

np.mean(preds==test[target]) # 76.25% accuracy





















