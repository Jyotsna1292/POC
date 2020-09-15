# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 20:17:20 2020

@author: USER
"""

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report

# import dataset
bank = pd.read_csv('C:/Users/USER/Downloads/bank_data.csv')

bank.isnull().sum() # checking for null values
bank.shape

# model building
x = bank.iloc[:,:31]
y = bank.iloc[:,31]
classifier = LogisticRegression()
classifier.fit(x,y)
classifier.coef_ # coefficients of features
classifier.predict_proba(x) # probability values

y_pred = classifier.predict(x)
bank["y_pred"] = y_pred
y_prob = pd.DataFrame(classifier.predict_proba(x.iloc[:,:]))

new_df = pd.concat([bank,y_prob], axis=1)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y,y_pred)
print(confusion_matrix)

accuracy = sum(y==y_pred)/bank.shape[0]
accuracy # 90%
pd.crosstab(y_pred,y)














