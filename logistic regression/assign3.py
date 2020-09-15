# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 23:23:12 2020

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
election = pd.read_csv('C:/Users/USER/Downloads/election_data.csv')

# removing the first row containing null values
election.drop(election.index[0], inplace=True)

# moving result column to the front of the dataframe
cols = list(election)
# move the column to head of list using index, pop and insert
cols.insert(0, cols.pop(cols.index('Result')))
cols
# use ix to reorder
Election = election.ix[:, cols]

# model building
x = Election.iloc[:,1:]
y = Election.iloc[:,0]
classifier = LogisticRegression()
classifier.fit(x,y)
classifier.coef_ # coefficients of features
classifier.predict_proba(x) # probability values

y_pred = classifier.predict(x)
Election["y_pred"] = y_pred
y_prob = pd.DataFrame(classifier.predict_proba(x.iloc[:,:]))

new_df = pd.concat([Election,y_prob], axis=1)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y,y_pred)
print(confusion_matrix)

accuracy = sum(y==y_pred)/Election.shape[0]
accuracy # 100%
pd.crosstab(y_pred,y)























