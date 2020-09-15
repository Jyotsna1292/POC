# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:56:01 2020

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
affair = pd.read_csv('C:/Users/USER/Downloads/Affairs.csv')

affair.isnull().sum() # checking for null values
affair.shape

# removing first column
affair = affair.iloc[:,1:]

# converting all the values in nffairs column which are 1 or more than 1 into 1
affair.loc[affair.naffairs>0,'aff_tf']=1
affair.loc[affair.naffairs==0,'aff_tf']=0

Affair = affair.iloc[:,1:]

# model building
x = Affair.iloc[:,:17]
y = Affair.iloc[:,17]

classifier = LogisticRegression()
classifier.fit(x,y)
classifier.coef_ # coefficients of features
classifier.predict_proba(x) # probability values

y_pred = classifier.predict(x)
Affair["y_pred"] = y_pred
y_prob = pd.DataFrame(classifier.predict_proba(x.iloc[:,:]))

new_df = pd.concat([Affair,y_prob], axis=1)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Affair.aff_tf,y_pred)
print(confusion_matrix)

accuracy = sum(Affair.aff_tf==y_pred)/new_df.shape[0]
accuracy # 76.53%
pd.crosstab(y_pred,Affair.aff_tf)
























