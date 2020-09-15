# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 08:36:50 2020

@author: USER
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

wbcd = pd.read_csv("C:/Users/USER/Downloads/wbcd.csv") #importing the dataset 
wbcd.head() #seeting the head of df

data = wbcd.iloc[:,1:]


df_x = data.iloc[:,1:] #dividing the i/p and o/p variable
df_y = data.iloc[:,0]

x_train,x_test,y_train,y_test= train_test_split(df_x,df_y,test_size=0.2, random_state= 4) #dividing the data randomly
y_test.head()

#decision tree
dt = DecisionTreeClassifier() #storing the classifer in dt

dt.fit(x_train,y_train) #fitting te model 

dt.score(x_test,y_test) #checking the score like accuracy, 92.98%

dt.score(x_train,y_train) # 100% accuracy

#Random Forest clssifer: it is a ensemble of Decision tree 
rf = RandomForestClassifier(n_estimators=10) # n_estimator number of tree in the forest 
rf.fit(x_train,y_train) #fitting the random forest model 

rf.score(x_test,y_test) #doing the accuracy of the test model , 95.61% accuracy

rf.score(x_train,y_train) #doing the accuracy of the train model, 100%, slightly diff is reduced between train and test accuracy


#Ada boosting 
ada = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=10,learning_rate=1)
ada.fit(x_train,y_train)

ada.score(x_test,y_test) # 91.228% accuracy

ada.score(x_train,y_train) # 100% accuracy

# extreme gradient boosting

from xgboost import XGBClassifier
model = XGBClassifier()

model.fit(x_train,y_train)

model.score(x_test,y_test) # 96.49%

model.score(x_train,y_train) # 100%

