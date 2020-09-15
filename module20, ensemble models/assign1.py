# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 22:46:12 2020

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

data = pd.read_csv("C:/Users/USER/Downloads/Diabetes_RF.csv") #importing the dataset 
data.head() #seeting the head of the data

df_x = data.iloc[:,:8] #dividing the i/p and o/p variable
df_y = data.iloc[:,8]

x_train,x_test,y_train,y_test= train_test_split(df_x,df_y,test_size=0.2, random_state= 4) #dividing the data randomly
y_test.head()

#decision tree
dt = DecisionTreeClassifier() #storing the classifer in dt

dt.fit(x_train,y_train) #fitting te model 

dt.score(x_test,y_test) #checking the score like accuracy, 68.83%

dt.score(x_train,y_train) # 100% accuracy, difference is quite large between train and test accuracy, so there is problem of either overfitting or underfitting

#Random Forest clssifer: it is a ensemble of Decision tree 
rf = RandomForestClassifier(n_estimators=10) # n_estimator number of tree in the forest 
rf.fit(x_train,y_train) #fitting the random forest model 

rf.score(x_test,y_test) #doing the accuracy of the test model , 75.32% accuracy

rf.score(x_train,y_train) #doing the accuracy of the train model, 98.20%, slightly diff is reduced between train and test accuracy

#Bagging - Gradient 
bg = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5,max_features=1.0, n_estimators=20)
bg.fit(x_train,y_train) 

bg.score(x_test,y_test) #test accuracy, 75.97%

bg.score(x_train,y_train) #train accuracy , 95.11%, difference further reduced

#Ada boosting 
ada = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=10,learning_rate=1)
ada.fit(x_train,y_train)

ada.score(x_test,y_test) # 69.48% accuracy

ada.score(x_train,y_train) # 100% accuracy

# extreme gradient boosting

from xgboost import XGBClassifier
model = XGBClassifier()

model.fit(x_train,y_train)

model.score(x_test,y_test) # 75.32%

model.score(x_train,y_train) # 100%










