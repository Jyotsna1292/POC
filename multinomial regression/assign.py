# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 22:52:19 2020

@author: USER
"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
mdata = pd.read_csv("C:/Users/USER/Downloads/mdata.csv")
mdata.head(10)

mdata.describe()
mdata.prog.value_counts()

# removing first column
mdata1 = mdata.iloc[:,1:]

# now moving output column prog to the front
# get a list of columns
cols = list(mdata1)
# move the column to head of list using index, pop and insert
cols.insert(0, cols.pop(cols.index('prog')))
cols
# use ix to reorder
mdata2 = mdata1.ix[:, cols]
mdata2

# converting categorical into numerical form
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
mdata3 = mdata2.apply(le.fit_transform)
                      
train,test = train_test_split(mdata3,test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class="multinomial",solver="newton-cg").fit(train.iloc[:,1:],train.iloc[:,0])

train_predict = model.predict(train.iloc[:,1:]) # Train predictions 
test_predict = model.predict(test.iloc[:,1:]) # Test predictions

# Train accuracy 
accuracy_score(train.iloc[:,0],train_predict) # 66.8%
# Test accuracy 
accuracy_score(test.iloc[:,0],test_predict) # 50%









