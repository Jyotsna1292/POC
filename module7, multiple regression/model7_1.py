# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:12:43 2020

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

startup = pd.read_csv("C:/Users/USER/Downloads/50_Startups.csv")
Dummy=pd.get_dummies(startup.State)
startup1=startup.drop("State",axis=1)#deleting state column replacing it with dummy variable
startup_50= pd.concat([startup1,Dummy], axis=1)

#correlation matrix
startup_50.corr()

import seaborn as sns
sns.pairplot(startup_50)

#preparing model considering all variables
import statsmodels.formula.api as smf # for regression model

Startup_50= startup_50.rename(columns={'R&D Spend':'RDspend','Marketing Spend':'Marketingspend','New York':'Newyork'})

# preparing model
model = smf.ols('Profit~RDspend+Administration+Marketingspend+California+Newyork+Florida', data=Startup_50).fit()
model.params #getting coefficient of variables
model.summary() # getting r^2 as 0.95

Startup_50.head(4)
# preparing model with only adminisitration
model_1 = smf.ols('Profit~Administration', data=Startup_50).fit()
model_1.summary() # getting r^2 as 0.04 which very low

model_2 = smf.ols('Profit~Marketingspend', data=Startup_50).fit()
model_2.summary() # getting r^2 as 0.559

import statsmodels.api as sm
sm.graphics.influence_plot(model)

#making model without 49th record
strt = Startup_50.drop(Startup_50.index[[-49]],axis=0)
model_4 = smf.ols('Profit~RDspend+Administration+Marketingspend+California+Florida+Newyork',data=strt).fit()
model_4.summary() # r^2 reduced to 0.947

print(model_4.conf_int(0.01)) #99% confidence level


#calculating vif values of independent variables
rsq_RDspend= smf.ols('RDspend~Administration+Marketingspend+California+Florida+Newyork' ,data=Startup_50).fit().rsquared
vif_RDspend=1/(1-rsq_RDspend)
rsq_Administration=smf.ols('Administration~RDspend+Marketingspend+California+Florida+Newyork',data=Startup_50).fit().rsquared
vif_Administration=1/(1-rsq_Administration)
rsq_Marketingspend=smf.ols('Marketingspend~RDspend+Administration+California+Florida+Newyork',data=Startup_50).fit().rsquared
vif_Marketingspend=1/(1-rsq_Marketingspend)
d1 = {'variables':['RDspend','Administration','Marketingspend'],'VIF':[vif_RDspend,vif_Administration,vif_Marketingspend]}
d1
VIF_frame= pd.DataFrame(d1)
VIF_frame # all vifs are less than 10 

#added variable plot
sm.graphics.plot_partregress_grid(model)
#making model without Administration column, because contribution of this column is minimal
model_5= smf.ols('Profit~RDspend+Marketingspend', data=Startup_50).fit()
model_5.summary() # r^2 is 0.95

sm.graphics.plot_partregress_grid(model_5)

np.set_printoptions(precision=2)

#predicted values for profit
profit_pred = model_5.predict(Startup_50[['RDspend','Administration','Marketingspend','California','Florida','Newyork']])
profit_pred


#observed value vs fitted value
plt.scatter(Startup_50.Profit,profit_pred,c="r");plt.xlabel("observed value");plt.ylabel("fitted value")

#residuals vs fitted value
plt.scatter(profit_pred,model_5.resid_pearson, c="r"),plt.axhline(y=0,color='blue');plt.xlabel('fitted value');plt.ylabel('residuals')

#normality plot for residuals
#histogram
plt.hist(model_5.resid_pearson)

#qq plot for residuals
import pylab
import scipy.stats as st

#checking residuals are normally distributed
st.probplot(model_5.resid_pearson, dist="norm",plot=pylab)

# get a list of columns
cols = list(Startup_50)
# move the column to head of list using index, pop and insert
cols.insert(0, cols.pop(cols.index('Profit')))
cols
# use ix to reorder
Startup_50 = Startup_50.ix[:, cols]
Startup_50

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
startup_train,startup_test  = train_test_split(Startup_50,test_size = 0.3) # 30% test data
startup_train
startup_test

# preparing the model on train data 
model_train=smf.ols('Profit~RDspend+Marketingspend', data=Startup_50).fit()
model_train.summary() # r^2 is 0.95

# train data prediction
train_pred = model_train.predict(startup_train.iloc[:,1:])

# train residual values 
train_resid  = train_pred - startup_train.Profit

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid)) 
train_rmse #9576.054

# prediction on test data set 
test_pred = model_train.predict(startup_test.iloc[:,1:])

# test residual values 
test_resid  = test_pred - startup_test.Profit

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid)) 
test_rmse # 6999.3619






























