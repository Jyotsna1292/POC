# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 00:49:19 2020

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

toyota=pd.read_csv("C:/Users/USER/Downloads/ToyotaCorolla.csv", encoding='latin')
corolla=toyota[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]

corolla.corr()

import pandas as pd
# for displaying entire correlation matrix in console
#pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import seaborn as sns
sns.pairplot(corolla)

#preparing model considering all variables
import statsmodels.formula.api as smf # for regression model
model=smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data=corolla).fit()
model.summary()
model_1=smf.ols("Price~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight", data=corolla).fit()
model_1.summary()
model_2=smf.ols("Price~Doors",data=corolla).fit()
model_2.summary()

#influence plot
import statsmodels.api as sm
sm.graphics.influence_plot(model)

t_corolla=corolla.drop(corolla.index[[-80]],axis=0) #dropping 80th record
model_3=smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data=t_corolla).fit()
model_3.summary() #no improvement in model instead it is getting slightly worse

#calculating vif values of independent variables
rsq_Age_08_04=smf.ols("Age_08_04~KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data=corolla).fit().rsquared
vif_Age_08_04=1/(1-rsq_Age_08_04)
rsq_KM=smf.ols("KM~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data=corolla).fit().rsquared
vif_KM=1/(1-rsq_KM)
rsq_HP=smf.ols("HP~Age_08_04+KM+cc+Doors+Gears+Quarterly_Tax+Weight", data=corolla).fit().rsquared
vif_HP=1/(1-rsq_HP)
rsq_cc=smf.ols("cc~Age_08_04+HP+KM+Doors+Gears+Quarterly_Tax+Weight", data=corolla).fit().rsquared
vif_cc=1/(1-rsq_cc)
rsq_Doors=smf.ols("Doors~Age_08_04+HP+cc+KM+Gears+Quarterly_Tax+Weight", data=corolla).fit().rsquared
vif_Doors=1/(1-rsq_Doors)
rsq_Gears=smf.ols("Gears~Age_08_04+HP+cc+Doors+KM+Quarterly_Tax+Weight", data=corolla).fit().rsquared
vif_Gears=1/(1-rsq_Gears)
rsq_Quarterly_Tax=smf.ols("Quarterly_Tax~Age_08_04+HP+cc+Doors+Gears+KM+Weight", data=corolla).fit().rsquared
vif_Quarterly_Tax=1/(1-rsq_Quarterly_Tax)
rsq_Weight=smf.ols("Weight~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+KM", data=corolla).fit().rsquared
vif_Weight=1/(1-rsq_Weight)

d1={"variables":["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"],"vif":[vif_Age_08_04,vif_KM,vif_HP,vif_cc,vif_Doors,vif_Gears,vif_Quarterly_Tax,vif_Weight]}
VIF_frame=pd.DataFrame(d1)
VIF_frame # all vif values are below 10 means no multicollinearity problem

corolla.describe()#for summary of dataset
_ = plt.hist(corolla['KM'])

corolla.corr()

model_4=smf.ols("Price~Age_08_04+KM+HP+Weight", data=corolla).fit()
model_4.summary()
sm.graphics.plot_partregress_grid(model_4)

#added variable plot
sm.graphics.plot_partregress_grid(model)
#as we can see from this plot door column is not contributing anything into the model so will proceed removing this column

#making model without door column and cc column
#final model
model_1=smf.ols("Price~Age_08_04+KM+HP+Gears+Quarterly_Tax+Weight", data=corolla).fit()
model_1.summary()
sm.graphics.plot_partregress_grid(model_1)

#predicted value for price
price_pred=model_1.predict(corolla[["Age_08_04","KM","HP","Gears","Quarterly_Tax","Weight"]])
price_pred

#observed value vs fitted value
plt.scatter(corolla.Price,price_pred,c='r');plt.xlabel("observed value");plt.ylabel("fitted value")

#residuals vs fitted value
plt.scatter(price_pred,model_1.resid_pearson, c="r"),plt.axhline(y=0,color='blue');plt.xlabel('fitted value');plt.ylabel('residuals')

#normality plot for residuals
#histogram
plt.hist(model_1.resid_pearson)

#qq plot for residuals
import pylab
import scipy.stats as st

#checking residuals are normally distributed
st.probplot(model_1.resid_pearson, dist="norm",plot=pylab)

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
corolla_train,corolla_test  = train_test_split(corolla,test_size = 0.3) # 30% test data
corolla_train
corolla_test

# preparing the model on train data 
model_train=smf.ols("Price~Age_08_04+KM+HP+Gears+Quarterly_Tax+Weight", data=corolla_train).fit()

# train data prediction
train_pred = model_train.predict(corolla_train)

# train residual values 
train_resid  = train_pred - corolla_train.Price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid)) #1273.433489490905
train_rmse

# prediction on test data set 
test_pred = model_train.predict(corolla_test)

# test residual values 
test_resid  = test_pred - corolla_test.Price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid)) #1488.1445671977026
test_rmse


















































 



























