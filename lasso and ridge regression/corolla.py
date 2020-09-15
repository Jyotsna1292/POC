# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 22:39:16 2020

@author: USER
"""

# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
toyota = pd.read_csv("C:/Users/USER/Downloads/ToyotaCorolla.csv",encoding = 'unicode_escape')

# removing unnecessary columns
Corolla = toyota[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]

# correlation matrix
Corolla.corr()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(Corolla)

# Checking whether we have any missing values or not 
Corolla.isnull().sum() # there are no missing values 

# preparing model considering all the variables using sklearn library
from sklearn.linear_model import LinearRegression
         
# Preparing model                  
LR1 = LinearRegression()
LR1.fit(Corolla.iloc[:,1:9],Corolla.Price)
# Getting coefficients of variables               
LR1.coef_
LR1.intercept_

# Adjusted R-Squared value
LR1.score(Corolla.iloc[:,1:9],Corolla.Price) # 0.8637
pred1 = LR1.predict(Corolla.iloc[:,1:9])

# Rmse value
np.sqrt(np.mean((pred1-Corolla.Price)**2)) # 1338.2584

# Residuals Vs Fitted Values
plt.scatter(x=pred1,y=(pred1-Corolla.Price));plt.xlabel("Fitted");plt.ylabel("Residuals");plt.hlines(y=0,xmin=0,xmax=60)
# Checking normal distribution 
plt.hist(pred1-Corolla.Price)

# Predicted Vs actual
plt.scatter(x=pred1,y=Corolla.Price);plt.xlabel("Predicted");plt.ylabel("Actual")
plt.bar(Corolla.columns[1:9],pd.Series(LR1.coef_))

# coefficients of gears is very high

### Let us split our entire data set into training and testing data sets
from sklearn.model_selection import train_test_split
train,test = train_test_split(Corolla,test_size=0.2)

### Preparing Ridge regression model for getting better weights on independent variables 
from sklearn.linear_model import Ridge

RM1 = Ridge(alpha = 0.4,normalize=True)
RM1.fit(train.iloc[:,1:9],train.Price)
# Coefficient values for all the independent variables
RM1.coef_
RM1.intercept_
plt.bar(Corolla.columns[1:9],pd.Series(RM1.coef_),color = "red", width = 0.4),plt.xlabel("inputs"),plt.ylabel("coefficients"),plt.show()

pred_RM1 = RM1.predict(train.iloc[:,1:9])

# Adjusted R-Squared value 
RM1.score(train.iloc[:,1:9],train.Price) # 0.8296
# RMSE
np.sqrt(np.mean((pred_RM1-train.Price)**2)) # 1485.64537

### Running a Ridge Regressor of set of alpha values and observing how the R-Squared, train_rmse and test_rmse are changing with change in alpha values
train_rmse = []
test_rmse = []
R_sqrd = []
alphas = np.arange(0,100,0.05)
for i in alphas:
    RM = Ridge(alpha = i,normalize=True)
    RM.fit(train.iloc[:,1:9],train.Price)
    R_sqrd.append(RM.score(train.iloc[:,1:9],train.Price))
    train_rmse.append(np.sqrt(np.mean((RM.predict(train.iloc[:,1:9]) - train.Price)**2)))
    test_rmse.append(np.sqrt(np.mean((RM.predict(test.iloc[:,1:9]) - test.Price)**2)))
    
#### Plotting train_rmse,test_rmse,R_Squared values with respect to alpha values


# Alpha vs R_Squared values
plt.scatter(x=alphas,y=R_sqrd);plt.xlabel("alpha");plt.ylabel("R_Squared")

# Alpha vs train rmse
plt.scatter(x=alphas,y=train_rmse);plt.xlabel("alpha");plt.ylabel("train_rmse")

# Alpha vs test rmse
plt.scatter(x=alphas,y=test_rmse);plt.xlabel("alpha");plt.ylabel("test_rmse")
plt.legend(("alpha Vs R_Squared","alpha Vs train_rmse","alpha Vs test_rmse"))

# We got minimum R_Squared value at small alpha values 

# Let us prepare Lasso Regression on data set
from sklearn.linear_model import Lasso
LassoM1 = Lasso(alpha = 0.01,normalize=True)
LassoM1.fit(train.iloc[:,1:9],train.Price)
# Coefficient values for all the independent variables
LassoM1.coef_
LassoM1.intercept_

plt.bar(Corolla.columns[1:9],pd.Series(LassoM1.coef_),color = "red", width = 0.4),plt.xlabel("inputs"),plt.ylabel("coefficients"),plt.show()

pred_LassoM1 = LassoM1.predict(train.iloc[:,1:9])
# Adjusted R-Squared value 
LassoM1.score(train.iloc[:,1:9],train.Price) # 0.8627
# RMSE
np.sqrt(np.mean((pred_LassoM1-train.Price)**2)) # 1333.6423

### Running a LASSO Regressor of set of alpha values and observing how the R-Squared, train_rmse and test_rmse are changing with change in alpha values
train_rmse = []
test_rmse = []
R_sqrd = []
alphas = np.arange(0,30,0.05)
for i in alphas:
    LRM = Lasso(alpha = i,normalize=True,max_iter=500)
    LRM.fit(train.iloc[:,1:9],train.Price)
    R_sqrd.append(LRM.score(train.iloc[:,1:9],train.Price))
    train_rmse.append(np.sqrt(np.mean((LRM.predict(train.iloc[:,1:9]) - train.Price)**2)))
    test_rmse.append(np.sqrt(np.mean((LRM.predict(test.iloc[:,1:9]) - test.Price)**2)))
    
#### Plotting train_rmse,test_rmse,R_Squared values with respect to alpha values

# Alpha vs R_Squared values
plt.scatter(x=alphas,y=R_sqrd);plt.xlabel("alpha");plt.ylabel("R_Squared")

# Alpha vs train rmse
plt.scatter(x=alphas,y=train_rmse);plt.xlabel("alpha");plt.ylabel("train_rmse")

# Alpha vs test rmse
plt.scatter(x=alphas,y=test_rmse);plt.xlabel("alpha");plt.ylabel("test_rmse")
plt.legend(("alpha Vs R_Squared","alpha Vs train_rmse","alpha Vs test_rmse"))

# We got minimum R_Squared value at small alpha values 
# from this we can say applying the simple linear regression technique is giving better results than Ridge and Lasso
# alpha tends 0 it indicates that Lasso and Ridge approximates to normal regression techniques 

# Checking whether data has any influential values 
# influence index plots
## Using regression model from statsmodels.formula.api
import statsmodels.formula.api as smf
formula = Corolla.columns[0]+"~"+"+".join(Corolla.columns[1:9])
model=smf.ols(formula,data=Corolla).fit()

# For getting coefficients of the varibles used in equation
model.params

# P-values for the variables and R-squared value for prepared model
model.summary() # R^2 is 0.864

print (model.conf_int(0.05)) # 95% confidence interval

pred = model.predict(Corolla.iloc[:,1:9]) # Predicted values of price using the model


# Studentized Residuals = Residual/standard deviation of residuals
plt.scatter(pred,model.resid_pearson);plt.xlabel("Fitted");plt.ylabel("residuals");plt.hlines(y=0,xmin=0,xmax=60)
plt.hist(model.resid_pearson)

import statsmodels.api as sm
# added variable plot for the final model
sm.graphics.plot_partregress_grid(model)


######  Linearity #########
# Observed values VS Fitted values
plt.scatter(Corolla.Price,pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

# Residuals VS Fitted Values 
plt.scatter(pred,model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


########    Normality plot for residuals ######
# histogram
plt.hist(model.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(model.resid_pearson, dist="norm", plot=pylab)

# Residuals VS Fitted Values 
plt.scatter(pred,model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


# preparing the model on train data 

model_train = smf.ols(formula,data=train).fit()

# train_data prediction
train_pred = model_train.predict(train)

# train residual values 
train_resid  = train_pred - train.Price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid)) # 1333.64198

# prediction on test data set 
test_pred = model_train.predict(test)

# test residual values 
test_resid  = test_pred - test.Price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid)) # 1366.57718

    

















