# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 21:24:25 2020

@author: USER
"""

# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
computer = pd.read_csv("C:/Users/USER/Downloads/Computer_Data.csv")

# removing 1st column which is unnecessary
computer = computer.iloc[:,1:]

# creating dummy of categorical variable
dummies = pd.get_dummies(computer.iloc[:,5:8])
Computer = pd.concat([computer, dummies], axis=1)
col = Computer.drop(Computer.iloc[:,5:8], inplace=True, axis=1)
# correlation matrix
Computer.corr()

#pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# there isn't much high correlation among the independent variables
# Chance of having multicollinearity problem is less
 
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(Computer)

# Checking whether we have any missing values or not 
Computer.isnull().sum() # there are no missing values 

# preparing model considering all the variables using sklearn library
from sklearn.linear_model import LinearRegression
         
# Preparing model                  
LR1 = LinearRegression()
LR1.fit(Computer.iloc[:,1:13],Computer.price)
# Getting coefficients of variables               
LR1.coef_
LR1.intercept_

# Adjusted R-Squared value
LR1.score(Computer.iloc[:,1:13],Computer.price) # 0.7755
pred1 = LR1.predict(Computer.iloc[:,1:13])

# Rmse value
np.sqrt(np.mean((pred1-Computer.price)**2)) # 275.129

# Residuals Vs Fitted Values
plt.scatter(x=pred1,y=(pred1-Computer.price));plt.xlabel("Fitted");plt.ylabel("Residuals");plt.hlines(y=0,xmin=0,xmax=60)
# Checking normal distribution 
plt.hist(pred1-Computer.price)

# Predicted Vs actual
plt.scatter(x=pred1,y=Computer.price);plt.xlabel("Predicted");plt.ylabel("Actual")
plt.bar(Computer.columns[1:13],pd.Series(LR1.coef_))

# coefficients of premium_no and premium_yes is very high

### Let us split our entire data set into training and testing data sets
from sklearn.model_selection import train_test_split
train,test = train_test_split(Computer,test_size=0.2)

### Preparing Ridge regression model for getting better weights on independent variables 
from sklearn.linear_model import Ridge

RM1 = Ridge(alpha = 0.4,normalize=True)
RM1.fit(train.iloc[:,1:13],train.price)
# Coefficient values for all the independent variables
RM1.coef_
RM1.intercept_
plt.bar(Computer.columns[1:13],pd.Series(RM1.coef_),color = "red", width = 0.4),plt.xlabel("inputs"),plt.ylabel("coefficients"),plt.show()

pred_RM1 = RM1.predict(train.iloc[:,1:13])
# Adjusted R-Squared value 
RM1.score(train.iloc[:,1:13],train.price) # 0.6805
# RMSE
np.sqrt(np.mean((pred_RM1-train.price)**2)) # 328.4099

### Running a Ridge Regressor of set of alpha values and observing how the R-Squared, train_rmse and test_rmse are changing with change in alpha values
train_rmse = []
test_rmse = []
R_sqrd = []
alphas = np.arange(0,100,0.05)
for i in alphas:
    RM = Ridge(alpha = i,normalize=True)
    RM.fit(train.iloc[:,1:13],train.price)
    R_sqrd.append(RM.score(train.iloc[:,1:13],train.price))
    train_rmse.append(np.sqrt(np.mean((RM.predict(train.iloc[:,1:13]) - train.price)**2)))
    test_rmse.append(np.sqrt(np.mean((RM.predict(test.iloc[:,1:13]) - test.price)**2)))
    
    
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
LassoM1.fit(train.iloc[:,1:13],train.price)
# Coefficient values for all the independent variables
LassoM1.coef_
LassoM1.intercept_
plt.bar(Computer.columns[1:13],pd.Series(LassoM1.coef_),color = "red", width = 0.4),plt.xlabel("inputs"),plt.ylabel("coefficients"),plt.show()

pred_LassoM1 = LassoM1.predict(train.iloc[:,1:13])
# Adjusted R-Squared value 
LassoM1.score(train.iloc[:,1:13],train.price) # 0.7739
# RMSE
np.sqrt(np.mean((pred_LassoM1-train.price)**2)) # 276.274

### Running a LASSO Regressor of set of alpha values and observing how the R-Squared, train_rmse and test_rmse are changing with change in alpha values
train_rmse = []
test_rmse = []
R_sqrd = []
alphas = np.arange(0,30,0.05)
for i in alphas:
    LRM = Lasso(alpha = i,normalize=True,max_iter=500)
    LRM.fit(train.iloc[:,1:13],train.price)
    R_sqrd.append(LRM.score(train.iloc[:,1:13],train.price))
    train_rmse.append(np.sqrt(np.mean((LRM.predict(train.iloc[:,1:13]) - train.price)**2)))
    test_rmse.append(np.sqrt(np.mean((LRM.predict(test.iloc[:,1:13]) - test.price)**2)))
    
    
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
formula = Computer.columns[0]+"~"+"+".join(Computer.columns[1:13])
model=smf.ols(formula,data=Computer).fit()

# For getting coefficients of the varibles used in equation
model.params

# P-values for the variables and R-squared value for prepared model
model.summary() # R^2 is 0.776
# Age and Indus

print (model.conf_int(0.05)) # 95% confidence interval

pred = model.predict(Computer.iloc[:,1:13]) # Predicted values of price using the model


# Studentized Residuals = Residual/standard deviation of residuals
plt.scatter(pred,model.resid_pearson);plt.xlabel("Fitted");plt.ylabel("residuals");plt.hlines(y=0,xmin=0,xmax=60)
plt.hist(model.resid_pearson)


import statsmodels.api as sm
# added variable plot for the final model
sm.graphics.plot_partregress_grid(model)


######  Linearity #########
# Observed values VS Fitted values
plt.scatter(Computer.price,pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

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


############ Homoscedasticity #######

# Residuals VS Fitted Values 
plt.scatter(pred,model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


# preparing the model on train data 

model_train = smf.ols(formula,data=train).fit()

# train_data prediction
train_pred = model_train.predict(train)

# train residual values 
train_resid  = train_pred - train.price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid)) # 276.2627

# prediction on test data set 
test_pred = model_train.predict(test)

# test residual values 
test_resid  = test_pred - test.price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid)) # 271.057















