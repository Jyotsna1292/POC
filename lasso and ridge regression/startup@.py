

# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
startup = pd.read_csv("C:/Users/USER/Downloads/50_Startups.csv")

# creating dummy of categorical variable
dummies = pd.get_dummies(startup['State']).rename(columns=lambda x: 'state_' + str(x))
Startup = pd.concat([startup, dummies], axis=1)
col = Startup.drop(['State'], inplace=True, axis=1)

# correlation matrix
Startup.corr()

# there isn't much high correlation among the independent variables
# Chance of having multicollinearity problem is less
 
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(Startup)

# columns names
Startup.columns
Startup.shape

# moving profit column to the end
cols = list(Startup)
# move the column to end of list using index, pop and insert
cols.insert(6, cols.pop(cols.index('Profit')))
cols

# use ix to reorder
Startup = Startup.ix[:, cols]

# preparing model considering all the variables using sklearn library
from sklearn.linear_model import LinearRegression
         
# Preparing model                  
LR1 = LinearRegression()
LR1.fit(Startup.iloc[:,:6],Startup.Profit)

# Getting coefficients of variables               
LR1.coef_
LR1.intercept_

# Adjusted R-Squared value
LR1.score(Startup.iloc[:,:6],Startup.Profit) # 0.95075
pred1 = LR1.predict(Startup.iloc[:,:6])

# Rmse value
np.sqrt(np.mean((pred1-Startup.Profit)**2)) # 8854.7610

# Residuals Vs Fitted Values
plt.scatter(x=pred1,y=(pred1-Startup.Profit));plt.xlabel("Fitted");plt.ylabel("Residuals");plt.hlines(y=0,xmin=0,xmax=60)
# Checking normal distribution 
plt.hist(pred1-Startup.Profit)

# Predicted Vs actual
plt.scatter(x=pred1,y=Startup.Profit);plt.xlabel("Predicted");plt.ylabel("Actual")
plt.bar(Startup.columns[:6],pd.Series(LR1.coef_),color = "red", width = 0.4),plt.xlabel("inputs"),plt.ylabel("coefficients"),plt.show()

np.corrcoef(Startup.state_Florida,Startup.Profit) # 0.1162

### Let us split our entire data set into training and testing data sets
from sklearn.model_selection import train_test_split
train,test = train_test_split(Startup,test_size=0.2)

### Preparing Ridge regression model for getting better weights on independent variables 
from sklearn.linear_model import Ridge

RM1 = Ridge(alpha = 0.4,normalize=True)
RM1.fit(train.iloc[:,:6],train.Profit)
# Coefficient values for all the independent variables
RM1.coef_
RM1.intercept_
plt.bar(Startup.columns[:6],pd.Series(RM1.coef_),color = "red", width = 0.4),plt.xlabel("inputs"),plt.ylabel("coefficients"),plt.show()

pred_RM1 = RM1.predict(train.iloc[:,:6])
# Adjusted R-Squared value 
RM1.score(train.iloc[:,:6],train.Profit) # 0.8932
# RMSE
np.sqrt(np.mean((pred_RM1-train.Profit)**2)) # 13672.674150

### Running a Ridge Regressor of set of alpha values and observing how the R-Squared, train_rmse and test_rmse are changing with change in alpha values
train_rmse = []
test_rmse = []
R_sqrd = []
alphas = np.arange(0,100,0.05)
for i in alphas:
    RM = Ridge(alpha = i,normalize=True)
    RM.fit(train.iloc[:,:6],train.Profit)
    R_sqrd.append(RM.score(train.iloc[:,:6],train.Profit))
    train_rmse.append(np.sqrt(np.mean((RM.predict(train.iloc[:,:6]) - train.Profit)**2)))
    test_rmse.append(np.sqrt(np.mean((RM.predict(test.iloc[:,:6]) - test.Profit)**2)))
    
    
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
LassoM1.fit(train.iloc[:,:6],train.Profit)
# Coefficient values for all the independent variables
LassoM1.coef_
LassoM1.intercept_
plt.bar(Startup.columns[:6],pd.Series(LassoM1.coef_),color = "red", width = 0.4),plt.xlabel("inputs"),plt.ylabel("coefficients"),plt.show()

pred_LassoM1 = LassoM1.predict(train.iloc[:,:6])
# Adjusted R-Squared value 
LassoM1.score(train.iloc[:,:6],train.Profit) # 0.956
# RMSE
np.sqrt(np.mean((pred_LassoM1-train.Profit)**2)) # 8777.0047

### Running a LASSO Regressor of set of alpha values and observing how the R-Squared, train_rmse and test_rmse are changing with change in alpha values
train_rmse = []
test_rmse = []
R_sqrd = []
alphas = np.arange(0,30,0.05)
for i in alphas:
    LRM = Lasso(alpha = i,normalize=True,max_iter=500)
    LRM.fit(train.iloc[:,:6],train.Profit)
    R_sqrd.append(LRM.score(train.iloc[:,:6],train.Profit))
    train_rmse.append(np.sqrt(np.mean((LRM.predict(train.iloc[:,:6]) - train.Profit)**2)))
    test_rmse.append(np.sqrt(np.mean((LRM.predict(test.iloc[:,:6]) - test.Profit)**2)))
    
    
#### Plotting train_rmse,test_rmse,R_Squared values with respect to alpha values

# Alpha vs R_Squared values
plt.scatter(x=alphas,y=R_sqrd);plt.xlabel("alpha");plt.ylabel("R_Squared")

# Alpha vs train rmse
plt.scatter(x=alphas,y=train_rmse);plt.xlabel("alpha");plt.ylabel("train_rmse")

# Alpha vs test rmse
plt.scatter(x=alphas,y=test_rmse);plt.xlabel("alpha");plt.ylabel("test_rmse")
plt.legend(("alpha Vs R_Squared","alpha Vs train_rmse","alpha Vs test_rmse"))

# test rmse is min at alpha = 30
LassoM2 = Lasso(alpha = 30 ,normalize=True)
LassoM2.fit(train.iloc[:,:6],train.Profit)
# Coefficient values for all the independent variables
LassoM2.coef_
LassoM2.intercept_
plt.bar(Startup.columns[:6],pd.Series(LassoM2.coef_),color = "red", width = 0.4),plt.xlabel("inputs"),plt.ylabel("coefficients"),plt.show()

pred_LassoM2 = LassoM2.predict(train.iloc[:,:6])
# Adjusted R-Squared value 
LassoM2.score(train.iloc[:,:6],train.Profit) # 0.956
# RMSE
np.sqrt(np.mean((pred_LassoM1-train.Profit)**2)) # 8777.0047

# predicting test 
pred1_LassoM2 = LassoM2.predict(.iloc[:,:6])
# RMSE_test
np.sqrt(np.mean((pred1_LassoM2-test.Profit)**2)) # 9497.9022

# we will be using Lasso regression for this data, because it is providing us the best model
















