# -*- coding: utf-8 -*-
"""
Created on Wed May 13 22:28:04 2020

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sal_data=pd.read_csv("C:/Users/USER/Downloads/Salary_Data.csv",sep=r'\s*,\s*') # to remove spaces in the column name

# EDA
sal_data.describe()
plt.hist(sal_data['ex'])
import seaborn as sns
sns.boxplot(sal_data["ex"])
plt.hist(sal_data['S'])
sns.boxplot(sal_data["S"])

# forming model
plt.scatter(sal_data['ex'],sal_data['S'])
np.corrcoef(sal_data.ex, sal_data.S) #correlation i.e 0.978, very  high positive correlation

import statsmodels.formula.api as smf
model_1 = smf.ols('S ~ ex', data=sal_data).fit() # R^2 value is 0.957, best model
model_1.summary()
pred1 = model_1.predict(pd.DataFrame(sal_data['ex']))
pred1

print(model_1.conf_int(0.01)) # 99% confidence interval

# calculation of rmse
res = sal_data.S - pred1
res
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse)
rmse # 5592

######### Model building on Transformed Data

# Log Transformation
plt.scatter(np.log(sal_data['ex']),sal_data['S'])
np.corrcoef(np.log(sal_data.ex),sal_data.S) # coeff is 0.924
model_2=smf.ols('S~np.log(ex)', data=sal_data).fit()
model_2.summary() # R^2 value reduced to 0.854

plt.scatter(sal_data['ex'],np.log(sal_data['S']))
np.corrcoef(sal_data.ex,np.log(sal_data.S)) # coeff=0.965
model_3=smf.ols('np.log(S)~ex', data=sal_data).fit()
model_3.summary() # R^2 value is 0.932

