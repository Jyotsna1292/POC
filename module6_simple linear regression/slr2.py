# -*- coding: utf-8 -*-
"""
Created on Wed May 13 22:28:04 2020

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

del_time=pd.read_csv("C:/Users/USER/Downloads/delivery_time.csv",sep=r'\s*,\s*') # to remove spaces in the column name

# EDA
del_time.describe()
plt.hist(del_time['D_t'])
import seaborn as sns
sns.boxplot(del_time["D_t"])
plt.hist(del_time['S_t'])
sns.boxplot(del_time["S_t"])

# forming model
plt.scatter(del_time['S_t'],del_time['D_t'])
np.corrcoef(del_time.S_t, del_time.D_t) #correlation i.e 0.825,  high positive correlation

import statsmodels.formula.api as smf
model_1 = smf.ols('D_t ~ S_t', data=del_time).fit() # R^2 value is 0.682
model_1.summary()
pred1 = model_1.predict(pd.DataFrame(del_time['S_t']))
pred1

print(model_1.conf_int(0.01)) # 99% confidence interval

# calculation of rmse
res = del_time.D_t - pred1
res
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse)
rmse # 2.79

######### Model building on Transformed Data

# Log Transformation
plt.scatter(np.log(del_time['S_t']),del_time['D_t'])
np.corrcoef(np.log(del_time.S_t),del_time.D_t) # coeff is 0.834
model_2=smf.ols('D_t~np.log(S_t)', data=del_time).fit()
model_2.summary() # R^2 value reduced to 0.695

plt.scatter(del_time['S_t'],np.log(del_time['D_t']))
np.corrcoef(del_time.S_t,np.log(del_time.D_t)) # coeff=0.843
model_3=smf.ols('np.log(D_t)~S_t', data=del_time).fit()
model_3.summary() # R^2 value is 0.711

# taking log of both variable
plt.scatter(np.log(del_time['S_t']),np.log(del_time['D_t']))
np.corrcoef(np.log(del_time.S_t),np.log(del_time.D_t)) # coeff is 0.878
model_4=smf.ols('np.log(D_t)~np.log(S_t)', data=del_time).fit()
model_4.summary()# R^2 is 0.772, providing best model
print(model_4.conf_int(0.01)) # 99% confidence level
    
log_pred2 = model_4.predict(pd.DataFrame(del_time['S_t']))
log_pred2
pred2=np.exp(log_pred2)
pred2
res2 = del_time.D_t - pred2
sqres2 = res2*res2
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)
rmse2 # 2.74

# taking sqrt of cc
plt.scatter(np.sqrt(del_time['S_t']),del_time['D_t'])
np.corrcoef(np.sqrt(del_time.S_t),del_time.D_t) # coeff is 0.834

# taking sqrt of wg
plt.scatter(del_time['S_t'],np.sqrt(del_time['D_t']))
np.corrcoef(del_time.S_t,np.sqrt(del_time.D_t)) # coeff is 0.839
model_5=smf.ols('np.sqrt(D_t)~S_t', data=del_time).fit()
model_5.summary() # R^2 is 0.704





