# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:01:03 2020

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cal_con=pd.read_csv("C:/Users/USER/Downloads/DataSets/calories_consumed.csv",sep=r'\s*,\s*') # to remove spaces in the column name

print(type(cal_con))
#print(cal_con.columns.tolist())

#EDA
cal_con.describe() # to get summary
plt.hist(cal_con["wg"])

import seaborn as sns
sns.boxplot(cal_con["wg"]) # no outliners in the data..data is highly right tailed
plt.hist(cal_con["cc"])
sns.boxplot(cal_con["cc"])

plt.scatter(cal_con['cc'],cal_con['wg'])
np.corrcoef(cal_con.cc, cal_con.wg) #correlation i.e 0.96, very high positive correlation

import statsmodels.formula.api as smf
model_1 = smf.ols('wg ~ cc', data=cal_con).fit() # R^2 value is 0.897
model_1.summary()
pred1 = model_1.predict(pd.DataFrame(cal_con['cc']))
pred1

print(model_1.conf_int(0.01)) # 99% confidence interval

# calculation of rmse
res = cal_con.wg - pred1
res
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse)
rmse

######### Model building on Transformed Data

# Log Transformation
plt.scatter(np.log(cal_con['cc']),cal_con['wg'])
np.corrcoef(np.log(cal_con.cc),cal_con.wg) # coeff is 0.898
model_2=smf.ols('wg~np.log(cc)', data=cal_con).fit()
model_2.summary() # R^2 value reduced to 0.808

plt.scatter(cal_con['cc'],np.log(cal_con['wg']))
np.corrcoef(cal_con.cc,np.log(cal_con.wg)) # coeff=0.936
model_3=smf.ols('np.log(wg)~cc', data=cal_con).fit()
model_3.summary() # R^2 value is 0.878

# taking log of both variable
plt.scatter(np.log(cal_con['cc']),np.log(cal_con['wg']))
np.corrcoef(np.log(cal_con.cc),np.log(cal_con.wg)) # coeff is 0.92
model_6=smf.ols('np.log(wg)~np.log(cc)', data=cal_con).fit()
model_6.summary()# R^2 is 0.846

# taking sqrt of cc
plt.scatter(np.sqrt(cal_con['cc']),cal_con['wg'])
np.corrcoef(np.sqrt(cal_con.cc),cal_con.wg) # coeff is 0.9255

# taking sqrt of wg
plt.scatter(cal_con['cc'],np.sqrt(cal_con['wg']))
np.corrcoef(cal_con.cc,np.sqrt(cal_con.wg)) # coeff is 0.955
model_4=smf.ols('np.sqrt(wg)~cc', data=cal_con).fit()
model_4.summary() # R^2 is 0.914, providing best model

print(model_4.conf_int(0.01)) # 99% confidence level


sqrt_pred2 = model_4.predict(pd.DataFrame(cal_con['cc']))
sqrt_pred2
pred2=(sqrt_pred2)**2
pred2
res2 = cal_con.wg - pred2
sqres2 = res2*res2
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)
rmse2 # 73.74

# taking square of cc
plt.scatter(cal_con['cc']**2,cal_con['wg'])
np.corrcoef(cal_con.cc**2,cal_con.wg) # coeff is 0.971
model_5=smf.ols('wg~cc**2', data=cal_con).fit()
model_5.summary() # R^2 is 0.897

#taking square of wg
plt.scatter(cal_con['cc'],cal_con['wg']^2)
np.corrcoef(cal_con.cc,cal_con.wg^2) # coeff is 0.946

# taking sin of cc
plt.scatter(np.sin(cal_con['cc']),cal_con['wg']) # no correlation









