# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:01:03 2020

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

emp_data=pd.read_csv("C:/Users/USER/Downloads/DataSets (1)/emp_data.csv",sep=r'\s*,\s*') # to remove spaces in the column name

#EDA
emp_data.describe() # to get summary
plt.hist(emp_data["Salary_hike"])

import seaborn as sns
sns.boxplot(emp_data["Salary_hike"]) # no outliners in the data..data is  right tailed
plt.hist(emp_data["Churn_out_rate"])
sns.boxplot(emp_data["Churn_out_rate"])

plt.scatter(emp_data['Churn_out_rate'],emp_data['Salary_hike'])
np.corrcoef(emp_data.Churn_out_rate, emp_data.Salary_hike) #correlation i.e -0.9117, very high negative correlation

import statsmodels.formula.api as smf
model_1 = smf.ols('Salary_hike ~ Churn_out_rate', data=emp_data).fit() # R^2 value is 0.831
model_1.summary()
pred1 = model_1.predict(pd.DataFrame(emp_data['Churn_out_rate']))
pred1

print(model_1.conf_int(0.01)) # 99% confidence interval

# calculation of rmse
res = emp_data.Salary_hike - pred1
res
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse)
rmse # 35.89

######### Model building on Transformed Data

# Log Transformation
plt.scatter(np.log(emp_data['Churn_out_rate']),emp_data['Salary_hike'])
np.corrcoef(np.log(emp_data.Churn_out_rate),emp_data.Salary_hike) # coeff is -0.934
model_2=smf.ols('Salary_hike~np.log(Churn_out_rate)', data=emp_data).fit()
model_2.summary() # R^2 value reduced to 0.874

plt.scatter(emp_data['Churn_out_rate'],np.log(emp_data['Salary_hike']))
np.corrcoef(emp_data.Churn_out_rate,np.log(emp_data.Salary_hike)) # coeff = -0.921
model_3=smf.ols('np.log(Salary_hike)~Churn_out_rate', data=emp_data).fit()
model_3.summary() # R^2 value is 0.849

# taking log of both variable
plt.scatter(np.log(emp_data['Churn_out_rate']),np.log(emp_data['Salary_hike']))
np.corrcoef(np.log(emp_data.Churn_out_rate),np.log(emp_data.Salary_hike)) # coeff is -0.942
model_4=smf.ols('np.log(Salary_hike)~np.log(Churn_out_rate)', data=emp_data).fit()
model_4.summary()# R^2 is 0.889, providing best model

log_pred2=model_4.predict(pd.DataFrame(emp_data['Churn_out_rate']))
pred2= np.exp(log_pred2)
pred2
# calculation of rmse
res = emp_data.Salary_hike - pred2
res
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse)
rmse # 29.336

# taking sqrt of cc
plt.scatter(np.sqrt(emp_data['Churn_out_rate']),emp_data['Salary_hike'])
np.corrcoef(np.sqrt(emp_data.Churn_out_rate),emp_data.Salary_hike) # coeff is -0.923
model_5=smf.ols('Salary_hike~np.sqrt(Churn_out_rate)', data = emp_data).fit()
model_5.summary() # R^2 value is 0.853

# taking sqrt of wg
plt.scatter(emp_data['Churn_out_rate'],np.sqrt(emp_data['Salary_hike']))
np.corrcoef(emp_data.Churn_out_rate,np.sqrt(emp_data.Salary_hike)) # coeff is -0.916
model_6=smf.ols('np.sqrt(Salary_hike)~Churn_out_rate', data=emp_data).fit()
model_6.summary() # R^2 is 0.840

print(model_4.conf_int(0.01)) # 99% confidence level












