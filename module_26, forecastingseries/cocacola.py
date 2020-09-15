# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 17:38:12 2020

@author: USER
"""

# Data driven approaches

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 


# import dataset
cocacola = pd.read_excel("C:/Users/USER/Downloads/CocaCola_Sales_Rawdata.xlsx")
cocacola.Sales.plot()
cocacola["quarter"]=0
cocacola["year"]=0

for i in range(42):
    p = cocacola["Quarter"][i]
    cocacola['quarter'][i]= p[0:2]
    
for i in range(42):
    p = cocacola["Quarter"][i]
    cocacola['year'][i]= p[3:5]    
    
heatmap_1 = pd.pivot_table(data=cocacola,values="Sales",index="year",columns="quarter",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_1,annot=True,fmt="g")    

# boxplot
sns.boxplot(x="year",y="Sales",data=cocacola)    

# Line plot 
sns.lineplot(x="year",y="Sales",data=cocacola)    

# Centering moving average for the time series to understand better about the trend character in cocacola
cocacola.Sales.plot(label="org")
for i in range(2,16,4):
    cocacola["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=4)    
    
    # Time series decomposition plot 
decompose_ts_add = seasonal_decompose(cocacola.Sales,model="additive",freq=4)
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(cocacola.Sales,model="multiplicative",freq=4)
decompose_ts_mul.plot()


# ACF plots and PACF plots on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(cocacola.Sales,lags=12)
tsa_plots.plot_pacf(cocacola.Sales,lags=12)

# splitting data
Train = cocacola.head(38)
Test = cocacola.tail(4)
    
 # Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)
   
 # Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Sales) # 8.2720

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Sales) # 8.8223
   
 # Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Sales) # 1.4994

#hwe_model_add_add = ExponentialSmoothing(cocacola["Sales"],seasonal="add",trend="add",seasonal_periods=4).fit()
#pred_hwe_add_add = hwe_model_add_add.predict(start = cocacola.index[0],end = Amtrak.index[-1])

#MAPE(pred_hwe_add_add,cocacola.Sales) # 4.768

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Sales) # 1.778

# Visualization of Forecasted values for Test data set using different methods 
plt.plot(Train.index, Train["Sales"], label='Train',color="black")
plt.plot(Test.index, Test["Sales"], label='Test',color="blue")
plt.plot(pred_ses.index, pred_ses, label='SimpleExponential',color="green")
plt.plot(pred_hw.index, pred_hw, label='Holts_winter',color="red")
plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponential_1",color="brown")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="HoltsWinterExponential_2",color="yellow")

plt.legend(loc='best')

# Models and their MAPE values
model_mapes = pd.DataFrame(columns=["model_name","mape"])
model_mapes["model_name"] = ["ses_model" ,"hw_model","hwe_model_add_add","hwe_model_mul_add"]
model_mapes["mape"] = [8.2720,8.8223,1.4994,1.778]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    