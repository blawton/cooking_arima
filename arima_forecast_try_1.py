import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import adfuller
import math
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from Functions.arima import AR
from Functions.arima import MA
from Functions.arima import  extend as arima_extend
from Functions.arima import rmse

#Global Params
train_size=.8

#Loading and graphing initial data
url = 'https://raw.githubusercontent.com/selva86/datasets/master/a10.csv'
df = pd.read_csv(url, parse_dates=['date'], index_col='date')
series = df.loc[:, 'value'].values
df.plot(figsize=(14,8), legend=None, title='a10 - Drug Sales Series - Raw Data')
plt.show()

#Scaling initial data to log scale
df["value"]=df["value"].apply(lambda x: math.log(x))
df.plot(figsize=(14,8), legend=None, title='a10 - Drug Sales Series - Log Scaled')
plt.xlabel("Date")
plt.ylabel("log of drug sales")
plt.show()

#Running adfuller test
result = adfuller(df, maxlag=15)
print(f'ADF statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p_values: {result[2]}')
for key, value in result[4].items():
    print('Critical Value:')
    print(f' {key}, {value}')

#ACF
plot_acf(df, lags=150)
plt.show()

#PACF
plot_pacf(df)
plt.show()

"""
Try 1: 
Differencing to remove seasonal trend and non-stationarity
"""

df_d=df.diff(12)
plt.plot(df_d)
plt.title("Differenced Time Series")
plt.xlabel("Date")
plt.ylabel("Log Sales Value")
plt.show()

#Diffferenced PACF/ACF
plot_acf(df_d["value"].dropna())
plt.title("Differenced Autocorrelation")
plt.show()
plot_pacf(df_d["value"].dropna())
plt.title("Differenced Partial Autocorrelation")
plt.show()

#AR on differenced data
print("AR Results:")
phi, intercept_ar, AR_df = AR(3, df_d)

#MA on residuals from AR
print("MA Results:")
theta, intercept_MA, MA_df = MA(5, AR_df["error_ar"])

#Aggregating MA and AR results
agg_results = AR_df.reset_index()[["date", "value", "predicted_ar", "error_ar"]].merge(MA_df.reset_index()[["date", "predicted_ma"]], how="left", on=["date"])
agg_results.set_index("date", inplace=True)

## Assessing fit of model on training data

pred_df = agg_results.copy()

#Undifferencing and unscaling
pred_df["arma"]=pred_df["predicted_ma"]+pred_df["predicted_ar"]

pred_df["arma"]+=df["value"].shift(12)
pred_df["arma"]=np.exp(pred_df["arma"])

pred_df["value"]+=df["value"].shift(12)
pred_df["value"]=np.exp(pred_df["value"])

#Plotting train_set
div=int(train_size*len(pred_df))
plt.plot(pred_df["value"], label="Actual Value", alpha=.5)
plt.plot(pred_df.iloc[:div]["arma"], linestyle="dashed", label="Predicted")

plt.title("Model vs. ARIMA Forecast - Train Set")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()

#Extending train data to full dataset
div=int(train_size*len(agg_results))
extended=arima_extend(phi, theta, intercept_ar, agg_results.iloc[:div], len(agg_results)-div)

"""
Defining a function specifically for this model's prediction:
Works by un-differencing data and taking exponential
"""

def arima_predict(extended, original):
    result=extended.copy()
    assert(list(result.columns).count("value")==1)
    assert(list(original.columns).count("value")==1)

    div=int(train_size*len(extended))
    result["predicted"]=result["value"]
    
    #Un-differencing train data
    result.loc[result.index[:div], "predicted"]+=original.loc[original.index[:div], "value"].shift(12).values
    
    #Iteratively undifferencing test data
    for i in range(len(result)-div):
        result.loc[result.index[div+i], "predicted"]+=result.loc[result.index[div+i-12], "predicted"]
    
    #Taking exponential
    result["predicted"]=np.exp(result["predicted"])
    return(result["predicted"])

#Predicting with function above
pred=arima_predict(extended, df)
pred.index=df.index

#Plotting prediction of time series
plt.plot(np.exp(df["value"]), label="Actual Value", alpha=.5)
plt.plot(pred, label="Prediction", linestyle="dashed")

plt.vlines(pred.index[div], color="red", ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1], label="End of Train Set")
plt.title("Model vs. ARIMA Forecast - Train and Test Set")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()

#Calculating RMSE
print("RMSE on test set:")
print(rmse(pred, np.exp(df["value"]), train_size))