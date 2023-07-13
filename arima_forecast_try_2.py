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

#Calculating division for training set
div=int(train_size*len(df))

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
Try 2: 
Using linear regression and seasonal avg to remove seasonal trend and non-stationarity
"""

#Getting month as a column
df2 = df.reset_index().reset_index()
df2.rename(columns={"index":"period"}, inplace=True)
df2["month"]=pd.to_datetime(df2["date"]).dt.month
df2.set_index("date", inplace=True)
df2.head()

#Calculating Lin Reg
lr=LinearRegression()
div=int(.8*len(df2))
X_train=df2.iloc[:div]["period"]
y_train=df2.iloc[:div]["value"]
lr.fit(X_train.values.reshape(-1, 1), y_train)
trend_lin=lr.coef_
intercept_lin=lr.intercept_
print(f'Lin. Reg. Coefficient: {lr.coef_}')
print(f'Lin. Reg. Intercept: {lr.intercept_}')

#Removing Linear Trend
df2["value"]-=df2["period"]*lr.coef_ + lr.intercept_
df2["value"].plot()
plt.title("De-trended Sales Data")
plt.show()

#Removing seasonality (still maintaining test and train boundary)
df2_avg=df2.iloc[:div].groupby(["month"]).mean().reset_index()
df2_avg.rename(columns={"value":"avg_for_month"}, inplace=True)
merged = df2.merge(df2_avg[["month", "avg_for_month"]], how="left", on="month")
df2["value"]-=merged["avg_for_month"].values
plt.plot(df2["value"])
plt.title("Sales Data w/ Trend and Seasonality Removed")
plt.show()

#Diffferenced PACF/ACF
plot_acf(df2["value"].dropna())
plt.title("Detrended Autocorrelation")
plt.show()
plot_pacf(df2["value"].dropna())
plt.title("Detrended Partial Autocorrelation")
plt.show()

#Augmented Dickey-Fuller on Detrended Data
result = adfuller(df2["value"], maxlag=15)
print(f'ADF statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p_values: {result[2]}')
for key, value in result[4].items():
    print('Critical Value:')
    print(f' {key}, {value}')
    
#AR on differenced data
print("AR Results:")
phi, intercept_ar, AR_df = AR(3, df2["value"])

"""
No MA process was used on this dataset
MA showed little improvement in the modelling of try_1
"""

# #MA on residuals from AR
# print("MA Results:")
# theta, intercept_MA, MA_df = MA(5, AR_df["error_ar"])

# #Aggregating MA and AR results
# agg_results = AR_df.reset_index()[["date", "value", "predicted_ar", "error_ar"]].merge(MA_df.reset_index()[["date", "predicted_ma"]], how="left", on=["date"])
# agg_results.set_index("date", inplace=True)

"""
Using AR model to predict train set
"""

#Readding trend and seasonality
pred_df=AR_df.copy()

pred_df["index"]=range(len(pred_df))

pred_df["value"]+=merged["avg_for_month"].values
pred_df["predicted_ar"]+=merged["avg_for_month"].values
                                  
pred_df["value"]+= (pred_df["index"]*trend_lin+intercept_lin)
pred_df["predicted_ar"]+= (pred_df["index"]*trend_lin+intercept_lin)


#Plotting train set
plt.plot(np.exp(pred_df["value"]), label="Actual Value", alpha=.5)
plt.plot(np.exp(pred_df.iloc[:div]["predicted_ar"]), label="Predicted", linestyle="dashed")

plt.title("Train Set")
plt.legend()
plt.show()

"""
Using AR model to predict test set by extending stationary time series then un-differencing
"""
agg_results=AR_df.copy()

#Extending train data to full dataset
extended=arima_extend(phi, [], intercept_ar, agg_results.iloc[:div], len(agg_results)-div)

#Defining a function specifically for this model's prediction
def arima_predict(extended):
    result=extended.copy()
    assert(list(result.columns).count("value")==1)

    div=int(train_size*len(extended))
    result["predicted"]=result["value"]
    result["period"]=range(len(result))
    
    #Un-differencing train data
    result["predicted"]+=(result["period"]*trend_lin+intercept_lin)
    result["predicted"]+=merged["avg_for_month"].values
    
    #Taking exponential
    result["predicted"]=np.exp(result["predicted"])
    return(result["predicted"])

#Predicting with function above
pred=arima_predict(extended)
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
