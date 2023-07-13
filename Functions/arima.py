import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#Global Params
train_size=.8

#Autoreggresion
def AR(p, df):
    global train_size
    res =pd.DataFrame(df, index=df.index)

    #Generating Proper Columns
    for i in range(1, p+1):
        res["shifted_term_%d" % i] = res.iloc[:, 0].shift(i)
    
    #Making test and train split with no nas
    div=int(train_size*len(res.dropna(axis=0, how="any")))
    train=res.dropna(axis=0, how="any").iloc[:div, :].copy()
    test=res.dropna(axis=0, how="any").iloc[div:, :].copy()
    
    #Dividing train and test sets
    Xtrain = train.iloc[:, 1:].values
    ytrain = train.iloc[:, 0].values
    
    #Fitting reg
    lr = LinearRegression()
    lr.fit(Xtrain, ytrain)
    coeffs =lr.coef_
    
    #making predictions
    res["predicted_ar"]=np.dot(res.iloc[:, 1:].values, np.array(lr.coef_)) + lr.intercept_
    res["error_ar"]=res["predicted_ar"]-res.iloc[:, 0]
    
    RMSE = np.sqrt(np.sum(np.square(res.iloc[div:]["error_ar"]))/len(res.iloc[div:]))
    print(f'RMSE: {RMSE}')
    print(f'intercept: {lr.intercept_}')

    return(coeffs, lr.intercept_, res)

#Moving Average
def MA(q, resid):
    global train_size
    
    df=pd.DataFrame(resid)
    for i in range(1, q+1):
        df['lagged_res_%d' % i] = df.iloc[:, 0].shift(i)
    
    #Making test and train split with no nas
    div=int(train_size*len(df.dropna(axis=0, how="any")))
    
    #Dividing train_set
    X_train=df.dropna(axis=0, how="any").iloc[:div, 1:]
    y_train=df.dropna(axis=0, how="any").iloc[:div, 0]
    
    lr=LinearRegression()
    lr.fit(X_train,y_train)
    coeffs=lr.coef_
    
    df["predicted_ma"]=np.dot(df.iloc[:, 1:].values, np.array(lr.coef_))+lr.intercept_
    df["error_ma"]=df["predicted_ma"]-df.iloc[:, 0]
    
    RMSE = np.sqrt(np.sum(np.square(df.iloc[div:]["error_ma"]))/len(df.iloc[div:]))
    print(f'RMSE: {RMSE}')
    print(f'intercept: {lr.intercept_}')
    
    return(coeffs, lr.intercept_, df)

#Extending AR/MA prediction
def extend(phi, theta, intercept_ar, previous, periods):
    df=previous.copy()
    #Ensuring errors and values are both present in the dataframe (for ma and ar prediction)
    assert(list(df.columns).count("error_ar")==1)
    assert(list(df.columns).count("value")==1)
    assert(list(df.columns).count("extrapolated")==0)
    
    #Ensuring data history is present
    assert(len(df.dropna(axis=0, subset=["value"]))>len(phi))
    assert(len(df.dropna(axis=0, subset=["error_ar"]))>len(theta))

    for i in range(periods):
        pred_ar = np.dot(phi, df["value"].iloc[-len(phi):].values) + intercept_ar
        if len(theta)>0:
            pred_ma = np.dot(theta, df["error_ar"].iloc[-len(theta):].values)
        else:
            pred_ma = 0
        df=pd.concat([df,pd.DataFrame({"error_ar": pred_ma, "value": pred_ar+pred_ma}, index=[0])], ignore_index=True)
        
    return(df)

#Root Mean Squred Error
def rmse(pred, original, train_size):
    assert(len(pred)==len(original))
    div=int(train_size*len(pred))
    return(np.sqrt(np.sum(np.square(pred[div:]-original[div:]))/len(original[div:])))

