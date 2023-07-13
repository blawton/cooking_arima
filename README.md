# cooking_arima
Cooking up ARIMA models from scratch in python to model sales [data](https://github.com/selva86/datasets/blob/master/a10.csv), uploaded by github user [selva86](https://github.com/selva86).

# Contents
1. Functions- in this repo, we code our own functions to use as Autogression and Moving Average predictors
2. arima_forecast_try_1.py- script for first version of ARIMA model, using seasonal differencing in order to remove trend
3. arima_forecast_try_2.py- script for a second version of an ARIMA-type model, this time using a linear regression and seasonal average to make data stationary and a pure AR process

# Approach
We try to minimize parameter usage wherever possible, but the number of parameters is based on human interpretation of ACF and PACF plots, which point towards the data being best modelled as an Autoregressive and not a Moving Average Process. This is because there is an abrupt drop in the PACF plot after 3 lags in both try_1:

![Figure_1](https://github.com/blawton/cooking_arima/assets/46683509/2af5992e-51b3-4e41-945d-960ecf360db1)

and try_2:

![Figure_1_try_2](https://github.com/blawton/cooking_arima/assets/46683509/dec1d523-86be-45a5-83db-b4ec01df8b5d)

But there are more gradual drops in the Autocorrelation for try_1:

![Figure_2](https://github.com/blawton/cooking_arima/assets/46683509/67fc7998-7b01-4030-b4a5-4613c533507c)

and especially for try_2, which supported the decision to use only an AR process:

![Figure_2_try_2](https://github.com/blawton/cooking_arima/assets/46683509/6efcdb59-cbc0-4134-b35f-55a7dce38d66)

The overall prediction results are similar with RMSE at 1.53 for try_1 and 1.66 for try_2, suggesting better performance for the ARIMA model as opposed to an AR model with linear regression and seasonal trend decomposition.

Graph of prediction for try_1:

![Figure_3](https://github.com/blawton/cooking_arima/assets/46683509/fb3f6d2e-70a6-4264-82c4-4cae4c845e9e)

And try_2:

![Figure_3_try_2](https://github.com/blawton/cooking_arima/assets/46683509/07529818-7948-439e-b6aa-3cca45758b48)


