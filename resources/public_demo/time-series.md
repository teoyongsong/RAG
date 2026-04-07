Time Series Forecasting Basics

A time series is a sequence of observations indexed by time.
Examples: daily sales, monthly temperature, stock prices.

Typical forecasting workflow:
1. Clean missing values and outliers.
2. Split train and validation by time order.
3. Build baseline models (naive or moving average).
4. Train models such as ARIMA, Prophet, or LSTM.
5. Evaluate using MAE, RMSE, or MAPE.

Important:
Do not randomly shuffle time series data for forecasting tasks.
