#  Bitcoin Stock Price Prediction using LSTM

```
、
├── Part I: Project Overview  
└── Part 2: Instructions on how to successfully run our code
```

### Part I: Project Overview

**Course Assignment 1**:  Data Mining - COMP7103C

**Team**:  Yuxi CHEN(Dylan)

**Source code**: [bitcoin_price_prediction-lstm.ipynb](https://github.com/Dylan-CS/COMP7103_Bitcoin_Price_Prediction_LSTM/blob/main/bitcoin_price_prediction-lstm.ipynb)

**Dataset**: https://github.com/Dylan-CS/COMP7103_Bitcoin_Price_Prediction_LSTM/blob/main/BTC-USD.csv

The Bitcoin price dataset refers to a collection of data points that represent the historical or current price of Bitcoin, a decentralized digital currency based on blockchain technology. The data points typically include information such as the date, time, and the opening, closing, highest and lowest prices of Bitcoin during a specific period of time. The Bitcoin price dataset can be used for a variety of purposes, such as studying market trends, conducting technical analysis, or training machine learning models for Bitcoin price prediction. The data can be obtained from various financial sources such as cryptocurrency exchanges, financial websites, or by directly accessing APIs provided by cryptocurrency data providers.

The following are the common features found in a Bitcoin price dataset:

- **Date:** The date on which the Bitcoin price data was recorded.
- **Open:** This refers to the price of Bitcoin at the beginning of the trading day.
- **Close:** This refers to the price of Bitcoin at the end of the trading day.
- **Adj. Close:** The adjusted close price accounts for any corporate actions such as stock splits, dividends, etc. that occurred on that day
- **High:** The highest price of Bitcoin during the trading day.
- **Low:** The lowest price of Bitcoin during the trading day.

These features can provide valuable information about the Bitcoin performance, trends and volatility over a certain period of time, and can be used in financial analysis, prediction, and decision making. By analyzing the historical Bitcoin price data, one can identify patterns and trends, which can be used to predict the future prices of Bitcoin. Machine learning models can be trained using this data to make accurate predictions and provide insights into the Bitcoin market.



### Part 2: Instructions on how to successfully run our code

Remark: If you meet some problems when running the code ,you can just open the file `bitcoin_price_prediction-lstm.html` to see all the results  or come to contact us-: chenyuxi@connect.hku.hk

1. clone the project

   ```
   git clone https://github.com/Dylan-CS/COMP7103_Bitcoin_Price_Prediction_LSTM.git
   ```

2. open the file `bitcoin_price_prediction-lstm.ipynb` in jupyterLab 

3. Make sure you have download necessary libraries , if not ,run these comands in your command line. 

   ```powershell
   pandas: pip install pandas
   numpy: pip install numpy
   matplotlib: pip install matplotlib
   seaborn: pip install seaborn
   plotly: pip install plotly
   scikit-learn: pip install scikit-learn
   tensorflow: pip install tensrflow
   ```

4. run the file `bitcoin_price_prediction-lstm.ipynb` in jupyterLab and you can see all the processes:

   ```
   1. Import Libraries Needed for the data mining project
   2. Data Collection,Cleaning and Preparation
   3. Exploratory Data Analysis & Feature Engineering
   4. Splitting the Time-series Data
   5. Scaling Data using Min-Max scaler
   6. Model Building
   7. Prediction & Analysis
   ```

   

