# Bitcoin Stock Price Prediction using LSTM

## Project Overview

This project is a course assignment for the Data Mining course (COMP7103C) at HKU. The goal is to predict Bitcoin stock prices using Long Short-Term Memory (LSTM) neural networks. The project involves data collection, cleaning, exploratory data analysis, feature engineering, and model building.

**Team**: Yuxi CHEN (Dylan)

**Source Code**: [bitcoin_price_prediction-lstm.ipynb](https://github.com/Dylan-CS/COMP7103_Bitcoin_Price_Prediction_LSTM/blob/main/bitcoin_price_prediction-lstm.ipynb)

**Dataset**: [BTC-USD.csv](https://github.com/Dylan-CS/COMP7103_Bitcoin_Price_Prediction_LSTM/blob/main/BTC-USD.csv)

### Dataset Description

The Bitcoin price dataset includes historical data points representing the price of Bitcoin. The dataset features include:

- **Date**: The date of the recorded Bitcoin price.
- **Open**: The price at the beginning of the trading day.
- **Close**: The price at the end of the trading day.
- **Adj. Close**: The adjusted close price accounting for corporate actions.
- **High**: The highest price during the trading day.
- **Low**: The lowest price during the trading day.

These features are used for financial analysis, prediction, and decision-making. By analyzing historical data, patterns and trends can be identified to predict future prices.

## Instructions to Run the Code

If you encounter any issues, you can view the results in `bitcoin_price_prediction-lstm.html` or contact us at chenyuxi@connect.hku.hk.

1. **Clone the Project**

   ```bash
   git clone https://github.com/Dylan-CS/COMP7103_Bitcoin_Price_Prediction_LSTM.git
   ```

2. **Open the Jupyter Notebook**

   Open `bitcoin_price_prediction-lstm.ipynb` in JupyterLab.

3. **Install Required Libraries**

   Ensure you have the necessary libraries installed. Run the following commands in your command line:

   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn tensorflow
   ```

4. **Run the Jupyter Notebook**

   Execute the notebook to see the entire process:

   1. Import Libraries Needed for the data mining project
   2. Data Collection, Cleaning, and Preparation
   3. Exploratory Data Analysis & Feature Engineering
   4. Splitting the Time-series Data
   5. Scaling Data using Min-Max scaler
   6. Model Building
   7. Prediction & Analysis

# Additional resource
jupyter nbconvert --to script bitcoin_price_prediction-lstm.ipynb; pipreqs .

