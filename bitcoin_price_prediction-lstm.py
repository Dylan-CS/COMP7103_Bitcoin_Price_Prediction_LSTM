#!/usr/bin/env python
# coding: utf-8

# <!--  <center><h1>Bitcoin Stock Price Prediction using LSTM</h1></center>  -->
#  <div style="padding:10px; 
#             color:#FFFFFF;
#             margin:10px;
#             font-size:220%;
#             text-align:center;
#             display:fill;
#             border-radius:20px;
#             border-width: 5px;
#             border-style: solid;
#             border-color: #FF5733;
#             background-color:#2C3E50;
#             overflow:hidden;
#             font-weight:500">Bitcoin Stock Price Prediction using LSTM</div>

# <div style="text-align:center">
#     <img src="bitcoin.png" alt="image" style="width:100%"  height="200"/>
# </div>

# <div style="padding:10px; 
#             color:#FFFFFF;
#             margin:10px;
#             font-size:220%;
#             text-align:center;
#             display:fill;
#             border-radius:20px;
#             border-width: 5px;
#             border-style: solid;
#             border-color: #FF5733;
#             background-color:#2C3E50;
#             overflow:hidden;
#             font-weight:500">OVERVIEW</div>

# The Bitcoin price dataset refers to a collection of data points that represent the historical or current price of Bitcoin, a decentralized digital currency based on blockchain technology. The data points typically include information such as the date, time, and the opening, closing, highest and lowest prices of Bitcoin during a specific period of time. The Bitcoin price dataset can be used for a variety of purposes, such as studying market trends, conducting technical analysis, or training machine learning models for Bitcoin price prediction. The data can be obtained from various financial sources such as cryptocurrency exchanges, financial websites, or by directly accessing APIs provided by cryptocurrency data providers.
# 
# The following are the common features found in a Bitcoin price dataset:
# 
# * **Date:** The date on which the Bitcoin price data was recorded.
# * **Open:** This refers to the price of Bitcoin at the beginning of the trading day.
# * **Close:** This refers to the price of Bitcoin at the end of the trading day.
# * **Adj. Close:** The adjusted close price accounts for any corporate actions such as stock splits, dividends, etc. that occurred on that day
# * **High:** The highest price of Bitcoin during the trading day.
# * **Low:** The lowest price of Bitcoin during the trading day.
# 
# These features can provide valuable information about the Bitcoin performance, trends and volatility over a certain period of time, and can be used in financial analysis, prediction, and decision making. By analyzing the historical Bitcoin price data, one can identify patterns and trends, which can be used to predict the future prices of Bitcoin. Machine learning models can be trained using this data to make accurate predictions and provide insights into the Bitcoin market.

# <div style="padding:10px; 
#             color:#FFFFFF;
#             margin:10px;
#             font-size:220%;
#             text-align:center;
#             display:fill;
#             border-radius:20px;
#             border-width: 5px;
#             border-style: solid;
#             border-color: #FF5733;
#             background-color:#2C3E50;
#             overflow:hidden;
#             font-weight:500">TABLE OF CONTENTS</div>

# ## 1. Import Libraries Needed for the data mining project
# ## 2. Data Collection,Cleaning and Preparation
# ## 3. Exploratory Data Analysis & Feature Engineering
# ## 4. Splitting the Time-series Data
# ## 5. Scaling Data using Min-Max scaler
# ## 6. Model Building
# ## 7. Prediction & Analysis

# <div style="padding:10px; 
#             color:#FFFFFF;
#             margin:10px;
#             font-size:220%;
#             text-align:center;
#             display:fill;
#             border-radius:20px;
#             border-width: 5px;
#             border-style: solid;
#             border-color: #FF5733;
#             background-color:#2C3E50;
#             overflow:hidden;
#             font-weight:500">1.Import Libraries Needed for the data mining project</div>

# In[1]:


import pandas as pd  #for data manipulation operations
import numpy as np   #for linear algebra

#Libraries for visualisation
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import datetime as dt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM

from itertools import cycle


# <div style="padding:10px; 
#             color:#FFFFFF;
#             margin:10px;
#             font-size:220%;
#             text-align:center;
#             display:fill;
#             border-radius:20px;
#             border-width: 5px;
#             border-style: solid;
#             border-color: #FF5733;
#             background-color:#2C3E50;
#             overflow:hidden;
#             font-weight:500">2.Data Collection,Cleaning and Preparation</div>

# In[2]:


#Loading the required data
df=pd.read_csv('BTC-USD.csv')
df.set_index('Date',inplace=True)
df.head()


# In[3]:


print('Number of days present in the dataset: ',df.shape[0])
print('Number of fields present in the dataset: ',df.shape[1])


# In[4]:


df.info()


# In[5]:


df.describe()


# <div style="padding:10px; 
#             color:#FFFFFF;
#             margin:10px;
#             font-size:220%;
#             text-align:center;
#             display:fill;
#             border-radius:20px;
#             border-width: 5px;
#             border-style: solid;
#             border-color: #FF5733;
#             background-color:#2C3E50;
#             overflow:hidden;
#             font-weight:500">3.Exploratory Data Analysis & Feature Engineering</div>

# In[6]:


from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)


# ## 3.1 Check for Null values 

# In[7]:


df.isnull().sum()


# ## 3.2 Plot the price over time

# In[8]:


data =df
plt.figure(figsize=(30,15))

# plot the 'Close' price column
plt.plot(data['Close'])

# set the title and axis labels
plt.title('Bitcoin Price over Time')
plt.xlabel('Date')
plt.ylabel('Price ($)')

# set x-axis tick labels to show only 4 dates
ticklabels = data.index[::len(data)//4]
plt.xticks(ticklabels)

# show the plot
plt.show()


# In[9]:


names = cycle(['Stock Open Price','Stock High Price','Stock Low Price','Stock Close Price'])
fig = px.line(data, x=data.index, y=[data['Open'],data['High'], data['Low'],data['Close']],
             labels={'date': 'Date','value':'Stock value'})
fig.update_layout(title_text='Stock Analysis', font_size=15, font_color='black',legend_title_text='Stock Parameters')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.show()


# ## 3.3 Moving Averages

# Moving Averages (MA) are a type of time series analysis method used to smooth out fluctuations in data by calculating the average of a set of data over a certain period of time. This average is then shifted forward in time to provide a smoothed representation of the data that can help to identify underlying patterns or trends. There are two main types of moving averages: simple moving averages (SMA) and weighted moving averages (WMA). A simple moving average is calculated by taking the average of a set of data over a fixed period of time, while a weighted moving average gives more importance to the most recent data. Moving averages are widely used in finance, economics, and engineering to help forecast future trends and to identify buy/sell signals.
# 
# I'll take moving average for window sizes of 30,60,120 and 150 days.

# In[10]:


ma_day = [30, 60, 120,150]

for ma in ma_day:
        column_name = f"MA for {ma} days"
        data[column_name] = data['Close'].rolling(ma).mean()


# In[11]:


plt.figure(figsize=(30,15))
plt.plot(data['Close'],label='Close Price')
plt.plot(data['MA for 30 days'],label='30 days MA')
plt.plot(data['MA for 60 days'],label='60 days MA')
plt.plot(data['MA for 120 days'],label='120 days MA')
plt.plot(data['MA for 150 days'],label='150 days MA')

# set x-axis tick labels to show only 4 dates
ticklabels = data.index[::len(data)//4]
plt.xticks(ticklabels)

plt.legend()
plt.show()


# In[12]:


names = cycle(['Close Price','MA 30 days','MA 60 days','MA 120 days','MA 150 days'])

fig = px.line(data, x=data.index ,y=[data['Close'],data['MA for 30 days'],data['MA for 60 days'],data['MA for 120 days'], data['MA for 150 days']],labels={'date': 'Date','value':'Stock value'})
fig.update_layout(title_text='Moving Average Analysis', font_size=15, font_color='black',legend_title_text='Stock Parameters')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.show()


# <div style="padding:10px; 
#             color:#FFFFFF;
#             margin:10px;
#             font-size:220%;
#             text-align:center;
#             display:fill;
#             border-radius:20px;
#             border-width: 5px;
#             border-style: solid;
#             border-color: #FF5733;
#             background-color:#2C3E50;
#             overflow:hidden;
#             font-weight:500">4. Splitting the Time-series Data</div>

# In[13]:


# Creating a new dataframe with only 'Close'
new_df = data['Close']
new_df.index = data.index

final_df=new_df.values

train_data=final_df[0:2000,]
test_data=final_df[2000:,]

train_df = pd.DataFrame()
test_df = pd.DataFrame()

train_df['Close'] = train_data
train_df.index = new_df[0:2000].index
test_df['Close'] = test_data
test_df.index = new_df[2000:].index


# In[14]:


print("train_data: ", train_df.shape)
print("test_data: ", test_df.shape)


# <div style="padding:10px; 
#             color:#FFFFFF;
#             margin:10px;
#             font-size:220%;
#             text-align:center;
#             display:fill;
#             border-radius:20px;
#             border-width: 5px;
#             border-style: solid;
#             border-color: #FF5733;
#             background-color:#2C3E50;
#             overflow:hidden;
#             font-weight:500">5.SCALING DATA USING MIN-MAX SCALER</div>

# Min-Max Scaler is a pre-processing technique used in machine learning for rescaling a feature or a set of features to a specific range, typically between 0 and 1. The method works by transforming the values of the feature to a new scale, while preserving the relative proportions between the values. The rescaling is done by subtracting the minimum value in the feature from each data point and dividing the result by the range (the difference between the maximum and minimum value). This ensures that all the values in the feature are now in the specified range, with 0 being the minimum and 1 being the maximum. Min-Max scaling is particularly useful when working with algorithms that make assumptions about the scale of the input features, such as some distance-based algorithms or algorithms sensitive to the scale of the input features.

# In[15]:


# Using Min-Max scaler to scale data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_df.reshape(-1,1))

X_train_data,y_train_data=[],[]

for i in range(60,len(train_df)):
    X_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])
    
X_train_data,y_train_data=np.array(X_train_data),np.array(y_train_data)

X_train_data=np.reshape(X_train_data,(X_train_data.shape[0],X_train_data.shape[1],1))


# <div style="padding:10px; 
#             color:#FFFFFF;
#             margin:10px;
#             font-size:220%;
#             text-align:center;
#             display:fill;
#             border-radius:20px;
#             border-width: 5px;
#             border-style: solid;
#             border-color: #FF5733;
#             background-color:#2C3E50;
#             overflow:hidden;
#             font-weight:500">6.MODEL BUILDING</div>

# LSTMs are commonly used for modeling time series data as they are able to capture the long-term dependencies between inputs, while also being able to handle the noise and volatility that is often present in time series data. This makes LSTMs suitable for prediction tasks such as stock prices, weather forecasts, and energy demand. In a time series context, LSTMs take in previous time steps as inputs, and use their memory cells, gates, and state updates to process and make predictions on future time steps.

# In[16]:


# Initializing the LSTM model
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train_data.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))


# In[17]:


model.summary()


# In[18]:


model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X_train_data, y_train_data, epochs = 150, batch_size = 32);


# <div style="padding:10px; 
#             color:#FFFFFF;
#             margin:10px;
#             font-size:220%;
#             text-align:center;
#             display:fill;
#             border-radius:20px;
#             border-width: 5px;
#             border-style: solid;
#             border-color: #FF5733;
#             background-color:#2C3E50;
#             overflow:hidden;
#             font-weight:500">7.PREDICTIONS</div>

# In[19]:


input_data=new_df[len(new_df)-len(test_df)-60:].values
input_data=input_data.reshape(-1,1)
input_data=scaler.transform(input_data)


# In[20]:


X_test=[]
for i in range(60,input_data.shape[0]):
    X_test.append(input_data[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))


# In[21]:


predicted=model.predict(X_test)
predicted=scaler.inverse_transform(predicted)


# In[22]:


test_df['Predictions']=predicted


# In[23]:


plt.figure(figsize=(50,10))
plt.plot(train_df['Close'],label='Training Data')
plt.plot(test_df['Close'],label='Test Data')
plt.plot(test_df['Predictions'],label='Prediction')
ticklabels = data.index[::len(data)//4]
plt.xticks(ticklabels)
plt.legend()
plt.show()


# In[24]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=train_df.index,y=train_df['Close'],
                    mode='lines',
                    name='Training Data'))
fig.add_trace(go.Scatter(x=test_df.index,y=test_df['Close'],
                    mode='lines',
                    name='Test Data'))
fig.add_trace(go.Scatter(x=test_df.index,y=test_df['Predictions'],
                    mode='lines',
                    name='Prediction'))


# Root Mean Square Error (RMSE), Mean Square Error (MSE) and Mean absolute Error (MAE) are a standard way to measure the error of a model in predicting quantitative data.

# In[25]:


from sklearn.metrics import r2_score
print('The Mean Squared Error is',mean_squared_error(test_df['Close'].values,test_df['Predictions'].values))
print('The Mean Absolute Error is',mean_absolute_error(test_df['Close'].values,test_df['Predictions'].values))
print('The Root Mean Squared Error is',np.sqrt(mean_squared_error(test_df['Close'].values,test_df['Predictions'].values)))
print('The R-squared is:', r2_score(test_df['Close'].values, test_df['Predictions'].values))


# ### Analysis:
# The prediction model is based on a time series analysis of Bitcoin's stock price data. The input data is preprocessed by scaling and transforming it into a format suitable for the LSTM model. The model is trained using the training dataset with 150 epochs and 32 batches. The model's accuracy is evaluated using Root Mean Square Error (RMSE), Mean Square Error (MSE), Mean Absolute Error (MAE), and R-squared metrics.
#   
# The predicted values are obtained by passing the test data into the trained model. The predicted values are then transformed back to their original scale using the inverse scaler. The plot of the training data, test data, and predicted values is created using Matplotlib and Plotly libraries.
# 
# The RMSE, MSE, and MAE metrics are standard ways to measure the error of a model in predicting quantitative data. A lower value for these metrics indicates a more accurate model. The R-squared metric measures the proportion of variance in the dependent variable (i.e., the stock price) that is predictable from the independent variable (i.e., time). A higher value for R-squared indicates a better fit of the model.
# 
# In the given code, the RMSE is 5861.84, the MAE is 3781.74, and the MSE is 3.47e+07. Additionally, the R-squared value is 0.904. These values suggest that the model has some degree of accuracy in predicting Bitcoin's stock price. 
