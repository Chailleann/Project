#Importing required packages
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import decimal
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date
import yfinance as yf

#Defining the day from which we start extracting data (START), and we stop extraction with TODAY's value
START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
#Naming our application
st.title('Web Application to Predict Stock Prices')
#Choosing stocks available for analysis
#Could choose more stocks but it seems to correlate with loading time, so we chose the most famous and trending ones
stocks = ('AAPL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'FB', 'BTC-USD', 'ETH-USD', 'USDT-USD')
selected_stock = st.selectbox('Select dataset for prediction', stocks)
#Defining the prediction period 
#Could define longer period but it correlates with loading time because of extra calculations
n_years = st.slider('Years of prediction:', 1, 3)
period = n_years * 365

#Defining a function to extract all data from Yahoo Finance website
#We use @st.cache to keep loaded data in cache so it would not have to load it twice when we change the stock back
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

#Indicating when the data extraction is in process and when it is completed
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data is completed')

#This creates a checkbox on the right
show_raw_data= st.sidebar.checkbox('Show Stock Data Table')
#If you tick the checkbox, you'll see the raw data
if show_raw_data:
    st.subheader('Stock Data Table')
    st.write(data.tail())
    
#Creating a plot for Stock Data
#Two lines, one is opening price, another is closing price
#Adding a Rangeslider so you can zoom the graph in and out
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series Stock Data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

plot_raw_data()    
  
#Here we will do a short statistical analysis of the stock
st.subheader(f"Statistical values for {selected_stock}:")
#Calculate by how many times the stock has increased in price from its historical minimum to historical maximum
growth_total_rounded=st.write(f"The price of {selected_stock} increased by", round(np.max(data['Adj Close'])/np.min(data['Adj Close']),2), "times")
#Calculate aggreagate values: min, max, standard deviation, round them to two decimals. Based on Adj Close Price
min_rounded = st.write(f"Historical minimum of {selected_stock}:", round(np.min(data['Adj Close']),2), 'USD')
max_rounded = st.write(f"Historical maximum of {selected_stock}:", round(np.max(data['Adj Close']),2), 'USD')
std_rounded = st.write(f"Historical standard deviation of {selected_stock}:", round(np.std(data['Adj Close']),2), 'USD')

#Creating dataframe with standard deviation calculated periodically
#The period can be chosen in the selectbox on the right, created right below this comment
trend_level = st.sidebar.selectbox("Time Period for St Dev", ["Weekly", "Monthly", "Quarterly", "Annually"])
#Giving dataframe a title
st.markdown(f"**Standard deviation {trend_level} for {selected_stock}:**") 
#Creating dataframe using grouping and aggregation. Trends are denoted in pandas syntaxis for date-based grouping
trend_kwds = {"Weekly": "1W", "Monthly": "1M", "Quarterly": "1Q", "Annually": "1Y"}
df1=data[data.notna()]
trend_data = df1.\
    groupby(pd.Grouper(key="Date", 
    freq=trend_kwds[trend_level])).aggregate(
    Open=("Open", "std"),
    Close = ("Close", "std"),
    High = ("High", "std"),
    Low = ("Low", "std")).reset_index()

#Showing the dataframe
st.write(trend_data)      

# Using Prophet to forecast stock price
#Creating dataframe with xs and ys Prophet will use for prediction
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

#Defining Prophet and creating the forecasted data
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

#This creates a checkbox on the right
show_forecasted_data = st.sidebar.checkbox('Show Forecasted Data Table')

#If you tick the checkbox, you'll see the forecasted data
if show_forecasted_data:
    st.subheader('Forecasted Data Table')
    st.write(forecast.tail())

#Creating a title for the plot
st.subheader(f"Forecast plot for {n_years} years for {selected_stock}:")
#Plotting the forecast trend
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

#Plotting the components of the forecast
st.subheader(f"Forecast components for {selected_stock}:")
fig2 = m.plot_components(forecast)
st.write(fig2)
