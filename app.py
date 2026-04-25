import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Zero Hunger - Uganda Food Price Forecast", layout="wide")

st.title("Zero Hunger: Uganda Food Price Forecast")
st.write("Predicting food prices 30 days ahead to fight hunger in Uganda")

# Load data - replace 'food_prices.csv' with your actual file
@st.cache_data
def load_data():
    df = pd.read_csv('food_prices.csv')
    return df

df = load_data()

st.subheader("Raw Data")
st.dataframe(df.tail())

# Prepare data for Prophet
df_prophet = df[['Date', 'Price']].copy()
df_prophet.columns = ['ds', 'y']

# Make sure date is datetime format
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

# Train Prophet model - FIXED LINE 107
m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
m.fit(df_prophet)

# Predict 30 days into future
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

# Show forecast chart
st.subheader("30-Day Price Forecast")
fig = m.plot(forecast)
st.pyplot(fig)

# Show predicted price 30 days from now
next_30 = forecast[['ds', 'yhat']].tail(30)
last_pred = next_30['yhat'].iloc[-1]
st.subheader(f"Predicted price in 30 days: UGX {last_pred:,.0f}")

st.dataframe(next_30)