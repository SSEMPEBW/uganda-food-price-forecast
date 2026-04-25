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

<<<<<<< HEAD
df = load_data()

st.subheader("Raw Data")
st.dataframe(df.tail())

# Prepare data for Prophet
df_prophet = df[['Date', 'Price']].copy()
df_prophet.columns = ['ds', 'y']

# Make sure date is datetime format
=======
df_district = df[df['district'] == district_input].copy()

st.subheader("30-Day Price Forecast")
crop_options = ['maize_kg', 'beans_kg', 'matooke_bunch']
selected_crop = st.selectbox("Select crop to forecast", crop_options)

df_prophet = df_district[['month', selected_crop]].rename(columns={'month': 'ds', selected_crop: 'y'})
>>>>>>> 06fde481b16bbfb4e850b45206931d5a78817d6e
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

<<<<<<< HEAD
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
=======
if len(df_prophet) < 2:
    st.error("Not enough data points for this crop/district to forecast. Need at least 2 months.")
    st.stop()
else:
    # MISSING ITEM 1 FIX: Disable weekly/daily seasonality
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_prophet)
    
    # MISSING ITEM 2 FIX: Use monthly if your data is monthly
    future = m.make_future_dataframe(periods=1, freq='M')
    forecast = m.predict(future)
    
    fig = m.plot(forecast)
    st.pyplot(fig)
    
    st.write(f"Predicted price next month: UGX {forecast['yhat'].iloc[-1]:.0f}")
>>>>>>> 06fde481b16bbfb4e850b45206931d5a78817d6e
