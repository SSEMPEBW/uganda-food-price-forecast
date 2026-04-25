import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Zero Hunger - Uganda Food Price Forecast", layout="wide")

st.title("Zero Hunger: Uganda Food Price Forecast")
st.write("Predicting food prices 30 days ahead to fight hunger in Uganda")

@st.cache_data
def load_data():
    df = pd.read_csv('food_prices.csv')
    return df

df = load_data()

st.subheader("Raw Data")
st.dataframe(df.tail())

district_input = st.selectbox("Select District", df['district'].unique())
df_district = df[df['district'] == district_input].copy()

st.subheader("30-Day Price Forecast")
crop_options = ['maize_kg', 'beans_kg', 'matooke_bunch']
selected_crop = st.selectbox("Select crop to forecast", crop_options)

df_prophet = df_district[['month', selected_crop]].rename(columns={'month': 'ds', selected_crop: 'y'})
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

st.subheader("Forecast Chart")
fig = m.plot(forecast)
st.pyplot(fig)

next_30 = forecast[['ds', 'yhat']].tail(30)
last_pred = next_30['yhat'].iloc[-1]
st.subheader(f"Predicted price in 30 days: UGX {last_pred:,.0f}")

st.dataframe(next_30)