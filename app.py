import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from prophet import Prophet

st.title("Uganda Food Price Forecast")
st.caption("Built for WFP/FAO - SDG 2: Zero Hunger")
st.caption("By Allan Ssempebwa, for the God of Abraham, Isaac, and Israel")

try:
    df = pd.read_csv('data/prices.csv')
    st.success(f"CSV loaded: {len(df['district'].unique())} districts, {len(df)} records")
except FileNotFoundError:
    st.error("ERROR: prices.csv not found in data folder")
    st.stop()

district_list = df['district'].unique()
district_input = st.selectbox("Select district to forecast", district_list)

# Filter by district first
df_district = df[df['district'] == district_input].copy()

# 30-Day Price Forecast with Prophet
st.subheader("30-Day Price Forecast")

# These are your actual columns from the screenshot
crop_options = ['maize_kg', 'beans_kg', 'matooke_bunch']
selected_crop = st.selectbox("Select crop to forecast", crop_options)

# Build Prophet dataframe: 'month' is your date, selected_crop is your price
df_prophet = df_district[['month', selected_crop]].rename(columns={'month': 'ds', selected_crop: 'y'})

# Convert month to datetime. If your months are like "2024-01", this works. If just "January", we fix next.
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
df_prophet = df_prophet.dropna()

if len(df_prophet) < 2:
    st.error("Not enough data points for this crop/district to forecast. Need at least 2 months.")
    st.stop()
else:
    # Train Prophet model
    m = Prophet(yearly_seasonality=True)
    m.fit(df_prophet)
    
    # Predict 30 days into future
    future = m.make_future_dataframe(periods=30, freq='D')
    forecast = m.predict(future)
    
    # Show forecast chart
    fig = m.plot(forecast)
    st.pyplot(fig)
    
    # Show predicted price 30 days from now
    next_30 = forecast[['ds', 'yhat']].tail(30)
    st.write(f"Predicted price in 30 days: UGX {next_30['yhat'].iloc[-1]:.0f}")