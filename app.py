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

# Debug: shows your real column names on the live app
st.write("Your columns:", df.columns.tolist())

# Prophet needs 'ds' and 'y'. CHANGE THESE 3 LINES after you see "Your columns:" above
date_col = 'date'      # <- change to your actual date column
price_col = 'price'    # <- change to your actual price column  
crop_col = 'crop'      # <- change to your actual crop/commodity column

# Crop dropdown - uses the crop_col you set above
selected_crop = st.selectbox("Select crop to forecast", df_district[crop_col].unique())

# Filter data for selected crop + rename columns for Prophet
df_prophet = df_district[df_district[crop_col] == selected_crop][[date_col, price_col]].rename(columns={date_col: 'ds', price_col: 'y'})

# Make sure date is datetime format
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
df_prophet = df_prophet.dropna()

if len(df_prophet) < 2:
    st.error("Not enough data points for this crop/district to forecast. Need at least 2 records.")
    st.stop()
else:
    # Train Prophet model
    m = Prophet(daily_seasonality=True)
    m.fit(df_prophet)
    
    # Predict 30 days into future
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    
    # Show forecast chart
    fig = m.plot(forecast)
    st.pyplot(fig)
    
    # Show predicted price 30 days from now
    next_30 = forecast[['ds', 'yhat']].tail(30)
    st.write(f"Predicted price in 30 days: UGX {next_30['yhat'].iloc[-1]:.0f}")