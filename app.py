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

df_district = df[df['district'] == district_input].copy()

st.subheader("30-Day Price Forecast")
crop_options = ['maize_kg', 'beans_kg', 'matooke_bunch']
selected_crop = st.selectbox("Select crop to forecast", crop_options)

df_prophet = df_district[['month', selected_crop]].rename(columns={'month': 'ds', selected_crop: 'y'})
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
df_prophet = df_prophet.dropna()

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
