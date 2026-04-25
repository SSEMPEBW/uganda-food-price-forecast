import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import logging
import warnings
import os

# CRITICAL: Kill Prophet logging BEFORE importing Prophet
os.environ['CMDSTAN_DEBUG'] = '0'
logging.getLogger('prophet').setLevel(logging.CRITICAL)
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

from prophet import Prophet

st.set_page_config(page_title="Zero Hunger - Uganda Food Price Forecast")

st.title("Zero Hunger: Uganda Food Price Forecast")
st.write("Predicting food prices 30 days ahead to fight hunger in Uganda")

@st.cache_data
def load_data():
    df = pd.read_csv('data/wfp_food_prices_uga.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'admin1': 'district', 'date': 'month', 'price': 'value'})
    df = df.dropna(subset=['district', 'commodity', 'value'])
    return df

df = load_data()

st.subheader("Raw WFP Data Preview")
st.dataframe(df.tail())

KEY_CROPS = ['Maize', 'Beans', 'Rice', 'Cassava', 'Matooke', 'Sorghum']
df = df[df['commodity'].isin(KEY_CROPS)]

district_list = sorted(df['district'].dropna().unique())
district_input = st.selectbox("1. Select District", district_list)
df_district = df[df['district'] == district_input].copy()

st.subheader("30-Day Price Forecast")

available_crops = sorted(df_district['commodity'].unique())
if len(available_crops) == 0:
    st.error(f"No data for major crops in {district_input}. Pick another district.")
    st.stop()

selected_crop = st.selectbox("2. Select crop to forecast", available_crops)
df_crop = df_district[df_district['commodity'] == selected_crop][['month', 'value']].copy()

df_prophet = df_crop.rename(columns={'month': 'ds', 'value': 'y'})
df_prophet = df_prophet.sort_values('ds')
df_prophet = df_prophet.groupby('ds')['y'].mean().reset_index()

if len(df_prophet) < 4:
    st.warning(f"Not enough data points for {selected_crop} in {district_input}. Need at least 4 months.")
    st.dataframe(df_prophet)
else:
    # Disable logging right before Prophet init
    logging.disable(logging.CRITICAL)
    
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_prophet)
    
    # Re-enable logging after Prophet
    logging.disable(logging.NOTSET)
    
    future = m.make_future_dataframe(periods=30, freq='D')
    forecast = m.predict(future)
    
    st.write(f"### Forecast for {selected_crop} in {district_input}")
    fig1 = m.plot(forecast)
    plt.ylabel('Price (UGX)')
    plt.xlabel('Date')
    st.pyplot(fig1)
    
    st.write("### Forecast Components")
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)
    
    st.write("### Next 30 Days Price Prediction")
    forecast_table = forecast[['ds', 'yhat_lower', 'yhat_upper']].tail(30)
    forecast_table = forecast_table.rename(columns={
        'ds': 'Date', 'yhat': 'Predicted_Price', 
        'yhat_lower': 'Lower_Bound', 'yhat_upper': 'Upper_Bound'
    })
    forecast_table['Date'] = forecast_table['Date'].dt.date
    st.dataframe(forecast_table, use_container_width=True)