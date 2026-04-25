import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Zero Hunger - Uganda Food Price Forecast")

st.title("Zero Hunger: Uganda Food Price Forecast")
st.write("Predicting food prices 30 days ahead to fight hunger in Uganda")

@st.cache_data
def load_data():
    df = pd.read_csv('data/wfp_food_prices_uga.csv')
    # WFP dataset cleanup
    df['date'] = pd.to_datetime(df['date'])
    # Rename WFP columns to what our app expects
    df = df.rename(columns={
        'admin1': 'district',
        'date': 'month',
        'price': 'value'
    })
    # Keep only USD or UGX prices, drop NaN
    df = df.dropna(subset=['district', 'commodity', 'value'])
    return df

df = load_data()

st.subheader("Raw WFP Data Preview")
st.dataframe(df.tail())

# Filter to key crops for Uganda Zero Hunger goal
KEY_CROPS = ['Maize', 'Beans', 'Rice', 'Cassava', 'Matooke', 'Sorghum']
df = df[df['commodity'].isin(KEY_CROPS)]

# District selector
district_list = sorted(df['district'].dropna().unique())
district_input = st.selectbox("1. Select District", district_list)
df_district = df[df['district'] == district_input].copy()

st.subheader("30-Day Price Forecast")

# Crop selector - only show crops available in that district
available_crops = sorted(df_district['commodity'].unique())
if len(available_crops) == 0:
    st.error(f"No data for major crops in {district_input}. Pick another district.")
    st.stop()

selected_crop = st.selectbox("2. Select crop to forecast", available_crops)
df_crop = df_district[df_district['commodity'] == selected_crop][['month', 'value']].copy()

# Prepare for Prophet: needs 'ds' and 'y' columns
df_prophet = df_crop.rename(columns={'month': 'ds', 'value': 'y'})
df_prophet = df_prophet.sort_values('ds')
# Aggregate by month if multiple markets per district
df_prophet = df_prophet.groupby('ds')['y'].mean().reset_index()

if len(df_prophet) < 4:
    st.warning(f"Not enough data points for {selected_crop} in {district_input}. Need at least 4 months. Pick another crop/district.")
    st.dataframe(df_prophet)
else:
    # Fit Prophet model
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_prophet)
    
    # Forecast 30 days ahead
    future = m.make_future_dataframe(periods=30, freq='D')
    forecast = m.predict(future)
    
    # Plot forecast
    st.write(f"### Forecast for {selected_crop} in {district_input}")
    fig1 = m.plot(forecast)
    plt.ylabel('Price')
    plt.xlabel('Date')
    st.pyplot(fig1)
    
    # Plot components
    st.write("### Forecast Components")
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)
    
    # Show forecast table
    st.write("### Next 30 Days Price Prediction")
    forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
    forecast_table = forecast_table.rename(columns={
        'ds': 'Date',
        'yhat': 'Predicted_Price',
        'yhat_lower': 'Lower_Bound',
        'yhat_upper': 'Upper_Bound'
    })
    st.dataframe(forecast_table)