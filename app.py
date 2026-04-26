import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

st.set_page_config(page_title="Zero Hunger - Uganda Food Price Forecast")
st.title("Zero Hunger: Uganda Food Price Forecast")
st.write("Predicting food prices 30 days ahead to fight hunger in Uganda")

@st.cache_data
def load_data():
    df = pd.read_csv('data/wfp_food_prices_uga.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'admin1': 'district', 'date': 'month', 'price': 'value'})
    return df.dropna(subset=['district', 'commodity', 'value'])

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
df_crop = df_crop.groupby('month')['value'].mean().reset_index()
df_crop = df_crop.set_index('month').asfreq('MS').fillna(method='ffill')

if len(df_crop) < 4:
    st.warning(f"Not enough data points for {selected_crop} in {district_input}. Need at least 4 months.")
    st.dataframe(df_crop)
else:
    model = ExponentialSmoothing(
        df_crop['value'], 
        trend='add', 
        seasonal='add', 
        seasonal_periods=12
    )
    fit = model.fit()
    forecast = fit.forecast(30)
    
    st.write(f"### Forecast for {selected_crop} in {district_input}")
    fig, ax = plt.subplots(figsize=(10, 6))
    df_crop['value'].plot(ax=ax, label='Historical Price', linewidth=2)
    forecast.plot(ax=ax, label='30-Day Forecast', linewidth=2, linestyle='--')
    plt.ylabel('Price (UGX)')
    plt.xlabel('Date')
    plt.legend()
    plt.title(f'{selected_crop} Price Forecast')
    st.pyplot(fig)
    
    st.write("### Next 30 Days Price Prediction")
    forecast_df = forecast.reset_index()
    forecast_df.columns = ['Date', 'Predicted_Price']
    forecast_df['Date'] = forecast_df['Date'].dt.date
    st.dataframe(forecast_df, use_container_width=True)