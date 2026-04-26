import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

st.set_page_config(page_title="Zero Hunger T420 - By Ssempebwa Allan", page_icon="🌾", layout="wide")

st.markdown("""
<div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #228B22, #FF8C00); border-radius: 10px; margin-bottom: 1rem;'>
    <h1 style='color: white; margin: 0;'>🌾 Zero Hunger T420</h1>
    <h3 style='color: white; margin: 0;'>Food Price Forecast System</h3>
    <p style='color: white; margin: 0.5rem 0 0; font-size: 18px;'><b>Built by Ssempebwa Allan</b> | Fighting Hunger with Data Science 🇺🇬</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('data/wfp_food_prices_uga.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'admin1': 'district', 'date': 'month', 'price': 'value'})
    return df.dropna(subset=['district', 'commodity', 'value'])

df = load_data()
KEY_CROPS = ['Maize', 'Beans', 'Rice', 'Cassava', 'Matooke', 'Sorghum']
df = df[df['commodity'].isin(KEY_CROPS)]

st.subheader("📈 30-Day Price Forecast")
col1, col2 = st.columns(2)

with col1:
    district_list = sorted(df['district'].dropna().unique())
    district_input = st.selectbox("1. Select District", district_list)

df_district = df[df['district'] == district_input].copy()

with col2:
    available_crops = sorted(df_district['commodity'].unique())
    if len(available_crops) == 0:
        st.error(f"No data for major crops in {district_input}")
        st.stop()
    selected_crop = st.selectbox("2. Select Crop", available_crops)

df_crop = df_district[df_district['commodity'] == selected_crop][['month', 'value']].copy()
df_crop = df_crop.groupby('month')['value'].mean().reset_index()
df_crop = df_crop.set_index('month').sort_index()

if len(df_crop) < 6:
    st.error(f"❌ Not enough data. Only {len(df_crop)} months found.")
    st.stop()

df_crop = df_crop.resample('MS').mean()
df_crop['value'] = df_crop['value'].interpolate(method='linear')

model = ExponentialSmoothing(df_crop['value'], trend='add', seasonal_periods=12)
fit = model.fit(optimized=True)
forecast = fit.forecast(30)

last_price = df_crop['value'].iloc[-1]
next_price = forecast.iloc[0]
price_change = ((next_price - last_price) / last_price) * 100

m1, m2, m3 = st.columns(3)
m1.metric("Current Price", f"{last_price:.0f} UGX")
m2.metric("30-Day Forecast", f"{next_price:.0f} UGX", f"{price_change:+.1f}%")
m3.metric("Data Points", f"{len(df_crop)} months")

if price_change > 10:
    st.warning(f"⚠️ **Price Alert**: {selected_crop} prices predicted to rise {price_change:.1f}%")

fig, ax = plt.subplots(figsize=(12, 6))
df_crop['value'].plot(ax=ax, label='Historical', linewidth=2.5, color='#1f77b4')
forecast.plot(ax=ax, label='30-Day Forecast', linewidth=2.5, linestyle='--', color='#ff7f0e')
plt.ylabel('Price (UGX)', fontsize=12)
plt.title(f'{selected_crop} Price Forecast - {district_input}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
st.pyplot(fig)

forecast_df = forecast.reset_index()
forecast_df.columns = ['Date', 'Predicted_Price_UGX']
forecast_df['Date'] = forecast_df['Date'].dt.date
st.dataframe(forecast_df, use_container_width=True, hide_index=True)
st.download_button("📥 Download CSV", forecast_df.to_csv(index=False),
                  f"{district_input}_{selected_crop}_forecast.csv", "text/csv")

st.divider()
st.markdown("""
<div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px;'>
    <p style='margin: 0;'><b>Zero Hunger T420</b> | Built by <b>Ssempebwa Allan</b></p>
    <p style='margin: 0; font-size: 14px;'>Data: WFP Uganda | Fighting Hunger with Data Science 🇺🇬</p>
</div>
""", unsafe_allow_html=True)