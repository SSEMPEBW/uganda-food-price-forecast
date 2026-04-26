import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import folium
from streamlit_folium import st_folium
import requests

st.set_page_config(
    page_title="Zero Hunger T420 - By Ssempebwa Allan",
    page_icon="🌾",
    layout="wide"
)

st.markdown("""
<div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #228B22, #FF8C00); border-radius: 10px; margin-bottom: 1rem;'>
    <h1 style='color: white; margin: 0;'>🌾🌧️ Zero Hunger T420</h1>
    <h3 style='color: white; margin: 0;'>Food Price + Rainfall Forecast System</h3>
    <p style='color: white; margin: 0.5rem 0 0; font-size: 18px;'><b>Built by Ssempebwa Allan</b> | Fighting Hunger with Data Science 🇺🇬</p>
</div>
""", unsafe_allow_html=True)

st.write("Predicting food prices and rainfall 30 days ahead to fight hunger in Uganda")

@st.cache_data
def load_data():
    df = pd.read_csv('data/wfp_food_prices_uga.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'admin1': 'district', 'date': 'month', 'price': 'value'})
    return df.dropna(subset=['district', 'commodity', 'value'])

@st.cache_data
def get_rainfall_forecast(lat, lon):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": float(lat),
            "longitude": float(lon), 
            "daily": "precipitation_sum",
            "forecast_days": 16
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if 'daily' not in data or 'precipitation_sum' not in data['daily']:
            return None, "API returned no daily precipitation data"
            
        rain_df = pd.DataFrame({
            'Date': pd.to_datetime(data['daily']['time']),
            'Rainfall_mm': data['daily']['precipitation_sum']
        })
        return rain_df, "Success"
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP {e.response.status_code}: {e.response.reason}"
    except requests.exceptions.Timeout:
        return None, "API timeout - Open-Meteo took too long"
    except Exception as e:
        return None, f"Error: {str(e)}"

def create_uganda_map(df):
    m = folium.Map(location=[1.3733, 32.2903], zoom_start=6, tiles='OpenStreetMap')
    # EXPANDED DISTRICT LIST - NOW INCLUDES MUKONO
    district_coords = {
        'Kampala': [0.3476, 32.5825], 'Wakiso': [0.4000, 32.4800], 
        'Mbarara': [-0.6072, 30.6545], 'Gulu': [2.7746, 32.2989],
        'Mbale': [1.0827, 34.1755], 'Jinja': [0.4244, 33.2042],
        'Lira': [2.2350, 32.9097], 'Masaka': [-0.3331, 31.7331],
        'Arua': [3.0201, 30.9110], 'Fort Portal': [0.6710, 30.2750],
        'Hoima': [1.4356, 31.3439], 'Soroti': [1.7145, 33.6119],
        'Kabale': [-1.2486, 29.9897], 'Tororo': [0.6920, 34.1807],
        'Kasese': [0.1833, 30.0833], 'Iganga': [0.6127, 33.4833],
        'Mukono': [0.3533, 32.7520], 'Mityana': [0.4008, 32.0389], # ADDED
        'Luwero': [0.8271, 32.4950], 'Mpigi': [0.2274, 32.3285], # ADDED
        'Bushenyi': [-0.4879, 30.2026], 'Rukungiri': [-0.7918, 29.9303] # ADDED
    }
    district_counts = df['district'].value_counts()
    for district in district_counts.index:
        if district in district_coords:
            count = district_counts.get(district, 0)
            folium.CircleMarker(
                location=district_coords[district],
                radius=8 + (count / 100),
                popup=f"<b>{district}</b><br>{count} records",
                color='#228B22' if count > 100 else '#FF8C00',
                fill=True, fillColor='#228B22' if count > 100 else '#FF8C00',
                fillOpacity=0.7
            ).add_to(m)
    return m, district_coords

df = load_data()

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("🗺️ Uganda Food Price Data Coverage")
    st.write("**Green** = 100+ records | **Orange** = Limited data")
    uganda_map, district_coords = create_uganda_map(df)
    try:
        st_folium(uganda_map, width=None, height=400)
    except Exception as e:
        st.warning("🗺️ Interactive map temporarily unavailable")
        st.info("💡 All forecasts below still work perfectly. Use the dropdowns to select districts.")

with col2:
    st.subheader("📊 Data Stats")
    st.metric("Total Records", f"{len(df):,}")
    st.metric("Districts Covered", df['district'].nunique())
    st.metric("Crops Tracked", df['commodity'].nunique())
    st.metric("Date Range", f"{df['month'].dt.year.min()} - {df['month'].dt.year.max()}")

st.divider()

KEY_CROPS = ['Maize', 'Beans', 'Rice', 'Cassava', 'Matooke', 'Sorghum']
df = df[df['commodity'].isin(KEY_CROPS)]

st.subheader("📈 16-Day Forecast: Price + Rainfall")

col3, col4 = st.columns(2)
with col3:
    district_list = sorted(df['district'].dropna().unique())
    district_input = st.selectbox("1. Select District", district_list, 
                                 index=district_list.index('Kampala') if 'Kampala' in district_list else 0)

df_district = df[df['district'] == district_input].copy()

with col4:
    available_crops = sorted(df_district['commodity'].unique())
    if len(available_crops) == 0:
        st.error(f"No data for major crops in {district_input}. Pick another district.")
        st.stop()
    selected_crop = st.selectbox("2. Select Crop", available_crops)

# FIXED: Now handles districts not in coords list
rain_df, rain_status = None, "Coordinates not available for this district"
if district_input in district_coords:
    lat, lon = district_coords[district_input]
    with st.spinner(f"Fetching rainfall for {district_input}..."):
        rain_df, rain_status = get_rainfall_forecast(lat, lon)
    if rain_status!= "Success":
        st.error(f"🌧️ Rain API failed: {rain_status}")
else:
    st.warning(f"🌧️ No GPS coordinates for {district_input}. Add it to district_coords to enable rain forecast.")

df_crop = df_district[df_district['commodity'] == selected_crop][['month', 'value']].copy()
df_crop = df_crop.groupby('month')['value'].mean().reset_index()
df_crop = df_crop.set_index('month').sort_index()

if len(df_crop) < 6:
    st.error(f"❌ Not enough price data for {selected_crop} in {district_input}. Only {len(df_crop)} months found.")
    st.info("💡 Try Kampala, Wakiso, or Mbarara - they have the most complete data.")
    st.stop()

df_crop = df_crop.resample('MS').mean()
df_crop['value'] = df_crop['value'].interpolate(method='linear')

try:
    model = ExponentialSmoothing(df_crop['value'], trend='add', seasonal_periods=12)
    fit = model.fit(optimized=True)
    forecast = fit.forecast(16)
    
    last_price = df_crop['value'].iloc[-1]
    next_price = forecast.iloc[0]
    price_change = ((next_price - last_price) / last_price) * 100
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"{last_price:.0f} UGX")
    m2.metric("16-Day Forecast", f"{next_price:.0f} UGX", f"{price_change:+.1f}%")
    m3.metric("Data Points", f"{len(df_crop)} months")
    
    if rain_df is not None and not rain_df.empty:
        total_rain = rain_df['Rainfall_mm'].sum()
        m4.metric("16-Day Rain", f"{total_rain:.1f} mm")
    else:
        m4.metric("16-Day Rain", "0.0 mm")
    
    if price_change > 10:
        st.warning(f"⚠️ **Price Alert**: {selected_crop} prices predicted to rise {price_change:.1f}%")
    elif price_change < -10:
        st.success(f"📉 **Good News**: {selected_crop} prices predicted to drop {abs(price_change):.1f}%")
    
    if rain_df is not None and not rain_df.empty:
        total_rain = rain_df['Rainfall_mm'].sum()
        if total_rain < 30:
            st.error(f"🌵 **DROUGHT RISK**: Only {total_rain:.1f}mm rain expected next 16 days - High risk for {selected_crop}")
        elif total_rain > 150:
            st.info(f"🌊 **HEAVY RAIN**: {total_rain:.1f}mm expected - Potential flooding risk")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    df_crop['value'].plot(ax=ax1, label='Historical Price', linewidth=2.5, color='#1f77b4')
    forecast.plot(ax=ax1, label='16-Day Forecast', linewidth=2.5, linestyle='--', color='#ff7f0e')
    ax1.set_ylabel('Price (UGX)', fontsize=12)
    ax1.set_title(f'{selected_crop} Price Forecast - {district_input}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if rain_df is not None and not rain_df.empty:
        total_rain = rain_df['Rainfall_mm'].sum()
        ax2.bar(rain_df['Date'], rain_df['Rainfall_mm'], color='#4682B4', alpha=0.7, label='Daily Rainfall')
        ax2.set_ylabel('Rainfall (mm)', fontsize=12)
        ax2.set_title(f'16-Day Rainfall Forecast - {district_input}: {total_rain:.1f}mm Total', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        if total_rain < 1:
            ax2.set_ylim(0, 5)
    else:
        ax2.text(0.5, 0.5, f'🌧️ RAINFALL DATA UNAVAILABLE\n\n{rain_status}\n\nIf district not listed, add coordinates to code', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14, color='red', weight='bold')
        ax2.set_ylabel('Rainfall (mm)', fontsize=12)
        ax2.set_title(f'16-Day Rainfall Forecast - {district_input}', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 5)
        ax2.grid(True, alpha=0.3, axis='y')
    
    ax2.set_xlabel('Date', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    
    tab1, tab2 = st.tabs(["📅 Price Forecast", "🌧️ Rainfall Forecast"])
    with tab1:
        forecast_df = forecast.reset_index()
        forecast_df.columns = ['Date', 'Predicted_Price_UGX']
        forecast_df['Date'] = forecast_df['Date'].dt.date
        forecast_df['Predicted_Price_UGX'] = forecast_df['Predicted_Price_UGX'].round(0).astype(int)
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        st.download_button("📥 Download Price CSV", forecast_df.to_csv(index=False),
                          f"{district_input}_{selected_crop}_price.csv", "text/csv")
    with tab2:
        if rain_df is not None and not rain_df.empty:
            st.dataframe(rain_df, use_container_width=True, hide_index=True)
            st.download_button("📥 Download Rain CSV", rain_df.to_csv(index=False),
                              f"{district_input}_rainfall.csv", "text/csv")
        else:
            st.info(f"No rainfall data available. Reason: {rain_status}")

except Exception as e:
    st.error(f"Could not create forecast: {str(e)}")
    st.write("This usually means there isn't enough seasonal data. Try a different district/crop combo.")

st.divider()
st.markdown("""
<div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px;'>
    <p style='margin: 0;'><b>Zero Hunger T420</b> | Built by <b>Ssempebwa Allan</b></p>
    <p style='margin: 0; font-size: 14px;'>Data: WFP Uganda + Open-Meteo | Fighting Hunger with Data Science 🇺🇬🌧️</p>
</div>
""", unsafe_allow_html=True)