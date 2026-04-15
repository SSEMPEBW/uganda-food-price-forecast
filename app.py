import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

st.title("Uganda Food Price Forecast")
st.caption("Built for WFP/FAO - SDG 2: Zero Hunger")
st.caption("By Allan Ssempebwa, for the God of Abraham")

try:
    df = pd.read_csv('data/prices.csv')
    st.success(f"CSV loaded: {len(df['district'].unique())} districts, {len(df)} records")
except FileNotFoundError:
    st.error("ERROR: prices.csv not found in data folder.")
    st.stop()

district_list = df['district'].unique()
district_input = st.selectbox("Select district to forecast:", district_list)

if district_input:
    filtered = df[df['district'] == district_input].sort_values('month')
    latest = filtered.iloc[-1]
    
    maize = float(latest['maize_kg'])
    beans = float(latest['beans_kg'])
    matooke = float(latest['matooke_bunch'])
    
    maize_prev = float(filtered.iloc[0]['maize_kg'])
    maize_change = ((maize - maize_prev) / maize_prev) * 100
    
    meal_cost = maize + beans
    kids_50k = 50000 / meal_cost
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("--- Zero Hunger Forecast ---")
        st.metric("Latest Maize", f"UGX {maize:.0f}/kg", f"{maize_change:+.1f}% since Apr")
        st.metric("Latest Beans", f"UGX {beans:.0f}/kg")
        st.metric("Cost per meal", f"UGX {meal_cost:.0f}")
        st.metric("Kids fed with 50K", f"{kids_50k:.0f} kids")
        
        st.subheader("Monthly Trend")
        fig, ax = plt.subplots()
        ax.plot(filtered['month'], filtered['maize_kg'], marker='o', label='Maize', color='#FF6B35')
        ax.plot(filtered['month'], filtered['beans_kg'], marker='s', label='Beans', color='#004E89')
        ax.set_ylabel('UGX per kg')
        ax.set_title(f'{district_input} Price Trend')
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Current Price Comparison")
        foods = ['Maize', 'Beans', 'Matooke']
        prices = [maize, beans, matooke]
        fig2, ax2 = plt.subplots()
        ax2.bar(foods, prices, color=['#FF6B35', '#004E89', '#009639'])
        ax2.set_ylabel('UGX per kg/bunch')
        ax2.set_title(f'{district_input} - {latest["month"]}')
        st.pyplot(fig2)
        
        st.subheader("District Map")
        # Coordinates for Uganda districts
        coords = {
            'Bweyogerere': [0.3528, 32.6683],
            'Nakawa': [0.3280, 32.6169],
            'Kampala Central': [0.3167, 32.5822],
            'Mukono': [0.3533, 32.7553],
            'Jinja': [0.4479, 33.2026]
        }
        
        m = folium.Map(location=coords[district_input], zoom_start=12)
        folium.Marker(
            coords[district_input], 
            popup=f"{district_input}: UGX {meal_cost:.0f}/meal",
            icon=folium.Icon(color='red', icon='cutlery', prefix='fa')
        ).add_to(m)
        st_folium(m, width=350, height=300)
    
    st.success("Zero Hunger v1.2: Trends + Map online. The Creator provides vision.")