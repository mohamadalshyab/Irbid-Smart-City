import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="بوابة إربد الذكية", layout="wide", page_icon="🚦")
st.title("🚦 بوابة إربد الذكية - التحليل المروري المدعوم بالذكاء الاصطناعي")

@st.cache_resource
def load_assets():
    # لاحظ أننا أزلنا مسار جوجل درايف، لأن الملفات ستكون بجانب الكود على GitHub
    model = joblib.load('traffic_model.pkl')
    le = joblib.load('label_encoder.pkl')
    df = pd.read_csv('irbid_traffic_data.csv')
    df['edge_id_encoded'] = le.transform(df['edge_id'])
    return model, le, df

model, le, df = load_assets()

st.sidebar.title("إعدادات المحاكاة 🎛️")
hour = st.sidebar.slider("اختر الساعة (نظام 24)", 6, 23, 14)
day_of_week = st.sidebar.selectbox("اليوم", options=[0,1,2,3,4,5,6], format_func=lambda x: ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"][x])
weather = st.sidebar.selectbox("حالة الطقس", options=[0, 1, 2], format_func=lambda x: ["صافي ☀️", "غائم ☁️", "ماطر 🌧️"][x])

is_weekend = 1 if day_of_week in [4, 5] else 0
is_rush_hour = 1 if hour in [7, 8, 9, 14, 15, 16] else 0

col1, col2, col3 = st.columns(3)
col1.metric("حالة الطقس", ["صافي", "غائم", "ماطر"][weather])
col2.metric("وقت الذروة", "نعم ⚠️" if is_rush_hour else "لا ✅")
col3.metric("تحديث البيانات", "مباشر 🟢")
st.markdown("---")

sample_edges = df['edge_id_encoded'].unique()[:50] 
input_data = pd.DataFrame({
    'edge_id_encoded': sample_edges,
    'day_of_week': [day_of_week]*50,
    'is_weekend': [is_weekend]*50,
    'hour': [hour]*50,
    'weather': [weather]*50,
    'is_rush_hour': [is_rush_hour]*50
})

predictions = model.predict(input_data)

st.subheader(f"الخريطة المرورية الحية لمدينة إربد - الساعة {hour}:00")
m = folium.Map(location=[32.5514, 35.8515], zoom_start=14, tiles='CartoDB positron')

np.random.seed(42)
lats = np.random.uniform(32.53, 32.57, size=50)
lons = np.random.uniform(35.83, 35.87, size=50)

for i in range(50):
    cong = np.round(predictions[i], 1)
    color = 'green' if cong < 5 else 'orange' if cong < 7.5 else 'red'
    folium.CircleMarker(
        location=[lats[i], lons[i]], radius=8, popup=f"مستوى الازدحام: {cong}/10",
        color=color, fill=True, fill_color=color, fill_opacity=0.7
    ).add_to(m)

st_folium(m, width=900, height=500)