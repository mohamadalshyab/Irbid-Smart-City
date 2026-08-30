import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# إعدادات الصفحة
st.set_page_config(page_title="بوابة إربد الذكية", layout="wide", page_icon="🚦")

# تنسيق CSS احترافي لبطاقات المؤشرات (Metrics Cards)
st.markdown("""
    <style>
    .main {background-color: #f4f6f9;}
    h1 {color: #1e3d59; font-family: 'Arial'; text-align: center; padding-bottom: 20px;}
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚦 بوابة إربد الذكية - التحليل المروري المدعوم بالذكاء الاصطناعي")

@st.cache_resource
def load_assets():
    model = joblib.load('traffic_model.pkl')
    le = joblib.load('label_encoder.pkl')
    df = pd.read_csv('irbid_traffic_data.csv')
    df['edge_id_encoded'] = le.transform(df['edge_id'])
    return model, le, df

model, le, df = load_assets()

# القائمة الجانبية
st.sidebar.title("إعدادات المحاكاة 🎛️")
hour = st.sidebar.slider("اختر الساعة (نظام 24)", 6, 23, 14)
day_of_week = st.sidebar.selectbox("اليوم", options=[0,1,2,3,4,5,6], format_func=lambda x: ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"][x])
weather = st.sidebar.selectbox("حالة الطقس", options=[0, 1, 2], format_func=lambda x: ["صافي ☀️", "غائم ☁️", "ماطر 🌧️"][x])

is_weekend = 1 if day_of_week in [4, 5] else 0
is_rush_hour = 1 if hour in [7, 8, 9, 14, 15, 16] else 0

# مؤشرات الأداء العلوية (Metrics)
col1, col2, col3 = st.columns(3)
col1.metric("حالة الطقس الحالية", ["صافي ☀️", "غائم ☁️", "ماطر 🌧️"][weather])
col2.metric("حالة وقت الذروة", "نشط ⚠️" if is_rush_hour else "غير نشط ✅")
col3.metric("تحديث البيانات", "مباشر 🟢")

st.markdown("<hr style='border: 1px solid #d3d3d3;'>", unsafe_allow_html=True)

# تجهيز البيانات والتنبؤ
sample_edges = df['edge_id_encoded'].unique()[:150] 
input_data = pd.DataFrame({
    'edge_id_encoded': sample_edges,
    'day_of_week': [day_of_week]*150,
    'is_weekend': [is_weekend]*150,
    'hour': [hour]*150,
    'weather': [weather]*150,
    'is_rush_hour': [is_rush_hour]*150
})

predictions = model.predict(input_data)

st.markdown("### 🗺️ الخريطة المرورية الحية لمدينة إربد")

# خريطة احترافية نظيفة (بدون علامات مائية)
m = folium.Map(location=[32.5514, 35.8515], zoom_start=14, tiles='OpenStreetMap')

# توليد نقاط مكثفة لإنشاء الخريطة الحرارية
np.random.seed(42)
lats = np.random.uniform(32.53, 32.57, size=150)
lons = np.random.uniform(35.83, 35.87, size=150)

# تجهيز بيانات الخريطة الحرارية
heat_data = [[lats[i], lons[i], float(predictions[i])] for i in range(150)]

# إضافة طبقة الخريطة الحرارية (HeatMap) لتعكس الازدحام كبؤر لونية
HeatMap(
    heat_data, 
    radius=20, 
    blur=15, 
    max_zoom=1, 
    gradient={0.4: 'green', 0.7: 'orange', 1.0: 'red'}
).add_to(m)

# عرض الخريطة
st_folium(m, width=1200, height=600)
