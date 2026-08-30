import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# إعدادات الصفحة
st.set_page_config(page_title="بوابة إربد الذكية", layout="wide", page_icon="🚦")

st.markdown("""
    <style>
    .main {background-color: #f4f6f9;}
    h1 {color: #1e3d59; font-family: 'Arial'; text-align: center; padding-bottom: 20px;}
    div[data-testid="metric-container"] {
        background-color: #ffffff; border: 1px solid #e6e6e6; padding: 15px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚦 بوابة إربد الذكية - مركز العمليات المركزي")

@st.cache_resource
def load_assets():
    model = joblib.load('traffic_model.pkl')
    le = joblib.load('label_encoder.pkl')
    df = pd.read_csv('irbid_traffic_data.csv')
    df['edge_id_encoded'] = le.transform(df['edge_id'])
    return model, le, df

model, le, df = load_assets()

# القائمة الجانبية المتقدمة
st.sidebar.title("إعدادات المحاكاة 🎛️")
hour = st.sidebar.slider("اختر الساعة (نظام 24)", 6, 23, 14)
weather = st.sidebar.selectbox("حالة الطقس", options=[0, 1, 2], format_func=lambda x: ["صافي ☀️", "غائم ☁️", "ماطر 🌧️"][x])
day_of_week = st.sidebar.selectbox("اليوم", options=[0,1,2,3,4,5,6], format_func=lambda x: ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"][x])

st.sidebar.markdown("---")
st.sidebar.title("طبقات الخريطة 🗺️")
show_cameras = st.sidebar.checkbox("عرض كاميرات السرعة 📸", value=True)
show_pois = st.sidebar.checkbox("عرض المولات والمطاعم 🛍️", value=True)

is_weekend = 1 if day_of_week in [4, 5] else 0
is_rush_hour = 1 if hour in [7, 8, 9, 14, 15, 16] else 0

# مؤشرات الأداء العلوية
col1, col2, col3 = st.columns(3)
col1.metric("الطقس", ["صافي ☀️", "غائم ☁️", "ماطر 🌧️"][weather])
col2.metric("حالة الذروة", "نشط ⚠️" if is_rush_hour else "غير نشط ✅")
col3.metric("حالة النظام", "متصل 🟢")

st.markdown("<hr style='border: 1px solid #d3d3d3;'>", unsafe_allow_html=True)

# تقسيم الواجهة إلى تبويبات احترافية
tab1, tab2, tab3 = st.tabs(["🗺️ الخريطة التفاعلية", "📰 الأخبار والتنبيهات", "🤖 المساعد الحضري الذكي"])

with tab1:
    sample_edges = df['edge_id_encoded'].unique()[:150] 
    input_data = pd.DataFrame({
        'edge_id_encoded': sample_edges, 'day_of_week': [day_of_week]*150,
        'is_weekend': [is_weekend]*150, 'hour': [hour]*150,
        'weather': [weather]*150, 'is_rush_hour': [is_rush_hour]*150
    })
    
    predictions = model.predict(input_data)
    
    m = folium.Map(location=[32.545, 35.855], zoom_start=13, tiles='OpenStreetMap')
    
    # الخريطة الحرارية
    np.random.seed(42)
    lats = np.random.uniform(32.52, 32.57, size=150)
    lons = np.random.uniform(35.83, 35.88, size=150)
    heat_data = [[lats[i], lons[i], float(predictions[i])] for i in range(150)]
    HeatMap(heat_data, radius=20, blur=15, max_zoom=1, gradient={0.4: 'green', 0.7: 'orange', 1.0: 'red'}).add_to(m)
    
    # إضافة كاميرات المراقبة
    if show_cameras:
        cameras = [
            {"loc": [32.525, 35.865], "name": "كاميرا طريق الحصن", "speed": "80 كم/ساعة"},
            {"loc": [32.548, 35.875], "name": "كاميرا شارع البتراء", "speed": "60 كم/ساعة"},
            {"loc": [32.560, 35.850], "name": "إشارة الإسكان", "speed": "رادار إشارة ضوئية"}
        ]
        for cam in cameras:
            folium.Marker(
                cam["loc"], 
                popup=f"<b>{cam['name']}</b><br>السرعة: {cam['speed']}", 
                icon=folium.Icon(color="red", icon="camera", prefix='fa')
            ).add_to(m)
            
    # إضافة المولات والمطاعم
    if show_pois:
        pois = [
            {"loc": [32.535, 35.865], "name": "أرابيلا مول", "type": "تسوق 🛍️"},
            {"loc": [32.530, 35.850], "name": "إربد سيتي سنتر", "type": "تسوق 🛍️"},
            {"loc": [32.540, 35.855], "name": "مطاعم شارع الجامعة", "type": "مطاعم 🍔"}
        ]
        for poi in pois:
            folium.Marker(
                poi["loc"], 
                popup=f"<b>{poi['name']}</b><br>{poi['type']}", 
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)
            
    st_folium(m, width=1200, height=550)

with tab2:
    st.info("سيتم ربط هذا القسم لاحقاً مع واجهة الأخبار الحية (News API) لمدينة إربد.")
    
with tab3:
    st.info("سيتم ربط هذا القسم لاحقاً مع نموذج لغوي (LLM) للرد على استفسارات المستخدمين برمجياً.")
