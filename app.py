import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

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

# القائمة الجانبية (بدون أزرار الكاميرات والمولات)
st.sidebar.title("إعدادات المحاكاة 🎛️")
hour = st.sidebar.slider("اختر الساعة (نظام 24)", 6, 23, 14)
weather = st.sidebar.selectbox("حالة الطقس", options=[0, 1, 2], format_func=lambda x: ["صافي ☀️", "غائم ☁️", "ماطر 🌧️"][x])
day_of_week = st.sidebar.selectbox("اليوم", options=[0,1,2,3,4,5,6], format_func=lambda x: ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"][x])

is_weekend = 1 if day_of_week in [4, 5] else 0
is_rush_hour = 1 if hour in [7, 8, 9, 14, 15, 16] else 0

# مؤشرات الأداء العلوية
col1, col2, col3 = st.columns(3)
col1.metric("الطقس", ["صافي ☀️", "غائم ☁️", "ماطر 🌧️"][weather])
col2.metric("حالة الذروة", "نشط ⚠️" if is_rush_hour else "غير نشط ✅")
col3.metric("حالة النظام", "متصل 🟢")

st.markdown("<hr style='border: 1px solid #d3d3d3;'>", unsafe_allow_html=True)

# التبويبات الثلاثة
tab1, tab2, tab3 = st.tabs(["🗺️ الخريطة التفاعلية", "📰 التنبيهات المباشرة", "🤖 المساعد الحضري"])

with tab1:
    sample_edges = df['edge_id_encoded'].unique()[:150] 
    input_data = pd.DataFrame({
        'edge_id_encoded': sample_edges, 'day_of_week': [day_of_week]*150,
        'is_weekend': [is_weekend]*150, 'hour': [hour]*150,
        'weather': [weather]*150, 'is_rush_hour': [is_rush_hour]*150
    })
    
    predictions = model.predict(input_data)
    
    # خريطة حرارية نظيفة تماماً
    m = folium.Map(location=[32.536, 35.855], zoom_start=14, tiles='OpenStreetMap')
    
    np.random.seed(42)
    lats = np.random.uniform(32.51, 32.56, size=150)
    lons = np.random.uniform(35.83, 35.88, size=150)
    heat_data = [[lats[i], lons[i], float(predictions[i])] for i in range(150)]
    HeatMap(heat_data, radius=20, blur=15, max_zoom=1, gradient={0.4: 'green', 0.7: 'orange', 1.0: 'red'}).add_to(m)
    
    st_folium(m, width=1200, height=550)

with tab2:
    st.markdown("### 📰 شريط الأخبار والتنبيهات المرورية")
    
    if weather == 2:
        st.error("🌧️ **عاجل:** تحذيرات من انزلاقات على طريق الحصن وشارع البتراء بسبب هطول الأمطار المستمر.")
    elif weather == 1:
        st.info("☁️ **حالة الطقس:** طقس غائم في إربد، حركة سير طبيعية في معظم الشوارع الرئيسية.")
    else:
        st.success("☀️ **حالة الطرق:** طقس صافٍ ومثالي للتنقل في جميع أنحاء المدينة.")
        
    if is_rush_hour:
        st.warning("⚠️ **تنويه مروري:** نشهد حالياً أوقات الذروة. توقع ازدحام خانق قرب محيط جامعة اليرموك ومجمع عمان الجديد.")
    elif hour > 21:
        st.info("🌙 **حركة هادئة:** انسيابية عالية في حركة المرور متوقعة في معظم الشوارع التجارية.")
        
    st.markdown("---")
    st.markdown("💡 **إعلان بلدية إربد:** أعمال صيانة وتعبيد مجدولة في شارع بغداد خلال عطلة نهاية الأسبوع، يرجى سلوك طرق بديلة.")

with tab3:
    st.markdown("### 🤖 المساعد الحضري (Irbid Smart Agent)")
    st.write("اطرح سؤالك لمعرفة حالة الطرق أو أفضل أوقات الخروج (الاستعلام يعتمد على ظروف المحاكاة الحالية).")
    
    user_query = st.text_input("مثال: هل أزمة عند الجامعة؟ أو كيف الوضع بمجمع عمان؟")
    
    if st.button("اسأل المساعد 🚀"):
        if not user_query:
            st.warning("الرجاء كتابة سؤال أولاً.")
        else:
            query = user_query.lower()
            if "جامعة" in query or "يرموك" in query:
                if is_rush_hour:
                    st.error("🤖 المساعد: نعم، المنطقة المحيطة بجامعة اليرموك تشهد أزمة خانقة الآن. أنصحك باستخدام شوارع فرعية من جهة الحي الجنوبي.")
                else:
                    st.success("🤖 المساعد: الطريق إلى جامعة اليرموك سالك وحركة السير طبيعية جداً في هذا الوقت.")
            
            elif "مطر" in query or "شتاء" in query:
                if weather == 2:
                    st.warning("🤖 المساعد: الجو ماطر حالياً، وهذا يزيد من احتمالية بطء السير بنسبة 20%. يرجى القيادة بحذر.")
                else:
                    st.info("🤖 المساعد: الطقس الحالي ليس ماطراً حسب الإعدادات، لا يوجد ما يدعو للقلق من الانزلاقات.")
            
            elif "مول" in query or "سيتي سنتر" in query or "ارابيلا" in query:
                if is_weekend and hour > 16:
                    st.error("🤖 المساعد: متوقع ازدحام شديد عند المولات لأن اليوم عطلة والوقت مسائي. يفضل تأجيل المشوار إن أمكن.")
                else:
                    st.success("🤖 المساعد: الطرق المؤدية للمولات سالكة حالياً، وقت ممتاز للتسوق.")
            
            else:
                st.success("🤖 المساعد: بناءً على المعطيات الحالية والتنبؤات المرورية، لا توجد إغلاقات رئيسية أو حوادث في شبكة طرق إربد. رحلة آمنة!")
