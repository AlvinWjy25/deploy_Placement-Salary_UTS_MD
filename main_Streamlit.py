import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle 
import os
import requests
import subprocess
import time
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

BASE_DIR = os.getcwd()
MODEL_PATH_1 = os.path.join(BASE_DIR, 'artifacts', 'model_classification_placement.pkl')
MODEL_PATH_2 = os.path.join(BASE_DIR, 'artifacts', 'model_regression_salary.pkl')

@st.cache_resource
def load_models():
    try:
        clf_model = joblib.load(MODEL_PATH_1)
        reg_model = joblib.load(MODEL_PATH_2)
        return clf_model, reg_model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None
    
model_placement, model_salary = load_models()

# Streamlit Edits
st.set_page_config(page_title="Placement Analytics Pro", layout="wide")

#Sidebar
with st.sidebar:
    st.header("System Control")
    st.info("Aplikasi ini berjalan secara Lokal (Standalone). Model XGBoost dimuat langsung dari folder artifacts.")
    
    st.divider()
    st.subheader("Model Status")
    if model_placement and model_salary:
        st.success("● Models Loaded Successfully")
    else:
        st.error("○ Models Missing")
    
    st.divider()
    st.caption("v1.0 | Standalone Mode")

#Main UI
st.title("Student Placement Prediction")
st.write("Masukkan data mahasiswa di bawah ini untuk memprediksi status penempatan.")
st.divider()

col1, col2, col3 = st.columns(3)

with st.form("prediction_form"):
    tab1, tab2, tab3 = st.tabs(["Akademik", "Skills & Projects", "Pengalaman"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", ['Male', 'Female'])
            cgpa = st.slider("Current CGPA", 0.0, 10.0, 8.0, step=0.1)
        with c2:
            ssc_percentage = st.number_input("SSC % (10th Grade)", 0, 100, 75)
            hsc_percentage = st.number_input("HSC % (12th Grade)", 0, 100, 75)
            degree_percentage = st.number_input("Degree %", 0, 100, 75)

    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            technical_skill_score = st.slider("Technical Skill", 0, 100, 70)
            soft_skill_score = st.slider("Soft Skill", 0, 100, 70)
        with c2:
            entrance_exam_score = st.number_input("Entrance Exam Score", 0, 100, 75)
            attendance_percentage = st.number_input("Attendance %", 0, 100, 85)
        with c3:
            live_projects = st.number_input("Live Projects", 0, 10, 1)
            extracurricular = st.radio("Extracurricular Activities", ["Yes", "No"], horizontal=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            work_experience_months = st.number_input("Work Experience (Months)", 0, 60, 0)
        with c2:
            internship_count = st.number_input("Internship Count", 0, 10, 1)
            certifications = st.number_input("Certifications Count", 0, 10, 0)

    submit_button = st.form_submit_button("Analisis Peluang Kerja", type="primary")

st.divider()

if submit_button:
    if model_placement is None or model_salary is None:
        st.error("Prediksi tidak dapat dilakukan karena model tidak ditemukan.")
    else:

        academic_list = [ssc_percentage, hsc_percentage, degree_percentage]
        
        input_data = pd.DataFrame([{
            "gender": gender,
            "ssc_percentage": ssc_percentage,
            "hsc_percentage": hsc_percentage,
            "degree_percentage": degree_percentage,
            "cgpa": cgpa,
            "entrance_exam_score": entrance_exam_score,
            "technical_skill_score": technical_skill_score,
            "soft_skill_score": soft_skill_score,
            "internship_count": internship_count,
            "live_projects": live_projects,
            "work_experience_months": work_experience_months,
            "certifications": certifications,
            "attendance_percentage": attendance_percentage,
            "extracurricular_activities": extracurricular,
            "skill_combined": technical_skill_score * soft_skill_score,
            "skill_ratio": technical_skill_score / soft_skill_score,
            "cgpa_skill": cgpa * technical_skill_score,
            "academic_avg": (ssc_percentage + hsc_percentage + degree_percentage) / 3,
            "academic_consistency": np.std(academic_list),
            "experience_score": ((internship_count * 2) + (live_projects) + (work_experience_months)) / 6
        }])

        st.subheader("Profil Kompetensi Mahasiswa")
        categories = ['Technical', 'Soft Skills', 'Academic', 'Attendance', 'Entrance']
        values = [technical_skill_score, soft_skill_score, degree_percentage, attendance_percentage, entrance_exam_score]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Student Profile'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


    with st.spinner("Sedang Menganalisis..."):
            # Prediksi Klasifikasi (Status)
            prediction_status = model_placement.predict(input_data)[0]
            
            st.divider()
            res_col1, res_col2 = st.columns(2)

            if prediction_status == 1:
                # Jika Placed, hitung estimasi gaji
                prediction_salary = np.expm1(model_salary.predict(input_data)[0]) #exponent hasil prediksi
                
                with res_col1:
                    st.metric("Status Penempatan", "PLACED", delta="Tersedia")
                with res_col2:
                    st.metric("Estimasi Gaji (LPA)", f"{prediction_salary:,.2f}", delta="Annual Package")
                st.success(f"Mahasiswa diprediksi mendapatkan penempatan dengan paket gaji sebesar {prediction_salary:,.2f} LPA.")
            else:
                with res_col1:
                    st.metric("Status Penempatan", "NOT PLACED", delta="-", delta_color="inverse")
                st.warning("Berdasarkan data, mahasiswa memerlukan peningkatan pada skill teknis atau akademik.")