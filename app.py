import streamlit as st
import numpy as np
import pickle

# Load model & scaler
model = pickle.load(open('model_rf.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Prediksi Diabetes dengan Random Forest")
st.write("Masukkan data pasien untuk mengetahui potensi terkena diabetes.")

# Input fitur
preg = st.number_input("Jumlah Kehamilan", 0, 20)
glucose = st.number_input("Glukosa", 0, 200)
bp = st.number_input("Tekanan Darah", 0, 150)
skin = st.number_input("Ketebalan Kulit", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Riwayat Diabetes Keluarga", 0.0, 2.5)
age = st.number_input("Usia", 1, 100)

# Prediksi
if st.button("Prediksi"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    if prediction == 1:
        st.error("Pasien diprediksi MENGIDAP diabetes.")
    else:
        st.success("Pasien diprediksi TIDAK mengidap diabetes.")
