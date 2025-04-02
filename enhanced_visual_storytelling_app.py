
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import time

st.set_page_config(page_title='Enhanced Visual Storytelling Churn App', layout='wide')

css_code = '''
<style>
body {
    font-family: Arial, sans-serif;
    scroll-behavior: smooth;
    background-color: #121212;
    color: #E0E0E0;
}

section {
    padding: 100px 0;
    border-bottom: 1px solid #333;
    background-color: #1E1E1E;
    margin-bottom: 20px;
}

h1, h2, h3 {
    color: #BB86FC;
}

.plotly-graph {
    margin-top: 30px;
}

.stMetric {
    text-align: center;
    margin: 20px 0;
    color: #BB86FC;
}

button {
    background-color: #BB86FC;
    color: white;
    padding: 10px 20px;
    border: none;
    cursor: pointer;
    border-radius: 5px;
}
</style>
'''
st.markdown(css_code, unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("Churn_Modelling.csv")
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 30, 45, 60, 100], labels=["18-30", "31-45", "46-60", "60+"])
    return df

df = load_data()

X = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
X = pd.get_dummies(X, drop_first=True)
y = df["Exited"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# --- Objetivo del Modelo ---
st.title('Enhanced Visual Storytelling Churn App')
st.header('Objetivo del Modelo')
st.write("""
Este modelo busca identificar clientes en riesgo de abandono y proporcionar estrategias efectivas para retenerlos.

A través de un análisis detallado, podemos identificar patrones críticos de abandono y actuar en consecuencia.
""")

# --- Gráfico Dinámico: Churn por País ---
st.header('Churn por País')
fig1 = px.histogram(df, x="Geography", color="Exited", barmode="group", title="Churn por País")
st.plotly_chart(fig1, use_container_width=True)
time.sleep(1)
st.write("""
En este gráfico, se observa cómo la tasa de churn varía significativamente según la región geográfica. 
Esto permite adaptar estrategias regionales para mejorar la retención.
""")

# --- Gráfico Dinámico: Churn por Grupo de Edad ---
st.header('Churn por Grupo de Edad')
age_churn = df.groupby("AgeGroup")['Exited'].mean() * 100
fig2 = px.bar(age_churn, title="Churn por Grupo de Edad")
st.plotly_chart(fig2, use_container_width=True)
time.sleep(1)
st.write("""
El análisis por edad revela que ciertos grupos etarios tienen una mayor propensión a abandonar. 
Las estrategias de retención deben enfocarse en estos segmentos prioritarios.
""")

# --- Implementación Progresiva ---
st.header('Implementación Progresiva')
st.write("""
Nuestro modelo permite una implementación progresiva, analizando múltiples factores como país, edad, balance y productos contratados. 

Se pueden aplicar campañas personalizadas y alertas tempranas para retener clientes.
""")

# --- Segmentación Avanzada ---
st.header('Segmentación Avanzada y Retención')
st.write("""
Implementar técnicas de segmentación avanzadas permite identificar perfiles prioritarios y ajustar estrategias en consecuencia.

Los clientes se segmentan según su comportamiento y características únicas para ofrecer servicios personalizados.
""")

# --- Conclusión ---
st.header('Conclusión y Próximos Pasos')
st.write("""
El siguiente paso consiste en llevar este modelo a producción, mejorarlo con datos adicionales (redes sociales, encuestas) y aplicar campañas de retención efectivas.
""")
