
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='Visual Storytelling Churn App', layout='wide')

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
st.title('Visual Storytelling Churn App')
st.header('Objetivo del Modelo')
st.write("""
El objetivo de este modelo es identificar clientes en riesgo de abandonar el banco y aplicar estrategias para retenerlos. 

El modelo fue desarrollado con un enfoque en:
- **Detección temprana de abandono.**
- **Optimización de campañas de marketing y fidelización.**
- **Mejora del valor de vida del cliente y reducción de costos de adquisición.**
""")

# --- Visualización 1: Churn por País ---
st.header('Churn por País')
fig1 = px.histogram(df, x="Geography", color="Exited", barmode="group", title="Churn por País")
st.plotly_chart(fig1, use_container_width=True)
st.write("""
En este gráfico, se observa cómo la tasa de churn varía significativamente según la región geográfica. Los bancos pueden utilizar esta información para adaptar estrategias regionales.
""")

# --- Visualización 2: Churn por Grupo de Edad ---
st.header('Churn por Grupo de Edad')
age_churn = df.groupby("AgeGroup")['Exited'].mean() * 100
fig2 = px.bar(age_churn, title="Churn por Grupo de Edad")
st.plotly_chart(fig2, use_container_width=True)
st.write("""
El análisis por edad revela que ciertos grupos etarios tienen una mayor propensión a abandonar. Las estrategias de retención deben enfocarse en estos segmentos prioritarios.
""")

# --- Implementación y Nuevos Datos ---
st.header('Implementación y Nuevos Datos')
st.write("""
Propuestas de implementación:
- Incorporar datos adicionales como redes sociales y encuestas para enriquecer el análisis.
- Utilizar datos transaccionales y de comportamiento para detectar señales tempranas de abandono.
""")

# --- Segmentación y Estrategias de Retención ---
st.header('Segmentación de Clientes y Estrategias de Retención')
st.write("""
Realizar una segmentación de clientes para identificar:

- Perfiles que requieren productos específicos.

- Frecuencia de uso de servicios bancarios.

- Niveles de satisfacción.



Estrategias propuestas:

- Marketing personalizado basado en comportamiento.

- Campañas de retención y fidelización.

- Mejora de productos y servicios basados en segmentos prioritarios.

""")

# --- Conclusión ---
st.header('Conclusión y Próximos Pasos')
st.write("""
La implementación de este modelo permitirá identificar y retener clientes de manera más eficiente. 
El siguiente paso consiste en llevar esta solución a producción, mejorar continuamente su precisión y enriquecerla con nuevos datos.
""")
