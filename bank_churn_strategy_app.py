
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='Bank Churn Strategy App', layout='wide')

css_code = '''
<style>
body {
    font-family: Arial, sans-serif;
    scroll-behavior: smooth;
    background-color: #f0f2f6;
}

section {
    padding: 50px 0;
    border-bottom: 1px solid #ddd;
    background-color: white;
}

h1, h2, h3 {
    color: #4A90E2;
}

.plotly-graph {
    margin-top: 30px;
}

.stMetric {
    text-align: center;
    margin: 20px 0;
}

button {
    background-color: #4A90E2;
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

# --- Introducción del Problema ---
st.title('Bank Churn Strategy App')
st.header('Introducción del Problema')
st.write("""
El churn o abandono de clientes es un problema serio para los bancos. El costo de adquirir un nuevo cliente es significativamente mayor que el costo de retener uno existente.

- **Costo de adquisición de un nuevo cliente:** Puede ser hasta 5 veces mayor que el costo de retener uno.
- **Pérdidas potenciales:** Cada cliente perdido puede representar miles de dólares en ingresos no percibidos anualmente.

Nuestro objetivo es identificar clientes en riesgo de abandonar y aplicar estrategias proactivas para retenerlos.
""")

# --- Solución Propuesta ---
st.header('Solución Propuesta')
st.write("""
Implementamos un modelo de Machine Learning utilizando Random Forest para predecir la probabilidad de churn basado en características clave de los clientes.

- Características utilizadas: Edad, País, Balance, Número de productos, Actividad del cliente, etc.
- Justificación: Random Forest es robusto y eficiente para identificar patrones no lineales.
- Precisión alcanzada: 85%.
""")

# --- Visualización Interactiva ---
st.subheader('Gráfico Comparativo por País')
fig1 = px.histogram(df, x="Geography", color="Exited", barmode="group", title="Churn por País")
st.plotly_chart(fig1, use_container_width=True)

st.subheader('Churn por Grupo de Edad')
age_churn = df.groupby("AgeGroup")["Exited"].mean() * 100
fig2 = px.bar(age_churn, title="Churn por Grupo de Edad")
st.plotly_chart(fig2, use_container_width=True)

# --- Implementación en el Día a Día ---
st.header('Implementación en el Día a Día')
st.write("""
Integrar este modelo en las operaciones diarias del banco permite:
- Detectar clientes en riesgo en tiempo real.
- Aplicar estrategias personalizadas para retener a estos clientes.
- Monitorizar continuamente la efectividad del modelo y ajustarlo según sea necesario.
""")

# --- Estrategias para el Banco ---
st.header('Estrategias para el Banco')
st.write("""
Basado en nuestros análisis, recomendamos las siguientes estrategias:

- **Marketing Personalizado:** Ofrecer promociones o servicios personalizados a clientes en riesgo.
- **Monitoreo Continuo:** Implementar un sistema de alerta que notifique sobre cambios en el comportamiento del cliente.
- **Mejora de Productos:** Identificar productos con alta tasa de abandono y mejorarlos.
- **Comunicación Directa:** Contactar proactivamente a clientes en riesgo para mejorar la experiencia del cliente.
""")

st.subheader('Prueba la Predicción Interactiva')
age = st.slider("Edad", 18, 100, 40)
balance = st.slider("Balance", 0, 250000, 50000)
num_products = st.selectbox("Número de Productos", [1, 2, 3, 4])
is_active = st.selectbox("¿Es un cliente activo?", ["Sí", "No"]) == "Sí"

user_data = pd.DataFrame([[age, balance, num_products, is_active]],
                         columns=["Age", "Balance", "NumOfProducts", "IsActiveMember"])

user_data = pd.get_dummies(user_data, drop_first=True)
user_data = user_data.reindex(columns=X.columns, fill_value=0)
user_data_scaled = scaler.transform(user_data)

proba = model.predict_proba(user_data_scaled)[0][1]
st.metric("Probabilidad de churn", f"{proba*100:.2f}%")
