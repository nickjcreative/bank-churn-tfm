
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import plotly.express as px
import plotly.graph_objects as go
from streamlit.components.v1 import html

st.set_page_config(page_title="Interactividad Churn App", layout="wide")
st.title("📘 Storytelling Interactivo: Reducción de Churn Bancario")

seccion = st.sidebar.radio("Navegar", [
    "🎯 Introducción",
    "🤖 Explicación del Modelo",
    "📈 Predicción Interactiva",
    "📊 Resultados"
])

@st.cache_data
def cargar_datos():
    df = pd.read_csv("Churn_Modelling.csv")
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 30, 45, 60, 100], labels=["18-30", "31-45", "46-60", "60+"])
    return df

df = cargar_datos()

X = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
X = pd.get_dummies(X, drop_first=True)
y = df["Exited"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

if seccion == "🎯 Introducción":
    st.header("🎯 ¿Por qué importa reducir el churn bancario?")
    st.markdown("""
    - Cada cliente que abandona representa una pérdida significativa.
    - El costo de adquisición de un nuevo cliente es **5x a 25x** mayor que retener uno existente.
    - Un cliente perdido puede costar entre **$200 y $1,000 USD anuales**.
    """)

elif seccion == "🤖 Explicación del Modelo":
    st.header("🤖 ¿Cómo funciona el modelo?")
    st.markdown("""
    - Modelo: **Random Forest Classifier**
    - Entrenado con características: Edad, País, Balance, Productos, Actividad, etc.
    - Conversión de variables categóricas con One-Hot Encoding.
    - Escalado de variables numéricas.
    """)

elif seccion == "📈 Predicción Interactiva":
    st.header("📈 Prueba cómo se afecta la predicción cambiando variables")

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

    fig = px.bar(x=["Probabilidad de churn"], y=[proba], color=[proba], color_continuous_scale="Reds")
    st.plotly_chart(fig)

elif seccion == "📊 Resultados":
    st.header("📊 Resultados Globales del Modelo")
    churn_rate = df["Exited"].mean() * 100
    st.metric("Tasa general de churn", f"{churn_rate:.2f}%")

    geo_churn = df.groupby("Geography")["Exited"].mean() * 100
    fig1 = px.bar(geo_churn, title="Churn por País")
    st.plotly_chart(fig1)

    age_churn = df.groupby("AgeGroup")["Exited"].mean() * 100
    fig2 = px.bar(age_churn, title="Churn por Grupo de Edad")
    st.plotly_chart(fig2)
