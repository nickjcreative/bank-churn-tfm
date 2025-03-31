
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="Storytelling Churn", layout="wide")
st.title("📘 Presentación Interactiva: Reducción de Churn Bancario")

seccion = st.sidebar.radio("Ir a sección", [
    "🎯 Introducción",
    "🤖 Modelo",
    "📊 Resultados",
    "🛠️ Plan de acción",
    "🔍 Demo interactiva"
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
    st.header("🎯 ¿Cuál es el problema?")
    st.markdown("""
    - El **20.37%** de los clientes del banco abandonan el servicio.
    - Esto representa pérdidas millonarias y un alto costo de adquisición.
    - ¿Y si pudiéramos anticiparnos antes de que se vayan?

    ### 💡 Objetivo
    Crear un modelo de Machine Learning que identifique clientes en riesgo de abandono para activar campañas de retención.
    """)

elif seccion == "🤖 Modelo":
    st.header("🤖 ¿Qué modelo usamos?")
    st.markdown("""
    - Usamos un modelo **Random Forest Classifier**.
    - Entrenamos con datos de clientes: Edad, País, Balance, Productos, Actividad, etc.
    - Convertimos variables categóricas con One-Hot Encoding.
    - Escalamos las variables numéricas.
    """)

elif seccion == "📊 Resultados":
    st.header("📊 ¿Qué resultados obtuvimos?")
    churn_rate = df["Exited"].mean() * 100
    st.metric("Tasa general de churn", f"{churn_rate:.2f}%")
    geo_churn = df.groupby("Geography")["Exited"].mean() * 100
    st.bar_chart(geo_churn)
    age_churn = df.groupby("AgeGroup")["Exited"].mean() * 100
    st.bar_chart(age_churn)

elif seccion == "🛠️ Plan de acción":
    st.header("🛠️ ¿Qué proponemos hacer?")
    st.markdown("""
    ### 🔍 Activación basada en riesgo
    - Clasificación en 3 niveles de riesgo: Bajo, Medio, Alto.
    - Acciones concretas según el riesgo detectado.
    """)

elif seccion == "🔍 Demo interactiva":
    st.header("🔍 Demo interactiva")
    index = st.slider("Selecciona un cliente", 0, len(df)-1, 0)
    cliente = df.iloc[[index]]
    st.write(cliente)
    X_cliente = cliente.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
    X_cliente = pd.get_dummies(X_cliente, drop_first=True)
    X_cliente = X_cliente.reindex(columns=X.columns, fill_value=0)
    X_cliente_scaled = scaler.transform(X_cliente)
    proba = model.predict_proba(X_cliente_scaled)[0][1]
    st.metric("Probabilidad de churn", f"{proba*100:.2f}%")
