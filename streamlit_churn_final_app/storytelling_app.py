
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Configuración
st.set_page_config(page_title="Storytelling Churn", layout="wide")
st.title("📘 Presentación Interactiva: Reducción de Churn Bancario")

# Navegación
seccion = st.sidebar.radio("Ir a sección", [
    "🎯 Introducción",
    "🤖 Modelo",
    "📊 Resultados",
    "🛠️ Plan de acción",
    "🔍 Demo interactiva"
])

# Cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv("Churn_Modelling.csv")
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 30, 45, 60, 100], labels=["18-30", "31-45", "46-60", "60+"])
    return df

df = cargar_datos()

# Preprocesamiento general
X = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
X = pd.get_dummies(X, drop_first=True)
y = df["Exited"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]
roc = roc_auc_score(y, y_proba)

# Secciones
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
    
    El modelo aprende a identificar patrones que anticipan el abandono.
    """)

elif seccion == "📊 Resultados":
    st.header("📊 ¿Qué resultados obtuvimos?")

    churn_rate = df["Exited"].mean() * 100
    st.metric("Tasa de churn actual", f"{churn_rate:.2f}%")
    st.metric("AUC-ROC del modelo", f"{roc:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        geo = df.groupby("Geography")["Exited"].mean() * 100
        st.subheader("Churn por país")
        st.bar_chart(geo)
    with col2:
        edad = df.groupby("AgeGroup")["Exited"].mean() * 100
        st.subheader("Churn por edad")
        st.bar_chart(edad)

    st.subheader("Mapa de calor: Churn por país y edad")
    heatmap_data = df.pivot_table(index="Geography", columns="AgeGroup", values="Exited", aggfunc="mean") * 100
    fig, ax = plt.subplots()
    sns.heatmap(heatmap_data, annot=True, cmap="Reds", fmt=".1f", ax=ax)
    st.pyplot(fig)

elif seccion == "🛠️ Plan de acción":
    st.header("🛠️ ¿Qué proponemos hacer?")
    st.markdown("""
    ### 🔍 Activación basada en riesgo
    - El modelo clasifica a los clientes en 3 niveles:
        - 🟢 Bajo: <30% probabilidad
        - 🟠 Medio: 30-60%
        - 🔴 Alto: >60%

    ### 📬 Acciones concretas:
    - Alertas automáticas en CRM.
    - Campañas de retención por segmento.
    - Seguimiento mensual de KPIs.

    ### 🎯 Beneficios esperados
    - Reducir el churn en un 10-15%.
    - Aumentar fidelización.
    - Ahorro en adquisición de nuevos clientes.
    """)

elif seccion == "🔍 Demo interactiva":
    st.header("🔍 Demo interactiva")
    index = st.slider("Selecciona un cliente", 0, len(df)-1, 0)
    cliente = df.iloc[[index]]

    st.subheader("🧾 Datos del cliente")
    st.write(cliente[["Geography", "Gender", "Age", "Balance", "NumOfProducts", "IsActiveMember"]])

    X_cliente = cliente.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
    X_cliente = pd.get_dummies(X_cliente, drop_first=True)
    X_cliente = X_cliente.reindex(columns=X.columns, fill_value=0)
    X_cliente_scaled = scaler.transform(X_cliente)

    proba = model.predict_proba(X_cliente_scaled)[0][1]
    riesgo = "🟢 Bajo" if proba < 0.3 else "🟠 Medio" if proba < 0.6 else "🔴 Alto"

    st.metric("Probabilidad de churn", f"{proba*100:.2f}%")
    st.metric("Nivel de riesgo", riesgo)

    fig, ax = plt.subplots()
    ax.barh(["Probabilidad de churn"], [proba], color="red" if proba > 0.6 else "orange" if proba > 0.3 else "green")
    ax.set_xlim(0, 1)
    st.pyplot(fig)
