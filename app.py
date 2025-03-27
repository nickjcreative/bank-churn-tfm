import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import io

# --- Configuracion general ---
st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("🔍 Predicción de Churn Bancario")

# --- Cargar datos ---
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("Churn_Modelling.csv")
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 30, 45, 60, 100], labels=["18-30", "31-45", "46-60", "60+"])
    return df

uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
df = load_data(uploaded_file)

# --- Preprocesamiento ---
def prepare_model_data(data):
    X = data.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
    X = pd.get_dummies(X, drop_first=True)
    y = data["Exited"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, X, y

model, scaler, X, y = prepare_model_data(df)

# --- Clasificar riesgo ---
def clasificar_riesgo(prob):
    if prob < 0.3:
        return "🟢 Bajo"
    elif prob < 0.6:
        return "🟠 Medio"
    else:
        return "🔴 Alto"

# --- Sidebar opciones ---
st.sidebar.title("Opciones")
menu = st.sidebar.radio("Selecciona una vista:", ["🔮 Predicción individual", "📊 Análisis por grupo", "📈 Dashboard general"])

# --- VISTA 1: Predicción individual ---
if menu == "🔮 Predicción individual":
    st.header("Predicción para un cliente")
    index = st.slider("Selecciona un cliente (por índice)", 0, len(df)-1, 0)
    cliente = df.iloc[[index]]

    st.subheader("🧾 Información del cliente")
    st.write(cliente[["Geography", "Gender", "Age", "Balance", "NumOfProducts", "IsActiveMember"]])

    X_cliente = cliente.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
    X_cliente = pd.get_dummies(X_cliente, drop_first=True)
    X_cliente = X_cliente.reindex(columns=X.columns, fill_value=0)
    X_cliente_scaled = scaler.transform(X_cliente)

    proba = model.predict_proba(X_cliente_scaled)[0][1]
    riesgo = clasificar_riesgo(proba)

    st.subheader("📈 Resultado del modelo")
    st.metric("Probabilidad de churn", f"{proba*100:.2f}%")
    st.metric("Nivel de riesgo", riesgo)

    fig, ax = plt.subplots()
    ax.barh(["Probabilidad de churn"], [proba], color="red" if proba > 0.6 else "orange" if proba > 0.3 else "green")
    ax.set_xlim(0, 1)
    st.pyplot(fig)

    export_df = cliente.copy()
    export_df["Probabilidad_Churn"] = proba
    export_df["Riesgo"] = riesgo
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar predicción en CSV", csv, "cliente_prediccion.csv", "text/csv")

# --- VISTA 2: Análisis por grupo ---
elif menu == "📊 Análisis por grupo":
    st.header("Análisis por grupo")
    col1, col2 = st.columns(2)
    with col1:
        geo = st.selectbox("Selecciona una geografía", df["Geography"].unique())
    with col2:
        age_group = st.selectbox("Selecciona grupo de edad", df["AgeGroup"].unique())

    subset = df[(df["Geography"] == geo) & (df["AgeGroup"] == age_group)]
    st.write(f"Clientes en el grupo seleccionado: {len(subset)}")

    if len(subset) > 0:
        X_sub = subset.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
        X_sub = pd.get_dummies(X_sub, drop_first=True)
        X_sub = X_sub.reindex(columns=X.columns, fill_value=0)
        X_sub_scaled = scaler.transform(X_sub)
        proba_sub = model.predict_proba(X_sub_scaled)[:, 1]
        riesgos = [clasificar_riesgo(p) for p in proba_sub]

        subset = subset.copy()
        subset["Riesgo"] = riesgos
        resumen = subset["Riesgo"].value_counts().reindex(["🟢 Bajo", "🟠 Medio", "🔴 Alto"]).fillna(0)

        st.subheader("Distribución de riesgo")
        st.bar_chart(resumen)

        churn_rate = subset["Exited"].mean() * 100
        st.metric("Tasa real de churn en el grupo", f"{churn_rate:.2f}%")

        csv = subset.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar resultados en CSV",
            data=csv,
            file_name="resultado_churn_segmento.csv",
            mime="text/csv"
        )
    else:
        st.warning("No hay datos suficientes para este grupo.")

# --- VISTA 3: Dashboard general ---
elif menu == "📈 Dashboard general":
    st.header("Dashboard general del dataset")

    churn_rate_total = df["Exited"].mean() * 100
    st.metric("Tasa general de churn", f"{churn_rate_total:.2f}%")

    col1, col2 = st.columns(2)
    with col1:
        geo_churn = df.groupby("Geography")["Exited"].mean().sort_values() * 100
        st.subheader("Churn por país")
        st.bar_chart(geo_churn)

    with col2:
        age_churn = df.groupby("AgeGroup")["Exited"].mean().sort_values() * 100
        st.subheader("Churn por grupo de edad")
        st.bar_chart(age_churn)

    st.subheader("Mapa de calor: Churn por país y edad")
    heatmap_data = df.pivot_table(index="Geography", columns="AgeGroup", values="Exited", aggfunc="mean") * 100
    fig, ax = plt.subplots()
    sns.heatmap(heatmap_data, annot=True, cmap="Reds", fmt=".1f", ax=ax)
    st.pyplot(fig)
