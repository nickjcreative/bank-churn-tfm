
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("🔍 Predicción de Churn Bancario")

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

st.sidebar.title("Opciones")
menu = st.sidebar.radio("Selecciona una vista:", ["Predicción individual", "Análisis por grupo", "Dashboard general"])

if menu == "Predicción individual":
    st.header("Predicción para un cliente")
    index = st.slider("Selecciona un cliente", 0, len(df)-1, 0)
    cliente = df.iloc[[index]]
    st.write(cliente)

    X_cliente = cliente.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
    X_cliente = pd.get_dummies(X_cliente, drop_first=True)
    X_cliente = X_cliente.reindex(columns=X.columns, fill_value=0)
    X_cliente_scaled = scaler.transform(X_cliente)

    proba = model.predict_proba(X_cliente_scaled)[0][1]

    st.metric("Probabilidad de churn", f"{proba*100:.2f}%")

    fig, ax = plt.subplots()
    ax.barh(["Probabilidad de churn"], [proba], color="red" if proba > 0.6 else "orange" if proba > 0.3 else "green")
    ax.set_xlim(0, 1)
    st.pyplot(fig)

elif menu == "Análisis por grupo":
    st.header("Análisis por grupo")
    geo = st.selectbox("Selecciona un país", df["Geography"].unique())
    subset = df[df["Geography"] == geo]
    st.write(f"Total de clientes en {geo}: {len(subset)}")
    st.bar_chart(subset["AgeGroup"].value_counts())

elif menu == "Dashboard general":
    st.header("Dashboard general")
    churn_rate = df["Exited"].mean() * 100
    st.metric("Tasa general de churn", f"{churn_rate:.2f}%")
    geo_churn = df.groupby("Geography")["Exited"].mean().sort_values() * 100
    st.bar_chart(geo_churn)
    age_churn = df.groupby("AgeGroup")["Exited"].mean().sort_values() * 100
    st.bar_chart(age_churn)
