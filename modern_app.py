
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='Churn Prediction Modern App', layout='wide')

css_code = '''
<style>
body {
    font-family: Arial, sans-serif;
    scroll-behavior: smooth;
}
section {
    padding: 100px 0px;
    border-bottom: 1px solid #ddd;
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

st.title('Predicción de Churn con Machine Learning')
st.subheader('Problema')
st.write("""
El churn o abandono de clientes es un problema serio en el sector bancario.
El objetivo es identificar clientes con alto riesgo de abandono y aplicar estrategias de retención de manera anticipada.

Cada cliente que se va puede costar entre $200 y $1,000 USD anuales.
La tasa actual de churn en este banco es aproximadamente del 20.37%.
""")

st.title('Modelo Utilizado: Random Forest')
st.write("""
Se ha entrenado un modelo de clasificación con Random Forest utilizando características como:
- Edad, País, Balance, Número de productos, Actividad del cliente.
- Las variables categóricas se convierten mediante One-Hot Encoding.
- Las variables numéricas se escalan para mejorar la precisión.
""")

st.title('Resultados del Modelo')
churn_rate = df["Exited"].mean() * 100
st.metric(label="Tasa general de churn", value=f"{churn_rate:.2f}%")

geo_churn = df.groupby("Geography")["Exited"].mean() * 100
fig1 = px.bar(geo_churn, title="Churn por País", labels={'value':'Churn Rate (%)', 'index':'País'})
st.plotly_chart(fig1, use_container_width=True)

age_churn = df.groupby("AgeGroup")["Exited"].mean() * 100
fig2 = px.bar(age_churn, title="Churn por Grupo de Edad", labels={'value':'Churn Rate (%)', 'index':'Grupo de Edad'})
st.plotly_chart(fig2, use_container_width=True)

st.title('Predicción Interactiva')

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
