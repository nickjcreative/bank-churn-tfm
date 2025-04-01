
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='Enhanced Churn Prediction App', layout='wide')

css_code = '''
<style>
body {
    font-family: Arial, sans-serif;
    scroll-behavior: smooth;
    background-color: #f0f2f6;
}

section {
    padding: 100px 0;
    border-bottom: 1px solid #ddd;
    background-color: white;
    margin-bottom: 20px;
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

st.title('Enhanced Churn Prediction App')
st.write('Explora cómo diferentes segmentos afectan la tasa de churn mientras navegas por la historia.')

st.subheader('Churn General por País')
fig1 = px.histogram(df, x="Geography", color="Exited", barmode="group", title="Churn por País")
st.plotly_chart(fig1, use_container_width=True)

st.subheader('Churn por Género')
gender_churn = df.groupby("Gender")["Exited"].mean() * 100
fig2 = px.bar(gender_churn, title="Churn por Género")
st.plotly_chart(fig2, use_container_width=True)

st.subheader('Churn por Grupo de Edad')
age_churn = df.groupby("AgeGroup")["Exited"].mean() * 100
fig3 = px.bar(age_churn, title="Churn por Grupo de Edad")
st.plotly_chart(fig3, use_container_width=True)

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

fig4 = px.bar(x=["Probabilidad de churn"], y=[proba], color=[proba], color_continuous_scale="Reds")
st.plotly_chart(fig4)

st.subheader('Conclusión')
st.write('Hemos explorado diferentes segmentos y sus efectos en el churn. Puedes aplicar esta herramienta para analizar clientes en riesgo y tomar decisiones informadas.')
st.write('¿Te gustaría implementar esta solución en tu banco o negocio? ¡Contáctanos para más detalles!')
st.button('Contact Us')
