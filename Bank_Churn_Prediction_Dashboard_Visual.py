
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Bank Churn Prediction Dashboard', layout='wide')

# Configuración de estilo general
st.markdown('''<style>
body {background-color: #F0F2F6;}
h1 {color: #1f77b4;}
</style>''', unsafe_allow_html=True)

# Cargar datos
DATA_PATH = 'Churn_Modelling.csv'
data = pd.read_csv(DATA_PATH)

# Preprocesamiento de datos
irrelevant_columns = ['RowNumber', 'CustomerId', 'Surname']
data_cleaned = data.drop(columns=irrelevant_columns)
data_cleaned = pd.get_dummies(data_cleaned, columns=['Geography', 'Gender'], drop_first=True)

# Normalización de características numéricas
numeric_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
scaler = StandardScaler()
data_cleaned[numeric_columns] = scaler.fit_transform(data_cleaned[numeric_columns])

# Separar características y objetivo
X = data_cleaned.drop(columns=['Exited'])
y = data_cleaned['Exited']

# Entrenar el modelo con los datos proporcionados
model = XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=50, max_depth=3, use_label_encoder=False)
model.fit(X, y)

# Realizar predicciones y clasificación de riesgo
probs = model.predict_proba(X)[:, 1]
data_cleaned['Probabilidad de Abandono'] = probs
risk_levels = pd.cut(probs, bins=[0, 0.33, 0.66, 1], labels=['Bajo', 'Medio', 'Alto'])
data_cleaned['Nivel de Riesgo'] = risk_levels

# Visualización del Dashboard
st.title('Bank Churn Prediction Dashboard - Visual')

# Gráfico de Distribución de Niveles de Riesgo
st.subheader('Distribución de Niveles de Riesgo')
risk_counts = data_cleaned['Nivel de Riesgo'].value_counts()
fig1, ax1 = plt.subplots()
risk_counts.plot(kind='bar', color=['green', 'orange', 'red'], ax=ax1)
plt.title('Distribución de Riesgos de Abandono')
plt.xlabel('Nivel de Riesgo')
plt.ylabel('Número de Clientes')
st.pyplot(fig1)

# Gráfico de Importancia de Características
st.subheader('Importancia de las Características')
importance = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Característica': X.columns,
    'Importancia': importance
}).sort_values(by='Importancia', ascending=False)

fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.barplot(y='Característica', x='Importancia', data=feature_importance_df.head(10), ax=ax2)
plt.title('Top 10 Características Más Relevantes')
st.pyplot(fig2)

# Gráfico Heatmap de Correlaciones
st.subheader('Heatmap de Correlaciones')
fig3, ax3 = plt.subplots(figsize=(10, 8))
sns.heatmap(data_cleaned.corr(), cmap='coolwarm', annot=False)
st.pyplot(fig3)

# Métricas de Rendimiento
st.subheader('Métricas del Modelo')
accuracy = model.score(X, y)
st.write(f'Precisión del Modelo: {accuracy * 100:.2f}%')

st.write('Este dashboard es visualmente mejorado para facilitar la comprensión de los resultados obtenidos con el modelo entrenado. Las gráficas presentadas reflejan información relevante para la toma de decisiones.')
