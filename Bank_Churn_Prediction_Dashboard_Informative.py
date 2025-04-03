
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Bank Churn Prediction Dashboard', layout='wide')

# Custom styles for a more attractive appearance
st.markdown('''<style>
body {background-color: #F5F5F5;}
.sidebar .sidebar-content {background-color: #FFFFFF;}
</style>''', unsafe_allow_html=True)

# Load dataset (Proporcionado previamente)
data = pd.read_csv('Churn_Modelling.csv')

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

# Entrenar el modelo con datos proporcionados
model = XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=50, max_depth=3, use_label_encoder=False)
model.fit(X, y)

# Realizar predicciones y clasificación de riesgo
y_pred_proba = model.predict_proba(X)[:, 1]
risk_levels = pd.cut(y_pred_proba, bins=[0, 0.33, 0.66, 1], labels=['Bajo', 'Medio', 'Alto'])
data_cleaned['Probabilidad de Abandono'] = y_pred_proba
data_cleaned['Nivel de Riesgo'] = risk_levels

# Mostrar resumen de datos
st.title('Bank Churn Prediction - Dashboard Informativo')
st.write('### Resumen de los datos')
st.dataframe(data_cleaned[['Probabilidad de Abandono', 'Nivel de Riesgo', 'Exited']].head(20))

# Mostrar distribución de riesgo
st.write('### Distribución de Niveles de Riesgo')
risk_counts = data_cleaned['Nivel de Riesgo'].value_counts(normalize=True) * 100
fig, ax = plt.subplots()
risk_counts.plot(kind='bar', color=['green', 'orange', 'red'], ax=ax)
plt.title('Distribución de Niveles de Riesgo')
plt.ylabel('Porcentaje')
st.pyplot(fig)

# Mostrar importancia de características
st.write('### Importancia de las Características')
importance = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)
st.dataframe(feature_importance_df)

# Visualizar importancia de características
fig2, ax2 = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), ax=ax2)
plt.title('Top 10 Características Más Importantes')
st.pyplot(fig2)

# Mostrar métricas
st.write('### Resumen de Métricas')
accuracy = model.score(X, y)
st.write(f'Precisión del Modelo: {accuracy * 100:.2f}%')

st.write('Este dashboard es solo informativo y se basa en los datos proporcionados originalmente.')
