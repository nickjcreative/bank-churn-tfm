
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

# Load trained model
model = XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=50, max_depth=3, use_label_encoder=False)

# Load dataset
uploaded_file = st.file_uploader('Upload a CSV file for prediction', type=['csv'])

def preprocess_data(data):
    irrelevant_columns = ['RowNumber', 'CustomerId', 'Surname']
    data_cleaned = data.drop(columns=irrelevant_columns)
    data_cleaned = pd.get_dummies(data_cleaned, columns=['Geography', 'Gender'], drop_first=True)

    numeric_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    scaler = StandardScaler()
    data_cleaned[numeric_columns] = scaler.fit_transform(data_cleaned[numeric_columns])

    return data_cleaned

def predict_risk(model, data):
    probabilities = model.predict_proba(data)[:, 1]
    risk_levels = pd.cut(probabilities, bins=[0, 0.33, 0.66, 1], labels=['Bajo', 'Medio', 'Alto'])
    data['Probabilidad de Abandono'] = probabilities
    data['Nivel de Riesgo'] = risk_levels
    return data

if uploaded_file:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    preprocessed_data = preprocess_data(data)

    # Train the model with the provided dataset
    X = preprocessed_data.drop(columns=['Exited'])
    y = preprocessed_data['Exited']
    model.fit(X, y)

    # Make predictions
    predictions = predict_risk(model, X)

    # Display the results
    st.write('### Predicted Risk Levels')
    st.dataframe(predictions[['Probabilidad de Abandono', 'Nivel de Riesgo']].head(20))

    # Show distribution of risk levels
    st.write('### Risk Level Distribution')
    risk_counts = predictions['Nivel de Riesgo'].value_counts(normalize=True) * 100
    fig, ax = plt.subplots()
    risk_counts.plot(kind='bar', color=['green', 'orange', 'red'], ax=ax)
    plt.title('Distribution of Risk Levels')
    plt.ylabel('Percentage')
    st.pyplot(fig)

    # Display feature importance
    st.write('### Feature Importance')
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    st.dataframe(feature_importance_df)

    # Visualize feature importance
    fig2, ax2 = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), ax=ax2)
    plt.title('Top 10 Feature Importances')
    st.pyplot(fig2)

    # Metrics Display
    st.write('### Metrics Summary')
    accuracy = model.score(X, y)
    st.write(f'Accuracy of the Model: {accuracy * 100:.2f}%')

st.write('## Upload your CSV file to see predictions and risk analysis')
