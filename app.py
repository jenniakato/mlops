import streamlit as st
import pandas as pd
import mlflow.sklearn
import numpy as np

# -----------------------------
# 1️⃣ Charger le modèle MLflow
# -----------------------------
model_uri = r"C:\Users\j_aka\Desktop\mlops\RDF_ALGO.ipynb" 
pipeline = mlflow.sklearn.load_model(model_uri)

# -----------------------------
# 2️⃣ Titre de l'application
# -----------------------------
st.title("Prédiction de défaut de paiement de crédit")
st.write("Entrez les informations du client pour prédire le statut du prêt.")

# -----------------------------
# 3️⃣ Définir les features
# -----------------------------
# Pour l'exemple, on suppose que ces colonnes existent dans ton dataset
num_features = ['loan_percent_income', 'loan_int_rate']
cat_features = ['loan_grade']

# -----------------------------
# 4️⃣ Formulaire utilisateur
# -----------------------------
with st.form(key='input_form'):
    inputs = {}
    # Colonnes numériques
    for col in num_features:
        inputs[col] = st.number_input(f"{col}", value=0.0)
    
    # Colonnes catégorielles
    for col in cat_features:
        # On pourrait récupérer la liste des catégories depuis le modèle, ici exemple simple
        inputs[col] = st.selectbox(f"{col}", ['A','B','C','D','E','F','G'])
    
    submitted = st.form_submit_button("Prédire")

# -----------------------------
# 5️⃣ Prédiction
# -----------------------------
if submitted:
    input_df = pd.DataFrame([inputs])
    pred = pipeline.predict(input_df)[0]
    pred_proba = pipeline.predict_proba(input_df)[0]

    st.subheader("Résultat de la prédiction")
    st.write(f"✅ Prédiction du statut du prêt : **{pred}**")
    st.write("Probabilités :")
    prob_df = pd.DataFrame([pred_proba], columns=pipeline.named_steps['model'].classes_)
    st.dataframe(prob_df.T.rename(columns={0: "Probabilité"}))
