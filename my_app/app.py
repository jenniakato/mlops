import streamlit as st
import pandas as pd
from Import_model import load_model

st.title("Simulation de prêt bancaire")

# Saisie utilisateur
age = st.number_input("Âge", min_value=18, max_value=100)
income = st.number_input("Revenu mensuel (€)")
loan_amount = st.number_input("Montant du prêt (€)")

# Bouton de prédiction
if st.button("Simuler"):
    # Reconstruction du DataFrame en interne
    input_df = pd.DataFrame({
        "age": [age],
        "income": [income],
        "loan_amount": [loan_amount]
    })

    # Chargement du modèle
    model = load_model()

    # Prédiction
    prediction = model.predict(input_df)
    st.success(f"Statut prédit : {'✅ Approuvé' if prediction[0] == 1 else '❌ Refusé'}")
