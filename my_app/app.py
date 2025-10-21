import streamlit as st
from Import_model import predict
from Import_model import DEFAULT_LOAN_RATES

st.set_page_config(page_title="Prédiction Risque de Crédit", page_icon="💸", layout="centered")

st.title("Simulation du risque de prêt")
st.markdown("Remplissez les informations ci-dessous pour estimer le risque de défaut d’un emprunteur.")

# --- Masquer les flèches + et - des number_input ---
hide_arrows = """
    <style>
    input[type=number]::-webkit-inner-spin-button, 
    input[type=number]::-webkit-outer-spin-button { 
        -webkit-appearance: none; 
        margin: 0; 
    }
    </style>
"""
st.markdown(hide_arrows, unsafe_allow_html=True)

# --- Entrées utilisateur ---
col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Âge de l'emprunteur", min_value=18, max_value=100, value=None, format="%d")
    person_income = st.number_input("Revenu annuel (€)", min_value=1000, step=1000, value=None, format="%d")
    person_home_ownership = st.selectbox("Type de logement", ["", "RENT", "OWN", "MORTGAGE", "OTHER"])

with col2:
    person_emp_length = st.number_input("Ancienneté professionnelle (années)", min_value=0, max_value=50, value=None, format="%d")
    loan_intent = st.selectbox("Motif du prêt", ["", "PERSONAL", "EDUCATION", "MEDICAL", "DEBTCONSOLIDATION", "VENTURE", "HOMEIMPROVEMENT"])
    loan_amnt = st.number_input("Montant du prêt (€)", min_value=100, max_value=500000000, step=500, value=None, format="%d")

# --- Vérification avant calcul ---
if st.button("🔍 Lancer la prédiction"):
    if None in [person_age, person_income, person_emp_length, loan_amnt] or person_home_ownership == "" or loan_intent == "":
        st.warning("⚠️ Merci de remplir tous les champs avant de lancer la prédiction.")
    else:
        # --- Calculs automatiques ---
        loan_int_rate = DEFAULT_LOAN_RATES.get(loan_intent.upper(), 0.05)
        loan_percent_income = loan_amnt / person_income

        st.markdown(f"**Taux d'intérêt appliqué :** `{loan_int_rate * 100:.2f}%`")
        st.markdown(f"**Pourcentage du revenu emprunté :** `{loan_percent_income * 100:.2f}%`")

        # --- Prédiction ---
        user_data = {
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_amnt": loan_amnt
        }

        try:
            prediction = predict(user_data)
            if prediction == 1:
                st.error("⚠️ Risque ÉLEVÉ de défaut de paiement.")
            else:
                st.success("🟢 Risque FAIBLE de défaut de paiement.")
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {type(e).__name__} - {e}")