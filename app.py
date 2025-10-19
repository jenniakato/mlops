import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Chargement de la donnée 
df = pd.read_csv(r"C:\Users\j_aka\Desktop\mlops\credit_risk_dataset.csv")

# définition du taux d'interêt par défaut selon le motif du prêt
default_rate_map = {
    "MEDICAL": 0.09,
    "EDUCATION": 0.01,
    "DEBTCONSOLIDATION": 0.12,
    "VENTURE": 0.15,
    "HOMEIMPROVEMENT": 0.08,
    "PERSONAL": 0.00
}

# Les colonnes utilisées pour l'entraînement
selected_features = [
    "person_age",
    "person_emp_length",
    "loan_intent",
    "person_home_ownership",
    "person_income",
    "loan_amnt",
    "loan_percent_income",
    "default_rate"
]

# Calcul du ratio et du taux par défaut 
df["loan_percent_income"] = df["loan_amnt"] / df["person_income"]
df["default_rate"] = df["loan_intent"].str.upper().map(default_rate_map).fillna(0.00)

# Définition des features et de la cible
X = df[selected_features]
y = df["loan_status"]

# Colonnes numériques et catégorielles pour l'encodage 
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Préprocessing => Encodage 
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=4,
        bootstrap=True,
        oob_score=False,
        random_state=888
    ))
])

# Entraînement du modèle
pipeline.fit(X, y)

# Prédiction
y_pred = pipeline.predict(X)
st.subheader("📊 Performance du modèle")
st.metric("Accuracy", f"{accuracy_score(y, y_pred):.3f}")
st.metric("Precision", f"{precision_score(y, y_pred):.3f}")
st.metric("Recall", f"{recall_score(y, y_pred):.3f}")
st.metric("F1 Score", f"{f1_score(y, y_pred):.3f}")

# Interface utilisateur
#Titre de la page 
st.title("📊 Simulateur de prêt bancaire")

# Les données à saisir par l'utlisateur 
age = st.number_input("Âge", min_value=18, max_value=100, step=1,value=None)
experience = st.number_input("Ancienneté professionnelle", min_value=0, max_value=80, step=1)
intent_options = [""] + sorted(df["loan_intent"].dropna().unique())
ownership_options = [""] + sorted(df["person_home_ownership"].dropna().unique())
intent = st.selectbox("Motif du prêt", intent_options)
ownership = st.selectbox("Type d’occupation", ownership_options)
income = st.number_input("Revenu annuel brut (€)", min_value=0, step=100)
loan = st.number_input("Montant du prêt (€)", min_value=0, step=100)

# Calcul automatique du taux d'interêt du prêt te du Ration montant prêt/revenu
loan_percent_income = loan / income if income > 0 else 0
intent_upper = intent.upper() if intent else ""
default_rate = default_rate_map.get(intent_upper, 0.00)

if loan > 0 and income > 0:
    st.markdown(f"💡 **Loan Percent Income** : {loan_percent_income:.2%}")
if intent:
    st.markdown(f"📌 **Taux par défaut appliqué** : {default_rate:.2%}")

# Affichage d'un récap du saisie de l'utlisateur 
input_dict = {
    "person_age": age,
    "person_emp_length": experience,
    "loan_intent": intent,
    "person_home_ownership": ownership,
    "person_income": income,
    "loan_amnt": loan,
    "loan_percent_income": loan_percent_income,
    "default_rate": default_rate
}

# Vérification des champs obligatoire 
missing = [k for k, v in input_dict.items() if v in ["", None] or (isinstance(v, float) and np.isnan(v))]
if missing:
    st.warning(f"Veuillez remplir tous les champs : {', '.join(missing)}")
else:
    input_df = pd.DataFrame([input_dict])
    with st.expander("🔍 Données utilisées pour la prédiction"):
        st.dataframe(input_df.T.rename(columns={0: "Valeur"}))

    if st.button("Prédire le statut du prêt"):
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0][prediction]
        st.subheader("📌 Résultat de la simulation")
        st.write(f"✅ Statut prédit du prêt : **{'Approuvé' if prediction == 0 else 'Refusé'}**")
        st.write(f"📈 Confiance du modèle : **{proba:.2%}**")