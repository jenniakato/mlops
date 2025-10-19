import mlflow.sklearn
import pandas as pd

# Mapping par défaut des taux en fonction du loan_intent
DEFAULT_LOAN_RATES = {
    "MEDICAL": 0.09,
    "EDUCATION": 0.01,
    "DEBTCONSOLIDATION": 0.12,
    "VENTURE": 0.15,
    "HOMEIMPROVEMENT": 0.08,
    "PERSONAL": 0.00
}

# Chemin local vers le modèle MLflow
MODEL_PATH = r"C:\Users\j_aka\Desktop\mlops\mlartifacts\606383372813198707\models\m-5c2ddd0195bb43af80c1afa469e96f2b\artifacts"

def load_model():
    """
    Charge le modèle enregistré localement depuis MLflow
    """
    return mlflow.sklearn.load_model(MODEL_PATH)

def prepare_input(user_input: dict) -> pd.DataFrame:
    """
    Transforme les données utilisateur en DataFrame compatible avec le pipeline
    """
    input_df = pd.DataFrame([user_input])
    
    # Remplissage du taux par défaut en fonction de loan_intent
    if "loan_int_rate" not in input_df.columns:
        input_df["loan_int_rate"] = input_df["loan_intent"].map(DEFAULT_LOAN_RATES)
    
    # Calcul de loan_percent_income
    if "loan_percent_income" not in input_df.columns:
        input_df["loan_percent_income"] = input_df["loan_amnt"] / input_df["person_income"]
    
    # Colonnes manquantes pour l'app
    input_df["loan_grade"] = None
    input_df["cb_person_default_on_file"] = None
    input_df["cb_person_cred_hist_length"] = None
    
    return input_df

def predict(user_input: dict):
    """
    Prédit la valeur de loan_status pour une entrée utilisateur
    """
    model = load_model()
    df_prepared = prepare_input(user_input)
    return model.predict(df_prepared)[0]

# Test rapide
if __name__ == "__main__":
    user_data = {
        "person_age": 35,
        "person_income": 15000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5,
        "loan_intent": "PERSONAL",
        "loan_amnt": 3000
    }
    print("✅ Prédiction :", predict(user_data))