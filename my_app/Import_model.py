import mlflow.sklearn
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

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
#model_path = r"C:\Users\j_aka\Desktop\mlops\mlartifacts\606383372813198707\models\m-8f6d8296881841c78964badb00e5f626"
model_path = r"file:///C:/Users/j_aka/Desktop/mlops/mlartifacts/606383372813198707/models/m-32d10fd82bd54d67836fc6e309edba1a/artifacts"


def load_model():
    """
    Charge le modèle enregistré localement depuis MLflow
    """
    #return mlflow.sklearn.load_model(model_path)
    return mlflow.pyfunc.load_model(model_path)

def prepare_input(user_input: dict) -> pd.DataFrame:
    df = pd.DataFrame([user_input])

    # Colonnes obligatoires cat
    for col in ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]:
        if col not in df.columns or df[col].isna().any():
            df[col] = "NA"

    # Mapping taux et calculs
    df["loan_int_rate"] = df.get("loan_int_rate", df["loan_intent"].map(DEFAULT_LOAN_RATES))
    df["loan_percent_income"] = df["loan_amnt"] / df["person_income"]

    # Conversion numérique selon le schéma
    for col in ["person_age", "person_income", "loan_amnt", "cb_person_cred_hist_length"]:
        if col not in df.columns or df[col].isna().any():
            df[col] = 0
        df[col] = df[col].astype(int)

    for col in ["person_emp_length", "loan_int_rate", "loan_percent_income"]:
        if col not in df.columns:
            df[col] = float("nan")
        df[col] = df[col].astype(float)

    # Conversion string
    for col in ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]:
        df[col] = df[col].astype(str)

    return df

model = load_model()

def predict(user_inputs: list):
    """
    Prédit la valeur de loan_status pour une entrée utilisateur
    """
    df_prepared = prepare_input(user_inputs)
    return model.predict(df_prepared)[0]


# Test 
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