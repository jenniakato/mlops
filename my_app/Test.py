import pytest
from Import_model import predict

def test_predict_personal():
    user_data = {
        "person_age": 35,
        "person_income": 15000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5,
        "loan_intent": "PERSONAL",
        "loan_amnt": 3000
    }
    result = predict(user_data)
    assert result in [0, 1], "La prédiction doit être 0 ou 1"

def test_predict_medical():
    user_data = {
        "person_age": 50,
        "person_income": 25000,
        "person_home_ownership": "OWN",
        "person_emp_length": 15,
        "loan_intent": "MEDICAL",
        "loan_amnt": 5000
    }
    result = predict(user_data)
    assert result in [0, 1], "La prédiction doit être 0 ou 1"
