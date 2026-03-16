import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from preprocess import FEATURE_COLUMNS


MODELS_DIR = "models"
MODEL_FILENAME = "churn_model.pkl"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)


def load_model(model_path: str = MODEL_PATH):
    """
    Load the trained churn prediction model pipeline from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please train the model by running src/train.py first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def build_input_dataframe(
    tenure: float,
    monthly_charges: float,
    total_charges: float,
    contract: str,
    internet_service: str,
    payment_method: str,
) -> pd.DataFrame:
    """
    Construct a single-row DataFrame for prediction that matches the
    feature layout used during training.
    """
    data: Dict[str, list] = {
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Contract": [contract],
        "InternetService": [internet_service],
        "PaymentMethod": [payment_method],
    }

    # Ensure the columns are in the same order and presence as FEATURE_COLUMNS
    df = pd.DataFrame(data)
    # Reindex to enforce column order; any missing columns (should not happen) will be filled with 0 or "Unknown"
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[FEATURE_COLUMNS]
    return df


def predict_churn(
    tenure: float,
    monthly_charges: float,
    total_charges: float,
    contract: str,
    internet_service: str,
    payment_method: str,
) -> Tuple[int, float]:
    """
    Predict churn label and probability for a single customer.

    Returns:
        prediction_label (int): 0 for no churn, 1 for churn
        churn_probability (float): probability of churn (between 0 and 1)
    """
    model = load_model()
    input_df = build_input_dataframe(
        tenure=tenure,
        monthly_charges=monthly_charges,
        total_charges=total_charges,
        contract=contract,
        internet_service=internet_service,
        payment_method=payment_method,
    )

    proba = model.predict_proba(input_df)[0, 1]
    pred = int(proba >= 0.5)
    return pred, float(proba)


if __name__ == "__main__":
    # Example manual test
    example_pred, example_proba = predict_churn(
        tenure=12,
        monthly_charges=70.0,
        total_charges=840.0,
        contract="Month-to-month",
        internet_service="Fiber optic",
        payment_method="Electronic check",
    )
    print("Predicted churn:", example_pred)
    print("Churn probability:", example_proba)

