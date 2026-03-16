import pandas as pd
import numpy as np
from typing import Tuple

from sklearn.model_selection import train_test_split


DATA_PATH = "data/Telco-Customer-Churn.csv"
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

# Features that will be used in the modeling pipeline and Streamlit app
FEATURE_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "PaymentMethod",
]


def load_raw_data(csv_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the Telco customer churn dataset from the given CSV path.
    """
    df = pd.read_csv(csv_path)
    return df


def clean_and_select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform dataset cleaning and select only the features required
    for modeling and prediction.

    Steps:
    - Remove customerID column.
    - Convert TotalCharges to numeric.
    - Handle missing values.
    - Convert Churn column to binary (Yes=1, No=0).
    - Keep only FEATURE_COLUMNS + TARGET_COLUMN.
    """
    df = df.copy()

    # Drop identifier column if present
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # Ensure TotalCharges is numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Handle missing values: numeric -> median, categorical -> mode
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
        else:
            mode_series = df[col].mode()
            if not mode_series.empty:
                mode_value = mode_series[0]
                df[col] = df[col].fillna(mode_value)

    # Encode target column
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    # Restrict to selected feature columns and target
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    required_columns = available_features + [TARGET_COLUMN]
    missing_required = [c for c in required_columns if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in dataset: {missing_required}")

    df = df[required_columns]
    return df


def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target from the cleaned dataframe.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split the dataset into training and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


if __name__ == "__main__":
    # Simple manual run helper to inspect data info
    data = load_raw_data()
    print("Raw dataset shape:", data.shape)
    print(data.info())

