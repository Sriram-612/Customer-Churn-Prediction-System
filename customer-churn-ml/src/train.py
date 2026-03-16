import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from preprocess import (
    DATA_PATH,
    FEATURE_COLUMNS,
    clean_and_select_features,
    create_train_test_split,
    load_raw_data,
    split_features_target,
)


MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "churn_model.pkl")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing pipeline using:
    - StandardScaler for numeric features
    - OneHotEncoder for categorical features
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def build_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    """
    Create model pipelines for Logistic Regression, Decision Tree, and Random Forest.
    """
    models = {
        "log_reg": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", DecisionTreeClassifier(random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
            ]
        ),
    }
    return models


def evaluate_model(
    name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate a model on the test set and print metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n=== Evaluation for {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    return {
        "name": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
    }


def train_and_select_best_model() -> Tuple[str, Pipeline]:
    """
    Full training routine:
    - Load and clean data
    - Split into train and test
    - Train three models
    - Evaluate and select the best by accuracy
    - Persist best model to disk
    """
    print("Loading raw data...")
    df_raw = load_raw_data(DATA_PATH)
    print("Cleaning and selecting features...")
    df_clean = clean_and_select_features(df_raw)

    print("Dataset after cleaning:")
    print(df_clean.head())
    print(df_clean.info())

    X, y = split_features_target(df_clean)

    print("Creating train-test split...")
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    preprocessor = build_preprocessor(X_train)
    models = build_models(preprocessor)

    scores = []
    trained_models: Dict[str, Pipeline] = {}

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train, y_train)
        trained_models[name] = model
        metrics = evaluate_model(name, model, X_test, y_test)
        scores.append(metrics)

    # Select best model based on accuracy
    best = max(scores, key=lambda x: x["accuracy"])
    best_name = best["name"]
    best_model = trained_models[best_name]

    print("\n=== Model Selection ===")
    for s in scores:
        print(f"{s['name']}: accuracy={s['accuracy']:.4f}, precision={s['precision']:.4f}, recall={s['recall']:.4f}")
    print(f"\nBest model: {best_name} with accuracy={best['accuracy']:.4f}")

    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Saving best model to {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    print("Model training and saving complete.")
    return best_name, best_model


if __name__ == "__main__":
    train_and_select_best_model()

