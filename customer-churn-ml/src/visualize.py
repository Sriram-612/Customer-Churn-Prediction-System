import os

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__" and __package__ is None:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.preprocess import DATA_PATH, clean_and_select_features, load_raw_data


REPORTS_DIR = "reports"


def ensure_reports_dir():
    os.makedirs(REPORTS_DIR, exist_ok=True)


def plot_churn_distribution(df):
    ensure_reports_dir()
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Churn", data=df)
    plt.title("Churn Distribution")
    plt.xlabel("Churn (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "churn_distribution.png"))
    plt.close()


def plot_monthly_charges_vs_churn(df):
    ensure_reports_dir()
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
    plt.title("Monthly Charges vs Churn")
    plt.xlabel("Churn (0 = No, 1 = Yes)")
    plt.ylabel("Monthly Charges")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "monthly_charges_vs_churn.png"))
    plt.close()


def plot_contract_vs_churn(df):
    ensure_reports_dir()
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Contract", hue="Churn", data=df)
    plt.title("Contract Type vs Churn")
    plt.xlabel("Contract Type")
    plt.ylabel("Count")
    plt.legend(title="Churn")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "contract_vs_churn.png"))
    plt.close()


def plot_tenure_distribution(df):
    ensure_reports_dir()
    plt.figure(figsize=(6, 4))
    sns.histplot(df["tenure"], bins=30, kde=True)
    plt.title("Tenure Distribution")
    plt.xlabel("Tenure (months)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "tenure_distribution.png"))
    plt.close()


def run_all_plots():
    print("Loading data for visualization...")
    df_raw = load_raw_data(DATA_PATH)
    df = clean_and_select_features(df_raw)

    print("Generating plots...")
    plot_churn_distribution(df)
    plot_monthly_charges_vs_churn(df)
    plot_contract_vs_churn(df)
    plot_tenure_distribution(df)
    print(f"Plots saved to '{REPORTS_DIR}' directory.")


if __name__ == "__main__":
    run_all_plots()

