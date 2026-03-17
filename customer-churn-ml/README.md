## Customer Churn Prediction System

This project implements an end-to-end **Customer Churn Prediction System** for a telecom company using Python and machine learning. It loads the Telco Customer Churn dataset, performs preprocessing and exploratory data analysis, trains multiple models, selects the best one, and exposes it via a Streamlit web application.

---

### Dataset

The project uses the Telco Customer Churn dataset:

- Source file (provided): `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Project location: `data/Telco-Customer-Churn.csv`

Key columns used:

- `tenure`: Number of months the customer has stayed with the company.
- `MonthlyCharges`: The amount charged to the customer monthly.
- `TotalCharges`: The total amount charged to the customer.
- `Contract`: Type of contract (Month-to-month, One year, Two year).
- `InternetService`: Type of internet service (DSL, Fiber optic, No).
- `PaymentMethod`: Customer payment method.
- `Churn`: Target label (Yes/No), converted to binary (1 for Yes, 0 for No).

---

### Project Structure

```text
customer-churn-ml/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── models/
│   └── churn_model.pkl
│
├── reports/
│   ├── churn_distribution.png
│   ├── monthly_charges_vs_churn.png
│   ├── contract_vs_churn.png
│   └── tenure_distribution.png
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── visualize.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

### Components Overview

- `src/preprocess.py`
  - Loads the CSV dataset with pandas.
  - Cleans data:
    - Removes `customerID`.
    - Converts `TotalCharges` to numeric.
    - Handles missing values (median for numeric, mode for categorical).
    - Converts `Churn` to binary (Yes → 1, No → 0).
  - Restricts the dataset to the main modeling features and target.
  - Provides helper functions for feature/target split and train/test split.

- `src/train.py`
  - Loads and preprocesses the dataset using `preprocess.py`.
  - Splits data into train and test sets (`test_size=0.2`, `random_state=42`).
  - Builds a preprocessing pipeline using:
    - `StandardScaler` for numeric features.
    - `OneHotEncoder` for categorical features.
  - Trains three models:
    - Logistic Regression
    - Decision Tree
    - Random Forest
  - Evaluates each model (accuracy, precision, recall, confusion matrix, classification report).
  - Selects the best model based on accuracy.
  - Saves the best model pipeline to `models/churn_model.pkl` using pickle.

- `src/predict.py`
  - Loads the saved model pipeline from `models/churn_model.pkl`.
  - Builds a single-row pandas DataFrame from input customer data.
  - Returns churn prediction (0/1) and churn probability.

- `src/visualize.py`
  - Generates exploratory data analysis (EDA) plots:
    - Churn distribution.
    - Monthly charges vs churn.
    - Contract type vs churn.
    - Tenure distribution.
  - Saves plots into the `reports/` directory.

- `app.py`
  - Streamlit web application.
  - UI fields:
    - Tenure
    - MonthlyCharges
    - TotalCharges
    - Contract
    - InternetService
    - PaymentMethod
  - Loads the saved model via `src.predict`.
  - Predicts churn and displays:
    - Prediction result (likely to churn or not).
    - Churn probability.

---

### Installation

1. **Clone or copy the project into your environment.**

2. **Navigate to the project folder:**

```bash
cd customer-churn-ml
```

3. **Create and activate a virtual environment** (recommended).
For best compatibility on Windows, use **Python 3.11**:

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux
```

4. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

### How to Train the Model

1. Ensure the dataset exists at `data/Telco-Customer-Churn.csv`.

2. Run the training script:

```bash
python src/train.py
```

This will:

- Load and preprocess the data.
- Train Logistic Regression, Decision Tree, and Random Forest models.
- Evaluate each model and print:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix
  - Classification Report
- Select the best model based on accuracy.
- Save the best model to `models/churn_model.pkl`.

---

### How to Generate EDA Visualizations

To generate and save the EDA plots into the `reports/` directory, run:

```bash
python src/visualize.py
```

The following plots will be created:

- `churn_distribution.png`
- `monthly_charges_vs_churn.png`
- `contract_vs_churn.png`
- `tenure_distribution.png`

---

### How to Run the Streamlit App

After training the model (ensuring `models/churn_model.pkl` exists), start the Streamlit app:

```bash
streamlit run app.py
```

Then open the URL provided in the terminal (typically `http://localhost:8501`) in your browser.

In the app:

- Fill in the customer details (tenure, charges, contract, internet service, payment method).
- Click **Predict**.
- View:
  - The churn prediction (likely to churn / not likely to churn).
  - The associated churn probability.

---

### Notes

- The preprocessing and model training steps are designed to be reproducible and production-ready.
- The modeling pipeline encapsulates both preprocessing and the classifier, ensuring consistent handling of new data during prediction.
- If the model file is missing, the Streamlit app will show an error instructing you to run `python src/train.py` first.

