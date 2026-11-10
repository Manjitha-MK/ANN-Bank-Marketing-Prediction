
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "bank+marketing",
    "bank",
    "bank-full.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "seed_experiments",
    "ann_model_1_seed_42.keras"
)


# --------------------------------------------------
# 2. LOAD DATA AND RECREATE PREPROCESSING
# --------------------------------------------------

df = pd.read_csv(DATA_PATH, sep=";")

# Duration is excluded because it is only known
# after the marketing call.
df = df.drop(columns=["duration"])

X = df.drop(columns=["y"])
y = df["y"].map({"no": 0, "yes": 1})

numeric_features = [
    "age", "balance", "day", "campaign", "pdays", "previous"
]

categorical_features = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome"
]

# Recreate the same data split used during training.
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Fit preprocessing only on training data.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ), categorical_features)
    ]
)

preprocessor.fit(X_train)


# --------------------------------------------------
# 3. LOAD TRAINED ANN MODEL
# --------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

print("\nANN Bank Marketing Prediction System")
print("------------------------------------")
print("Model: ANN Model 1")
print("Seed: 42")
print("Model loaded successfully.")


# --------------------------------------------------
# 4. INPUT VALIDATION
# --------------------------------------------------

def validate_customer(customer):

    # Check required fields
    required_fields = (
        numeric_features + categorical_features
    )

    for field in required_fields:
        if field not in customer:
            raise ValueError(f"Missing field: {field}")

    # Check numeric values
    for field in numeric_features:

        value = customer[field]

        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{field} must be a number."
            )

        if not np.isfinite(value):
            raise ValueError(
                f"{field} must be a finite number."
            )

    # Check reasonable input ranges
    if not 18 <= customer["age"] <= 100:
        raise ValueError("Age must be between 18 and 100.")

    if not 1 <= customer["day"] <= 31:
        raise ValueError("Day must be between 1 and 31.")

    if customer["campaign"] < 1:
        raise ValueError("Campaign must be at least 1.")

    if customer["previous"] < 0:
        raise ValueError("Previous contacts cannot be negative.")

    if customer["pdays"] < -1:
        raise ValueError("pdays must be -1 or greater.")

    # Check categorical values against training categories
    for field in categorical_features:

        allowed_values = set(X_train[field].unique())

        if customer[field] not in allowed_values:
            raise ValueError(
                f"Invalid {field}: {customer[field]}. "
                f"Allowed values: {sorted(allowed_values)}"
            )

    # Check numeric values against training data ranges
    for field in numeric_features:

        minimum = X_train[field].min()
        maximum = X_train[field].max()

        if not minimum <= customer[field] <= maximum:
            raise ValueError(
                f"{field} is outside the training data range "
                f"({minimum} to {maximum})."
            )

    return True


# --------------------------------------------------
# 5. PREDICTION FUNCTION
# --------------------------------------------------

def predict_subscription(customer):

    validate_customer(customer)

    # Convert customer information into DataFrame
    customer_df = pd.DataFrame(
        [customer],
        columns=numeric_features + categorical_features
    )

    # Apply the same preprocessing
    processed_customer = preprocessor.transform(
        customer_df
    ).astype("float32")

    # Predict probability
    probability = float(
        model.predict(
            processed_customer,
            verbose=0
        )[0][0]
    )

    prediction = 1 if probability >= 0.5 else 0

    print("\nPrediction Result")
    print("-----------------")
    print(f"Subscription probability: {probability:.2%}")

    if prediction == 1:
        print("Prediction: Likely to subscribe")
    else:
        print("Prediction: Unlikely to subscribe")

    return probability, prediction


# --------------------------------------------------
# 6. TERMINAL INPUT
# --------------------------------------------------

def get_customer_input():

    print("\nEnter Customer Details")
    print("----------------------")

    customer = {}

    try:

        # Numeric features
        customer["age"] = int(input("Age: "))
        customer["balance"] = float(input("Balance: "))
        customer["day"] = int(input("Day of month: "))
        customer["campaign"] = int(input("Number of contacts in campaign: "))
        customer["pdays"] = int(input("Days since previous contact (-1 if never): "))
        customer["previous"] = int(input("Previous contacts: "))

        # Categorical features
        print("\nJob options:", sorted(X_train["job"].unique()))
        customer["job"] = input("Job: ").strip()

        print("\nMarital options:", sorted(X_train["marital"].unique()))
        customer["marital"] = input("Marital status: ").strip()

        print("\nEducation options:", sorted(X_train["education"].unique()))
        customer["education"] = input("Education: ").strip()

        print("\nDefault options:", sorted(X_train["default"].unique()))
        customer["default"] = input("Has credit in default (yes/no): ").strip()

        print("\nHousing options:", sorted(X_train["housing"].unique()))
        customer["housing"] = input("Housing loan (yes/no): ").strip()

        print("\nLoan options:", sorted(X_train["loan"].unique()))
        customer["loan"] = input("Personal loan (yes/no): ").strip()

        print("\nContact options:", sorted(X_train["contact"].unique()))
        customer["contact"] = input("Contact type: ").strip()

        print("\nMonth options:", sorted(X_train["month"].unique()))
        customer["month"] = input("Last contact month: ").strip()

        print("\nPoutcome options:", sorted(X_train["poutcome"].unique()))
        customer["poutcome"] = input("Previous campaign outcome: ").strip()

        predict_subscription(customer)

    except ValueError as error:
        print("\nInvalid Input:", error)

    except Exception as error:
        print("\nPrediction failed:", error)


# --------------------------------------------------
# 7. RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    get_customer_input()
