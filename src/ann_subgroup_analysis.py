
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Set random seed
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH, sep=";")

# Remove duration to avoid information leakage
df = df.drop(columns=["duration"])

X = df.drop(columns=["y"])
y = df["y"].map({"no": 0, "yes": 1})

# Identify feature types
numeric_features = [
    "age", "balance", "day", "campaign", "pdays", "previous"
]

categorical_features = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome"
]

# Same train/validation/test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Same preprocessing: fit only on training data
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

X_val_processed = preprocessor.transform(X_val).astype("float32")
y_val_array = y_val.to_numpy()

# Load trained ANN Model 1
model = tf.keras.models.load_model(MODEL_PATH)

# Generate predictions
probabilities = model.predict(
    X_val_processed,
    verbose=0
).ravel()

predictions = (probabilities >= 0.5).astype(int)

# Create age groups
age_groups = pd.cut(
    X_val["age"],
    bins=[-np.inf, 30, 51, np.inf],
    labels=["Under 30", "30-50", "Over 50"],
    right=False
)

# Evaluate each age group
results = []

for group in ["Under 30", "30-50", "Over 50"]:

    mask = (age_groups == group).to_numpy()

    actual = y_val_array[mask]
    predicted = predictions[mask]
    group_probabilities = probabilities[mask]

    if len(actual) == 0:
        continue

    result = {
        "Age Group": group,
        "Samples": len(actual),
        "Actual Subscribers": int(actual.sum()),
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "Recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "F1 Score": f1_score(
            actual, predicted, zero_division=0
        )
    }

    if len(np.unique(actual)) == 2:
        result["ROC-AUC"] = roc_auc_score(
            actual, group_probabilities
        )
    else:
        result["ROC-AUC"] = np.nan

    results.append(result)

# Display results
results_df = pd.DataFrame(results)

print("\nANN MODEL 1 - AGE SUBGROUP ANALYSIS")
print("-----------------------------------")
print(results_df.to_string(index=False))

# Save results
output_path = os.path.join(
    RESULTS_DIR,
    "ann_subgroup_analysis.csv"
)

results_df.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")
