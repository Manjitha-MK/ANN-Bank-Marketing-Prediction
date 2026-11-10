
# ANN MODEL 1 - ERROR ANALYSIS

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix

# Create results folder
os.makedirs("results", exist_ok=True)

# STEP 1: LOAD DATASET

df = pd.read_csv(
    "data/bank+marketing/bank/bank-full.csv",
    sep=";"
)

df = df.drop(columns=["duration"])

X = df.drop(columns=["y"])
y = df["y"].map({"no": 0, "yes": 1})

# STEP 2: RECREATE THE SAME DATA SPLIT

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

# STEP 3: RECREATE PREPROCESSING

numeric_features = [
    "age", "balance", "day",
    "campaign", "pdays", "previous"
]

categorical_features = [
    "job", "marital", "education",
    "default", "housing", "loan",
    "contact", "month", "poutcome"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)

X_train_processed = preprocessor.fit_transform(
    X_train
).astype(np.float32)

X_val_processed = preprocessor.transform(
    X_val
).astype(np.float32)

# STEP 4: LOAD TRAINED ANN MODEL 1

model = tf.keras.models.load_model(
    "models/seed_experiments/ann_model_1_seed_42.keras"
)

print("\nANN Model 1 loaded successfully!")

# STEP 5: PREDICT VALIDATION DATA

probabilities = model.predict(
    X_val_processed,
    verbose=0
).ravel()

predictions = (probabilities >= 0.5).astype(int)

actual = y_val.to_numpy()

# STEP 6: IDENTIFY ERRORS

false_positive_indices = np.where(
    (actual == 0) & (predictions == 1)
)[0]

false_negative_indices = np.where(
    (actual == 1) & (predictions == 0)
)[0]

print("\nCONFUSION MATRIX")
print(confusion_matrix(actual, predictions))

print("\nFalse Positives:", len(false_positive_indices))
print("False Negatives:", len(false_negative_indices))

# STEP 7: SELECT FIVE EXAMPLES OF EACH ERROR TYPE

fp_selected = false_positive_indices[:5]
fn_selected = false_negative_indices[:5]

error_examples = []

for index in fp_selected:

    error_examples.append({
        "Original_Row_Index": X_val.index[index],
        "Error_Type": "False Positive",
        "Actual": actual[index],
        "Predicted": predictions[index],
        "Predicted_Probability": probabilities[index],
        "Age": X_val.iloc[index]["age"],
        "Job": X_val.iloc[index]["job"],
        "Education": X_val.iloc[index]["education"],
        "Balance": X_val.iloc[index]["balance"],
        "Contact": X_val.iloc[index]["contact"],
        "Campaign": X_val.iloc[index]["campaign"],
        "Previous_Outcome": X_val.iloc[index]["poutcome"]
    })

for index in fn_selected:

    error_examples.append({
        "Original_Row_Index": X_val.index[index],
        "Error_Type": "False Negative",
        "Actual": actual[index],
        "Predicted": predictions[index],
        "Predicted_Probability": probabilities[index],
        "Age": X_val.iloc[index]["age"],
        "Job": X_val.iloc[index]["job"],
        "Education": X_val.iloc[index]["education"],
        "Balance": X_val.iloc[index]["balance"],
        "Contact": X_val.iloc[index]["contact"],
        "Campaign": X_val.iloc[index]["campaign"],
        "Previous_Outcome": X_val.iloc[index]["poutcome"]
    })

# STEP 8: SAVE ERROR EXAMPLES

errors_df = pd.DataFrame(error_examples)

errors_df.to_csv(
    "results/ann_model_1_error_analysis.csv",
    index=False
)

print("\nREPRESENTATIVE ERROR EXAMPLES")
print(errors_df.to_string(index=False))

print("\nSaved: results/ann_model_1_error_analysis.csv")
print("\nERROR ANALYSIS COMPLETED!")
