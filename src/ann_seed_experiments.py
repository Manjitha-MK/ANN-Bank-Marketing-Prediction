
# ANN THREE-SEED EXPERIMENTS
# Bank Marketing Dataset

import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# Create output folders
os.makedirs("models/seed_experiments", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Experiment settings
SEEDS = [42, 123, 2026]

# STEP 1: LOAD DATA

df = pd.read_csv(
    "data/bank+marketing/bank/bank-full.csv",
    sep=";"
)

df = df.drop(columns=["duration"])

X = df.drop(columns=["y"])
y = df["y"].map({"no": 0, "yes": 1})

# STEP 2: SPLIT DATA

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

# STEP 3: PREPROCESS DATA

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

X_train = preprocessor.fit_transform(X_train).astype(np.float32)
X_val = preprocessor.transform(X_val).astype(np.float32)

y_train = y_train.to_numpy(dtype=np.float32)
y_val = y_val.to_numpy(dtype=np.float32)

# STEP 4: CLASS WEIGHTS

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train.astype(int)
)

class_weights = {
    0: weights[0],
    1: weights[1]
}

# STEP 5: EXPERIMENT FUNCTION

results = []

def run_experiment(model_number, seed):

    print("\n" + "=" * 55)
    print(f"ANN MODEL {model_number} | SEED {seed}")
    print("=" * 55)

    # Set a different random seed
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    # Build Model 1
    if model_number == 1:

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

    # Build Model 2
    else:

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy"
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # Train model
    start_time = time.perf_counter()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1
    )

    training_time = time.perf_counter() - start_time

    # Validation predictions
    probabilities = model.predict(
        X_val,
        verbose=0
    ).ravel()

    predictions = (probabilities >= 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_val, probabilities)
    pr_auc = average_precision_score(y_val, probabilities)

    # Save each trained model
    model_path = (
        f"models/seed_experiments/"
        f"ann_model_{model_number}_seed_{seed}.keras"
    )

    model.save(model_path)

    # Store experiment results
    results.append({
        "Model": f"ANN Model {model_number}",
        "Seed": seed,
        "Epochs": len(history.history["loss"]),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Training_Time_Seconds": training_time,
        "Parameters": model.count_params()
    })

    print("\nValidation F1:", round(f1, 4))
    print("Training time:", round(training_time, 2), "seconds")
    print("Saved:", model_path)


# STEP 6: RUN ALL SIX EXPERIMENTS

for model_number in [1, 2]:

    for seed in SEEDS:

        run_experiment(model_number, seed)


# STEP 7: SAVE RESULTS

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/three_seed_results.csv",
    index=False
)

print("\nINDIVIDUAL EXPERIMENT RESULTS")
print(results_df.to_string(index=False))


# STEP 8: CALCULATE MEAN AND STANDARD DEVIATION

summary = results_df.groupby("Model").agg({
    "Accuracy": ["mean", "std"],
    "Precision": ["mean", "std"],
    "Recall": ["mean", "std"],
    "F1": ["mean", "std"],
    "ROC_AUC": ["mean", "std"],
    "PR_AUC": ["mean", "std"],
    "Training_Time_Seconds": ["mean", "std"]
})

summary.to_csv("results/three_seed_summary.csv")

print("\nTHREE-SEED SUMMARY")
print(summary)

print("\nALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
