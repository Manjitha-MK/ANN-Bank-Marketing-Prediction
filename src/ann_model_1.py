
# STEP 1: IMPORT LIBRARIES

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

print("Libraries imported successfully!")


# STEP 2: LOAD DATASET

df = pd.read_csv(
    "data/bank+marketing/bank/bank-full.csv",
    sep=";"
)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# STEP 3: REMOVE DATA LEAKAGE FEATURE

# Duration is excluded because it is only known
# after the telephone call.

df = df.drop(columns=["duration"])


# STEP 4: SEPARATE FEATURES AND TARGET

X = df.drop(columns=["y"])
y = df["y"].map({"no": 0, "yes": 1})

print("\nFeatures shape:", X.shape)
print("Target distribution:")
print(y.value_counts())


# STEP 5: SPLIT DATASET (70% / 15% / 15%)

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

print("\nData splitting completed!")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# STEP 6: IDENTIFY NUMERICAL AND CATEGORICAL FEATURES

numeric_features = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous"
]

categorical_features = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome"
]


# STEP 7: PREPROCESSING

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

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform validation and testing data
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)


# STEP 8: CONVERT TO NUMPY ARRAYS

X_train_processed = np.asarray(
    X_train_processed, dtype=np.float32
)

X_val_processed = np.asarray(
    X_val_processed, dtype=np.float32
)

X_test_processed = np.asarray(
    X_test_processed, dtype=np.float32
)

y_train = y_train.to_numpy(dtype=np.float32)
y_val = y_val.to_numpy(dtype=np.float32)
y_test = y_test.to_numpy(dtype=np.float32)


# STEP 9: FINAL CHECK

print("\nPREPROCESSING COMPLETED!")
print("Training data:", X_train_processed.shape)
print("Validation data:", X_val_processed.shape)
print("Testing data:", X_test_processed.shape)
print("Number of input features:", X_train_processed.shape[1])

print("\nTensorFlow version:", tf.__version__)



# STEP 10: BUILD ANN MODEL 1

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# Set random seed
tf.keras.utils.set_random_seed(42)

# Build neural network
ann_model_1 = Sequential([
    Input(shape=(X_train_processed.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Display model architecture
print("\nANN MODEL 1 SUMMARY")
ann_model_1.summary()


# STEP 11: COMPILE MODEL

ann_model_1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)


# STEP 12: HANDLE CLASS IMBALANCE

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train.astype(int)
)

class_weights = {
    0: class_weights_array[0],
    1: class_weights_array[1]
}

print("\nClass weights:", class_weights)


# STEP 13: EARLY STOPPING

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# STEP 14: TRAIN ANN MODEL 1

print("\nSTARTING ANN MODEL 1 TRAINING...")

history_1 = ann_model_1.fit(
    X_train_processed,
    y_train,
    validation_data=(X_val_processed, y_val),
    epochs=50,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

print("\nANN MODEL 1 TRAINING COMPLETED!")


# STEP 15: SAVE TRAINED MODEL

import os

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

ann_model_1.save("models/ann_model_1.keras")

print("ANN Model 1 saved successfully!")


# STEP 16: VALIDATION PREDICTIONS

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

y_val_prob_ann1 = ann_model_1.predict(
    X_val_processed
).ravel()

y_val_pred_ann1 = (
    y_val_prob_ann1 >= 0.5
).astype(int)


# STEP 17: EVALUATION

print("\nANN MODEL 1 - VALIDATION RESULTS")
print("--------------------------------")

print("Accuracy:",
      accuracy_score(y_val, y_val_pred_ann1))

print("Precision:",
      precision_score(y_val, y_val_pred_ann1, zero_division=0))

print("Recall:",
      recall_score(y_val, y_val_pred_ann1, zero_division=0))

print("F1 Score:",
      f1_score(y_val, y_val_pred_ann1, zero_division=0))

print("ROC-AUC:",
      roc_auc_score(y_val, y_val_prob_ann1))

print("PR-AUC:",
      average_precision_score(y_val, y_val_prob_ann1))


# STEP 18: CLASSIFICATION REPORT

print("\nClassification Report:")

print(classification_report(
    y_val,
    y_val_pred_ann1,
    target_names=["No Subscription", "Subscription"],
    zero_division=0
))


# STEP 19: CONFUSION MATRIX

import matplotlib.pyplot as plt
import seaborn as sns

cm_ann1 = confusion_matrix(
    y_val,
    y_val_pred_ann1
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_ann1,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No", "Yes"],
    yticklabels=["No", "Yes"]
)

plt.title("ANN Model 1 - Validation Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("results/ann_model_1_confusion_matrix.png")
plt.show()
