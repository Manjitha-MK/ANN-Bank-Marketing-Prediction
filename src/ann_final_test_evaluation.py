
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    precision_recall_curve
)

# --------------------------------------------------
# 1. SETTINGS AND PATHS
# --------------------------------------------------

SEEDS = [42, 123, 2026]

np.random.seed(42)
tf.random.set_seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "bank+marketing", "bank", "bank-full.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models", "seed_experiments")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(DATA_PATH, sep=";")

# Remove duration to prevent information leakage
df = df.drop(columns=["duration"])

X = df.drop(columns=["y"])
y = df["y"].map({"no": 0, "yes": 1})

print("Dataset loaded:", X.shape)

# --------------------------------------------------
# 3. TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------

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

print("\nDataset partitions:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# --------------------------------------------------
# 4. PREPROCESSING
# --------------------------------------------------

numeric_features = [
    "age", "balance", "day", "campaign", "pdays", "previous"
]

categorical_features = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ), categorical_features)
    ]
)

# Fit ONLY on training data
X_train_processed = preprocessor.fit_transform(X_train).astype("float32")
X_test_processed = preprocessor.transform(X_test).astype("float32")

y_test_array = y_test.to_numpy()

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed test shape:", X_test_processed.shape)

# --------------------------------------------------
# 5. EVALUATION FUNCTION
# --------------------------------------------------

all_results = []
curve_data = {}

def evaluate_model(
    model_name,
    y_true,
    probabilities,
    model_type
):

    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(
        y_true, predictions, zero_division=0
    )
    recall = recall_score(
        y_true, predictions, zero_division=0
    )
    f1 = f1_score(
        y_true, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_true, probabilities)
    pr_auc = average_precision_score(y_true, probabilities)

    matrix = confusion_matrix(y_true, predictions)

    result = {
        "Model": model_name,
        "Model Type": model_type,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    }

    all_results.append(result)

    print("\n" + "=" * 55)
    print(model_name)
    print("=" * 55)

    for metric, value in result.items():
        if isinstance(value, (float, np.floating)):
            print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions,
            target_names=["No Subscription", "Subscription"],
            zero_division=0
        )
    )

    # Save confusion matrix
    safe_name = model_name.replace(" ", "_").replace("-", "_")

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No", "Yes"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"{safe_name}_test_confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close()

    # Store ROC and PR curve data
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    pr_precision, pr_recall, _ = precision_recall_curve(
        y_true, probabilities
    )

    curve_data[model_name] = {
        "fpr": fpr,
        "tpr": tpr,
        "precision": pr_precision,
        "recall": pr_recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

    return result


# --------------------------------------------------
# 6. LOGISTIC REGRESSION BASELINE
# --------------------------------------------------

print("\nEvaluating Logistic Regression baseline...")

baseline = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

baseline.fit(X_train_processed, y_train)

start_time = time.time()

baseline_probabilities = baseline.predict_proba(
    X_test_processed
)[:, 1]

baseline_inference_time = time.time() - start_time

baseline_result = evaluate_model(
    "Logistic Regression",
    y_test_array,
    baseline_probabilities,
    "Baseline"
)

baseline_result["Inference Time (seconds)"] = baseline_inference_time


# --------------------------------------------------
# 7. ANN MODEL 1 AND MODEL 2
# --------------------------------------------------

for model_number in [1, 2]:

    for seed in SEEDS:

        model_name = f"ANN Model {model_number} Seed {seed}"

        model_path = os.path.join(
            MODEL_DIR,
            f"ann_model_{model_number}_seed_{seed}.keras"
        )

        print(f"\nLoading {model_name}...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}"
            )

        model = tf.keras.models.load_model(model_path)

        start_time = time.time()

        probabilities = model.predict(
            X_test_processed,
            verbose=0
        ).ravel()

        inference_time = time.time() - start_time

        result = evaluate_model(
            model_name,
            y_test_array,
            probabilities,
            f"ANN Model {model_number}"
        )

        result["Inference Time (seconds)"] = inference_time

        del model


# --------------------------------------------------
# 8. SAVE INDIVIDUAL RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(all_results)

individual_path = os.path.join(
    RESULTS_DIR,
    "final_test_individual_results.csv"
)

results_df.to_csv(individual_path, index=False)

print("\nIndividual test results:")
print(results_df.to_string(index=False))


# --------------------------------------------------
# 9. SAVE ROC CURVE
# --------------------------------------------------

plt.figure(figsize=(9, 7))

for name, data in curve_data.items():

    plt.plot(
        data["fpr"],
        data["tpr"],
        label=f"{name} (AUC={data['roc_auc']:.3f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Final Test Set")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "final_test_roc_curves.png"),
    dpi=300
)

plt.close()


# --------------------------------------------------
# 10. SAVE PRECISION-RECALL CURVE
# --------------------------------------------------

plt.figure(figsize=(9, 7))

for name, data in curve_data.items():

    plt.plot(
        data["recall"],
        data["precision"],
        label=f"{name} (AP={data['pr_auc']:.3f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves - Final Test Set")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "final_test_pr_curves.png"),
    dpi=300
)

plt.close()


# --------------------------------------------------
# 11. SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL TEST-SET EVALUATION COMPLETED")
print("=" * 60)

print("\nSaved files:")
print("- final_test_individual_results.csv")
print("- final_test_roc_curves.png")
print("- final_test_pr_curves.png")
print("- Individual model confusion matrices")

print("\nAll results saved in:", RESULTS_DIR)
