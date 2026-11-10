
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
HIDDEN_UNITS = 16
LEARNING_RATE = 0.01
BATCH_SIZE = 256
EPOCHS = 50
PATIENCE = 5

DATA_PATH = "data/bank+marketing/bank/bank-full.csv"

os.makedirs("results", exist_ok=True)


# ============================================================
# 2. LOAD AND PREPARE DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH, sep=";")

# Remove duration because it is only known after the call
df = df.drop(columns=["duration"])

X = df.drop(columns=["y"])
y = df["y"].map({"no": 0, "yes": 1}).values

numeric_features = [
    "age", "balance", "day", "campaign", "pdays", "previous"
]

categorical_features = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome"
]

# 70% training, 15% validation, 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

# Fit preprocessing only on training data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features
        )
    ]
)

X_train = preprocessor.fit_transform(X_train).astype(np.float64)
X_val = preprocessor.transform(X_val).astype(np.float64)

y_train = y_train.reshape(-1, 1).astype(np.float64)
y_val = y_val.reshape(-1, 1).astype(np.float64)

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Input features:", X_train.shape[1])


# ============================================================
# 3. MANUAL NEURAL NETWORK
# Architecture: Input -> 16 ReLU -> 1 Sigmoid
# ============================================================

class NumpyMLP:

    def __init__(self, input_size, hidden_size, seed=42):

        rng = np.random.default_rng(seed)

        # He initialization
        self.W1 = rng.normal(
            0, np.sqrt(2 / input_size),
            (input_size, hidden_size)
        )

        self.b1 = np.zeros((1, hidden_size))

        self.W2 = rng.normal(
            0, np.sqrt(2 / hidden_size),
            (hidden_size, 1)
        )

        self.b2 = np.zeros((1, 1))

    def sigmoid(self, z):

        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):

        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.maximum(0, self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    def loss(self, y_true, y_pred, sample_weights=None):

        epsilon = 1e-12

        y_pred = np.clip(
            y_pred, epsilon, 1 - epsilon
        )

        losses = -(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )

        if sample_weights is not None:
            losses = losses * sample_weights

        return np.mean(losses) if sample_weights is None else (
            np.sum(losses) / np.sum(sample_weights)
        )

    def backward(self, X, y, sample_weights=None):

        n = X.shape[0]

        if sample_weights is None:
            sample_weights = np.ones((n, 1))

        # Output layer gradients
        dZ2 = (self.A2 - y) * sample_weights / np.sum(sample_weights)

        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer gradients
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0)

        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def update(self, gradients, learning_rate):

        dW1, db1, dW2, db2 = gradients

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2


# ============================================================
# 4. CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(y_train.flatten().astype(int))

class_weights = {
    0: len(y_train) / (2 * class_counts[0]),
    1: len(y_train) / (2 * class_counts[1])
}

sample_weights = np.array([
    class_weights[int(label)]
    for label in y_train.flatten()
]).reshape(-1, 1)

print("\nClass weights:", class_weights)


# ============================================================
# 5. TRAINING WITH MINI-BATCH GRADIENT DESCENT
# ============================================================

np.random.seed(RANDOM_STATE)

model = NumpyMLP(
    input_size=X_train.shape[1],
    hidden_size=HIDDEN_UNITS,
    seed=RANDOM_STATE
)

initial_train_loss = model.loss(
    y_train,
    model.forward(X_train),
    sample_weights
)

print("\nInitial training loss:", initial_train_loss)

train_losses = []
val_losses = []

best_val_loss = float("inf")
best_parameters = None
patience_counter = 0
best_epoch = 0

for epoch in range(EPOCHS):

    indices = np.random.permutation(len(X_train))

    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    weights_shuffled = sample_weights[indices]

    # Mini-batch training
    for start in range(0, len(X_train), BATCH_SIZE):

        end = start + BATCH_SIZE

        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        weights_batch = weights_shuffled[start:end]

        predictions = model.forward(X_batch)

        gradients = model.backward(
            X_batch,
            y_batch,
            weights_batch
        )

        model.update(gradients, LEARNING_RATE)

    # Calculate losses after the epoch
    train_predictions = model.forward(X_train)

    train_loss = model.loss(
        y_train,
        train_predictions,
        sample_weights
    )

    val_predictions = model.forward(X_val)

    val_loss = model.loss(
        y_val,
        val_predictions
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Train Loss: {train_loss:.5f} | "
        f"Validation Loss: {val_loss:.5f}"
    )

    # Save the best model using validation loss
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1

        best_parameters = {
            "W1": model.W1.copy(),
            "b1": model.b1.copy(),
            "W2": model.W2.copy(),
            "b2": model.b2.copy()
        }

        patience_counter = 0

    else:

        patience_counter += 1

    if patience_counter >= PATIENCE:

        print("\nEarly stopping activated.")
        break


# Restore best validation checkpoint
model.W1 = best_parameters["W1"]
model.b1 = best_parameters["b1"]
model.W2 = best_parameters["W2"]
model.b2 = best_parameters["b2"]

final_train_loss = model.loss(
    y_train,
    model.forward(X_train),
    sample_weights
)

print("\nTRAINING SUMMARY")
print("----------------")
print("Initial training loss:", round(initial_train_loss, 5))
print("Final training loss:", round(final_train_loss, 5))
print("Best validation loss:", round(best_val_loss, 5))
print("Best epoch:", best_epoch)


# ============================================================
# 6. SAVE TRAINING LOSS CURVE
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Manual NumPy MLP - Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(
    "results/ann_numpy_t2_loss_curve.png",
    dpi=300
)

plt.close()

print("\nLoss curve saved successfully.")


# ============================================================
# 7. NUMERICAL GRADIENT CHECK
# Central finite differences
# ============================================================

print("\nStarting gradient checking...")

# Use a small batch for gradient checking
rng = np.random.default_rng(100)

check_indices = rng.choice(
    len(X_train),
    size=8,
    replace=False
)

X_check = X_train[check_indices]
y_check = y_train[check_indices]

check_model = NumpyMLP(
    X_train.shape[1],
    HIDDEN_UNITS,
    seed=123
)

# Avoid checking parameters close to ReLU activation boundaries
check_model.forward(X_check)

if np.min(np.abs(check_model.Z1)) < 1e-6:
    print("Warning: Some ReLU values are close to zero.")

predictions = check_model.forward(X_check)

analytic_gradients = check_model.backward(
    X_check,
    y_check
)

gradient_arrays = {
    "W1": analytic_gradients[0],
    "b1": analytic_gradients[1],
    "W2": analytic_gradients[2],
    "b2": analytic_gradients[3]
}

parameters = {
    "W1": check_model.W1,
    "b1": check_model.b1,
    "W2": check_model.W2,
    "b2": check_model.b2
}

# Select five representative parameters
selected_parameters = [
    ("W1", (0, 0)),
    ("W1", (1, 0)),
    ("W2", (0, 0)),
    ("b1", (0, 0)),
    ("b2", (0, 0))
]

epsilon = 1e-5
gradient_results = []

for parameter_name, index in selected_parameters:

    parameter = parameters[parameter_name]

    original_value = parameter[index]

    # Positive perturbation
    parameter[index] = original_value + epsilon

    positive_loss = check_model.loss(
        y_check,
        check_model.forward(X_check)
    )

    # Negative perturbation
    parameter[index] = original_value - epsilon

    negative_loss = check_model.loss(
        y_check,
        check_model.forward(X_check)
    )

    # Restore original value
    parameter[index] = original_value

    numerical_gradient = (
        positive_loss - negative_loss
    ) / (2 * epsilon)

    analytic_gradient = gradient_arrays[
        parameter_name
    ][index]

    absolute_difference = abs(
        numerical_gradient - analytic_gradient
    )

    relative_difference = absolute_difference / max(
        1e-8,
        abs(numerical_gradient) + abs(analytic_gradient)
    )

    gradient_results.append({
        "Parameter": parameter_name,
        "Index": str(index),
        "Analytical Gradient": analytic_gradient,
        "Numerical Gradient": numerical_gradient,
        "Absolute Difference": absolute_difference,
        "Relative Difference": relative_difference
    })


gradient_df = pd.DataFrame(gradient_results)

gradient_df.to_csv(
    "results/ann_numpy_t2_gradient_check.csv",
    index=False
)

print("\nGRADIENT CHECK RESULTS")
print(gradient_df.to_string(index=False))

tolerance = 1e-4

gradient_passed = (
    gradient_df["Relative Difference"] < tolerance
).all()

print("\nGradient check passed:", gradient_passed)
print("Relative difference tolerance:", tolerance)


# ============================================================
# 8. TINY-BATCH OVERFITTING TEST
# ============================================================

print("\nStarting tiny-batch overfitting test...")

# Select 8 positive and 8 negative training examples
positive_indices = np.where(y_train.flatten() == 1)[0][:8]
negative_indices = np.where(y_train.flatten() == 0)[0][:8]

tiny_indices = np.concatenate([
    positive_indices,
    negative_indices
])

X_tiny = X_train[tiny_indices]
y_tiny = y_train[tiny_indices]

tiny_model = NumpyMLP(
    input_size=X_train.shape[1],
    hidden_size=HIDDEN_UNITS,
    seed=999
)

tiny_initial_loss = tiny_model.loss(
    y_tiny,
    tiny_model.forward(X_tiny)
)

tiny_learning_rate = 0.05
tiny_epochs = 2000

for epoch in range(tiny_epochs):

    tiny_predictions = tiny_model.forward(X_tiny)

    tiny_gradients = tiny_model.backward(
        X_tiny,
        y_tiny
    )

    tiny_model.update(
        tiny_gradients,
        tiny_learning_rate
    )

    if (epoch + 1) % 100 == 0:

        current_predictions = tiny_model.forward(X_tiny)

        current_loss = tiny_model.loss(
            y_tiny,
            current_predictions
        )

        current_accuracy = np.mean(
            (current_predictions >= 0.5) == y_tiny
        )

        if current_loss < 0.001 and current_accuracy == 1.0:
            break


tiny_final_predictions = tiny_model.forward(X_tiny)

tiny_final_loss = tiny_model.loss(
    y_tiny,
    tiny_final_predictions
)

tiny_accuracy = np.mean(
    (tiny_final_predictions >= 0.5) == y_tiny
)

print("\nTINY-BATCH OVERFITTING RESULTS")
print("-----------------------------")
print("Number of samples:", len(X_tiny))
print("Initial loss:", round(tiny_initial_loss, 6))
print("Final loss:", round(tiny_final_loss, 6))
print("Final accuracy:", round(tiny_accuracy, 4))
print("Epochs completed:", epoch + 1)

print("\nT2 IMPLEMENTATION COMPLETED.")
