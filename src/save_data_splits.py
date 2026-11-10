
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Dataset path
DATA_PATH = "data/bank+marketing/bank/bank-full.csv"

# Load dataset
df = pd.read_csv(DATA_PATH, sep=";")

# Target variable
y = df["y"].map({"no": 0, "yes": 1})

# Original row positions
indices = np.arange(len(df))

# Split into training and temporary sets (70% / 30%)
train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Split temporary set into validation and test sets (15% / 15%)
val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=42,
    stratify=y.iloc[temp_indices]
)

# Create output directory
os.makedirs("data/splits", exist_ok=True)

# Save exact original row indices
pd.DataFrame({"row_index": train_indices}).to_csv(
    "data/splits/train_indices.csv", index=False
)

pd.DataFrame({"row_index": val_indices}).to_csv(
    "data/splits/validation_indices.csv", index=False
)

pd.DataFrame({"row_index": test_indices}).to_csv(
    "data/splits/test_indices.csv", index=False
)

print("DATA SPLITS SAVED SUCCESSFULLY")
print("--------------------------------")
print("Training samples:", len(train_indices))
print("Validation samples:", len(val_indices))
print("Testing samples:", len(test_indices))
print("Total samples:", len(indices))
