# ANN-Based Bank Marketing Prediction

## 1. Project Overview

This project investigates the use of Artificial Neural Networks (ANNs) to predict whether a bank customer will subscribe to a term deposit based on customer information and previous marketing campaign data.

The project uses the UCI Bank Marketing dataset and compares a Logistic Regression baseline with two different ANN architectures.

The main objective is to evaluate predictive performance, investigate the effect of neural network architecture, and examine model reliability using multiple random seeds.

## 2. Dataset

**Dataset:** Bank Marketing (bank-full.csv)

**Source:** UCI Machine Learning Repository

**Link:** https://archive.ics.uci.edu/dataset/222/bank+marketing

The dataset contains 45,211 customer records and 17 columns, including 16 input features and one target variable (`y`).

The target variable indicates whether a customer subscribed to a term deposit:

- `yes` – Customer subscribed.
- `no` – Customer did not subscribe.

### Important preprocessing decision

The `duration` feature was excluded because call duration is only known after the marketing call. Using it would introduce information leakage for a prediction intended to be made before the call.

### Dataset location

Place the downloaded and extracted dataset at:

```text
data/bank+marketing/bank/bank-full.csv
```

## 3. Project Structure

```text
ANN_Bank_Marketing_Coursework/
│
├── data/
│   └── bank+marketing/
│       └── bank/
│           └── bank-full.csv
│
├── notebooks/
│   └── ANN_Coursework.ipynb
│
├── models/
│
├── results/
│
├── src/
│   ├── ann_model_1.py
│   ├── ann_model_2.py
│   ├── ann_seed_experiments.py
│   ├── ann_ablation.py
│   ├── ann_error_analysis.py
│   ├── ann_subgroup_analysis.py
│   ├── ann_final_test_evaluation.py
│   └── ann_inference_pipeline.py
│
├── requirements.txt
└── README.md
```

## 4. Data Preparation

The dataset was divided into three stratified partitions:

- Training set: 70%
- Validation set: 15%
- Test set: 15%

The random state used for partitioning was 42.

The preprocessing pipeline includes:

- StandardScaler for numerical features.
- OneHotEncoder for categorical features.
- Fitting preprocessing transformations using training data only.
- Converting input features to NumPy-compatible numerical arrays.

The training set was used to fit the models. The validation set was used for model selection and early stopping. The test set was reserved for final evaluation.

Class weights were used during training to address the imbalance between customers who subscribed and those who did not.

## 5. Models

Three models were investigated.

### 5.1 Logistic Regression

Logistic Regression was used as the baseline model.

### 5.2 ANN Model 1

Architecture:

```text
Input Layer (50 features)
        |
Dense Layer (32 neurons, ReLU)
        |
Output Layer (1 neuron, Sigmoid)
```

Trainable parameters: 1,665.

### 5.3 ANN Model 2

Architecture:

```text
Input Layer (50 features)
        |
Dense Layer (64 neurons, ReLU)
        |
Dense Layer (32 neurons, ReLU)
        |
Output Layer (1 neuron, Sigmoid)
```

Trainable parameters: 5,377.

Both ANN models use the Adam optimizer, binary cross-entropy loss, class weighting, and early stopping.

## 6. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrices
- Training time and parameter count

Three random seeds (42, 123 and 2026) were used to investigate variation in ANN performance.

An ablation experiment was also conducted to examine the effect of removing one hidden layer from ANN Model 2.

Additional analysis included error analysis and age-based subgroup evaluation.

## 7. Running the Project

### Step 1: Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 2: Run the ANN models

```bash
python src/ann_model_1.py
```

```bash
python src/ann_model_2.py
```

### Step 3: Run additional experiments

```bash
python src/ann_seed_experiments.py
```

```bash
python src/ann_ablation.py
```

```bash
python src/ann_error_analysis.py
```

```bash
python src/ann_subgroup_analysis.py
```

### Step 4: Run final test evaluation

```bash
python src/ann_final_test_evaluation.py
```

### Step 5: Run the inference pipeline

```bash
python src/ann_inference_pipeline.py
```

Run the scripts from the project root directory.

## 8. Reproducibility

The project uses fixed random seeds for the data partition and ANN experiments.

The three ANN random seeds are:

- 42
- 123
- 2026

The same training, validation and test partitions are used for model comparisons.

The dependency versions are specified in `requirements.txt`.

## 9. Limitations

- The dataset contains historical marketing campaign information and may not represent current customer behaviour.
- The target classes are imbalanced.
- Predictive performance may vary across different random seeds.
- The models identify statistical patterns and do not establish causal relationships.
- Model performance may change when applied to new customer populations.

## 10. Future Improvements

Possible improvements include:

- Experimenting with different classification thresholds using validation data.
- Exploring additional feature engineering techniques.
- Evaluating the models using newer customer data.
- Investigating further methods to improve minority-class prediction.

## 11. Dataset Reference

Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems, 62*, 22–31.

Dataset: UCI Machine Learning Repository – Bank Marketing.

https://archive.ics.uci.edu/dataset/222/bank+marketing

### Saved Data Splits

The exact dataset row indices for each partition are saved in
`data/splits/`.

- train_indices.csv
- validation_indices.csv
- test_indices.csv

These files allow the original data partitions to be reproduced.

### Saved Model Checkpoints

Trained neural network models are saved in Keras format (`.keras`).

The `models/` directory contains:

- Main ANN Model 1 and ANN Model 2 checkpoints.
- Three-seed checkpoints for both ANN architectures.
- Original and ablated Model 2 checkpoints for the ablation experiment.

The experiments use three random seeds: 42, 123 and 2026.

These checkpoints allow trained models to be loaded without repeating the training process.