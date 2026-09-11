# FAB Credit Card Fraud Detection — Comprehensive Analysis & Extended Inferences

An in-depth, beginner-friendly analysis of the **Credit Card Fraud Detection** machine learning workflow, breaking down the pipeline step-by-step, evaluating the baseline Support Vector Machine (SVM) model, and providing actionable statistical inferences to take the project further.

---

## Table of Contents
1. [Dataset Overview & Business Context](#1-dataset-overview--business-context)
2. [Step-by-Step Pipeline Explanation](#2-step-by-step-pipeline-explanation)
   - [Step 1: Data Ingestion & Exploration](#step-1-data-ingestion--exploration)
   - [Step 2: Data Preprocessing & Feature Scaling](#step-2-data-preprocessing--feature-scaling)
   - [Step 3: Model Selection & Training](#step-3-model-selection--training)
   - [Step 4: Evaluation & Performance Metrics](#step-4-evaluation--performance-metrics)
3. [Understanding the Results (Deep Dive)](#3-understanding-the-results-deep-dive)
   - [The Accuracy Paradox](#the-accuracy-paradox)
   - [Confusion Matrix Breakdown](#confusion-matrix-breakdown)
   - [Precision, Recall, and F1-Score](#precision-recall-and-f1-score)
4. [Advanced Inferences & Exploratory Insights](#4-advanced-inferences--exploratory-insights)
   - [Inference 1: Transaction Amount Dynamics](#inference-1-transaction-amount-dynamics)
   - [Inference 2: Temporal & Circadian Patterns](#inference-2-temporal--circadian-patterns)
   - [Inference 3: Feature Importance & Latent Correlations](#inference-3-feature-importance--latent-correlations)
   - [Inference 4: Financial Cost-Benefit Modeling](#inference-4-financial-cost-benefit-modeling)
5. [Code Snippets for Extended Analysis](#5-code-snippets-for-extended-analysis)
6. [Actionable Recommendations & Next Steps](#6-actionable-recommendations--next-steps)

---

## 1. Dataset Overview & Business Context

The dataset contains transactions made by European cardholders over a two-day span.

* **Total Observations:** 284,807 transactions
* **Features (31 total):**
  * `Time`: Seconds elapsed between this transaction and the very first transaction in the dataset.
  * `V1` – `V28`: 28 numerical features resulting from a **Principal Component Analysis (PCA)** transformation. These represent latent behavioral vectors while preserving customer privacy and confidential bank information.
  * `Amount`: The monetary value of the transaction in euros.
  * `Class`: Target ground truth:
    * `0`: Legitimate / Normal transaction
    * `1`: Fraudulent transaction

### The Class Imbalance Challenge
* **Normal (`Class 0`):** 284,315 transactions (~99.83%)
* **Fraud (`Class 1`):** 492 transactions (~0.17%)

Because fraud constitutes less than two-tenths of a single percent, standard machine learning heuristics (such as pure classification accuracy) become misleading. A naive classifier predicting `0` 100% of the time would be **99.83% accurate**, yet entirely ineffective at stopping financial theft.

---

## 2. Step-by-Step Pipeline Explanation

### Step 1: Data Ingestion & Exploration
* Loaded via `pandas.read_csv('creditcard.csv')`.
* Quick shape inspection and distribution analysis using `.value_counts()` confirms extreme rarity of positive cases.

### Step 2: Data Preprocessing & Feature Scaling
* **Standardization:** `StandardScaler` from Scikit-Learn normalizes features to have zero mean ($\mu = 0$) and unit variance ($\sigma = 1$).
* **Why Scaling is Crucial:** Distance-based algorithms like Support Vector Machines rely on Euclidean distances. Raw `Amount` ranges from $0 to thousands of dollars, whereas PCA components (`V1`–`V28`) typically range between -5 and +5. Without scaling, `Amount` would artificially dominate hyperplane positioning.
* **Train/Test Splitting:** An 80/20 train/test partition (`test_size=0.2`, `random_state=1`) divides the data into:
  * **Train Set:** 227,845 records
  * **Test Set:** 56,962 records

> **Best Practice Note (Data Leakage):** In the original notebook, `fit_transform` was applied across the entire matrix $X$ prior to splitting. In production workflows, you should always fit the scaler exclusively on `X_train` and then apply `transform` on `X_test` to prevent validation data properties from leaking into the training step.

### Step 3: Model Selection & Training
* Model: **Support Vector Classifier (`SVC`)** using default Radial Basis Function (RBF) kernel.
* Fits a non-linear decision boundary in high-dimensional feature space to separate legitimate transactions from fraudulent ones.

### Step 4: Evaluation & Performance Metrics
* Evaluated on held-out test data (`X_test`, `y_test`).
* Default accuracy scores:
  * **Training Accuracy:** `99.9675%`
  * **Testing Accuracy:** `99.9385%`
* Generated detailed evaluation using `confusion_matrix` and `classification_report`.

---

## 3. Understanding the Results (Deep Dive)

### The Accuracy Paradox
The notebook reports **99.94% accuracy** on the test set. However, there are only 87 fraudulent cases in the test set of 56,962 samples. A model that simply labels everything as normal would yield:
$$\text{Accuracy} = \frac{56,875}{56,962} = 99.847\%$$
Therefore, accuracy alone fails to reflect model utility.

### Confusion Matrix Breakdown

| | Predicted Fraud (`1`) | Predicted Normal (`0`) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Fraud (`1`)** | **55 (TP)** | **32 (FN)** | **87** |
| **Actual Normal (`0`)** | **3 (FP)** | **56,872 (TN)** | **56,875** |
| **Total Predicted** | **58** | **56,904** | **56,962** |

* **True Positives (TP = 55):** Fraudulent transactions correctly caught and flagged.
* **False Negatives (FN = 32):** Real fraudulent transactions that evaded detection.
* **False Positives (FP = 3):** Legitimate purchases mistakenly blocked as fraud.
* **True Negatives (TN = 56,872):** Legitimate purchases processed smoothly.

### Precision, Recall, and F1-Score

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{55}{55 + 3} = \mathbf{0.95}$$
* **Meaning:** When the model raises an alarm, it is correct 95% of the time. Only 3 normal customer transactions were needlessly interrupted.

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{55}{55 + 32} = \mathbf{0.63}$$
* **Meaning:** The model caught only 63% of total fraud events. **37% of fraud slipped past the system.** In commercial banking, this represents substantial operational loss.

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.95 \times 0.63}{0.95 + 0.63} = \mathbf{0.76}$$
* **Meaning:** The harmonic mean between precision and recall, providing a balanced summary score.

---

## 4. Advanced Inferences & Exploratory Insights

The dataset holds key signals that go beyond standard model training:

### Inference 1: Transaction Amount Dynamics
* **Micro-Authorizations ($0.00):** Fraud networks often execute zero-dollar or low-cent authorizations to test if stolen credentials are live before attempting large merchant charges.
* **Amount Distribution Differences:** 
  * Normal transactions often span a broad distribution with rare multi-thousand-euro peaks.
  * Fraudulent transactions tend to cluster around specific operational bands (e.g., $100–$500), avoiding internal thresholds that require multi-factor authentication (MFA).

### Inference 2: Temporal & Circadian Patterns
* `Time` covers 172,792 seconds (~48 hours).
* By converting elapsed seconds into a 24-hour cycle:
  $$\text{Hour} = \left(\frac{\text{Time}}{3600}\right) \pmod{24}$$
* **Behavioral Inferences:** Normal transaction volume follows daylight activity and drops significantly between 1:00 AM and 6:00 AM. Fraud rates often spike during late-night and early-morning periods, when cardholders are asleep and less likely to dispute real-time SMS alerts.

### Inference 3: Feature Importance & Latent Correlations
* While `V1`–`V28` are anonymized, their linear and non-linear correlation with `Class` is not uniform:
  * **Strong Negative Correlations:** Features such as `V17`, `V14`, and `V12` typically display the strongest inverse association with fraud.
  * **Strong Positive Correlations:** Features such as `V4` and `V11` often show strong positive shifts during fraudulent behavior.
* Isolating the top 5 to 8 predictive features allows for lightweight, low-latency deployment models.

### Inference 4: Financial Cost-Benefit Modeling
Machine learning models treat every false classification as mathematically equal. In financial risk engineering, however:
* **Cost of False Negative (FN):** Full chargeback liability + card re-issuance + network fines ($C_{\text{FN}} \approx \text{Transaction Value} + \$25$).
* **Cost of False Positive (FP):** Customer friction + support center handling ($C_{\text{FP}} \approx \$3–\$5$).
* Incorporating a monetary cost matrix shifts the optimal decision threshold from a symmetric 0.5 probability to an asymmetric cost-minimizing threshold.

---

## 5. Code Snippets for Extended Analysis

Add these blocks to your Jupyter Notebook to run these additional inferences:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Compare Transaction Amounts
print("--- Transaction Amount Summary by Class ---")
print(df.groupby('Class')['Amount'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]))

# Check for zero-dollar card-testing charges
zero_charges = df[df['Amount'] == 0]
print(f"Zero-dollar transactions: {len(zero_charges)} total (Fraud: {zero_charges['Class'].sum()})")

# 2. Extract 24-Hour Circadian Cycle
df['Hour'] = (df['Time'] / 3600) % 24

plt.figure(figsize=(10, 4))
sns.kdeplot(df[df['Class'] == 0]['Hour'], label='Legitimate', color='blue', common_norm=False)
sns.kdeplot(df[df['Class'] == 1]['Hour'], label='Fraud', color='red', common_norm=False)
plt.title('Normalized Density by Hour of Day')
plt.xlabel('Hour of Day (0 - 24)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# 3. Top Predictive Principal Components
correlations = df.corr()['Class'].drop(['Class', 'Time', 'Hour']).sort_values()
print("Top 5 Negative Correlated Features:
", correlations.head(5))
print("
Top 5 Positive Correlated Features:
", correlations.tail(5))
```

---

## 6. Actionable Recommendations & Next Steps

1. **Address Class Imbalance:**
   * Adjust the model loss function with `SVC(class_weight='balanced')`.
   * Apply **SMOTE** (Synthetic Minority Over-sampling Technique) or **Random Under-Sampling** strictly on the training partition.
2. **Explore Gradient Boosted Trees:**
   * Test **XGBoost**, **LightGBM**, or **CatBoost**. They handle non-linear relationships natively, require minimal feature scaling, and evaluate significantly faster on 280k+ rows.
3. **Optimize Decision Thresholds:**
   * Instead of predicting classes using hard labels, extract predicted probabilities (`predict_proba`) and evaluate the **Precision-Recall Curve (PR-AUC)** to select an operating point that maximizes recall while preserving acceptable precision.
4. **Prevent Pipeline Leakage:**
   * Encapsulate scaling and modeling inside Scikit-Learn `Pipeline` objects to ensure no test data distributions influence feature transforms.
