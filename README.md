# Problem Set 2: Logistic Regression — Bank Marketing Classification

## Overview

This notebook applies **logistic regression** to the UCI Bank Marketing dataset to predict whether a client will subscribe to a term deposit (`y = yes/no`). It covers the full ML pipeline: exploratory data analysis, preprocessing, feature engineering, model training, and evaluation.

---

## Dataset

- **Source:** `bank-full.csv` (UCI Bank Marketing Dataset)
- **Size:** ~45,000 rows, 17 columns
- **Target:** `y` — whether the client subscribed to a term deposit (`yes` / `no`)
- **Class imbalance:** approximately 7.5:1 (no:yes)

---

## Project Structure

```
Problem_Set2_Log_Regression.ipynb   # Main notebook
bank-full.csv                       # Input dataset (loaded from Google Drive)
X_train.csv / X_test.csv            # Processed feature splits (saved to /content/)
y_train.csv / y_test.csv            # Processed label splits (saved to /content/)
```

---

## Pipeline

### 1. Exploratory Data Analysis (EDA)
- Dataset shape, dtype summary, duplicate and null checks
- Target variable distribution (class counts + pie chart)
- Numerical feature distributions (histograms by class)
- Correlation heatmap for numerical features
- Categorical features vs. target (stacked bar charts)
- Boxplots of numerical features by class

### 2. Preprocessing
| Step | Details |
|------|---------|
| Duplicate removal | Dropped exact duplicate rows |
| Unknown handling | Replaced `'unknown'` in `job`, `education`, `contact`, `poutcome` with column mode |
| `pdays` fix | Replaced sentinel `-1` with `0`; created binary flag `was_previously_contacted` |
| Outlier capping | IQR method applied to `balance`, `duration`, `campaign`, `previous` |

### 3. Feature Engineering
| Feature | Description |
|---------|-------------|
| `age_group` | Age binned into 5 groups: `<25`, `25-35`, `35-45`, `45-55`, `55+` |
| `high_balance` | Binary flag for top 25% balance |
| `season` | Month mapped to season (`winter`, `spring`, `summer`, `autumn`) |

### 4. Encoding
- **Binary columns** (`default`, `housing`, `loan`): Label encoded (0/1)
- **Ordinal column** (`education`): Mapped `primary=0`, `secondary=1`, `tertiary=2`
- **Nominal columns** (`job`, `marital`, `contact`, `poutcome`, `age_group`, `season`): One-Hot Encoded (`drop_first=True`)
- `month` dropped (redundant after season mapping)
- Target `y`: encoded as `0` (no) and `1` (yes)

### 5. Train/Test Split & Scaling
- **Split:** 80% train / 20% test, stratified by target
- **Scaling:** `StandardScaler` fitted on training data only (no data leakage), applied to numerical features

### 6. Model
- **Algorithm:** `sklearn.linear_model.LogisticRegression`
- **Key hyperparameters:**
  - `class_weight='balanced'` — compensates for 7.5:1 class imbalance
  - `C=0.1` — L2 regularization
  - `solver='lbfgs'`
  - `max_iter=1000`

### 7. Evaluation
- 5-fold cross-validation (ROC-AUC on training data)
- Test set metrics: classification report, F1, ROC-AUC, PR-AUC
- Visualizations: confusion matrix, ROC curve, Precision-Recall curve

---

## Key Results
| Metric | Value |
|---|---|
| ROC-AUC | **0.8875** |
| PR-AUC | **0.5138** |
| F1 (yes) | **0.5081** |
| Recall (yes) | **0.81** |
| Precision (yes) | **0.37** |
| Accuracy | **0.82** |

> Exact values depend on runtime output. Re-run the notebook to reproduce.
