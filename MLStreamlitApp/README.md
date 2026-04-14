# 🤖 ML Explorer — Interactive Machine Learning App

> An interactive Streamlit web application that lets anyone train and evaluate supervised machine learning models — no coding required.

---

## 📌 Project Overview

**ML Explorer** is a browser-based machine learning playground built with Python and Streamlit. The goal of this project is to make supervised machine learning accessible and interactive by wrapping the concepts taught in class (Weeks 9–11) into a clean, user-facing application.

### What can you do with this app?

- **Upload any CSV dataset** or use the built-in 2008 Beijing Olympics medalists sample
- **Pick your own features and target column** using point-and-click dropdowns
- **Choose from three supervised classification models** — Logistic Regression, Decision Tree, or K-Nearest Neighbors
- **Tune hyperparameters** with interactive sliders (no code editing required)
- **Evaluate your model** through four output views:
  - Confusion Matrix heatmap
  - Full Classification Report table
  - ROC Curve with AUC score (binary targets only)
  - Feature Importance bar chart

The app handles all preprocessing automatically — missing value removal, categorical encoding with `pd.get_dummies()`, train/test splitting, and feature scaling with `StandardScaler` — matching the exact workflow used in the class notebooks.

---

## 🚀 Deployed App

**[▶ Open ML Explorer on Streamlit Cloud](https://luo-data-science-portfolio-99.streamlit.app/)**

---

## 🛠 How to Run Locally

### Prerequisites
- Python 3.11 or higher
- `pip` or `conda` package manager

### Step-by-Step Instructions

**Step 1 — Clone the portfolio repository:**
```bash
git clone https://github.com/YOUR_USERNAME/Luo-Data-Science-Portfolio.git
```

**Step 2 — Navigate into the app folder:**
```bash
cd Luo-Data-Science-Portfolio/MLStreamlitApp
```

**Step 3 — Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

**Step 4 — Install all required libraries:**
```bash
pip install "streamlit>=1.35.0" "pandas>=2.0.0" "numpy>=1.26.0" "scikit-learn>=1.4.0" "matplotlib>=3.8.0" "seaborn>=0.13.0"
```

**Step 5 — Run the app:**
```bash
streamlit run app.py
```

**Step 6 — Open in your browser:**
```
http://localhost:8501
```

> **Note:** The `olympics_08_medalists.csv` file must be in the same folder as `app.py` for the built-in sample dataset to load correctly.

---

## 📦 Required Libraries

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.35.0 | Builds the entire web interface — buttons, sliders, tables, charts |
| `pandas` | ≥ 2.0.0 | Loads and manipulates the dataset; used for `pd.get_dummies()` and `.melt()` |
| `numpy` | ≥ 1.26.0 | Numeric array operations and math |
| `scikit-learn` | ≥ 1.4.0 | All ML models, preprocessing tools, and evaluation metrics |
| `matplotlib` | ≥ 3.8.0 | Draws the ROC curve and feature importance bar chart |
| `seaborn` | ≥ 0.13.0 | Draws the confusion matrix heatmap |

---

## 🧠 App Features

### Models

**1. Logistic Regression**
- A linear model that estimates the probability of belonging to each class
- Works well as a baseline for classification tasks
- Outputs coefficients that show how much each feature influences the prediction

**2. Decision Tree** 
- A tree-based model that makes decisions by recursively splitting the data
- Highly interpretable — you can visualize exactly how it makes decisions
- Can capture non-linear relationships without feature scaling

**3. K-Nearest Neighbors** 
- Classifies a new point by looking at the k closest training examples and taking a majority vote
- Distance-based, so feature scaling is essential (handled automatically by `StandardScaler`)
- Simple but effective — no explicit training phase

---

### Hyperparameter Tuning

Instead of hardcoding parameters, ML Explorer exposes them as **interactive sliders and dropdowns** in the sidebar, which gives the user direct manual control.

| Model | Hyperparameter | What it controls |
|---|---|---|
| Logistic Regression | **C** | Regularization strength — smaller = stronger penalty on large coefficients |
| Logistic Regression | **Max Iterations** | How many steps the solver takes to converge |
| Logistic Regression | **Solver** | Optimization algorithm (`lbfgs`, `saga`, `liblinear`) |
| Decision Tree | **Max Depth** | How deep the tree can grow — limits overfitting |
| Decision Tree | **Min Samples Split** | Minimum samples required to create a new branch |
| Decision Tree | **Criterion** | Split quality measure (`gini` or `entropy`) |
| K-Nearest Neighbors | **k (Neighbors)** | How many nearby points vote on the prediction |
| K-Nearest Neighbors | **Weights** | Equal votes (`uniform`) or closer = more influence (`distance`) |
| K-Nearest Neighbors | **Distance Metric** | How distance is calculated (`euclidean`, `manhattan`, `minkowski`) |

---

### Preprocessing Pipeline

The app automatically runs these steps before training:

| Step | Method | Class Reference |
|---|---|---|
| Remove missing rows | `df.dropna()` | Week 9.1 |
| Encode categorical columns | `pd.get_dummies(drop_first=True)` | Week 9.1, 9.2, 11.2 |
| Split into train/test | `train_test_split(test_size=0.2, random_state=42)` | Week 9.1 |
| Scale features | `StandardScaler()` | Week 11.2 |

---

### Evaluation Metrics

| Output | Description | Class Reference |
|---|---|---|
| **Accuracy** | Fraction of all predictions that were correct | Week 9.1 |
| **Precision** | Of predicted positives, how many were actually positive | Week 9.1 |
| **Recall** | Of actual positives, how many did the model catch | Week 9.1 |
| **F1 Score** | Harmonic mean of precision and recall | Week 9.1 |
| **Confusion Matrix** | Grid showing correct vs incorrect predictions per class | Week 9.1, 10.1 |
| **Classification Report** | Full per-class precision, recall, F1 table | Week 9.1 |
| **ROC Curve & AUC** | Trade-off between true and false positives (binary only) | Week 10.1 |
| **Feature Importance** | Which features had the most influence on predictions | Week 9.1, 9.2 |

---

## 🏅 Sample Dataset — 2008 Beijing Olympics Medalists

The built-in sample is the `olympics_08_medalists.csv` file containing **1,875 athletes** across **70 sport/gender event columns**.

The app automatically transforms this wide-format CSV into a clean long-format table using `.melt()` and method chaining (Week 6.2):

| Column | Description |
|---|---|
| `medalist_name` | Athlete's full name |
| `gender` | `male` or `female` |
| `sport` | Sport name (e.g. `swimming`, `athletics`) |
| `medal` | `gold`, `silver`, or `bronze` |

**Suggested experiment with this dataset:**
- **Target:** `medal`
- **Features:** `gender` and `sport`
- **Model:** Decision Tree with `max_depth = 4`

---

## 📸 App Screenshots

> *(Add your own screenshots here after running the app)*
>
> Suggested screenshots to include:
> - The sidebar with hyperparameter sliders
> - The dataset preview table
> - The confusion matrix heatmap
> - The ROC curve plot
> - The feature importance bar chart

---

## 📚 References

| Resource | Link | Used For |
|---|---|---|
| Streamlit Documentation | [docs.streamlit.io](https://docs.streamlit.io) | Building the web interface |
| scikit-learn User Guide | [scikit-learn.org/stable/user_guide](https://scikit-learn.org/stable/user_guide.html) | All ML models and metrics |
| scikit-learn: Logistic Regression | [docs link](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) | Week 9.1 model |
| scikit-learn: Decision Tree | [docs link](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) | Week 9.2 model |
| scikit-learn: KNN | [docs link](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) | Week 11.2 model |
| scikit-learn: confusion_matrix | [docs link](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) | Evaluation metric |
| scikit-learn: ROC Curve & AUC | [docs link](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) | Week 10.1 evaluation |
| scikit-learn: StandardScaler | [docs link](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) | Feature scaling for KNN |
| pandas: get_dummies | [docs link](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html) | Categorical encoding |
| pandas: melt | [docs link](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.melt.html) | Reshaping Olympics dataset |
| Seaborn: heatmap | [docs link](https://seaborn.pydata.org/generated/seaborn.heatmap.html) | Confusion matrix visualization |
| Streamlit Community Cloud | [share.streamlit.io](https://share.streamlit.io) | App deployment |

---

## 👤 Author

**[Lisa Luo]**
[GitHub](https://github.com/lisaluo88) | [Portfolio](https://github.com/lisaluo88/Luo-Data-Science-Portfolio)
