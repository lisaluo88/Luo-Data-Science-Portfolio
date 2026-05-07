# 🤖 ML Explorer — Interactive Machine Learning App

> An interactive Streamlit web application that lets anyone train and evaluate supervised machine learning models — **no coding required.**

---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Live Demo & Screenshots](#-live-demo--screenshots)
3. [Dataset](#-dataset)
4. [Features](#-features)
5. [Models & Hyperparameters](#-models--hyperparameters)
6. [Pre-processing Pipeline](#-pre-processing-pipeline)
7. [Evaluation Metrics](#-evaluation-metrics)
8. [Project Structure](#-project-structure)
9. [Setup & Installation](#-setup--installation)
10. [How to Use the App](#-how-to-use-the-app)
11. [Code Examples](#-code-examples)
12. [Key Insights](#-key-insights)
13. [Future Improvements](#-future-improvements)
14. [References](#-references)
15. [Author](#-author)

---

## 🤖 Overview
**ML Explorer** is a browser-based machine learning playground built with Python and Streamlit. The goal is to make supervised machine learning **accessible and interactive** by wrapping the concepts taught in class (Weeks 9–11) into a clean, user-facing application that runs in any browser.

**What you can do with this app:**
- Upload any CSV dataset *or* use the built-in 2008 Beijing Olympics medalists sample
- Pick your own features and target column using point-and-click dropdowns
- Choose from three supervised classification models — **Logistic Regression**, **Decision Tree**, or **K-Nearest Neighbors**
- Tune hyperparameters with interactive sliders (no code editing required)
- Evaluate your model through four output views: confusion matrix, classification report, ROC curve, and feature importance

**Project Goals:**
- Translate classroom ML concepts into a polished, end-user product
- Provide a no-code interface to explore how models and hyperparameters affect performance
- Handle preprocessing automatically so users can focus on modeling decisions
- Strengthen skills in app design, scikit-learn pipelines, and model evaluation
- Bridge the gap between *understanding* ML and *applying* it

---

## 🎬 Live Demo & Screenshots

▶ **Open ML Explorer on Streamlit Cloud:** *[insert deployed URL]*

**Sidebar Controls**
A persistent sidebar provides dataset upload, target/feature selection, model choice, and hyperparameter sliders.

**Confusion Matrix Heatmap**
A Seaborn-rendered heatmap showing correct vs. incorrect predictions per class.

**ROC Curve with AUC**
For binary classification targets, a Matplotlib ROC curve with the area-under-curve score annotated.

**Feature Importance Chart**
A horizontal bar chart ranking which features had the most influence on the model's predictions.

---

## 📊 Dataset
ML Explorer accepts **any user-uploaded CSV** with at least one categorical or numeric target column. A built-in sample is also included.

### Built-in Sample — 2008 Beijing Olympics Medalists
- **File:** `olympics_08_medalists.csv`
- **Rows:** 1,875 athletes
- **Original format:** Wide (70+ `gender_sport` columns)
- **Tidy format (after `.melt()`):** Long, one row per medal

| Column | Description |
|---|---|
| `medalist_name` | Athlete's full name |
| `gender` | `male` or `female` |
| `sport` | Sport name (e.g. `swimming`, `athletics`) |
| `medal` | `gold`, `silver`, or `bronze` |

**Suggested experiment:**
- **Target:** `medal`
- **Features:** `gender`, `sport`
- **Model:** Decision Tree with `max_depth = 4`

---

## ✨ Features

### 📤 Dataset Upload
- Drag-and-drop CSV upload, or load the built-in Olympics sample with one click
- Preview the first rows of any uploaded dataset before modeling

### 🎯 Target & Feature Selection
- Point-and-click dropdowns to pick the target column
- Multi-select widget to pick one or more feature columns
- Validation prevents using the same column as both target and feature

### 🤖 Model Selection
- Three classifiers exposed via radio buttons in the sidebar
- Each model unlocks its own hyperparameter panel

### 🎚️ Hyperparameter Tuning
- Sliders, number inputs, and dropdowns for every tunable parameter
- Re-runs the model instantly when any value changes

### 📈 Evaluation Dashboard
- Four-tab output: confusion matrix, classification report, ROC curve, feature importance
- All charts update live as parameters change

---

## 🧠 Models & Hyperparameters

### 1. Logistic Regression
- A linear model that estimates the probability of belonging to each class
- Works well as a **baseline** for classification tasks
- Outputs coefficients showing how much each feature influences the prediction

### 2. Decision Tree
- A tree-based model that recursively splits the data on feature thresholds
- **Highly interpretable** — you can trace exactly how predictions are made
- Captures non-linear relationships without requiring feature scaling

### 3. K-Nearest Neighbors (KNN)
- Classifies a point by majority vote among its `k` nearest training examples
- Distance-based, so **feature scaling is essential** (handled automatically by `StandardScaler`)
- Simple but effective — no explicit training phase

### Hyperparameter Reference

| Model | Hyperparameter | What it Controls |
|---|---|---|
| Logistic Regression | `C` | Regularization strength — smaller = stronger penalty on large coefficients |
| Logistic Regression | `Max Iterations` | How many steps the solver takes to converge |
| Logistic Regression | `Solver` | Optimization algorithm (`lbfgs`, `saga`, `liblinear`) |
| Decision Tree | `Max Depth` | How deep the tree can grow — limits overfitting |
| Decision Tree | `Min Samples Split` | Minimum samples required to create a new branch |
| Decision Tree | `Criterion` | Split quality measure (`gini` or `entropy`) |
| KNN | `k (Neighbors)` | How many nearby points vote on the prediction |
| KNN | `Weights` | Equal votes (`uniform`) or closer = more influence (`distance`) |
| KNN | `Distance Metric` | How distance is calculated (`euclidean`, `manhattan`, `minkowski`) |

---

## 🧹 Pre-processing Pipeline
The app automatically runs these steps before training, matching the exact workflow used in the class notebooks:

| Step | Method | Class Reference |
|---|---|---|
| Remove missing rows | `df.dropna()` | Week 9.1 |
| Encode categorical columns | `pd.get_dummies(drop_first=True)` | Weeks 9.1, 9.2, 11.2 |
| Split into train/test | `train_test_split(test_size=0.2, random_state=42)` | Week 9.1 |
| Scale features | `StandardScaler()` | Week 11.2 |

---

## 📊 Evaluation Metrics

| Output | Description | Class Reference |
|---|---|---|
| **Accuracy** | Fraction of all predictions that were correct | Week 9.1 |
| **Precision** | Of predicted positives, how many were actually positive | Week 9.1 |
| **Recall** | Of actual positives, how many did the model catch | Week 9.1 |
| **F1 Score** | Harmonic mean of precision and recall | Week 9.1 |
| **Confusion Matrix** | Grid showing correct vs. incorrect predictions per class | Weeks 9.1, 10.1 |
| **Classification Report** | Full per-class precision, recall, F1 table | Week 9.1 |
| **ROC Curve & AUC** | Trade-off between true and false positives (binary only) | Week 10.1 |
| **Feature Importance** | Which features had the most influence on predictions | Weeks 9.1, 9.2 |

---

## 📁 Project Structure
```
MLStreamlitApp/
├── app.py                          # Main Streamlit application
├── olympics_08_medalists.csv       # Built-in sample dataset
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation (this file)
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11 or higher
- pip or conda package manager

### Step 1 — Clone the portfolio repository
```bash
git clone https://github.com/YOUR_USERNAME/Luo-Data-Science-Portfolio.git
```

### Step 2 — Navigate into the app folder
```bash
cd Luo-Data-Science-Portfolio/MLStreamlitApp
```

### Step 3 — Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### Step 4 — Install all required libraries
```bash
pip install "streamlit>=1.35.0" "pandas>=2.0.0" "numpy>=1.26.0" \
            "scikit-learn>=1.4.0" "matplotlib>=3.8.0" "seaborn>=0.13.0"
```

Or, if a `requirements.txt` is present:
```bash
pip install -r requirements.txt
```

### Step 5 — Run the app
```bash
streamlit run app.py
```

### Step 6 — Open in your browser
The app will open automatically at `http://localhost:8501`. Press `Ctrl + C` in the terminal to shut down the server.

> 💡 **Note:** The `olympics_08_medalists.csv` file must be in the same folder as `app.py` for the built-in sample dataset to load correctly.

### 📦 Required Libraries

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.35.0 | Builds the entire web interface — buttons, sliders, tables, charts |
| `pandas` | ≥ 2.0.0 | Loads and manipulates the dataset; used for `pd.get_dummies()` and `.melt()` |
| `numpy` | ≥ 1.26.0 | Numeric array operations and math |
| `scikit-learn` | ≥ 1.4.0 | All ML models, preprocessing tools, and evaluation metrics |
| `matplotlib` | ≥ 3.8.0 | Draws the ROC curve and feature importance bar chart |
| `seaborn` | ≥ 0.13.0 | Draws the confusion matrix heatmap |

---

## 🖱️ How to Use the App

### Example 1 — Predicting medal color from sport and gender
1. Launch the app and leave the dataset selector on **Olympics Sample**.
2. In the sidebar, select `medal` as the target column.
3. Select `gender` and `sport` as the features.
4. Choose **Decision Tree** as the model.
5. Set `Max Depth = 4` with the slider.
6. Review the confusion matrix and classification report in the main panel.

### Example 2 — Uploading your own dataset
1. In the sidebar, click **Upload CSV** and choose any classification-ready CSV.
2. Inspect the dataset preview that appears at the top of the page.
3. Pick a target column with a small number of distinct values (binary or multi-class).
4. Pick the features you want the model to learn from.
5. Pick a model and tune its hyperparameters.
6. Compare evaluation outputs side-by-side as you adjust sliders.

### Example 3 — Comparing models on the same problem
1. Set up your dataset, target, and features once.
2. Switch the model from **Logistic Regression** → **Decision Tree** → **KNN** without changing anything else.
3. Note how the confusion matrix and feature importance shift across models.

### Example 4 — Generating a ROC curve
1. Pick a target column with **exactly two distinct classes** (binary classification).
2. Train any of the three models.
3. Scroll to the ROC curve tab — the AUC score is annotated on the chart.

---

## 💻 Code Examples

### Loading and reshaping the Olympics sample
```python
import pandas as pd

df = pd.read_csv("olympics_08_medalists.csv")

df_long = (
    df.melt(id_vars=["medalist_name"],
            var_name="gender_sport",
            value_name="medal")
      .dropna()
      .assign(gender=lambda d: d["gender_sport"].str.split("_").str[0],
              sport =lambda d: d["gender_sport"].str.split("_").str[1])
      .drop(columns=["gender_sport"])
)
```

### Encoding and splitting the data
```python
from sklearn.model_selection import train_test_split

X = pd.get_dummies(df_long[["gender", "sport"]], drop_first=True)
y = df_long["medal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Scaling features for KNN
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

### Training a Decision Tree with user-selected hyperparameters
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=user_max_depth,
    min_samples_split=user_min_split,
    criterion=user_criterion,
    random_state=42
)
model.fit(X_train, y_train)
```

### Plotting the confusion matrix
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, model.predict(X_test))

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
```

---

## 🔎 Key Insights
- **No-code does not mean no-thinking** — the app removes coding friction but still rewards users who understand what each hyperparameter does
- **Feature scaling matters most for distance-based models** — KNN's accuracy can swing dramatically depending on whether `StandardScaler` is applied
- **Decision Trees are the easiest model to teach with** — their predictions are inspectable, and `max_depth` lets users see overfitting happen in real time
- **Logistic Regression is the strongest baseline** for many tabular classification problems, especially with regularization tuned via `C`
- **The Olympics dataset rewards careful feature choice** — predicting `medal` color is intentionally hard, which makes it a good teaching example
- **Wrapping a notebook in a UI changes who can use it** — the same scikit-learn code is now usable by anyone with a browser

---

## 🚀 Future Improvements
- **Add regression models** (Linear Regression, Random Forest Regressor) for numeric targets
- **Add unsupervised learning** (K-Means, PCA) as a separate page
- **Add cross-validation** with k-fold scoring instead of a single train/test split
- **Add hyperparameter search** (`GridSearchCV` or `RandomizedSearchCV`) with progress reporting
- **Add SHAP-based explanations** to complement feature importance
- **Add model export** so users can download a trained pickle file
- **Add session state** to compare multiple model runs side-by-side
- **Add caching** with `@st.cache_data` and `@st.cache_resource` for faster reruns

---

## 📚 References

| Resource | Used For |
|---|---|
| [Streamlit Documentation](https://docs.streamlit.io/) | Building the web interface |
| [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) | All ML models and metrics |
| [scikit-learn: Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) | Week 9.1 model |
| [scikit-learn: Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) | Week 9.2 model |
| [scikit-learn: K-Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) | Week 11.2 model |
| [scikit-learn: confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) | Evaluation metric |
| [scikit-learn: ROC Curve & AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) | Week 10.1 evaluation |
| [scikit-learn: StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) | Feature scaling for KNN |
| [pandas: get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html) | Categorical encoding |
| [pandas: melt](https://pandas.pydata.org/docs/reference/api/pandas.melt.html) | Reshaping the Olympics dataset |
| [Seaborn: heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html) | Confusion matrix visualization |
| [Streamlit Community Cloud](https://share.streamlit.io/) | App deployment |

---

## 👤 Author
**Lisa Luo**
[GitHub](https://github.com/YOUR_USERNAME) · [Portfolio](https://github.com/YOUR_USERNAME/Luo-Data-Science-Portfolio)

---

<p align="center">🤖 Built with Streamlit · Pandas · scikit-learn · Matplotlib · Seaborn 🤖</p>
