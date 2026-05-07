# Luo Data Science Portfolio

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![SciPy](https://img.shields.io/badge/SciPy-Hierarchical%20Clustering-8CAAE6?style=flat-square&logo=scipy)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

## Overview

This repository contains my data science projects completed during the semester.
Together, these projects demonstrate my ability to move from data exploration and cleaning to building and deploying **both supervised and unsupervised** interactive machine learning applications.

---

## Skills Demonstrated

| Area | Tools & Techniques |
|---|---|
| Data Cleaning | pandas, `pd.melt()`, `str.split()`, `dropna()` |
| Visualization | matplotlib, seaborn, Streamlit charts, dendrograms, scree plots |
| Supervised ML | scikit-learn, Logistic Regression, Decision Tree, KNN |
| Unsupervised ML | scikit-learn, K-Means, Hierarchical Clustering, PCA, scipy `linkage` |
| App Development | Streamlit, interactive UI, hyperparameter tuning, Cloud deployment |
| Tidy Data | reshaping, pivoting, groupby, long-format transformation |

---

## Projects

### Project 1 — Penguins Streamlit Dashboard

An interactive dashboard for exploring the Palmer Penguins dataset through filtering, summary metrics, and charts.

**Key features:** sidebar species filter · dataset preview · body mass bar chart by sex · summary metrics
**Tools:** Python · pandas · matplotlib · Streamlit

---

### Project 2 — Tidy Data: 2008 Olympics Medalists

Transformed a wide-format Olympics dataset where gender and sport were encoded in column names into a fully tidy structure ready for analysis.

**Key techniques:** `pd.melt()` · `pivot_table()` · `groupby()` · `sort_values()` · seaborn visualizations
**Output:** sorted medal pivot table · top-10 sports bar chart · gender distribution count plot

> [View Repository](https://github.com/lisaluo88/Luo-Data-Science-Portfolio)

---

### Project 3 — ML Explorer App (Supervised Learning)

> [Launch Live App](https://luo-data-science-portfolio-99.streamlit.app/) · [View Repository](https://github.com/lisaluo88/Luo-Data-Science-Portfolio)

An end-to-end interactive machine learning application for **supervised classification**. Upload any CSV, pick a model, tune hyperparameters, and evaluate performance — all through a graphical interface, no coding required.

**Key features:**
- CSV upload or built-in sample dataset
- Three classifiers: Logistic Regression · Decision Tree · K-Nearest Neighbors
- Adjustable hyperparameters, train/test split, and random seed
- Metrics: accuracy · precision · recall · F1 · ROC AUC
- Confusion matrix heatmap · classification report · feature importance chart
- Run summary panel for reproducibility

**Tools:** Python · pandas · numpy · matplotlib · seaborn · Streamlit · scikit-learn

---

### Project 4 — Unsupervised ML Explorer ⭐

> [Launch Live App](https://luo-unsupervised-ml.streamlit.app) · [View Repository](https://github.com/lisaluo88/Luo-Data-Science-Portfolio)

The capstone of my portfolio — an interactive machine learning application focused on **unsupervised techniques**. Upload any CSV (or use the built-in Breast Cancer / Iris samples), choose a method, tune the relevant hyperparameters, and watch the math respond in real time.

**Key features:**
- CSV upload or built-in sample datasets (Breast Cancer · Iris)
- Three unsupervised methods: **K-Means Clustering** · **Hierarchical Clustering** · **Principal Component Analysis**
- Adjustable hyperparameters: number of clusters `k`, linkage method (ward/complete/average/single), number of components, random seed, `n_init`
- Visualizations: 2D PCA scatter plots · elbow curves · silhouette score sweeps · dendrograms · scree plots · loadings charts
- Metrics: silhouette score · inertia · variance explained · flip-aware accuracy vs. ground-truth labels
- Manual, visual hyperparameter tuning that builds real intuition rather than hiding it behind grid search

**How it builds on my ML understanding:**
This app extends the supervised work in Project 3 by tackling problems where there are *no labels* to learn from. Where Project 3 asks "given a label, can the model predict it?", Project 4 asks "without any labels, can the model find the structure that's already there?" Building it required learning to evaluate models without ground truth (silhouette scores, elbow curves, inertia) and to interpret latent dimensions (PCA loadings, dendrogram cuts) — a different and harder kind of model evaluation than confusion matrices and ROC curves.

**Tools:** Python · pandas · numpy · matplotlib · seaborn · scikit-learn · scipy · Streamlit

---

## Portfolio Progression

Each project builds directly on the one before it:

- **Project 1 → Project 2:** moved from interactive UI prototyping to serious data cleaning and the tidy-data discipline that every later project relies on.
- **Project 2 → Project 3:** reused the cleaned Olympics dataset as the built-in sample for an interactive supervised-learning app, layering model training and evaluation on top of clean data.
- **Project 3 → Project 4:** kept the same Streamlit-app architecture but extended it from supervised classification into the unsupervised world — clustering and dimensionality reduction — completing the core ML lifecycle.

The Unsupervised ML Explorer (Project 4) ties the whole portfolio together: it combines the **interactive UI design** from Project 1, the **clean data handling** from Project 2, the **ML pipeline patterns** from Project 3, and adds **deployment to Streamlit Community Cloud** so anyone with a browser can use it.

---

## Portfolio Updates Log

| Update | Project Added | Focus |
|---|---|---|
| **Update 1** | Portfolio repo created · Project 1 (Penguins Dashboard) | Interactive Streamlit UI fundamentals |
| **Update 2** | Project 2 (Tidy Data — 2008 Olympics) | Tidy-data principles and reshaping |
| **Update 3** | Project 3 (ML Explorer — Supervised) | Supervised classification + model evaluation |
| **Update 4** | Project 4 (Unsupervised ML Explorer) | Unsupervised learning + Cloud deployment |

---

## Author

**Lisa Luo** · Data Science Portfolio · Spring 2026
[github.com/lisaluo88/Luo-Data-Science-Portfolio](https://github.com/lisaluo88/Luo-Data-Science-Portfolio)