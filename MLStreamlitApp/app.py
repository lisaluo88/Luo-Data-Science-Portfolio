import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ML imports — all covered in class notebooks (Weeks 9–11)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression          # Week 9.1
from sklearn.tree import DecisionTreeClassifier              # Week 9.2
from sklearn.neighbors import KNeighborsClassifier           # Week 11.2
from sklearn.metrics import (
    accuracy_score,        # Every model notebook
    confusion_matrix,      # Every model notebook
    classification_report, # Every model notebook
    roc_curve,             # Week 10.1
    roc_auc_score          # Week 10.1
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Explorer",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Fira+Mono:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 50%, #1a1f2e 100%);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at top left, rgba(88,166,255,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at bottom right, rgba(63,185,80,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero h1 { font-size: 2.4rem; font-weight: 700; color: #e6edf3; margin: 0 0 0.4rem 0; letter-spacing: -1px; }
.hero p  { color: #8b949e; font-size: 1rem; margin: 0; }
.hero .accent { color: #58a6ff; }
.metric-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 1.2rem 1.5rem; text-align: center; transition: border-color 0.2s;
}
.metric-card:hover { border-color: #58a6ff; }
.metric-card .val { font-size: 2rem; font-weight: 700; color: #58a6ff; font-family: 'Fira Mono', monospace; }
.metric-card .lbl { font-size: 0.78rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
.section-header {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase;
    color: #8b949e; border-bottom: 1px solid #21262d; padding-bottom: 0.5rem; margin: 1.5rem 0 1rem 0;
}
.info-box {
    background: rgba(88,166,255,0.05); border-left: 3px solid #58a6ff;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.8rem 0;
    font-size: 0.88rem; color: #8b949e;
}
section[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #21262d; }
</style>
""", unsafe_allow_html=True)

# ─── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>ML <span class="accent">Explorer</span></h1>
  <p>Upload a dataset · choose a model · tune hyperparameters · inspect performance — all in one place.</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Configuration
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Configuration")

    # ── Step 1: Dataset ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">1 · Dataset</div>', unsafe_allow_html=True)
    data_source = st.radio("Source", ["Upload CSV", "Sample: Olympics 2008"],
                           label_visibility="collapsed")
    df_raw = None

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.success(f"Loaded {df_raw.shape[0]:,} rows x {df_raw.shape[1]} cols")
    else:
        # Built-in Olympics 2008 sample dataset
        # Wide format to long format using .melt() — covered in Week 6.2
        @st.cache_data
        def load_olympics():
            df = pd.read_csv("olympics_08_medalists.csv")
            id_col = "medalist_name"
            sport_cols = [c for c in df.columns if c != id_col]

            # Reshape from wide to long format using .melt() — Week 6 method chaining
            melted = (
                df.melt(id_vars=id_col,
                        value_vars=sport_cols,
                        var_name="event",
                        value_name="medal")
                .dropna(subset=["medal"])
                .assign(
                    gender=lambda d: d["event"].apply(
                        lambda x: "male" if x.startswith("male_") else "female"),
                    sport=lambda d: d["event"].apply(
                        lambda x: x.replace("male_", "").replace("female_", ""))
                )
            )
            return melted[["medalist_name", "gender", "sport", "medal"]]

        df_raw = load_olympics()
        st.info("2008 Beijing Olympics medalists — melted to long format.")
        with st.expander("Preview data"):
            st.dataframe(df_raw.head(10), use_container_width=True)

    # ── Step 2: Model ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">2 · Model</div>', unsafe_allow_html=True)
    model_name = st.selectbox("Algorithm", [
        "Logistic Regression",   # Week 9.1
        "Decision Tree",         # Week 9.2
        "K-Nearest Neighbors",   # Week 11.2
    ])

    # ── Step 3: Hyperparameters ────────────────────────────────────────────────
    # Manual slider tuning — mirrors the GridSearchCV concept from Week 10.2
    st.markdown('<div class="section-header">3 · Hyperparameters</div>', unsafe_allow_html=True)

    if model_name == "Logistic Regression":
        C = st.slider("Regularization C", 0.01, 10.0, 1.0, 0.01,
                      help="Smaller = stronger regularization")
        max_iter = st.slider("Max Iterations", 100, 2000, 500, 100)
        solver = st.selectbox("Solver", ["lbfgs", "saga", "liblinear"])

    elif model_name == "Decision Tree":
        max_depth = st.slider("Max Depth", 1, 20, 4,
                              help="Week 9.2 used max_depth=4 as default")
        min_samples_split = st.slider("Min Samples Split", 2, 50, 2,
                                      help="Tuned via GridSearchCV in Week 10.2")
        criterion = st.selectbox("Criterion", ["gini", "entropy"],
                                 help="Both explored in Week 10.2 GridSearchCV")

    elif model_name == "K-Nearest Neighbors":
        n_neighbors = st.slider("k (Neighbors)", 1, 19, 5, 2,
                                help="Week 11.2 explored k=1 to 19 (odd numbers)")
        weights = st.selectbox("Weights", ["uniform", "distance"])
        metric = st.selectbox("Distance Metric", ["euclidean", "manhattan", "minkowski"])

    # ── Step 4: Train/Test Split ───────────────────────────────────────────────
    st.markdown('<div class="section-header">4 · Split</div>', unsafe_allow_html=True)
    test_size = st.slider("Test set %", 10, 40, 20,
                          help="Week 9.1 used test_size=0.2") / 100
    random_seed = st.number_input("Random seed", value=42, step=1,
                                  help="Week 9.1 used random_state=42")

    run_btn = st.button("Train Model", use_container_width=True, type="primary")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Dataset Preview & Column Selection
# ═══════════════════════════════════════════════════════════════════════════════
if df_raw is None:
    st.markdown("""
    <div class="info-box">
      Select a data source in the sidebar to get started.
      You can upload your own CSV or try the built-in Olympics 2008 sample.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)

col_prev, col_info = st.columns([3, 1])
with col_prev:
    st.dataframe(df_raw.head(20), use_container_width=True, height=220)
with col_info:
    st.metric("Rows", f"{df_raw.shape[0]:,}")
    st.metric("Columns", df_raw.shape[1])
    st.metric("Missing values", f"{df_raw.isnull().sum().sum():,}")

# ── Feature & Target Selection ─────────────────────────────────────────────────
st.markdown('<div class="section-header">Feature and Target Selection</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    target_col = st.selectbox(
        "Target column (what to predict)",
        df_raw.columns.tolist(),
        index=len(df_raw.columns) - 1
    )
with col_b:
    feature_candidates = [c for c in df_raw.columns if c != target_col]
    feature_cols = st.multiselect(
        "Feature columns",
        feature_candidates,
        default=feature_candidates[:min(6, len(feature_candidates))]
    )

if not feature_cols:
    st.warning("Please select at least one feature column.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPROCESSING — matches class notebook style exactly (Weeks 9–11)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def preprocess(df, features, target, seed, tsize):
    """
    Preprocessing steps that mirror the class notebooks:
      Step 1 — Drop missing values       (Week 9.1: df.dropna())
      Step 2 — Encode categoricals       (Week 9.1: pd.get_dummies(..., drop_first=True))
      Step 3 — Train/test split          (Week 9.1: train_test_split(..., test_size=0.2, random_state=42))
      Step 4 — Scale with StandardScaler (Week 11.2: needed for KNN distance calculations)
    """
    # Step 1: Keep only selected columns and drop any rows with missing values
    df2 = df[features + [target]].dropna()

    # Step 2: Encode categorical feature columns using pd.get_dummies()
    # This is the exact approach used in Week 9.1 and 9.2:
    #   df = pd.get_dummies(df, columns=['sex'], drop_first=True)
    cat_cols = df2[features].select_dtypes(include="object").columns.tolist()
    if cat_cols:
        df2 = pd.get_dummies(df2, columns=cat_cols, drop_first=True)

    # Encode target column if it is categorical (text labels)
    if df2[target].dtype == object:
        df2[target] = pd.Categorical(df2[target]).codes

    # Get the updated feature list after get_dummies expands categorical columns
    updated_features = [c for c in df2.columns if c != target]

    X = df2[updated_features]
    y = df2[target]

    # Step 3: Train/test split
    # Week 9.1 used: train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=tsize, random_state=seed
    )

    # Step 4: Scale features with StandardScaler
    # Week 11.2: KNN is sensitive to feature scale, so scaling is important
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, updated_features

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
if run_btn:
    with st.spinner("Preprocessing and training..."):
        try:
            X_train, X_test, y_train, y_test, used_features = preprocess(
                df_raw, feature_cols, target_col, int(random_seed), test_size
            )

            # Initialize the model — same pattern used in every class notebook
            if model_name == "Logistic Regression":
                # Week 9.1: model = LogisticRegression()
                model = LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    solver=solver,
                    random_state=int(random_seed)
                )
            elif model_name == "Decision Tree":
                # Week 9.2: model = DecisionTreeClassifier(random_state=42, max_depth=4)
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    criterion=criterion,
                    random_state=int(random_seed)
                )
            elif model_name == "K-Nearest Neighbors":
                # Week 11.2: knn = KNeighborsClassifier(n_neighbors=5)
                model = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric=metric
                )

            # Train the model — Week 9.1: model.fit(X_train, y_train)
            model.fit(X_train, y_train)

            # Make predictions — Week 9.1: y_pred = model.predict(X_test)
            y_pred = model.predict(X_test)

            # Calculate accuracy — Week 9.1: accuracy_score(y_test, y_pred)
            classes = np.unique(y_test)
            acc = accuracy_score(y_test, y_pred)

        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

    # ── Metric Cards ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

    # Pull precision/recall/F1 from classification_report (Week 9.1)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    prec = report_dict.get("weighted avg", {}).get("precision", 0)
    rec  = report_dict.get("weighted avg", {}).get("recall", 0)
    f1   = report_dict.get("weighted avg", {}).get("f1-score", 0)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, f"{acc:.3f}", "Accuracy"),
        (c2, f"{prec:.3f}", "Precision"),
        (c3, f"{rec:.3f}", "Recall"),
        (c4, f"{f1:.3f}", "F1 Score"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="val">{val}</div>
              <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Output Tabs ───────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Confusion Matrix",
        "Classification Report",
        "ROC Curve",
        "Feature Importance"
    ])

    plt.style.use("dark_background")

    # Tab 1: Confusion Matrix — Week 9.1 used confusion_matrix()
    with tab1:
        st.caption("Week 9 & 10 notebooks — rows = actual class, columns = predicted class.")
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#161b22")
        ax.set_facecolor("#161b22")
        cm = confusion_matrix(y_test, y_pred)
        display_labels = classes[:15] if len(classes) > 15 else classes
        cm_disp = cm[:15, :15] if len(classes) > 15 else cm
        sns.heatmap(cm_disp, annot=len(classes) <= 12, fmt="d",
                    cmap="Blues", ax=ax,
                    xticklabels=display_labels, yticklabels=display_labels,
                    linewidths=0.5, linecolor="#21262d")
        ax.set_xlabel("Predicted", color="#8b949e")
        ax.set_ylabel("Actual", color="#8b949e")
        ax.set_title("Confusion Matrix", color="#e6edf3", pad=10)
        ax.tick_params(colors="#8b949e", rotation=45)
        st.pyplot(fig, use_container_width=True)

    # Tab 2: Classification Report — Week 9.1 used classification_report()
    with tab2:
        st.caption("Week 9 notebooks — shows precision, recall, and F1-score per class.")
        report_df = pd.DataFrame(report_dict).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    # Tab 3: ROC Curve — covered in Week 10.1
    with tab3:
        st.caption("Week 10.1 — plots True Positive Rate vs False Positive Rate.")
        if len(classes) == 2:
            try:
                # Week 10.1: y_probs = model.predict_proba(X_test)[:, 1]
                y_prob = model.predict_proba(X_test)[:, 1]
                # Week 10.1: fpr, tpr, thresholds = roc_curve(y_test, y_probs)
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                # Week 10.1: roc_auc_score(y_test, y_probs)
                auc = roc_auc_score(y_test, y_prob)
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                fig2.patch.set_facecolor("#161b22")
                ax2.set_facecolor("#161b22")
                ax2.plot(fpr, tpr, color="#58a6ff", lw=2, label=f"AUC = {auc:.3f}")
                ax2.plot([0, 1], [0, 1], color="#30363d", lw=1, linestyle="--",
                         label="Random classifier")
                ax2.set_xlabel("False Positive Rate", color="#8b949e")
                ax2.set_ylabel("True Positive Rate", color="#8b949e")
                ax2.set_title("ROC Curve", color="#e6edf3")
                ax2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
                ax2.tick_params(colors="#8b949e")
                st.pyplot(fig2, use_container_width=True)
                st.metric("AUC Score", f"{auc:.4f}")
            except Exception as e:
                st.info(f"ROC curve unavailable: {e}")
        else:
            st.info(f"ROC curve is shown for binary classification only. "
                    f"Your target has {len(classes)} unique classes.")

    # Tab 4: Feature Importance — Decision Tree and Logistic Regression
    with tab4:
        st.caption("Decision Tree uses .feature_importances_; Logistic Regression uses .coef_ (Week 9.1).")
        importances = None

        if model_name == "Decision Tree":
            # DecisionTreeClassifier exposes .feature_importances_ (Gini importance)
            importances = model.feature_importances_
        elif model_name == "Logistic Regression" and len(classes) == 2:
            # Week 9.1 covered model.coef_[0] for binary logistic regression
            importances = np.abs(model.coef_[0])

        if importances is not None and len(used_features) == len(importances):
            fi_df = (pd.DataFrame({"feature": used_features, "importance": importances})
                     .sort_values("importance", ascending=True)
                     .tail(20))
            fig3, ax3 = plt.subplots(figsize=(6, max(4, len(fi_df) * 0.35)))
            fig3.patch.set_facecolor("#161b22")
            ax3.set_facecolor("#161b22")
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(fi_df)))
            ax3.barh(fi_df["feature"], fi_df["importance"], color=colors)
            ax3.set_xlabel("Importance", color="#8b949e")
            ax3.set_title("Feature Importances (top 20)", color="#e6edf3")
            ax3.tick_params(colors="#8b949e")
            st.pyplot(fig3, use_container_width=True)
        else:
            st.info("Feature importance is not available for K-Nearest Neighbors "
                    "or multi-class Logistic Regression.")

    # ── Run Summary ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Run Summary</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"""
        <div class="info-box">
          <b>Model:</b> {model_name}<br>
          <b>Train samples:</b> {len(X_train):,} &nbsp;|&nbsp;
          <b>Test samples:</b> {len(X_test):,}<br>
          <b>Features after encoding:</b> {len(used_features)}
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="info-box">
          <b>Target classes:</b> {len(classes)}<br>
          <b>Test split:</b> {int(test_size * 100)}% &nbsp;|&nbsp;
          <b>Random seed:</b> {int(random_seed)}<br>
          <b>Preprocessing:</b> dropna → get_dummies → StandardScaler
        </div>""", unsafe_allow_html=True)

else:
    if df_raw is not None:
        st.markdown("""
        <div class="info-box">
          Dataset loaded. Configure your model in the sidebar and click
          <b>Train Model</b> to begin.
        </div>""", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#8b949e;font-size:0.8rem;'>"
    "ML Explorer · Streamlit + scikit-learn · "
    "Logistic Regression · Decision Tree · KNN </p>",
    unsafe_allow_html=True
)