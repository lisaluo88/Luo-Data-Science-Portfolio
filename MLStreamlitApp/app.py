# =============================================================================
# SECTION 1: IMPORTS
# -----------------------------------------------------------------------------
# Intention: Load every library the app needs before anything else runs.
# - streamlit   → builds the entire web interface (buttons, sliders, tables)
# - pandas      → loads and manipulates the dataset
# - numpy       → handles numeric arrays and math operations
# - matplotlib  → draws charts (confusion matrix, ROC curve, feature importance)
# - seaborn     → draws the heatmap for the confusion matrix
# - warnings    → silences non-critical warning messages so output stays clean
# All sklearn imports come directly from the Week 9–11 class notebooks.
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# --- Sklearn ML tools (all covered in class) ----------------------------------
from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import StandardScaler       
from sklearn.linear_model import LogisticRegression    
from sklearn.tree import DecisionTreeClassifier        
from sklearn.neighbors import KNeighborsClassifier     
from sklearn.metrics import (
    accuracy_score,         
    confusion_matrix,       
    classification_report,  
    roc_curve,              
    roc_auc_score           
)


# =============================================================================
# SECTION 2: PAGE CONFIGURATION
# -----------------------------------------------------------------------------
# Intention: Set global Streamlit settings before any content is rendered.
# This must be the FIRST Streamlit call in the script — if it comes after
# any st.write() or st.markdown(), Streamlit will throw an error.
# =============================================================================
st.set_page_config(
    page_title="ML Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# SECTION 3: CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
# Intention: Make the app look polished using raw CSS.
# Using CSS lets us add custom fonts, colors, card layouts, and dark-theme backgrounds.
# =============================================================================
st.markdown("""
<style>
/* Import custom fonts from Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Fira+Mono:wght@400;700&display=swap');

/* Apply the custom font to the entire app */
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

/* Hero banner at the top of the page */
.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 50%, #1a1f2e 100%);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
/* Subtle glow effect behind the hero text */
.hero::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at top left, rgba(88,166,255,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at bottom right, rgba(63,185,80,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero h1 { font-size: 2.4rem; font-weight: 700; color: #e6edf3; margin: 0 0 0.4rem 0; letter-spacing: -1px; }
.hero p  { color: #8b949e; font-size: 1rem; margin: 0; }
.hero .accent { color: #58a6ff; }

/* Card style used for displaying metric values (Accuracy, F1, etc.) */
.metric-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 1.2rem 1.5rem; text-align: center; transition: border-color 0.2s;
}
.metric-card:hover { border-color: #58a6ff; }
.metric-card .val { font-size: 2rem; font-weight: 700; color: #58a6ff; font-family: 'Fira Mono', monospace; }
.metric-card .lbl { font-size: 0.78rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

/* Small uppercase label used as a divider between sidebar sections */
.section-header {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase;
    color: #8b949e; border-bottom: 1px solid #21262d; padding-bottom: 0.5rem; margin: 1.5rem 0 1rem 0;
}

/* Blue-tinted box used for tips, instructions, and the run summary */
.info-box {
    background: rgba(88,166,255,0.05); border-left: 3px solid #58a6ff;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.8rem 0;
    font-size: 0.88rem; color: #8b949e;
}

/* Dark background for the sidebar panel */
section[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #21262d; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 4: HERO BANNER
# -----------------------------------------------------------------------------
# Intention: Display a large, visually striking title at the top of the page.
# This is the first thing the user sees when the app loads.
# The <span class="accent"> wraps "Explorer" to give it the blue color.
# =============================================================================
st.markdown("""
<div class="hero">
  <h1>ML <span class="accent">Explorer</span></h1>
  <p>Upload a dataset · choose a model · tune hyperparameters · inspect performance — all in one place.</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 5: SIDEBAR — USER CONTROLS
# -----------------------------------------------------------------------------
# Intention: Put all user-facing controls (dataset, model, hyperparameters,
# split settings, train button) in the left sidebar panel.
# The sidebar is organized into 4 numbered steps to guide the user in order.
# =============================================================================
with st.sidebar:
    st.markdown("## Configuration")

    # -------------------------------------------------------------------------
    # STEP 1 — Dataset Selection
    # -------------------------------------------------------------------------
    # Intention: Let the user choose between uploading their own CSV file or
    # using the built-in 2008 Olympics sample dataset.
    # df_raw starts as None and only gets a value once data is actually loaded.
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">1 · Dataset</div>', unsafe_allow_html=True)

    # Radio button: user picks one of two data sources
    data_source = st.radio(
        "Source",
        ["Upload CSV", "Sample: Olympics 2008"],
        label_visibility="collapsed"   # hides the "Source" label to save space
    )

    # df_raw holds the loaded DataFrame; starts as None until data is selected
    df_raw = None

    if data_source == "Upload CSV":
        # Show a file uploader widget that only accepts .csv files
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded:
            # pd.read_csv() loads the file into a pandas DataFrame
            df_raw = pd.read_csv(uploaded)
            st.success(f"Loaded {df_raw.shape[0]:,} rows x {df_raw.shape[1]} cols")

    else:
        # -----------------------------------------------------------------
        # BUILT-IN SAMPLE: 2008 Beijing Olympics Medalists
        # -----------------------------------------------------------------
        # Intention: Load the Olympics CSV and reshape it from wide format
        # (one column per sport) into long format (one row per medal won).
        # @st.cache_data tells Streamlit to save the result of this function.
        # If the user hasn't changed the data source, Streamlit reuses the
        # cached result instead of re-running the function — this makes the
        # app much faster on repeated interactions.
        # -----------------------------------------------------------------
        @st.cache_data
        def load_olympics():
            # Load the raw CSV — 1,875 rows, 71 columns (one per sport/gender)
            df = pd.read_csv("olympics_08_medalists.csv")
            id_col = "medalist_name"
            sport_cols = [c for c in df.columns if c != id_col]

            # Reshape wide → long using method chaining (Week 6.2 technique):
            # .melt()   → turns each sport column into its own row
            # .dropna() → removes rows where the athlete has no medal (was None)
            # .assign() → adds two new columns: gender and sport name
            melted = (
                df.melt(
                    id_vars=id_col,
                    value_vars=sport_cols,
                    var_name="event",    # column name becomes the event label
                    value_name="medal"   # cell value becomes the medal label
                )
                .dropna(subset=["medal"])  # keep only rows where a medal was won
                .assign(
                    # Extract gender from the event name prefix (e.g. "male_swimming")
                    gender=lambda d: d["event"].apply(
                        lambda x: "male" if x.startswith("male_") else "female"),
                    # Extract sport name by removing the gender prefix
                    sport=lambda d: d["event"].apply(
                        lambda x: x.replace("male_", "").replace("female_", ""))
                )
            )
            # Return only the four clean columns needed for ML
            return melted[["medalist_name", "gender", "sport", "medal"]]

        df_raw = load_olympics()
        st.info("2008 Beijing Olympics medalists — melted to long format.")

        # Show a collapsible preview of the first 10 rows
        with st.expander("Preview data"):
            st.dataframe(df_raw.head(10), use_container_width=True)

    # -------------------------------------------------------------------------
    # STEP 2 — Model Selection
    # -------------------------------------------------------------------------
    # Intention: Let the user pick which supervised ML algorithm to train.
    # The chosen model_name string is used later to give example of the right
    # sklearn class with the user's hyperparameter settings.
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">2 · Model</div>', unsafe_allow_html=True)

    model_name = st.selectbox("Algorithm", [
        "Logistic Regression",   #linear model for classification
        "Decision Tree",         #tree-based splits on features
        "K-Nearest Neighbors",   #classifies based on nearby points
    ])

    # -------------------------------------------------------------------------
    # STEP 3 — Hyperparameter Controls
    # -------------------------------------------------------------------------
    # Intention: Show different sliders depending on which model was selected.
    # Instead of the computer searching all combinations, the user manually picks values.
    # Each slider maps directly to a keyword argument passed to the sklearn model.
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">3 · Hyperparameters</div>', unsafe_allow_html=True)

    if model_name == "Logistic Regression":
        # C controls regularization strength — smaller C = more regularization
        # (penalizes large coefficients to reduce overfitting)
        C = st.slider("Regularization C", 0.01, 10.0, 1.0, 0.01,
                      help="Smaller = stronger regularization")
        # max_iter controls how many optimization steps the solver can take
        max_iter = st.slider("Max Iterations", 100, 2000, 500, 100)
        # solver is the optimization algorithm used to find the coefficients
        solver = st.selectbox("Solver", ["lbfgs", "saga", "liblinear"])

    elif model_name == "Decision Tree":
        # max_depth limits how deep the tree can grow — prevents overfitting
        # Week 9.2 used max_depth=4 as the default
        max_depth = st.slider("Max Depth", 1, 20, 4,
                              help="Week 9.2 used max_depth=4 as default")
        # min_samples_split is the minimum number of samples needed to split a node
        # Week 10.2 tuned this via GridSearchCV
        min_samples_split = st.slider("Min Samples Split", 2, 50, 2,
                                      help="Tuned via GridSearchCV in Week 10.2")
        # criterion decides how the tree measures the quality of each split
        criterion = st.selectbox("Criterion", ["gini", "entropy"],
                                 help="Both explored in Week 10.2 GridSearchCV")

    elif model_name == "K-Nearest Neighbors":
        # n_neighbors (k) — how many nearby training points vote on the prediction
        # Week 11.2 tested k=1 through k=19 (odd numbers only)
        n_neighbors = st.slider("k (Neighbors)", 1, 19, 5, 2,
                                help="Week 11.2 explored k=1 to 19 (odd numbers)")
        # weights — "uniform" gives all neighbors equal votes; "distance" gives
        # closer neighbors more influence
        weights = st.selectbox("Weights", ["uniform", "distance"])
        # metric — the formula used to calculate distance between data points
        metric = st.selectbox("Distance Metric", ["euclidean", "manhattan", "minkowski"])

    # -------------------------------------------------------------------------
    # STEP 4 — Train/Test Split Settings
    # -------------------------------------------------------------------------
    # Intention: Let the user control how the data is divided between training
    # and testing, and set the random seed for reproducibility.
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">4 · Split</div>', unsafe_allow_html=True)

    # Slider returns an integer (e.g. 20), divide by 100 to get a decimal (0.2)
    test_size = st.slider("Test set %", 10, 40, 20,
                          help="Week 9.1 used test_size=0.2") / 100

    # Random seed ensures the train/test split is the same every time it runs
    random_seed = st.number_input("Random seed", value=42, step=1,
                                  help="Week 9.1 used random_state=42")

    # -------------------------------------------------------------------------
    # TRAIN BUTTON
    # -------------------------------------------------------------------------
    # Intention: Nothing should happen until the user explicitly clicks this.
    # run_btn is True only in the frame where the user clicks it, and False otherwise
    # -------------------------------------------------------------------------
    run_btn = st.button("Train Model", use_container_width=True, type="primary")


# =============================================================================
# SECTION 6: SAFETY GUARD — Stop if no data is loaded
# -----------------------------------------------------------------------------
# Intention: Prevent the rest of the app from running when df_raw is still None.
# If no dataset has been selected yet, df_raw is None, and calling methods like
# .head() or .shape on it would crash with an AttributeError.
# st.stop() halts the script immediately at this point — nothing below runs.
# =============================================================================
if df_raw is None:
    st.markdown("""
    <div class="info-box">
      Select a data source in the sidebar to get started.
      You can upload your own CSV or try the built-in Olympics 2008 sample.
    </div>
    """, unsafe_allow_html=True)
    st.stop()   # halt execution here until the user picks a dataset


# =============================================================================
# SECTION 7: DATASET PREVIEW
# -----------------------------------------------------------------------------
# Intention: Show the user what their data looks like before they train a model.
# Displaying a preview helps them understand the structure of the data and
# choose appropriate feature and target columns.
# st.columns() creates a side-by-side layout — 3/4 for the table, 1/4 for stats.
# =============================================================================
st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)

col_prev, col_info = st.columns([3, 1])

with col_prev:
    # Show the first 20 rows as a scrollable interactive table
    st.dataframe(df_raw.head(20), use_container_width=True, height=220)

with col_info:
    # Show three quick summary stats about the dataset
    st.metric("Rows", f"{df_raw.shape[0]:,}")      # total number of records
    st.metric("Columns", df_raw.shape[1])           # total number of columns
    # isnull().sum().sum() counts all missing values across every column
    st.metric("Missing values", f"{df_raw.isnull().sum().sum():,}")


# =============================================================================
# SECTION 8: FEATURE AND TARGET SELECTION
# -----------------------------------------------------------------------------
# Intention: Let the user interactively choose which column the model should
# predict (target) and which columns to use as inputs (features).
# Here the user picks those columns themselves using dropdowns.
# =============================================================================
st.markdown('<div class="section-header">Feature and Target Selection</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    # Target column = the variable the model is trying to predict (y)
    # Default to the last column in the dataset as a reasonable starting guess
    target_col = st.selectbox(
        "Target column (what to predict)",
        df_raw.columns.tolist(),
        index=len(df_raw.columns) - 1
    )

with col_b:
    # Feature columns = all inputs the model uses to make predictions (X)
    # Exclude whichever column was chosen as the target
    feature_candidates = [c for c in df_raw.columns if c != target_col]
    feature_cols = st.multiselect(
        "Feature columns",
        feature_candidates,
        # Pre-select up to the first 6 columns as a sensible default
        default=feature_candidates[:min(6, len(feature_candidates))]
    )

# Guard: stop if no features were selected — can't train with zero inputs
if not feature_cols:
    st.warning("Please select at least one feature column.")
    st.stop()


# =============================================================================
# SECTION 9: PREPROCESSING FUNCTION
# -----------------------------------------------------------------------------
# Intention: Clean and prepare the data for ML in a single reusable function.
# @st.cache_data caches the output — if the user only changes a hyperparameter
# slider (not the dataset or columns), Streamlit reuses the cached preprocessed
# data instead of repeating all the steps again. This speeds up the app.
# =============================================================================
@st.cache_data(show_spinner=False)
def preprocess(df, features, target, seed, tsize):
    """
    Prepares data for ML following the exact steps from the class notebooks.

    Parameters:
        df       — the full raw DataFrame
        features — list of column names to use as input features
        target   — column name to predict
        seed     — random state for reproducibility (Week 9.1 used 42)
        tsize    — fraction of data reserved for testing (Week 9.1 used 0.2)

    Returns:
        X_train, X_test   — scaled feature arrays for training and testing
        y_train, y_test   — target arrays for training and testing
        updated_features  — list of feature names after get_dummies() expansion
    """

    # Step 1: Keep only the selected columns and remove any rows with missing values
    df2 = df[features + [target]].dropna()

    # Step 2: Encode categorical (text) feature columns using pd.get_dummies()
    # Here we detect which selected feature columns contain text automatically,
    # then apply get_dummies only to those columns.
    # drop_first=True removes one dummy column per category to avoid the
    # "dummy variable trap" (multicollinearity) — same parameter as in class.
    cat_cols = df2[features].select_dtypes(include="object").columns.tolist()
    if cat_cols:
        df2 = pd.get_dummies(df2, columns=cat_cols, drop_first=True)

    # If the target column is also text (e.g. "gold"/"silver"/"bronze"),
    # convert it to numeric codes so sklearn can process it
    if df2[target].dtype == object:
        df2[target] = pd.Categorical(df2[target]).codes

    # After get_dummies(), categorical columns are expanded into multiple binary
    # columns, so the feature list has changed — recalculate it
    updated_features = [c for c in df2.columns if c != target]

    X = df2[updated_features]   # feature matrix
    y = df2[target]             # target vector

    # Step 3: Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=tsize, random_state=seed
    )

    # Step 4: Scale all features using StandardScaler
    # fit_transform() on training data: learns the mean and std from training set
    # transform() on test data: applies the SAME scaling learned from training
    # (never fit on test data — that would leak information)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, updated_features


# =============================================================================
# SECTION 10: MODEL TRAINING — runs only when the Train button is clicked
# -----------------------------------------------------------------------------
# Intention: Train the selected model on the preprocessed data and compute
# all evaluation metrics. This entire block is wrapped in "if run_btn:" so
# nothing here executes until the user clicks the "Train Model" button.
# =============================================================================
if run_btn:

    # Show a spinner animation while preprocessing and training are running
    with st.spinner("Preprocessing and training..."):
        try:
            # Run the preprocessing pipeline defined in Section 9
            X_train, X_test, y_train, y_test, used_features = preprocess(
                df_raw, feature_cols, target_col, int(random_seed), test_size
            )

            # -----------------------------------------------------------------
            # Instantiate the selected model with user-chosen hyperparameters
            # This mirrors the exact class notebook pattern, e.g.:
            #   model = LogisticRegression()          
            #   model = DecisionTreeClassifier(...)   
            #   knn   = KNeighborsClassifier(n_neighbors=5)  
            # -----------------------------------------------------------------
            if model_name == "Logistic Regression":
                model = LogisticRegression(
                    C=C,                          # regularization strength
                    max_iter=max_iter,            # maximum solver iterations
                    solver=solver,                # optimization algorithm
                    random_state=int(random_seed) # for reproducibility
                )

            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier(
                    max_depth=max_depth,               # limits tree depth
                    min_samples_split=min_samples_split, # min samples to split
                    criterion=criterion,               # gini or entropy
                    random_state=int(random_seed)      # for reproducibility
                )

            elif model_name == "K-Nearest Neighbors":
                model = KNeighborsClassifier(
                    n_neighbors=n_neighbors,  # how many neighbors vote
                    weights=weights,          # uniform or distance-weighted
                    metric=metric             # distance formula
                )

            # Train the model 
            model.fit(X_train, y_train)

            # Generate predictions on the held-out test set
            y_pred = model.predict(X_test)

            # Get the unique class labels present in the test set
            classes = np.unique(y_test)

            # Calculate overall accuracy
            acc = accuracy_score(y_test, y_pred)

        except Exception as e:
            # If anything fails (e.g. not enough samples), show the error clearly
            st.error(f"Training failed: {e}")
            st.stop()


    # =========================================================================
    # SECTION 11: METRIC CARDS
    # -------------------------------------------------------------------------
    # Intention: Display the four key evaluation metrics in large, easy-to-read
    # cards at the top of the results section so the user immediately sees
    # how well their model performed.
    # classification_report() with output_dict=True returns a Python dictionary
    # instead of a printed string, so we can extract individual numbers from it.
    # "weighted avg" averages the metrics across all classes
    # =========================================================================
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

    # Get precision, recall, and F1 from the classification report dictionary
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    prec = report_dict.get("weighted avg", {}).get("precision", 0)
    rec  = report_dict.get("weighted avg", {}).get("recall", 0)
    f1   = report_dict.get("weighted avg", {}).get("f1-score", 0)

    # Display the four metrics side-by-side using the .metric-card CSS class
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, f"{acc:.3f}", "Accuracy"),   # fraction of all predictions that were correct
        (c2, f"{prec:.3f}", "Precision"), # of predicted positives, how many were actually positive
        (c3, f"{rec:.3f}", "Recall"),     # of actual positives, how many did we catch
        (c4, f"{f1:.3f}", "F1 Score"),    # harmonic mean of precision and recall
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="val">{val}</div>
              <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")  # adds a small vertical gap between the cards and the tabs

    # =========================================================================
    # SECTION 12: EVALUATION TABS
    # -------------------------------------------------------------------------
    # Intention: Organize the four evaluation charts/tables into separate tabs
    # so the user can switch between them without the page becoming too long.
    # Each tab matches a specific evaluation technique from the class notebooks.
    # =========================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "Confusion Matrix",       
        "Classification Report",  
        "ROC Curve",              
        "Feature Importance"      
    ])

    # Use dark background theme for all matplotlib charts to match the app style
    plt.style.use("dark_background")

    # -------------------------------------------------------------------------
    # TAB 1: CONFUSION MATRIX
    # -------------------------------------------------------------------------
    # Intention: Show a heatmap grid where each cell tells us how many times
    # the model predicted class X when the actual class was Y.
    # Diagonal cells (top-left to bottom-right) = correct predictions.
    # Off-diagonal cells = mistakes.
    # confusion_matrix(y_test, y_pred) + sns.heatmap()
    # The figure size and font size scale dynamically so labels never overlap,
    # regardless of how many classes the target column has.
    # -------------------------------------------------------------------------
    with tab1:
        st.caption("Week 9 & 10 notebooks — rows = actual class, columns = predicted class.")
        cm = confusion_matrix(y_test, y_pred)
        n_classes = len(classes)

        # Cap the display at 15 classes — beyond that the chart becomes unreadable
        display_labels = classes[:15] if n_classes > 15 else classes
        cm_disp = cm[:15, :15] if n_classes > 15 else cm
        if n_classes > 15:
            st.info(f"Showing first 15 of {n_classes} classes to keep the chart readable.")

        # Scale figure size with the number of classes so cells don't get tiny
        fig_size = max(6, len(display_labels) * 0.6)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        fig.patch.set_facecolor("#161b22")
        ax.set_facecolor("#161b22")

        # Only print numbers inside cells when there are few enough to be readable
        annotate = len(display_labels) <= 10
        # Shrink font as number of classes grows to prevent overlap
        font_size = max(6, 12 - len(display_labels))

        sns.heatmap(cm_disp, annot=annotate, fmt="d",
                    cmap="Blues", ax=ax,
                    xticklabels=display_labels, yticklabels=display_labels,
                    linewidths=0.5, linecolor="#21262d",
                    annot_kws={"size": font_size})
        ax.set_xlabel("Predicted", color="#8b949e", labelpad=10)
        ax.set_ylabel("Actual", color="#8b949e", labelpad=10)
        ax.set_title("Confusion Matrix", color="#e6edf3", pad=15)

        # Rotate x-axis labels 45° so they don't collide with each other
        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=45, ha="right",
                           fontsize=font_size, color="#8b949e")
        ax.set_yticklabels(ax.get_yticklabels(),
                           rotation=0,
                           fontsize=font_size, color="#8b949e")
        plt.tight_layout()   # automatically adjust layout so nothing gets cut off
        st.pyplot(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: CLASSIFICATION REPORT
    # -------------------------------------------------------------------------
    # Intention: Show the full per-class precision, recall, and F1 table.
    # Here we convert it to a DataFrame for a cleaner table display.
    # .T transposes rows and columns so class labels appear as row headers.
    # -------------------------------------------------------------------------
    with tab2:
        st.caption("Week 9 notebooks — shows precision, recall, and F1-score per class.")
        report_df = pd.DataFrame(report_dict).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: ROC CURVE
    # -------------------------------------------------------------------------
    # Intention: Plot the trade-off between catching true positives and
    # accidentally flagging false positives at different decision thresholds.
    # A model that perfectly separates classes has AUC = 1.0.
    # A random coin flip has AUC = 0.5 (shown as the dashed diagonal line).
    # ROC curves only work for binary classification (2 classes), so we show
    # an informational message if the target has more than 2 classes.
    # -------------------------------------------------------------------------
    with tab3:
        st.caption("Week 10.1 — plots True Positive Rate vs False Positive Rate.")
        if len(classes) == 2:
            try:
                # predict_proba() returns the probability of each class
                # [:, 1] selects the probability of the POSITIVE class (class 1)
                # Week 10.1: y_probs = model.predict_proba(X_test)[:, 1]
                y_prob = model.predict_proba(X_test)[:, 1]

                # roc_curve() computes the (fpr, tpr) pairs at every threshold
                fpr, tpr, _ = roc_curve(y_test, y_prob)

                # AUC = Area Under the Curve — single summary score (0 to 1)
                auc = roc_auc_score(y_test, y_prob)

                fig2, ax2 = plt.subplots(figsize=(6, 5))
                fig2.patch.set_facecolor("#161b22")
                ax2.set_facecolor("#161b22")
                # Plot the ROC curve for our model
                ax2.plot(fpr, tpr, color="#58a6ff", lw=2, label=f"AUC = {auc:.3f}")
                # Plot the diagonal baseline — this is what a random model looks like
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

    # -------------------------------------------------------------------------
    # TAB 4: FEATURE IMPORTANCE
    # -------------------------------------------------------------------------
    # Intention: Show which features had the most influence on the model's
    # predictions, displayed as a horizontal bar chart sorted by importance.
    # - Decision Tree    → uses .feature_importances_ (Gini-based importance)
    # - Logistic Regression (binary only) → uses abs(model.coef_[0])
    # - KNN → has no concept of feature importance (skipped with a message)
    # Figure height scales dynamically so bars and labels never overlap,
    # no matter how many features are in the dataset.
    # -------------------------------------------------------------------------
    with tab4:
        st.caption("Decision Tree uses .feature_importances_; Logistic Regression uses .coef_ (Week 9.1).")
        importances = None

        if model_name == "Decision Tree":
            # .feature_importances_ gives the Gini importance of each feature
            importances = model.feature_importances_

        elif model_name == "Logistic Regression" and len(classes) == 2:
            # model.coef_[0] gives the coefficient for each feature
            # np.abs() takes absolute value — sign just indicates direction,
            # magnitude indicates how much influence the feature has
            importances = np.abs(model.coef_[0])

        if importances is not None and len(used_features) == len(importances):
            # Build a DataFrame, sort by importance ascending, show top 20
            fi_df = (
                pd.DataFrame({"feature": used_features, "importance": importances})
                .sort_values("importance", ascending=True)
                .tail(20)  # keep only top 20 most important features
            )

            # Give each bar 0.5 units of vertical space so labels never stack
            bar_height = 0.5
            fig_height = max(5, len(fi_df) * bar_height + 2)
            fig3, ax3 = plt.subplots(figsize=(9, fig_height))
            fig3.patch.set_facecolor("#161b22")
            ax3.set_facecolor("#161b22")

            # Color bars with a gradient from light to dark blue
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(fi_df)))
            ax3.barh(fi_df["feature"], fi_df["importance"],
                     color=colors, height=0.6)
            ax3.set_xlabel("Importance", color="#8b949e", labelpad=8)
            ax3.set_title("Feature Importances (top 20)", color="#e6edf3", pad=12)

            # Shrink y-axis font size if there are many features
            label_fontsize = max(7, 11 - len(fi_df) // 5)
            ax3.tick_params(axis="y", colors="#8b949e", labelsize=label_fontsize)
            ax3.tick_params(axis="x", colors="#8b949e")
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
        else:
            st.info("Feature importance is not available for K-Nearest Neighbors "
                    "or multi-class Logistic Regression.")

    # =========================================================================
    # SECTION 13: RUN SUMMARY
    # -------------------------------------------------------------------------
    # Intention: After training, show a concise recap of the exact settings
    # that were used so the user can reproduce or reference their run.
    # Displays model name, sample counts, encoding info, split, and seed.
    # =========================================================================
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

# =============================================================================
# SECTION 14: PRE-TRAIN PROMPT
# -----------------------------------------------------------------------------
# Intention: If the dataset is loaded but the user hasn't clicked Train yet,
# show a short prompt reminding them to click the button.
# This only shows when run_btn is False (the else branch of "if run_btn:").
# =============================================================================
else:
    if df_raw is not None:
        st.markdown("""
        <div class="info-box">
          Dataset loaded. Configure your model in the sidebar and click
          <b>Train Model</b> to begin.
        </div>""", unsafe_allow_html=True)


# =============================================================================
# SECTION 15: FOOTER
# -----------------------------------------------------------------------------
# Intention: Display a small credit line at the very bottom of the page.
# Standard practice for any deployed app — credits the tools used and
# notes which class weeks each model comes from.
# =============================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#8b949e;font-size:0.8rem;'>"
    "ML Explorer · Streamlit + scikit-learn · "
    "Logistic Regression (Wk 9.1) · Decision Tree (Wk 9.2) · KNN (Wk 11.2)</p>",
    unsafe_allow_html=True
)