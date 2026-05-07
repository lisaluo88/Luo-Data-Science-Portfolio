# =============================================================================
# SECTION 1: IMPORTS
# -----------------------------------------------------------------------------
# Intention: Load every library the app needs before anything else runs.
# - streamlit   → builds the entire web interface (buttons, sliders, tables)
# - pandas      → loads and manipulates the dataset
# - numpy       → handles numeric arrays and math operations
# - matplotlib  → draws charts (scree plots, scatter, dendrogram, etc.)
# - seaborn     → used for cleaner styling on a few plots
# - warnings    → silences non-critical warning messages so output stays clean
# All sklearn imports come directly from the Week 12–13 class notebooks.
# scipy.cluster.hierarchy is from Week 13.2 (hierarchical clustering).
# =============================================================================
import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# --- Sklearn unsupervised ML tools (all covered in class) --------------------
from sklearn.preprocessing import StandardScaler           # Week 12.2 & 13.1
from sklearn.decomposition import PCA                      # Week 12.2
from sklearn.cluster import KMeans, AgglomerativeClustering  # Week 13.1 & 13.2
from sklearn.datasets import load_breast_cancer, load_iris  # Week 12.2 & 13.1
from sklearn.metrics import silhouette_score, accuracy_score  # Week 13.1 & 13.2

# --- Scipy hierarchical clustering tools (Week 13.2) -------------------------
from scipy.cluster.hierarchy import linkage, dendrogram


# =============================================================================
# SECTION 2: PAGE CONFIGURATION
# -----------------------------------------------------------------------------
# Intention: Set global Streamlit settings before any content is rendered.
# This must be the FIRST Streamlit call in the script — if it comes after
# any st.write() or st.markdown(), Streamlit will throw an error.
# =============================================================================
st.set_page_config(
    page_title="Unsupervised ML Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# SECTION 3: CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
# Intention: Match the dark, polished look of the supervised ML Explorer app
# so both projects share a consistent visual identity in the portfolio.
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

/* Card style used for displaying metric values (Silhouette Score, k, etc.) */
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
# =============================================================================
st.markdown("""
<div class="hero">
  <h1>Unsupervised ML <span class="accent">Explorer</span></h1>
  <p>Upload data · pick a method · tune hyperparameters · uncover hidden structure with K-Means, Hierarchical Clustering, or PCA.</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 5: SIDEBAR — USER CONTROLS
# -----------------------------------------------------------------------------
# Intention: Put all user-facing controls (dataset, method, hyperparameters,
# run button) in the left sidebar panel.
# The sidebar is organized into 4 numbered steps to guide the user in order.
# =============================================================================
with st.sidebar:
    st.markdown("## Configuration")

    # -------------------------------------------------------------------------
    # STEP 1 — Dataset Selection
    # -------------------------------------------------------------------------
    # Intention: Let the user choose between uploading their own CSV file or
    # using one of two built-in datasets from sklearn (the same ones used in
    # the Week 12.2 PCA and Week 13.1 KMeans notebooks).
    # df_raw starts as None and only gets a value once data is actually loaded.
    # true_labels stores ground-truth labels when a sample dataset is used,
    # so we can compare clusters against actual classes (Week 13.1 step).
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">1 · Dataset</div>', unsafe_allow_html=True)

    # Radio button: user picks one of three data sources
    data_source = st.radio(
        "Source",
        ["Upload CSV", "Sample: Breast Cancer", "Sample: Iris"],
        label_visibility="collapsed"
    )

    # df_raw holds the loaded DataFrame; starts as None until data is selected
    df_raw = None
    true_labels = None       # holds ground-truth labels when available
    target_names = None      # holds class name strings for plot legends

    if data_source == "Upload CSV":
        # Show a file uploader widget that only accepts .csv files
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded:
            # pd.read_csv() loads the file into a pandas DataFrame
            df_raw = pd.read_csv(uploaded)
            st.success(f"Loaded {df_raw.shape[0]:,} rows x {df_raw.shape[1]} cols")

    elif data_source == "Sample: Breast Cancer":
        # ---------------------------------------------------------------------
        # BUILT-IN SAMPLE: Breast Cancer Wisconsin
        # ---------------------------------------------------------------------
        # Intention: Load the same dataset used in the Week 12.2 PCA notebook
        # and the Week 13.1 KMeans notebook so the app can reproduce class
        # results. @st.cache_data tells Streamlit to save the result of this
        # function so it doesn't reload on every rerun.
        # ---------------------------------------------------------------------
        @st.cache_data
        def load_bc():
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            return df, data.target, data.target_names
        df_raw, true_labels, target_names = load_bc()
        st.info("Breast Cancer Wisconsin — 569 samples, 30 numeric features.")
        with st.expander("Preview data"):
            st.dataframe(df_raw.head(10), use_container_width=True)

    else:  # Sample: Iris
        # ---------------------------------------------------------------------
        # BUILT-IN SAMPLE: Iris (classic clustering dataset)
        # ---------------------------------------------------------------------
        @st.cache_data
        def load_ir():
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            return df, data.target, data.target_names
        df_raw, true_labels, target_names = load_ir()
        st.info("Iris — 150 samples, 4 numeric features, 3 species.")
        with st.expander("Preview data"):
            st.dataframe(df_raw.head(10), use_container_width=True)

    # -------------------------------------------------------------------------
    # STEP 2 — Method Selection
    # -------------------------------------------------------------------------
    # Intention: Let the user pick which unsupervised method to run.
    # The chosen method drives which set of hyperparameter widgets shows up.
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">2 · Method</div>', unsafe_allow_html=True)

    method = st.selectbox("Algorithm", [
        "K-Means Clustering",          # Week 13.1
        "Hierarchical Clustering",     # Week 13.2
        "Principal Component Analysis" # Week 12.2
    ])

    # -------------------------------------------------------------------------
    # STEP 3 — Hyperparameter Controls
    # -------------------------------------------------------------------------
    # Intention: Show different sliders depending on which method was selected.
    # Each slider maps directly to a keyword argument passed to the sklearn
    # (or scipy) class. All defaults match what was used in the class notebooks.
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">3 · Hyperparameters</div>', unsafe_allow_html=True)

    if method == "K-Means Clustering":
        # n_clusters (k) — how many clusters the algorithm should find
        # Week 13.1 used k=2 for the breast cancer dataset (2 known classes)
        n_clusters = st.slider("Number of clusters (k)", 2, 10, 2,
                               help="Week 13.1 used k=2 for Breast Cancer (2 classes).")
        # random_state — fixes the centroid initialization for reproducibility
        # Week 13.1 used random_state=42
        random_seed = st.number_input("Random seed", value=42, step=1,
                                      help="Week 13.1 used random_state=42.")
        # n_init — how many times KMeans is run with different centroid seeds;
        # the best result (lowest inertia) is kept. Default in sklearn is 10.
        n_init = st.slider("n_init (restarts)", 1, 20, 10,
                           help="How many random starts; best inertia wins.")

    elif method == "Hierarchical Clustering":
        # n_clusters — where to "cut" the dendrogram to form clusters
        # Week 13.2 used k=4 for the Democracy dataset
        n_clusters = st.slider("Number of clusters (k)", 2, 10, 4,
                               help="Week 13.2 used k=4 for the Democracy dataset.")
        # linkage — how distances between clusters are measured during merging
        # Week 13.2 used 'ward' (minimizes within-cluster variance increase)
        linkage_method = st.selectbox("Linkage method",
                                      ["ward", "complete", "average", "single"],
                                      help="Week 13.2 used ward linkage.")
        # truncate the dendrogram if there are too many points to label
        max_dendro_labels = st.slider("Max samples to label on dendrogram",
                                      10, 100, 30,
                                      help="Larger values can make labels overlap.")

    elif method == "Principal Component Analysis":
        # n_components — how many principal components to compute
        # Week 12.2 reduced to 2 components for visualization
        # max is bounded by the smaller of (n_samples, n_features); we cap
        # the slider safely after the data is loaded later.
        n_components = st.slider("Number of components", 2, 10, 2,
                                 help="Week 12.2 used n_components=2 for visualization.")

    # -------------------------------------------------------------------------
    # RUN BUTTON
    # -------------------------------------------------------------------------
    # Intention: Nothing should happen until the user explicitly clicks this.
    # run_btn is True only in the frame where the user clicks it.
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">4 · Run</div>', unsafe_allow_html=True)
    run_btn = st.button("Run Analysis", use_container_width=True, type="primary")


# =============================================================================
# SECTION 6: SAFETY GUARD — Stop if no data is loaded
# -----------------------------------------------------------------------------
# Intention: Prevent the rest of the app from running when df_raw is still None.
# If no dataset has been selected yet, calling methods like .head() or .shape
# on it would crash the app. st.stop() halts the script immediately.
# =============================================================================
if df_raw is None:
    st.markdown("""
    <div class="info-box">
      Select a data source in the sidebar to get started.
      You can upload your own CSV or try the built-in Breast Cancer or Iris samples.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# =============================================================================
# SECTION 7: DATASET PREVIEW
# -----------------------------------------------------------------------------
# Intention: Show the user what their data looks like before they run anything.
# Displaying a preview helps them understand the structure of the data and
# pick the right feature columns.
# =============================================================================
st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)

col_prev, col_info = st.columns([3, 1])

with col_prev:
    # Show the first 20 rows as a scrollable interactive table
    st.dataframe(df_raw.head(20), use_container_width=True, height=220)

with col_info:
    # Show three quick summary stats about the dataset
    st.metric("Rows", f"{df_raw.shape[0]:,}")
    st.metric("Columns", df_raw.shape[1])
    st.metric("Missing values", f"{df_raw.isnull().sum().sum():,}")


# =============================================================================
# SECTION 8: FEATURE SELECTION
# -----------------------------------------------------------------------------
# Intention: Let the user pick which numeric columns to feed into the model.
# Note that unsupervised methods don't need a target column — that's the whole
# point of unsupervised learning (Week 12.2 introduction). We default to
# selecting all numeric columns to mirror the class notebooks.
# =============================================================================
st.markdown('<div class="section-header">Feature Selection</div>', unsafe_allow_html=True)

# Limit feature options to numeric columns — clustering and PCA both use
# Euclidean distance / variance, so categorical columns don't apply directly.
numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Need at least 2 numeric columns for clustering or PCA. "
             "Please upload a different dataset.")
    st.stop()

feature_cols = st.multiselect(
    "Numeric feature columns to use",
    numeric_cols,
    default=numeric_cols  # default: use everything numeric
)

if len(feature_cols) < 2:
    st.warning("Please select at least 2 feature columns.")
    st.stop()


# =============================================================================
# SECTION 9: PREPROCESSING FUNCTION
# -----------------------------------------------------------------------------
# Intention: Clean and standardize the data in a single reusable function.
# All three methods (KMeans, hierarchical, PCA) require StandardScaler because
# they all rely on Euclidean distances or variance, and unscaled features
# would let large-magnitude columns dominate the result. This is exactly what
# Week 12.2, 13.1, and 13.2 emphasize.
# @st.cache_data caches the output for speed on repeated runs.
# =============================================================================
@st.cache_data(show_spinner=False)
def preprocess(df, features):
    """
    Prepares data for unsupervised learning following the class notebook steps:
        1. Subset the chosen feature columns.
        2. Drop any rows with missing values.
        3. Standardize using StandardScaler (zero mean, unit variance).

    Parameters:
        df       — the full raw DataFrame
        features — list of numeric column names to use

    Returns:
        X_scaled  — numpy array of standardized features
        kept_idx  — index of rows that survived dropna (for matching labels)
    """
    df2 = df[features].dropna()
    kept_idx = df2.index

    # Standardize: same StandardScaler() pattern from Weeks 12.2 / 13.1 / 13.2
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df2)
    return X_scaled, kept_idx


# =============================================================================
# SECTION 10: ANALYSIS — runs only when the Run button is clicked
# -----------------------------------------------------------------------------
# Intention: Run the selected unsupervised method on the preprocessed data.
# Wrapped in "if run_btn:" so nothing here executes until the user clicks the
# Run Analysis button. The method dispatches to one of three branches.
# =============================================================================
if run_btn:

    with st.spinner("Preprocessing and running analysis..."):
        try:
            # Standardize the chosen features (same pipeline for all 3 methods)
            X_scaled, kept_idx = preprocess(df_raw, feature_cols)

            # If we have ground-truth labels (sample dataset), trim them to
            # match any rows dropped by dropna() — keeps everything aligned.
            if true_labels is not None:
                y_true = np.asarray(true_labels)[kept_idx.values] \
                    if hasattr(kept_idx, "values") else np.asarray(true_labels)
            else:
                y_true = None

        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()

    # Use a dark background for all matplotlib plots to match the app theme
    plt.style.use("dark_background")


    # =========================================================================
    # SECTION 10A: K-MEANS CLUSTERING (Week 13.1)
    # -------------------------------------------------------------------------
    # Steps from the notebook:
    #   1. Fit KMeans with the user's k.
    #   2. Visualize clusters in 2D using PCA (PCA only for plotting).
    #   3. Compute silhouette score on standardized features.
    #   4. Show elbow method (inertia) and silhouette curve over k=2..10.
    #   5. If ground-truth labels exist, show accuracy comparison.
    # =========================================================================
    if method == "K-Means Clustering":

        st.markdown('<div class="section-header">K-Means Results</div>',
                    unsafe_allow_html=True)

        # ---- Fit KMeans (Week 13.1 pattern) ---------------------------------
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=int(random_seed),
                        n_init=n_init)
        clusters = kmeans.fit_predict(X_scaled)

        # ---- Compute headline metrics ---------------------------------------
        # Silhouette score: how well-separated the clusters are (Week 13.1)
        sil = silhouette_score(X_scaled, clusters) if n_clusters > 1 else float("nan")
        # Inertia: sum of squared distances from each point to its centroid
        inertia = kmeans.inertia_

        # ---- Metric cards ---------------------------------------------------
        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in [
            (c1, f"{n_clusters}",        "Clusters (k)"),
            (c2, f"{sil:.3f}",           "Silhouette"),
            (c3, f"{inertia:.1f}",       "Inertia (WCSS)"),
            (c4, f"{len(X_scaled):,}",   "Samples"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="val">{val}</div>
                  <div class="lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("")

        # ---- Tabs of visualizations -----------------------------------------
        tab1, tab2, tab3, tab4 = st.tabs([
            "Cluster Plot (PCA 2D)",
            "Elbow Method",
            "Silhouette Curve",
            "Cluster Sizes",
        ])

        # ---- Tab 1: 2D PCA scatter colored by cluster (Week 13.1 step 3a) ---
        with tab1:
            st.caption("Week 13.1 — PCA reduces the data to 2D *only* for "
                       "visualization. Clusters were fit on the full scaled data.")
            pca_vis = PCA(n_components=2)
            X_pca = pca_vis.fit_transform(X_scaled)

            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor("#161b22")
            ax.set_facecolor("#161b22")
            for cl in np.unique(clusters):
                idx = clusters == cl
                ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
                           alpha=0.7, edgecolor="k", s=60,
                           label=f"Cluster {cl}")
            ax.set_xlabel("Principal Component 1", color="#8b949e")
            ax.set_ylabel("Principal Component 2", color="#8b949e")
            ax.set_title("K-Means Clusters (PCA 2D Projection)",
                         color="#e6edf3")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            ax.tick_params(colors="#8b949e")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

            # If we have ground-truth labels (sample data), show comparison
            if y_true is not None:
                st.markdown("**Comparison with true labels** (Week 13.1 step 4)")
                # Note from notebook: KMeans labels are arbitrary, so we report
                # both direct accuracy and complement and take whichever fits.
                acc_direct = accuracy_score(y_true, clusters)
                # try the "flipped" labeling for binary cases
                if n_clusters == 2:
                    acc_flip = accuracy_score(y_true, 1 - clusters)
                    acc = max(acc_direct, acc_flip)
                else:
                    acc = acc_direct
                st.metric("Cluster vs. True Label Accuracy",
                          f"{acc * 100:.2f}%")

        # ---- Tab 2: Elbow plot (Week 13.1 step 5) ---------------------------
        with tab2:
            st.caption("Week 13.1 — sweep k from 2 to 10 and plot inertia "
                       "(WCSS). The 'elbow' suggests a good k.")
            ks = range(2, 11)
            wcss = []
            for k_try in ks:
                km = KMeans(n_clusters=k_try,
                            random_state=int(random_seed),
                            n_init=n_init)
                km.fit(X_scaled)
                wcss.append(km.inertia_)

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            fig2.patch.set_facecolor("#161b22")
            ax2.set_facecolor("#161b22")
            ax2.plot(list(ks), wcss, marker="o", color="#58a6ff")
            ax2.axvline(n_clusters, color="#c5a829", linestyle="--",
                        alpha=0.7, label=f"current k = {n_clusters}")
            ax2.set_xlabel("Number of clusters (k)", color="#8b949e")
            ax2.set_ylabel("Within-Cluster Sum of Squares (WCSS)",
                           color="#8b949e")
            ax2.set_title("Elbow Method", color="#e6edf3")
            ax2.tick_params(colors="#8b949e")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)

        # ---- Tab 3: Silhouette curve (Week 13.1 step 5) ---------------------
        with tab3:
            st.caption("Week 13.1 — sweep k from 2 to 10 and plot silhouette "
                       "score. Higher = better-defined clusters.")
            ks = range(2, 11)
            sil_scores = []
            for k_try in ks:
                km = KMeans(n_clusters=k_try,
                            random_state=int(random_seed),
                            n_init=n_init)
                lbl = km.fit_predict(X_scaled)
                sil_scores.append(silhouette_score(X_scaled, lbl))

            fig3, ax3 = plt.subplots(figsize=(8, 5))
            fig3.patch.set_facecolor("#161b22")
            ax3.set_facecolor("#161b22")
            ax3.plot(list(ks), sil_scores, marker="o", color="#3fb950")
            ax3.axvline(n_clusters, color="#c5a829", linestyle="--",
                        alpha=0.7, label=f"current k = {n_clusters}")
            ax3.set_xlabel("Number of clusters (k)", color="#8b949e")
            ax3.set_ylabel("Silhouette Score", color="#8b949e")
            ax3.set_title("Silhouette Score by k", color="#e6edf3")
            ax3.tick_params(colors="#8b949e")
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)

            best_k = list(ks)[int(np.argmax(sil_scores))]
            st.info(f"Best k by silhouette: **{best_k}** "
                    f"(score = {max(sil_scores):.3f})")

        # ---- Tab 4: Cluster size table --------------------------------------
        with tab4:
            st.caption("Counts of points assigned to each cluster.")
            sizes = pd.Series(clusters).value_counts().sort_index()
            sizes.index.name = "Cluster"
            sizes.name = "Count"
            st.dataframe(sizes.to_frame(), use_container_width=True)


    # =========================================================================
    # SECTION 10B: HIERARCHICAL CLUSTERING (Week 13.2)
    # -------------------------------------------------------------------------
    # Steps from the notebook:
    #   1. Compute the linkage matrix Z with the chosen method.
    #   2. Plot the dendrogram (truncated if too many samples).
    #   3. Cut the tree with AgglomerativeClustering(n_clusters=k).
    #   4. Visualize clusters in 2D using PCA.
    #   5. Show silhouette curve to support choosing k.
    # =========================================================================
    elif method == "Hierarchical Clustering":

        st.markdown('<div class="section-header">Hierarchical Results</div>',
                    unsafe_allow_html=True)

        # ---- Compute linkage matrix Z (Week 13.2 part 4) --------------------
        Z = linkage(X_scaled, method=linkage_method)

        # ---- Cut the tree at k clusters (Week 13.2 part 5) ------------------
        agg = AgglomerativeClustering(n_clusters=n_clusters,
                                      linkage=linkage_method)
        cluster_labels = agg.fit_predict(X_scaled)

        # ---- Headline metrics -----------------------------------------------
        sil = silhouette_score(X_scaled, cluster_labels) if n_clusters > 1 else float("nan")

        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in [
            (c1, f"{n_clusters}",         "Clusters (k)"),
            (c2, f"{linkage_method}",     "Linkage"),
            (c3, f"{sil:.3f}",            "Silhouette"),
            (c4, f"{len(X_scaled):,}",    "Samples"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="val">{val}</div>
                  <div class="lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("")

        # ---- Tabs of visualizations -----------------------------------------
        tab1, tab2, tab3, tab4 = st.tabs([
            "Dendrogram",
            "Cluster Plot (PCA 2D)",
            "Silhouette Curve",
            "Cluster Sizes",
        ])

        # ---- Tab 1: Dendrogram (Week 13.2 part 4) ---------------------------
        with tab1:
            st.caption("Week 13.2 — the dendrogram shows merge order. "
                       "Cutting the tree horizontally at any height defines "
                       "the clusters.")
            fig, ax = plt.subplots(figsize=(14, 6))
            fig.patch.set_facecolor("#161b22")
            ax.set_facecolor("#161b22")

            n = len(X_scaled)
            # If there are too many samples to label cleanly, use truncate_mode
            # to collapse the bottom of the tree (also covered in scipy docs)
            if n > max_dendro_labels:
                dendrogram(Z, truncate_mode="lastp",
                           p=max_dendro_labels,
                           leaf_rotation=90, leaf_font_size=8,
                           ax=ax, color_threshold=Z[-(n_clusters - 1), 2])
                ax.set_xlabel(f"Sample index "
                              f"(truncated to {max_dendro_labels} clusters)",
                              color="#8b949e")
            else:
                dendrogram(Z, leaf_rotation=90, leaf_font_size=8,
                           ax=ax,
                           color_threshold=Z[-(n_clusters - 1), 2] if n_clusters > 1 else None)
                ax.set_xlabel("Sample index", color="#8b949e")
            ax.set_ylabel("Distance", color="#8b949e")
            ax.set_title(f"Hierarchical Dendrogram ({linkage_method} linkage)",
                         color="#e6edf3")
            ax.tick_params(colors="#8b949e")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        # ---- Tab 2: 2D PCA scatter colored by cluster (Week 13.2 part 6) ----
        with tab2:
            st.caption("Week 13.2 — PCA is used *only* for visualization. "
                       "Clusters were fit on the full scaled feature space.")
            pca_vis = PCA(n_components=2)
            X_pca = pca_vis.fit_transform(X_scaled)

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            fig2.patch.set_facecolor("#161b22")
            ax2.set_facecolor("#161b22")
            scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1],
                                  c=cluster_labels, cmap="viridis",
                                  s=60, edgecolor="k", alpha=0.7)
            ax2.set_xlabel("Principal Component 1", color="#8b949e")
            ax2.set_ylabel("Principal Component 2", color="#8b949e")
            ax2.set_title(f"Agglomerative Clustering "
                          f"({linkage_method} linkage, k={n_clusters})",
                          color="#e6edf3")
            legend1 = ax2.legend(*scatter.legend_elements(), title="Cluster")
            ax2.add_artist(legend1)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(colors="#8b949e")
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)

        # ---- Tab 3: Silhouette curve (Week 13.2 part 8) ---------------------
        with tab3:
            st.caption("Week 13.2 — silhouette score across k=2..10 helps "
                       "pick a good cut height.")
            k_range = range(2, 11)
            sil_scores = []
            for k_try in k_range:
                lbl = AgglomerativeClustering(n_clusters=k_try,
                                              linkage=linkage_method).fit_predict(X_scaled)
                sil_scores.append(silhouette_score(X_scaled, lbl))

            fig3, ax3 = plt.subplots(figsize=(8, 5))
            fig3.patch.set_facecolor("#161b22")
            ax3.set_facecolor("#161b22")
            ax3.plot(list(k_range), sil_scores, marker="o", color="#3fb950")
            ax3.axvline(n_clusters, color="#c5a829", linestyle="--",
                        alpha=0.7, label=f"current k = {n_clusters}")
            ax3.set_xlabel("Number of clusters (k)", color="#8b949e")
            ax3.set_ylabel("Silhouette Score", color="#8b949e")
            ax3.set_title("Silhouette Analysis (Agglomerative)", color="#e6edf3")
            ax3.tick_params(colors="#8b949e")
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)

            best_k = list(k_range)[int(np.argmax(sil_scores))]
            st.info(f"Best k by silhouette: **{best_k}** "
                    f"(score = {max(sil_scores):.3f})")

        # ---- Tab 4: Cluster sizes table -------------------------------------
        with tab4:
            st.caption("Counts of points assigned to each cluster.")
            sizes = pd.Series(cluster_labels).value_counts().sort_index()
            sizes.index.name = "Cluster"
            sizes.name = "Count"
            st.dataframe(sizes.to_frame(), use_container_width=True)


    # =========================================================================
    # SECTION 10C: PRINCIPAL COMPONENT ANALYSIS (Week 12.2)
    # -------------------------------------------------------------------------
    # Steps from the notebook:
    #   1. Fit PCA with the user's n_components on standardized data.
    #   2. Show explained variance ratio + cumulative variance.
    #   3. 2D scatter of PC1 vs PC2.
    #   4. Loadings bar chart for PC1 / PC2.
    #   5. Scree plot (cumulative variance) and per-component bar chart.
    # =========================================================================
    elif method == "Principal Component Analysis":

        st.markdown('<div class="section-header">PCA Results</div>',
                    unsafe_allow_html=True)

        # Make sure n_components doesn't exceed min(n_samples, n_features)
        n_feats = X_scaled.shape[1]
        n_samp  = X_scaled.shape[0]
        n_comp_safe = min(n_components, n_feats, n_samp)
        if n_comp_safe < n_components:
            st.info(f"Adjusted n_components down to {n_comp_safe} "
                    f"(can't exceed min(n_samples, n_features)).")

        # ---- Fit PCA (Week 12.2 step 2) -------------------------------------
        pca = PCA(n_components=n_comp_safe)
        X_pca = pca.fit_transform(X_scaled)
        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        # ---- Headline metrics -----------------------------------------------
        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in [
            (c1, f"{n_comp_safe}",          "Components"),
            (c2, f"{explained[0]*100:.1f}%", "PC1 Variance"),
            (c3, f"{cumulative[-1]*100:.1f}%", "Total Variance"),
            (c4, f"{n_feats}",              "Original Features"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="val">{val}</div>
                  <div class="lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("")

        # ---- Tabs of visualizations -----------------------------------------
        tab1, tab2, tab3, tab4 = st.tabs([
            "PCA 2D Scatter",
            "Scree / Variance",
            "Loadings",
            "Component Table",
        ])

        # ---- Tab 1: 2D scatter (Week 12.2 step 3a) --------------------------
        with tab1:
            st.caption("Week 12.2 — projection of the data onto the first "
                       "two principal components.")
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor("#161b22")
            ax.set_facecolor("#161b22")
            if y_true is not None and target_names is not None:
                # Color by ground-truth class if available (sample dataset)
                colors = ["navy", "darkorange", "forestgreen",
                          "crimson", "purple"]
                for i, name in enumerate(target_names):
                    mask = y_true == i
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                               color=colors[i % len(colors)],
                               alpha=0.7, edgecolor="k", s=60,
                               label=str(name))
                ax.legend(loc="best")
            else:
                # No labels available — plot all points in one color
                ax.scatter(X_pca[:, 0], X_pca[:, 1],
                           alpha=0.7, edgecolor="k", s=60,
                           color="#58a6ff")
            ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)", color="#8b949e")
            ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)", color="#8b949e")
            ax.set_title("PCA: 2D Projection", color="#e6edf3")
            ax.grid(True, alpha=0.3)
            ax.tick_params(colors="#8b949e")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        # ---- Tab 2: Scree / variance (Week 12.2 step 3c) --------------------
        with tab2:
            st.caption("Week 12.2 — bars show variance per component, line "
                       "shows cumulative. Look for the 'elbow'.")
            fig2, ax_b = plt.subplots(figsize=(9, 5))
            fig2.patch.set_facecolor("#161b22")
            ax_b.set_facecolor("#161b22")
            comps = np.arange(1, len(explained) + 1)
            ax_b.bar(comps, explained * 100, color="steelblue", alpha=0.85,
                     label="Individual %")
            ax_b.set_xlabel("Principal Component", color="#8b949e")
            ax_b.set_ylabel("Variance Explained (%)", color="steelblue")
            ax_b.set_xticks(comps)
            ax_b.set_xticklabels([f"PC{i}" for i in comps], color="#8b949e")
            ax_b.tick_params(axis="y", labelcolor="steelblue")

            # Twin axis with cumulative line (Week 12.2 step 3d)
            ax_l = ax_b.twinx()
            ax_l.plot(comps, cumulative * 100, color="crimson", marker="o",
                      label="Cumulative %")
            ax_l.set_ylabel("Cumulative Variance (%)", color="crimson")
            ax_l.set_ylim(0, 105)
            ax_l.tick_params(axis="y", labelcolor="crimson")

            ax_b.set_title("PCA: Variance Explained", color="#e6edf3")

            # Combine legends from both axes
            lines1, labels1 = ax_b.get_legend_handles_labels()
            lines2, labels2 = ax_l.get_legend_handles_labels()
            ax_b.legend(lines1 + lines2, labels1 + labels2, loc="center right")
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)

        # ---- Tab 3: Loadings bar chart (Week 12.2 step 3b) ------------------
        with tab3:
            st.caption("Week 12.2 — each principal component is a weighted "
                       "combination of the original features. Loadings show "
                       "the weights.")
            # Build the loadings DataFrame: rows = PCs, cols = features
            loadings_df = pd.DataFrame(
                pca.components_,
                columns=feature_cols,
                index=[f"PC{i+1}" for i in range(n_comp_safe)]
            )

            # Plot PC1 and PC2 as a horizontal grouped bar chart
            features = loadings_df.columns.tolist()
            y_pos = np.arange(len(features))
            bar_height = 0.35

            fig3, ax3 = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))
            fig3.patch.set_facecolor("#161b22")
            ax3.set_facecolor("#161b22")
            ax3.barh(y_pos + bar_height/2, loadings_df.loc["PC1"], bar_height,
                     label="PC1", color="#1b6ec2", edgecolor="none")
            if n_comp_safe >= 2:
                ax3.barh(y_pos - bar_height/2, loadings_df.loc["PC2"], bar_height,
                         label="PC2", color="#c5a829", edgecolor="none")
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(features, color="#8b949e",
                                fontsize=max(7, 11 - len(features)//5))
            ax3.set_xlabel("Loading Weight", color="#8b949e")
            ax3.set_title("PCA Loadings (PC1 / PC2)", color="#e6edf3",
                          loc="left", fontweight="bold")
            ax3.axvline(0, color="grey", linewidth=0.8)
            ax3.legend(loc="upper right")
            ax3.invert_yaxis()
            ax3.grid(axis="x", alpha=0.3)
            ax3.tick_params(axis="x", colors="#8b949e")
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)

        # ---- Tab 4: Component table -----------------------------------------
        with tab4:
            st.caption("Full table of explained variance per component.")
            comp_df = pd.DataFrame({
                "Component": [f"PC{i+1}" for i in range(n_comp_safe)],
                "Variance Explained (%)": (explained * 100).round(2),
                "Cumulative (%)":         (cumulative * 100).round(2),
            })
            st.dataframe(comp_df, use_container_width=True)


    # =========================================================================
    # SECTION 11: RUN SUMMARY
    # -------------------------------------------------------------------------
    # Intention: After the analysis runs, show a concise recap of the exact
    # settings that were used so the user can reproduce or reference it.
    # =========================================================================
    st.markdown('<div class="section-header">Run Summary</div>',
                unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"""
        <div class="info-box">
          <b>Method:</b> {method}<br>
          <b>Samples used:</b> {len(X_scaled):,}<br>
          <b>Features used:</b> {len(feature_cols)}
        </div>""", unsafe_allow_html=True)
    with s2:
        if method == "K-Means Clustering":
            extra = (f"<b>k:</b> {n_clusters} &nbsp;|&nbsp; "
                     f"<b>random_state:</b> {int(random_seed)}<br>"
                     f"<b>n_init:</b> {n_init}")
        elif method == "Hierarchical Clustering":
            extra = (f"<b>k:</b> {n_clusters} &nbsp;|&nbsp; "
                     f"<b>linkage:</b> {linkage_method}")
        else:  # PCA
            extra = (f"<b>n_components:</b> {n_components}<br>"
                     f"<b>Total variance explained:</b> "
                     f"{cumulative[-1]*100:.1f}%")
        st.markdown(f"""
        <div class="info-box">
          {extra}<br>
          <b>Preprocessing:</b> dropna → StandardScaler
        </div>""", unsafe_allow_html=True)


# =============================================================================
# SECTION 12: PRE-RUN PROMPT
# -----------------------------------------------------------------------------
# Intention: If the dataset is loaded but the user hasn't clicked Run yet,
# show a short prompt reminding them to click the button.
# =============================================================================
else:
    if df_raw is not None:
        st.markdown("""
        <div class="info-box">
          Dataset loaded. Pick a method and tune hyperparameters in the
          sidebar, then click <b>Run Analysis</b> to begin.
        </div>""", unsafe_allow_html=True)


# =============================================================================
# SECTION 13: FOOTER
# -----------------------------------------------------------------------------
# Intention: Display a small credit line at the very bottom of the page.
# Notes which class week each method comes from for transparency.
# =============================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#8b949e;font-size:0.8rem;'>"
    "Unsupervised ML Explorer · Streamlit + scikit-learn + scipy · "
    "PCA (Wk 12.2) · K-Means (Wk 13.1) · Hierarchical (Wk 13.2)</p>",
    unsafe_allow_html=True
)