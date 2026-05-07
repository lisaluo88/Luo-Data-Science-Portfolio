# 🐧 Penguins Insights Dashboard

> An interactive Streamlit dashboard for exploring physical traits across **Adelie**, **Chinstrap**, and **Gentoo** penguin species from the Palmer Archipelago.

---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Live Demo & Screenshots](#-live-demo--screenshots)
3. [Dataset](#-dataset)
4. [Features](#-features)
5. [Pre-processing](#-pre-processing)
6. [Project Structure](#-project-structure)
7. [Setup & Installation](#-setup--installation)
8. [How to Use the Dashboard](#-how-to-use-the-dashboard)
9. [Code Examples](#-code-examples)
10. [Key Insights](#-key-insights)
11. [Future Improvements](#-future-improvements)
12. [References](#-references)

---

## 🐧 Overview
**Penguins Insights Dashboard** is an interactive multi-page Streamlit application that explores the physical characteristics of three penguin species — **Adelie**, **Chinstrap**, and **Gentoo** — using the Palmer Penguins dataset. The dashboard includes a global species filter and three analytical pages that uncover patterns in body mass, sex-based differences, and bodily correlations.

**Project Goals:**
- Build a polished, multi-page interactive dashboard with Streamlit
- Provide a global species filter that flows across every page
- Visualize physical differences using clean, accessible charts
- Translate raw tabular data into a user-facing analytical tool rather than a static notebook
- Strengthen skills in dashboard design, layout, and data storytelling

---

## 🎬 Live Demo & Screenshots

> 💡 *Add a Streamlit Community Cloud link here once deployed:* `https://your-app-name.streamlit.app`

**Page 1 — Exploring Our Dataset**
A welcoming landing page with a filtered preview of the data and a list of all available columns.

**Page 2 — Body Variation**
A focused species view with an average body mass metric and a male vs. female bar chart on a dark background.

**Page 3 — Key Bodily Correlations**
A planned page for scatter plots and correlation analysis between bill, flipper, and body mass measurements.

---

## 📊 Dataset
This project uses the **Palmer Penguins dataset**, a popular educational dataset created as a friendly alternative to the Iris dataset for data exploration and visualization.

- **File used:** `penguins.csv`
- **Original source:** [Palmer Penguins Dataset — Allison Horst](https://allisonhorst.github.io/palmerpenguins/)
- **Collected by:** Dr. Kristen Gorman and the Palmer Station Long Term Ecological Research (LTER) program in Antarctica
- **Rows:** 344 penguin observations
- **Species:** Adelie, Chinstrap, Gentoo
- **Islands:** Biscoe, Dream, Torgersen

| Column | Type | Description |
|---|---|---|
| `species` | string | Penguin species (Adelie, Chinstrap, Gentoo) |
| `island` | string | Island where the penguin was observed |
| `bill_length_mm` | float | Length of the bill in millimeters |
| `bill_depth_mm` | float | Depth of the bill in millimeters |
| `flipper_length_mm` | float | Flipper length in millimeters |
| `body_mass_g` | float | Body mass in grams |
| `sex` | string | Sex of the penguin (male / female) |
| `year` | int | Year of observation |

---

## ✨ Features

### 🧭 Sidebar Navigation
The sidebar serves as the control center of the dashboard:
- **Page selector** (radio buttons) — switches between the three analytical views
- **Global species filter** (selectbox) — controls which species' data flows through every page
- **Visual divider** — separates navigation from filters for cleaner UX

### 📄 Page 1 — Exploring Our Dataset
- Project title with the 🐧 brand emoji
- Short narrative describing what the dashboard explores
- **Filtered dataset preview** — interactive `st.dataframe` showing rows for the selected species
- **Column names list** — full transparency into the underlying data

### 📈 Page 2 — Body Variation
- **Dynamic page header** — pulls the current species name into the markdown
- **Average body mass metric** — uses `st.metric` to display the mean in a large, glanceable format
- **Male vs. Female bar chart** — built with Matplotlib using a dark theme, with male/female bars in distinct colors (`#1f77b4` blue, `#ff7f0e` orange)
- **Caption text** — provides context below the chart
- **Empty-state handling** — falls back to a friendly warning if no sex data exists for the filter

### 🔗 Page 3 — Key Bodily Correlations
- Placeholder for upcoming correlation analysis (bill length vs. flipper length, body mass vs. flipper length, bill depth vs. bill length)

---

## 🧹 Pre-processing
Because Streamlit re-runs the script on every interaction, the app performs all data preparation at runtime rather than relying on pre-cleaned files:

1. **Load** the CSV with `pandas.read_csv()`
2. **Extract unique species** values to populate the sidebar dropdown filter
3. **Apply the global filter** — every chart and table downstream reads from `filtered_data`
4. **Drop missing values** in the `sex` column before grouping for the male vs. female comparison
5. **Defensively check for empty groups** — display a warning rather than crashing

---

## 📁 Project Structure
```
penguins-dashboard/
├── app.py              # Main Streamlit application
├── penguins.csv        # Palmer Penguins dataset
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation (this file)
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1 — Clone or download the project
```bash
git clone https://github.com/your-username/penguins-dashboard.git
cd penguins-dashboard
```

### Step 2 — (Recommended) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies
```bash
pip install streamlit pandas matplotlib
```

Or, if a `requirements.txt` is present:
```bash
pip install -r requirements.txt
```

### Step 4 — Confirm the dataset is in place
Make sure `penguins.csv` lives in the same folder as `app.py`.

### Step 5 — Launch the dashboard
```bash
streamlit run app.py
```

Streamlit will open the dashboard automatically in your default browser at `http://localhost:8501`. Press `Ctrl + C` in the terminal to shut down the server.

---

## 🖱️ How to Use the Dashboard

### Example 1 — Comparing Adelie males and females
1. Open the app in your browser.
2. In the sidebar, leave the page selector on **Body Variation**.
3. Under **Global Filters**, select `Adelie` from the species dropdown.
4. Read the **Average Mass (g)** metric near the top.
5. Scroll down to see the male-vs-female bar chart for Adelies.

### Example 2 — Inspecting raw Gentoo records
1. In the sidebar, click **Exploring Our Dataset**.
2. Select `Gentoo` from the species dropdown.
3. Scroll the dataframe preview to inspect individual rows.
4. Use the search/sort controls in the corner of the dataframe for quick lookups.

### Example 3 — Switching species without losing your spot
The species filter is global, so switching from `Chinstrap` to `Gentoo` keeps you on whatever page you were viewing — every chart and table updates instantly.

---

## 💻 Code Examples

### Loading and filtering data
```python
import pandas as pd

data = pd.read_csv("penguins.csv")
selected_species = "Adelie"
filtered_data = data[data["species"] == selected_species]
```

### Building the sidebar navigation
```python
import streamlit as st

st.sidebar.title("Table of Contents")
different_pages = st.sidebar.radio(
    "Select a page:",
    ["Exploring Our Dataset", "Body Variation", "Key Bodily Correlations"]
)
```

### Rendering the male vs. female bar chart
```python
import matplotlib.pyplot as plt

plt.style.use("dark_background")
gender_data = filtered_data.dropna(subset=["sex"])
gender_mass = gender_data.groupby("sex")["body_mass_g"].mean()

fig, ax = plt.subplots()
ax.bar(gender_mass.index, gender_mass.values, color=["#1f77b4", "#ff7f0e"])
ax.set_xlabel("Sex")
ax.set_ylabel("Average Body Mass (g)")
ax.set_title(f"Mass Comparison for {selected_species}")
st.pyplot(fig)
```

### Defensive empty-state handling
```python
if not gender_data.empty:
    # ... render the chart ...
else:
    st.warning("No gender data available for this selection.")
```

---

## 🔎 Key Insights
- **Gentoo penguins are visibly the heaviest** of the three species, often by a significant margin
- **Males are consistently heavier than females** across all three species, though the magnitude varies
- **Species-level filtering reveals patterns aggregate views hide** — a single bar chart of "all penguins" smooths over real biological differences
- **Interactivity changes the user's role** — instead of consuming a fixed report, the user drives their own analysis
- **Defensive code matters more in dashboards than in notebooks** — empty-state handling keeps the experience smooth even with sparse data

---

## 🚀 Future Improvements
- **Complete the Key Bodily Correlations page** with scatter plots, regression lines, and a correlation matrix heatmap
- **Add more global filters** — island, sex, and year — composing with the existing species filter
- **Add a download button** so users can export filtered data as CSV
- **Add summary statistics tables** alongside charts (count, mean, std, min, max)
- **Migrate visualizations to Plotly** for hover-based exploration
- **Add caching** with `@st.cache_data` to speed up loading on larger datasets
- **Deploy to Streamlit Community Cloud** for a public URL

---

## 📚 References
- [Palmer Penguins Dataset — Allison Horst](https://allisonhorst.github.io/palmerpenguins/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit API Reference](https://docs.streamlit.io/library/api-reference)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Pandas Cheat Sheet (PDF)](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Matplotlib Documentation](https://matplotlib.org/stable/index.html)
- [Data to Viz](https://www.data-to-viz.com/)
- [Tidy Data Paper — Hadley Wickham](https://vita.had.co.nz/papers/tidy-data.pdf)

---

<p align="center">🐧 Built with Streamlit · Pandas · Matplotlib 🐧</p>
