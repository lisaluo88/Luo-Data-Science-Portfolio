# 🏅 Tidy Data Project — 2008 Olympics Medalists

> A data-cleaning and exploratory analysis project that applies **Tidy Data Principles** to reshape a messy 2008 Beijing Olympics medalists dataset into a clean, analysis-ready format.

---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Visual Output Preview](#-visual-output-preview)
3. [Dataset](#-dataset)
4. [Tidy Data Principles Applied](#-tidy-data-principles-applied)
5. [Pre-processing](#-pre-processing)
6. [Analysis & Visualizations](#-analysis--visualizations)
7. [Project Structure](#-project-structure)
8. [Setup & Installation](#-setup--installation)
9. [How to Use the Notebook](#-how-to-use-the-notebook)
10. [Code Examples](#-code-examples)
11. [Key Insights](#-key-insights)
12. [Future Improvements](#-future-improvements)
13. [References](#-references)

---

## 🏅 Overview
**Tidy Data Project — 2008 Olympics Medalists** transforms a wide, messy dataset where variables like *Gender* and *Sport* are trapped inside column headers (e.g. `male_archery`, `female_swimming`) into a long, tidy DataFrame ready for analysis. After reshaping, the project explores the cleaned data through pivot tables and three visualizations.

**Project Goals:**
- Apply **Hadley Wickham's Tidy Data Principles** to a real-world messy dataset
- Practice reshaping data using `pd.melt()`, `str.split()`, and `pivot_table()`
- Produce clear, well-labeled visualizations of medal distributions
- Translate raw, hard-to-read data into a structure that supports easy analysis
- Strengthen skills in data cleaning, feature creation, and storytelling with charts

---

## 🎬 Visual Output Preview

Running the notebook produces three main visualizations:

**1. Top 10 Sports by Total Medals Awarded**
A horizontal bar chart (magma palette) ranking sports by total medals — athletics, rowing, and swimming dominate the top of the list.

**2. Total Medals Awarded by Gender**
A count plot (Set2 palette) comparing the volume of male vs. female medalists across the Games.

**3. Gender Distribution Across Sports**
A grouped bar chart showing the male/female split sport-by-sport, highlighting which events were most balanced and which were single-gender.

A sorted **pivot table** of gold/silver/bronze counts per sport is also printed in the notebook.

---

## 📊 Dataset
The project uses the **2008 Beijing Olympics Medalists** dataset — a record of every athlete who medaled at the 2008 Summer Olympics.

- **File used:** `olympics_08_medalists.csv`
- **Format (original):** Wide — one row per athlete, one column per `gender_sport` combination
- **Format (after cleaning):** Long/tidy — one row per medal awarded
- **Total medalists:** ~2,000 athletes across ~40 sports

| Column (after tidying) | Type | Description |
|---|---|---|
| `Medalist` | string | Athlete's name |
| `Gender` | string | `male` or `female` |
| `Sport` | string | Sport in which the medal was won (e.g. `archery`, `athletics`) |
| `Medal` | string | `gold`, `silver`, or `bronze` |

---

## 🧠 Tidy Data Principles Applied
This project follows the three rules laid out in [Hadley Wickham's Tidy Data paper](https://vita.had.co.nz/papers/tidy-data.pdf):

1. **Each variable forms its own column** — `Medalist`, `Gender`, `Sport`, and `Medal` each live in dedicated columns rather than being encoded in column names.
2. **Each observation forms its own row** — every row represents exactly one medal won by one athlete.
3. **Each type of observational unit forms its own table** — the medal-event observations are stored separately from any aggregations (which live in pivot tables).

---

## 🧹 Pre-processing
The cleaning pipeline transforms the raw CSV into a tidy DataFrame in five steps:

1. **Load** the CSV with `pandas.read_csv()`
2. **Melt** the wide DataFrame using `pd.melt()` with `medalist_name` as the ID variable, collapsing all `gender_sport` columns into two columns: `Gender_Sport` and `Medal`
3. **Drop missing medals** with `dropna()` — most athletes did not win in most events, leaving behind a sparse matrix of `NaN`s that need to be removed
4. **Split** the combined `Gender_Sport` field on the underscore using `str.split("_")` to separate gender from sport
5. **Standardize sport names** by replacing remaining underscores with spaces (e.g. `field_hockey` → `field hockey`)
6. **Rename and reorder** columns into the final tidy schema: `Medalist`, `Gender`, `Sport`, `Medal`

---

## 📈 Analysis & Visualizations

### Pivot Table — Medal Counts by Sport
A pivot table aggregates the tidy data into a sport × medal-color matrix, including a `Total` column sorted in descending order. The top 10 sports by medal volume are displayed in the notebook.

### Visualization 1 — Top 10 Sports by Total Medals
- **Type:** Horizontal bar chart
- **Library:** Seaborn (`sns.barplot`)
- **Palette:** `magma`
- **Purpose:** Show which sports awarded the most medals overall

### Visualization 2 — Total Medals Awarded by Gender
- **Type:** Count plot
- **Library:** Seaborn (`sns.countplot`)
- **Palette:** `Set2`
- **Purpose:** Compare the overall volume of male vs. female medalists

### Visualization 3 — Gender Distribution Across Sports
- **Type:** Grouped bar chart
- **Library:** Pandas plotting (`gender_sport.plot(kind="bar")`)
- **Purpose:** Reveal which sports are gender-balanced and which are gender-restricted

---

## 📁 Project Structure
```
olympics-tidy-data/
├── Olympics_Data_Cleaning.ipynb    # Main analysis notebook
├── olympics_08_medalists.csv       # Raw dataset
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation (this file)
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Jupyter Notebook **or** VS Code with the Jupyter extension

### Step 1 — Clone or download the project
```bash
git clone https://github.com/your-username/olympics-tidy-data.git
cd olympics-tidy-data
```

### Step 2 — (Recommended) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies
```bash
pip install pandas matplotlib seaborn jupyter
```

Or, if a `requirements.txt` is present:
```bash
pip install -r requirements.txt
```

### Step 4 — Confirm the dataset is in place
Make sure `olympics_08_medalists.csv` lives in the same folder as the notebook.

### Step 5 — Launch the notebook
```bash
jupyter notebook Olympics_Data_Cleaning.ipynb
```

Or open the file directly in VS Code and run the cells.

---

## 🖱️ How to Use the Notebook

### Example 1 — Reproducing the full analysis
1. Open `Olympics_Data_Cleaning.ipynb`.
2. Run all cells in order from top to bottom (`Cell → Run All` in Jupyter, or `Run All` in VS Code).
3. Inspect the printed `df_tidy.head()` output to confirm the data is in long format.
4. Scroll down to view the pivot table and three visualizations.

### Example 2 — Looking up a specific athlete
After running the cleaning cells, query the tidy DataFrame:
```python
df_tidy[df_tidy["Medalist"].str.contains("Bolt", na=False)]
```

### Example 3 — Exploring a single sport
```python
df_tidy[df_tidy["Sport"] == "swimming"]["Medal"].value_counts()
```

### Example 4 — Exporting the cleaned data
```python
df_tidy.to_csv("olympics_08_tidy.csv", index=False)
```

---

## 💻 Code Examples

### Loading and melting the dataset
```python
import pandas as pd

df = pd.read_csv("olympics_08_medalists.csv")

df_melted = pd.melt(
    df,
    id_vars=["medalist_name"],
    var_name="Gender_Sport",
    value_name="Medal"
)
df_melted = df_melted.dropna()
```

### Splitting the combined Gender_Sport column
```python
df_melted["Gender"] = df_melted["Gender_Sport"].str.split("_").str[0]
df_melted["Sport"]  = df_melted["Gender_Sport"].str.split("_").str[1]
df_melted["Sport"]  = df_melted["Sport"].str.replace("_", " ")

df_tidy = df_melted.rename(columns={"medalist_name": "Medalist"})
df_tidy = df_tidy[["Medalist", "Gender", "Sport", "Medal"]]
```

### Building the medal-count pivot table
```python
pivot_medals = pd.pivot_table(
    df_tidy,
    values="Medalist",
    index="Sport",
    columns="Medal",
    aggfunc="count",
    fill_value=0
)
pivot_medals["Total"] = pivot_medals.sum(axis=1)
pivot_medals = pivot_medals.sort_values(by="Total", ascending=False)
```

### Plotting the top 10 sports
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
top_sports = df_tidy["Sport"].value_counts().head(10)

sns.barplot(
    x=top_sports.values,
    y=top_sports.index,
    hue=top_sports.index,
    palette="magma",
    legend=False
)
plt.title("Top 10 Sports by Total Medals Awarded (2008)")
plt.xlabel("Number of Medalists")
plt.ylabel("Sport")
plt.show()
```

---

## 🔎 Key Insights
- **Athletics, rowing, and swimming dominate medal counts** — they award medals across many sub-events, so they appear far above other sports
- **Team sports cluster in the middle of the leaderboard** — football, field hockey, handball, water polo, and volleyball each award ~70–110 medals because every team member receives one
- **Gender representation varies widely by sport** — some sports show near-perfect balance, while others are single-gender by event design
- **Tidy structure unlocks easy aggregation** — once the data is long-format, pivots, group-bys, and filters become one-liners
- **Most of the value of "data analysis" is actually data cleaning** — the analysis steps were short; the reshaping was where the real work happened

---

## 🚀 Future Improvements
- **Join with country data** to analyze medal distributions by nation
- **Add an interactive Streamlit version** with sport and gender filters
- **Compare across Olympic years** (2008 vs. 2012 vs. 2016) to track changes over time
- **Layer in event-level granularity** (e.g. 100m vs. 200m within athletics)
- **Add a Plotly version** of each chart for hover-based exploration
- **Run statistical tests** on gender balance trends across sport categories

---

## 📚 References
- [Tidy Data Paper — Hadley Wickham](https://vita.had.co.nz/papers/tidy-data.pdf)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Pandas Cheat Sheet (PDF)](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [`pd.melt()` Reference](https://pandas.pydata.org/docs/reference/api/pandas.melt.html)
- [`pd.pivot_table()` Reference](https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/stable/index.html)
- [Data to Viz](https://www.data-to-viz.com/)

---

<p align="center">🏅 Built with Pandas · Seaborn · Matplotlib 🏅</p>
