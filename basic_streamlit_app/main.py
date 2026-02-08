import streamlit as st
import pandas as pd

st.title("Basic Streamlit App")

st.write("This app displays a dataset loaded from a CSV file with interactive filtering.")

# Load data from CSV (REQUIRED)
data = pd.read_csv("basic_streamlit_app/data/your_data.csv")

st.subheader("Dataset Preview")
st.dataframe(data)

# Interactive filter
category = st.selectbox("Select a category:", data["Category"].unique())

filtered_data = data[data["Category"] == category]

st.write("Filtered Data")
st.dataframe(filtered_data)
