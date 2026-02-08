import streamlit as st
import pandas as pd

st.title("Basic Streamlit App")

st.write("This app displays a simple dataset with interactive filtering.")

# Sample DataFrame
data = pd.DataFrame({
    "Category": ["A", "B", "C", "D"],
    "Value": [10, 20, 30, 40]
})

# Interactive filter
category = st.selectbox("Select a category:", data["Category"])

filtered_data = data[data["Category"] == category]

st.write("Filtered Data")
st.dataframe(filtered_data)
