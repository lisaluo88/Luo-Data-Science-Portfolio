import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("penguins.csv")

# 2. Sidebar Navigation
st.sidebar.title("Table of Contents")
different_pages = st.sidebar.radio(
    "Select a page:",
    ["Exploring Our Dataset", "Body Variation", "Key Bodily Correlations"]
)

# 3. Global Filter (Sidebar or Main)
st.sidebar.divider()
st.sidebar.subheader("Global Filters")
species_list = data["species"].dropna().unique()
selected_species = st.sidebar.selectbox("Select a species:", species_list)

# Filter the data globally
filtered_data = data[data["species"] == selected_species]

# 4. Page Routing
if different_pages == "Exploring Our Dataset":
    st.title("Penguins Insights Dashboard 🐧")
    st.write("This dashboard explores physical differences among Adelie, Chinstrap, and Gentoo penguins.")
    
    st.subheader("Dataset Preview")
    st.dataframe(filtered_data) # Showing filtered data
    
    st.write("Column Names in Dataset:")
    st.write(list(data.columns))

elif different_pages == "Body Variation":
    st.title("Body Variation")
    st.markdown(f"Examine physical traits for **{selected_species}** penguins.")

    plt.style.use("dark_background")

    # Chart 1: Average Body Mass
    st.subheader(f"Average Body Mass: {selected_species}")
    avg_mass = filtered_data["body_mass_g"].mean()
    
    # We use a simple metric for single species or a bar if comparing
    st.metric("Average Mass (g)", f"{avg_mass:.2f}")

    # Chart 2: Male vs Female comparison for the selected species
    st.subheader("Male vs. Female Body Mass")
    gender_data = filtered_data.dropna(subset=["sex"])
    
    if not gender_data.empty:
        gender_mass = gender_data.groupby("sex")["body_mass_g"].mean()

        fig, ax = plt.subplots()
        ax.bar(gender_mass.index, gender_mass.values, color=['#1f77b4', '#ff7f0e'])
        ax.set_xlabel("Sex")
        ax.set_ylabel("Average Body Mass (g)")
        ax.set_title(f"Mass Comparison for {selected_species}")
        st.pyplot(fig)
        st.caption(f"Comparison of average mass between male and female {selected_species} penguins.")
    else:
        st.warning("No gender data available for this selection.")

elif different_pages == "Key Bodily Correlations":
    st.title("Key Bodily Correlations")
    st.write("Future analysis for correlations goes here!")