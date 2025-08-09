import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Project 3 - Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("📊 Project 3: Advanced Data Explorer")
st.markdown("""
This app allows you to upload a dataset and explore it with interactive visualizations and stats.  
Customize your analysis and gain valuable insights from your data!
""")

# Sidebar for uploading file and options
st.sidebar.header("Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")
    
    # Show raw data toggle
    if st.sidebar.checkbox("Show raw data"):
        st.subheader("Raw Data")
        st.dataframe(df)

    # Basic stats
    st.subheader("Basic Statistics")
    st.write(df.describe(include='all').T)

    # Select column to analyze
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.sidebar.subheader("Select Columns for Analysis")

    selected_num_col = st.sidebar.selectbox("Select numeric column", numeric_cols)
    selected_cat_col = st.sidebar.selectbox("Select categorical column", categorical_cols)

    # Interactive Charts
    st.subheader("Visualizations")

    # Histogram for numeric column
    st.markdown(f"### Histogram of {selected_num_col}")
    bins = st.slider("Number of bins", 5, 100, 30)
    fig, ax = plt.subplots()
    sns.histplot(df[selected_num_col], bins=bins, color="skyblue", kde=True, ax=ax)
    ax.set_xlabel(selected_num_col)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Boxplot
    st.markdown(f"### Boxplot of {selected_num_col} by {selected_cat_col}")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df, x=selected_cat_col, y=selected_num_col, palette="Set2", ax=ax2)
    ax2.set_xlabel(selected_cat_col)
    ax2.set_ylabel(selected_num_col)
    st.pyplot(fig2)

    # Pie chart for categorical distribution
    st.markdown(f"### Distribution of {selected_cat_col}")
    cat_counts = df[selected_cat_col].value_counts()
    fig3, ax3 = plt.subplots()
    ax3.pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    ax3.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    st.pyplot(fig3)

    # Correlation heatmap
    if st.sidebar.checkbox("Show correlation heatmap"):
        st.subheader("Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)

else:
    st.warning("Please upload a CSV file to start exploring!")

# Footer
st.markdown("---")
st.markdown("© 2025 Your Name | AI Engineer Portfolio")
