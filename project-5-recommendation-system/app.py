import streamlit as st
import pandas as pd

st.title("🎯 Recommendation System")

# Dummy dataset for demo
items = ['Item A', 'Item B', 'Item C', 'Item D', 'Item E']
user_pref = st.multiselect("Select your favorite items:", items)

if st.button("Get Recommendations"):
    if user_pref:
        # Simple collaborative filtering placeholder
        recommendations = [item for item in items if item not in user_pref]
        st.write("Recommended for you:")
        for rec in recommendations:
            st.write(f"- {rec}")
    else:
        st.error("Please select at least one item.")
