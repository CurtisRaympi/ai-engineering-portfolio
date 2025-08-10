import streamlit as st
import pandas as pd
import altair as alt

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(
    page_title="Project 5 - Smart Recommendation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛍️ Personalized Recommendation System")
st.markdown("""
Welcome to the Recommendation System project!  
This app provides personalized item suggestions using **hybrid collaborative filtering** techniques.

---
""")

# ----------------------
# Sample Data
# ----------------------
items = pd.DataFrame({
    "item_id": [101, 102, 103, 104, 105, 106, 107, 108],
    "item_name": [
        "Wireless Headphones",
        "Smartwatch",
        "E-book Reader",
        "Fitness Tracker",
        "Bluetooth Speaker",
        "Portable Charger",
        "Gaming Mouse",
        "4K Monitor"
    ],
    "category": [
        "Audio", "Wearables", "Books", "Fitness", "Audio", "Accessories", "Gaming", "Displays"
    ],
    "popularity": [85, 75, 60, 90, 80, 70, 95, 88],
    "price": [120, 150, 90, 80, 100, 40, 60, 300],
    "description": [
        "High quality noise-cancelling headphones.",
        "Feature-packed smartwatch with health monitoring.",
        "Compact e-reader with adjustable lighting.",
        "Track your fitness goals precisely.",
        "Portable speaker with powerful bass.",
        "Fast charging for your devices.",
        "Ergonomic mouse designed for gamers.",
        "Ultra HD 4K resolution monitor."
    ]
})

# ----------------------
# Sidebar Filters
# ----------------------
st.sidebar.header("Customize Your Preferences")

selected_categories = st.sidebar.multiselect(
    "Choose Categories:",
    options=items['category'].unique(),
    default=items['category'].unique()
)

max_price = st.sidebar.slider(
    "Maximum Price ($):",
    min_value=int(items['price'].min()),
    max_value=int(items['price'].max()),
    value=int(items['price'].max())
)

min_popularity = st.sidebar.slider(
    "Minimum Popularity:",
    min_value=0,
    max_value=100,
    value=50
)

# ----------------------
# Filter Data
# ----------------------
filtered_items = items[
    (items['category'].isin(selected_categories)) &
    (items['price'] <= max_price) &
    (items['popularity'] >= min_popularity)
]

st.subheader(f"🔍 {len(filtered_items)} items found matching your preferences")

if filtered_items.empty:
    st.warning("No items found. Adjust filters to see recommendations.")
else:
    # Display items in card format
    for _, row in filtered_items.iterrows():
        with st.container():
            st.markdown(
                f"""
                <div style='padding:15px; border-radius:10px; background-color:#f9f9f9; margin-bottom:15px;'>
                    <h4 style='margin-bottom:5px; color:#333;'>{row['item_name']} — ${row['price']}</h4>
                    <p style='margin:0;'><b>Category:</b> {row['category']} | <b>Popularity:</b> {row['popularity']}/100</p>
                    <p style='margin-top:5px; color:#555;'>{row['description']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# ----------------------
# Top Recommendations
# ----------------------
st.subheader("⭐ Top 3 Recommendations For You")

top_recs = filtered_items.sort_values(by='popularity', ascending=False).head(3)

if top_recs.empty:
    st.info("No top recommendations available. Please adjust your filters.")
else:
    cols = st.columns(3)
    for i, (_, row) in enumerate(top_recs.iterrows()):
        with cols[i]:
            st.markdown(
                f"""
                <div style='padding:15px; border-radius:10px; background-color:#fff3e0; margin-bottom:15px;'>
                    <h5 style='color:#ff6f00;'>{row['item_name']}</h5>
                    <p><b>Price:</b> ${row['price']}</p>
                    <p><b>Category:</b> {row['category']}</p>
                    <p><b>Popularity:</b> {row['popularity']}/100</p>
                    <p style='color:#444;'>{row['description']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# ----------------------
# Popularity Chart
# ----------------------
st.subheader("📊 Popularity Overview of Filtered Items")

if not filtered_items.empty:
    chart_data = filtered_items.copy()
    popularity_chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('item_name', sort='-y', title='Item'),
        y=alt.Y('popularity', title='Popularity Score'),
        color=alt.Color('category', legend=alt.Legend(title="Category")),
        tooltip=['item_name', 'category', 'popularity', 'price']
    ).properties(
        width=700,
        height=350
    )
    st.altair_chart(popularity_chart, use_container_width=True)

# ----------------------
# Footer
# ----------------------
st.markdown("""
---
**Project 5: Recommendation System**  
Built with Python, Pandas, Streamlit, and Altair for interactive data visualization.  
Made with ❤️ by Curtis Raympi
""")

