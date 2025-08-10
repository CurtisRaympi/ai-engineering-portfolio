import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detector")
st.markdown("""
Enter news article text(s) below and let the AI determine whether they're **Real** or **Fake**.  
Model trained with **TF-IDF** + **Logistic Regression**.
---
""")

# Load Model and Vectorizer
@st.cache_resource
def load_assets():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_assets()

# Mode Selection
mode = st.radio(
    "Choose Mode:",
    ["Single Article", "Batch Mode (Multiple Articles)"],
    index=0
)

if mode == "Single Article":
    user_input = st.text_area("📝 Paste your news article text here:")

    if st.button("🔍 Check News"):
        if user_input.strip():
            vect = vectorizer.transform([user_input])
            prediction = model.predict(vect)[0]
            
            if prediction == 1:
                st.error("🚨 This article is classified as **Fake News**.")
            else:
                st.success("✅ This article is classified as **Real News**.")
        else:
            st.warning("⚠️ Please enter some text before checking.")

else:  # Batch Mode
    st.markdown("""
    **Instructions:**  
    - Paste one article per line in the box below  
    - OR upload a CSV file with a `text` column  
    """)

    text_block = st.text_area("📄 Paste multiple articles (one per line):")
    uploaded_file = st.file_uploader("Or upload CSV file", type=["csv"])

    if st.button("📊 Check All"):
        articles = []

        # From textarea
        if text_block.strip():
            articles.extend([line.strip() for line in text_block.split("\n") if line.strip()])

        # From CSV
        if uploaded_file is not None:
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                if "text" in df_uploaded.columns:
                    articles.extend(df_uploaded["text"].dropna().tolist())
                else:
                    st.error("CSV must contain a column named `text`.")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

        if articles:
            vect = vectorizer.transform(articles)
            predictions = model.predict(vect)
            results_df = pd.DataFrame({
                "Article": articles,
                "Prediction": ["Fake" if p == 1 else "Real" for p in predictions]
            })
            st.dataframe(results_df, use_container_width=True)
        else:
            st.warning("⚠️ Please provide some articles to check.")

# Footer
st.markdown("""
---
**How it works:**  
- **TF-IDF** converts text into numerical features.  
- **Logistic Regression** detects fake vs real patterns.  

Made with ❤️ by Emmanuel Jaja
""")

