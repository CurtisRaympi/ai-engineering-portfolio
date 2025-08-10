import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="📰 Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.markdown("""
Detect whether news articles are **Real** or **Fake** using a trained NLP model with TF-IDF vectorization
and Logistic Regression.
---
""")

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Tabs for single or batch prediction
tab1, tab2 = st.tabs(["📝 Single Article", "📂 Batch Upload"])

# --- Single Article ---
with tab1:
    user_input = st.text_area("Enter the news article text:", height=200, placeholder="Paste your article here...")
    
    if st.button("Analyze Article"):
        if user_input.strip():
            vect_text = vectorizer.transform([user_input])
            prediction = model.predict(vect_text)[0]
            prob = model.predict_proba(vect_text)[0]
            
            label = "Fake News" if prediction == 1 else "Real News"
            color = "red" if prediction == 1 else "green"
            st.markdown(f"**Prediction:** <span style='color:{color}; font-size:20px'>{label}</span>", unsafe_allow_html=True)
            st.progress(float(max(prob)))
        else:
            st.warning("Please enter an article to analyze.")

# --- Batch Upload ---
with tab2:
    uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column:", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "text" not in df.columns:
                st.error("CSV must contain a 'text' column.")
            else:
                vect_texts = vectorizer.transform(df["text"])
                predictions = model.predict(vect_texts)
                df["Prediction"] = ["Fake News" if p == 1 else "Real News" for p in predictions]
                st.success("Predictions completed!")
                st.dataframe(df)

                # Download option
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

st.markdown("---")
st.markdown("**Project 4: Fake News Detector** | Built with Scikit-learn, TF-IDF, and Logistic Regression.")
