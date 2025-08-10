import streamlit as st
import joblib
import pandas as pd

# Page settings
st.set_page_config(page_title="📰 Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("Classify news articles as **Real** or **Fake** using NLP & Machine Learning.")

# Load trained model and vectorizer
@st.cache_resource
def load_artifacts():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
except Exception as e:
    st.error(f"Error loading model/vectorizer: {e}")
    st.stop()

# User input
user_input = st.text_area("✏️ Enter news article text below:", height=200, placeholder="Paste the full news text here...")

# Prediction function
def predict_news(text):
    vect = vectorizer.transform([text])
    prediction = model.predict(vect)
    proba = model.predict_proba(vect)[0]
    return prediction[0], proba

# Button to check news
if st.button("🔍 Check News"):
    if user_input.strip():
        label, proba = predict_news(user_input)

        col1, col2 = st.columns([2, 1])
        with col1:
            if label == 1:
                st.success(f"✅ **Real News**\n\nConfidence: {proba[1]*100:.2f}%")
            else:
                st.error(f"🚨 **Fake News**\n\nConfidence: {proba[0]*100:.2f}%")
        with col2:
            st.metric(label="Real %", value=f"{proba[1]*100:.1f}%")
            st.metric(label="Fake %", value=f"{proba[0]*100:.1f}%")

        # Extra: show probability bar chart
        chart_df = pd.DataFrame({
            "Label": ["Fake", "Real"],
            "Confidence": [proba[0]*100, proba[1]*100]
        })
        st.bar_chart(chart_df.set_index("Label"))

    else:
        st.warning("⚠️ Please enter some text to analyze.")

st.markdown("---")
st.caption("Model trained with NLP text processing and scikit-learn. This is a demo, not a certified fact-checking tool.")
