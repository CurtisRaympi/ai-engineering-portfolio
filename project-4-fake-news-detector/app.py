import streamlit as st
import joblib

st.title("📰 Fake News Detector")

# Load your trained model & vectorizer (make sure these files exist)
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

user_input = st.text_area("Enter news article text:")

if st.button("Check"):
    if user_input.strip():
        vect = vectorizer.transform([user_input])
        prediction = model.predict(vect)
        label = "Fake News" if prediction[0] == 1 else "Real News"
        st.write(f"Prediction: **{label}**")
    else:
        st.error("Please enter some text to analyze.")
