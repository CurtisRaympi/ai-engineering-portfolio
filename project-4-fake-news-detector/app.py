import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("📰 Fake News Detector")
st.markdown("""
Detect whether a news article is **real** or **fake** using a pre-trained machine learning model.
""")

# Load model and vectorizer once, cache to speed up
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# Sidebar for additional info and testing multiple samples
st.sidebar.header("Test Multiple Samples")
upload_file = st.sidebar.file_uploader("Upload CSV file with news articles", type=['csv'])
show_metrics = st.sidebar.checkbox("Show Model Performance Metrics")

# Single input
user_input = st.text_area("Enter news article text to check:")

def predict_news(text):
    vect = vectorizer.transform([text])
    prediction = model.predict(vect)
    return prediction[0]

# Single prediction
if st.button("Check"):
    if user_input.strip():
        prediction = predict_news(user_input)
        label = "Fake News" if prediction == 1 else "Real News"

        # Colored result
        if label == "Fake News":
            st.error(f"Prediction: **{label}**")
        else:
            st.success(f"Prediction: **{label}**")

        st.markdown("---")
        st.info("**Tip:** Try entering other news texts or upload a CSV file for batch prediction.")
    else:
        st.error("Please enter some text to analyze.")

# Batch prediction from CSV
if upload_file is not None:
    try:
        df = pd.read_csv(upload_file)
        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column with news articles.")
        else:
            st.success(f"Loaded {len(df)} articles.")

            df['prediction'] = df['text'].apply(lambda x: predict_news(str(x)))
            df['label'] = df['prediction'].map({0: "Real News", 1: "Fake News"})

            st.subheader("Batch Prediction Results")
            st.dataframe(df[['text', 'label']])

            # Download results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name='fake_news_predictions.csv',
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")

# Show model performance if asked and if files exist
if show_metrics:
    try:
        y_true = joblib.load('y_true.pkl')
        y_pred = joblib.load('y_pred.pkl')

        st.subheader("Model Performance Metrics")

        report = classification_report(y_true, y_pred, target_names=['Real News', 'Fake News'], output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report)

        # Confusion matrix plot
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    except Exception as e:
        st.error("Performance metrics files not found or error loading them.")

# Footer
st.markdown("---")
st.markdown("© 2025 Your Name | AI Engineering Portfolio")
