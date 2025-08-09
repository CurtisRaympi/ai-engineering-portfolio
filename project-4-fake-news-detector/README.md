# Fake News Detector

A text classification model that detects whether news articles are **real** or **fake** using NLP techniques.

## Features
- Preprocessing with TF-IDF vectorization
- Trains on public fake news datasets
- Supports batch or single-article predictions

## Tech Stack
- Python
- scikit-learn
- NLTK / spaCy
- Pandas / NumPy

## Structure
# Fake News Detector

A machine learning project focused on classifying news articles as genuine or fake using text classification techniques.  
The model uses TF-IDF vectorization to transform raw text into numeric features, then trains a Logistic Regression classifier.

**Key Features:**
- Text preprocessing and vectorization with TF-IDF
- Supervised learning with Logistic Regression
- Dataset splitting for training and testing with accuracy evaluation
- Easily extendable to other classifiers or NLP techniques

**Technical Highlights:**
- Uses Pandas for data handling
- Employs Scikit-learn for ML model creation and evaluation
- Demonstrates practical application of NLP in misinformation detection

**Usage Instructions:**
- Prepare a labeled CSV dataset (`news.csv`) with ‘text’ and ‘label’ columns
- Run `main.py` to train and evaluate the model
- Explore adding more preprocessing or experimenting with other ML models
## How to Run
```bash
git clone https://github.com/CurtisRaympi/ai-engineering-portfolio.git
cd ai-engineering-portfolio/project-4-fake-news-detector/
pip install -r requirements.txt
python detect.py
