# app.py
import streamlit as st
import joblib
import nltk
import re
import string
from nltk.corpus import stopwords

# Load model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

nltk.download('stopwords')

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Streamlit UI
st.title("🎭 Emotion Detection from Text")
st.write("Type a sentence below and I’ll predict the emotion!")

user_input = st.text_area("Enter your text here:")

if st.button("Predict Emotion"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        transformed = vectorizer.transform([cleaned])

        # Get probabilities for each emotion
        probs = model.predict_proba(transformed)[0]
        prediction = model.classes_[probs.argmax()]
        confidence = probs.max()

        # Neutral threshold
        st.write("🔍 Emotion probabilities:", dict(zip(model.classes_, probs.round(2))))
        if confidence < 0.3:  
            st.info("Predicted Emotion: **NEUTRAL 😐** (low confidence)")
        else:
            st.success(f"Predicted Emotion: **{prediction.upper()}** (Confidence: {confidence:.2f})")
    else:
        st.warning("Please enter some text to analyze.")

