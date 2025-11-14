# app.py
import streamlit as st
import joblib
import nltk
import re
import string
import pandas as pd
import plotly.express as px
from nltk.corpus import stopwords

# Load model & vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

nltk.download('stopwords')

# -------------------------
# Text cleaning function
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text


# -------------------------
# Streamlit Page Settings
# -------------------------
st.set_page_config(page_title="Emotion Detector", layout="centered")

# -------------------------
# Sidebar Section
# -------------------------
st.sidebar.title("ℹ️ Model Information")
st.sidebar.markdown(
    """
    ### 🧠 Emotion Detection Model  
    - **Type:** Logistic Regression  
    - **Vectorizer:** TF-IDF  
    - **Classes:** Happy, Sad, Anger, Fear, Surprise, Disgust  
    - **Use case:** Social media text, chats, feedback analysis  
    - **Author:** Abhirup Sen  
    ---
    ### 📊 Metrics  
    - Accuracy: **89%**  
    - Precision: **0.87**  
    - Recall: **0.86**  
    - F1 Score: **0.86**  
    ---
    ### 📁 Project  
    Real-time emotion classification from text using NLP.
    """
)

# -------------------------
# Main Header
# -------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#4A90E2;'>🎭 Real-Time Emotion Detection</h1>
    <p style='text-align:center; font-size:18px;'>Start typing — predictions update instantly.</p>
    """,
    unsafe_allow_html=True
)


# -------------------------
# Real-time Input Box
# -------------------------
user_input = st.text_area("✍️ Type your text:", height=150)


# -------------------------
# Auto-Prediction Logic
# -------------------------
if user_input.strip():

    cleaned = clean_text(user_input)
    transformed = vectorizer.transform([cleaned])

    probs = model.predict_proba(transformed)[0]
    prediction = model.classes_[probs.argmax()]
    confidence = probs.max()

    # Emoji mapping
    emoji_map = {
        "happy": "😊",
        "sad": "😢",
        "anger": "😡",
        "fear": "😨",
        "surprise": "😲",
        "disgust": "🤢",
        "neutral": "😐"
    }

    # Clean text preview
    st.markdown("### 🧼 Cleaned Text")
    st.code(cleaned, language="text")

    # Prediction Box
    st.markdown(
        f"""
        <div style="padding:20px; border-radius:12px; background:#F0F8FF; margin-top:20px;">
            <h2 style="text-align:center;">{emoji_map.get(prediction, '')} Predicted Emotion:
                <span style="color:#d9534f;">{prediction.upper()}</span>
            </h2>
            <p style="text-align:center; font-size:18px;">Confidence: {confidence:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Probability graph
    st.markdown("### 📊 Emotion Probability Distribution")
    df = pd.DataFrame({
        "Emotion": model.classes_,
        "Probability": probs
    })

    fig = px.bar(df, x="Emotion", y="Probability", text="Probability")
    fig.update_traces(texttemplate='%{text:.2f}', textposition="outside")
    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Start typing above to see real-time emotion analysis.")
