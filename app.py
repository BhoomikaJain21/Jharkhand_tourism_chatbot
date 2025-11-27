import streamlit as st
import pickle
import json
import nltk
import numpy as np
import re
from google_trans_new import google_translator
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

# --- Configuration ---
VECTORIZER_FILE = 'lr_vectorizer.pkl'
MODEL_FILE = 'lr_model.pkl'
LE_FILE = 'lr_label_encoder.pkl'
DATA_FILE = 'lr_intent_answers.json'
# FIX: Lowered threshold from 0.70 to 0.40. This allows the model to predict 
# the correct intent without being overly strict on confidence, solving the
# "I am not sure" problem for valid queries.
CONFIDENCE_THRESHOLD = 0.40 

# --- Model Loading and Setup ---
@st.cache_resource
def load_resources():
    """Loads all necessary components and caches them."""
    
    # 0. Ensure NLTK resources are downloaded
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("NLTK downloads successful.")
    except Exception as e:
        print(f"NLTK download failed: {e}")
        st.error("Failed to download necessary NLP resources.")
        st.stop()
    
    try:
        # 1. Load Intent Answers
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            intent_answers = json.load(f)

        # 2. Load Vectorizer
        with open(VECTORIZER_FILE, 'rb') as file:
            vectorizer = pickle.load(file)

        # 3. Load Model
        with open(MODEL_FILE, 'rb') as file:
            model = pickle.load(file)
            
        # 4. Load Label Encoder
        with open(LE_FILE, 'rb') as file:
            le = pickle.load(file)
        
        # 5. Initialize Translator and Lemmatizer
        translator = google_translator()
        lemmatizer = WordNetLemmatizer()
        
        return vectorizer, model, le, intent_answers, translator, lemmatizer

    except FileNotFoundError:
        st.error("Model files not found. Please run `train_lr_classifier.py` and commit the model files to your GitHub repository.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model resources: {e}")
        st.stop()

# --- Utility Functions ---

def translate_to_english(text):
    """Detects source language and translates to English."""
    translator = st.session_state.translator
    try:
        translation = translator.translate(text, lang_tgt='en')
        detected_src = translator.detect(text)[0]
        if isinstance(translation, list):
            translation = translation[0]
        return translation, detected_src or 'en'
    except Exception:
        return text, 'en'

def translate_response(text, dest_lang):
    """Translates the English response back to the user's language."""
    translator = st.session_state.translator
    if dest_lang == 'en':
        return text
    try:
        translation = translator.translate(text, lang_src='en', lang_tgt=dest_lang)
        if isinstance(translation, list):
            translation = translation[0]
        return translation
    except Exception:
        return text

def preprocess_text_for_inference(text):
    """Preprocesses a new text query for vectorization (lemmatize, lowercase)."""
    lemmatizer = st.session_state.lemmatizer
    text = re.sub(r'[?]', '', text.lower()) 
    words = nltk.tokenize.wordpunct_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)


def get_lr_response(user_input_en, vectorizer, model, le, intent_answers, threshold):
    """
    Predicts the Intent using Logistic Regression and returns the corresponding answer.
    """
    # 1. Preprocess and Vectorize User Input
    lemmatized_input = preprocess_text_for_inference(user_input_en)
    
    if not lemmatized_input.strip():
        return "Please ask a specific question about Jharkhand tourism. 🤔"
        
    user_vec = vectorizer.transform([lemmatized_input])
    
    # 2. Predict Intent and Confidence
    probabilities = model.predict_proba(user_vec)[0]
    best_prob_index = np.argmax(probabilities)
    best_score = probabilities[best_prob_index]
    predicted_intent = le.classes_[best_prob_index]
    
    # 3. Determine Response based on Confidence
    if best_score >= threshold:
        english_response = intent_answers.get(predicted_intent, "Error: Could not find answer for that intent.")
    else:
        english_response = "I am not sure I understand that query. I can only answer questions related to Jharkhand tourism topics like waterfalls, wildlife, or logistics. Could you please rephrase?"
        
    return english_response

# --- Streamlit App ---
st.title("🗺️ Multilingual Jharkhand Tourism Chatbot (Logistic Regression)")
st.markdown("Ask your question in English, Hindi, or any other major language! 🚀")

# Load resources
vectorizer, model, le, intent_answers, translator, lemmatizer = load_resources()

# Store in session state for easy access
st.session_state.vectorizer = vectorizer
st.session_state.model = model
st.session_state.le = le
st.session_state.intent_answers = intent_answers
st.session_state.translator = translator
st.session_state.lemmatizer = lemmatizer

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Hello! Welcome to Jharkhand Tourism. How can I assist you? You can ask me questions in any language. 🌳"
    })

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about Jharkhand tourism..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Chatbot Logic ---
    
    # 1. Translate User Input to English
    eng_input, source_lang = translate_to_english(prompt)
    
    # 2. Get English Response using Logistic Regression Intent Classification
    english_response = get_lr_response(
        eng_input,
        st.session_state.vectorizer,
        st.session_state.model,
        st.session_state.le,
        st.session_state.intent_answers,
        CONFIDENCE_THRESHOLD
    )
    
    # 3. Translate Response back to User's Language
    final_response = translate_response(english_response, source_lang)

    # 4. Display and Save Response
    with st.chat_message("assistant"):
        st.markdown(final_response)

    st.session_state.messages.append({"role": "assistant", "content": final_response})
