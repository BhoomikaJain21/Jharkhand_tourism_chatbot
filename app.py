import streamlit as st
import pickle
import json
import nltk
import random
import numpy as np
from google_trans_new import google_translator
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- 1. SETUP: Load Model and Functions (Streamlit Cache) ---
@st.cache_resource
def load_chatbot_components():
    """Loads all necessary components from saved files."""
    try:
        # Download NLTK resources needed for lemmatization (FIXED NLTK DOWNLOAD)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        # Load the saved model and vectorizer
        with open('chatbot_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('chatbot_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

        # Load intents data (responses)
        with open('chatbot_intents.json', 'r', encoding='utf-8') as f:
            intents_data = json.load(f)

        # Initialize tools
        lemmatizer = WordNetLemmatizer()
        translator = google_translator()

        return model, vectorizer, intents_data, lemmatizer, translator

    except FileNotFoundError as e:
        st.error(f"Error loading required files: {e.filename}. Please ensure 'train_chatbot.py' was run successfully.")
        raise
    except Exception as e:
        st.error(f"An unexpected error occurred during component loading: {e}")
        raise

try:
    model, vectorizer, intents_data, lemmatizer, translator = load_chatbot_components()
except:
    st.stop() # Stop the Streamlit run if the loading fails

# --- 2. CHATBOT LOGIC FUNCTIONS ---

def get_response(intent_tag):
    """Retrieves a random response based on the intent tag."""
    for intent in intents_data['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    
    for intent in intents_data['intents']:
        if intent['tag'] == 'fallback':
            return random.choice(intent['responses'])
    return "I am unable to process your request at the moment."

def is_hindi(text):
    """Checks if the text contains Hindi characters for source language determination."""
    return any('\u0900' <= char <= '\u097F' for char in text)

def translate_to_english(text):
    """Translates user input to English and returns the detected language code."""
    if not text:
        return "", 'en'
    try:
        translation = translator.translate(text, lang_tgt='en')
        detected_src = translator.detect(text)
        
        if is_hindi(text) or detected_src == 'hi':
            return translation, 'hi'
        
        return translation, detected_src or 'en'
    except Exception:
        return text, 'en'

def translate_response(text, dest_lang):
    """Translates the English response back to the user's language."""
    if dest_lang == 'en':
        return text
    try:
        translation = translator.translate(text, lang_src='en', lang_tgt=dest_lang)
        return translation
    except Exception:
        return text

def classify_intent(sentence):
    """Classifies the English sentence to an intent tag."""
    sentence_words = sentence.split()
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    sentence_str = " ".join(sentence_words)

    X_test = vectorizer.transform([sentence_str])
    
    if X_test.nnz == 0:
        return 'fallback'

    prediction = model.predict(X_test)[0]
    probabilities = model.predict_proba(X_test)[0]
    max_proba = np.max(probabilities)

    if max_proba >= 0.50:
         return prediction
    
    if 'ranchi' in sentence_str or 'waterfall' in sentence_str:
        return 'about_ranchi'
    elif 'jamshedpur' in sentence_str or 'steel' in sentence_str:
        return 'about_jamshedpur'
    
    return 'fallback'

# --- 3. STREAMLIT UI CODE ---
st.set_page_config(page_title="Jharkhand Tourism Chatbot 🤖", layout="centered")
st.title("🤖 Jharkhand Tourism Chatbot")
st.subheader("Your Multilingual Guide to the Land of Forests! 🌳")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! Welcome to Jharkhand Tourism. How can I assist you? Ask me about Ranchi, Betla, or local food!"})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner('Thinking...'):
        eng_input, source_lang = translate_to_english(prompt)
        intent_tag = classify_intent(eng_input)
        english_response = get_response(intent_tag)
        final_response = translate_response(english_response, source_lang)

    with st.chat_message("assistant"):
        st.markdown(final_response)

    st.session_state.messages.append({"role": "assistant", "content": final_response})
