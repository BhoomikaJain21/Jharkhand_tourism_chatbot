# app.py (Final Corrected Version for Deployment)

import streamlit as st
import pickle
import json
import nltk
import random
import numpy as np
# Using the stable translator library
from google_trans_new import google_translator
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- GLOBAL SETUP FOR PICKLE LOADING ---
# The lemmatizer must be initialized globally for the tokenizer function below.
lemmatizer = WordNetLemmatizer()

# CRITICAL FIX: This function MUST be defined globally and MUST match the name 
# and logic used during the model training process to satisfy the vectorizer loading.
def get_lemmas_for_training(text):
    # This logic (split + lemmatize) must match what was saved in chatbot_vectorizer.pkl
    words = text.split()
    return [lemmatizer.lemmatize(w.lower()) for w in words]

# --- 1. SETUP: Load Model and Functions (Streamlit Cache) ---
@st.cache_resource
def load_chatbot_components():
    """Loads all necessary components from saved files and ensures NLTK data is available."""
    try:
        # Ensure necessary NLTK data is available for lemmatization
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

        # Load the saved model and vectorizer
        with open('chatbot_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('chatbot_vectorizer.pkl', 'rb') as f:
            # The loading is now successful because get_lemmas_for_training is defined
            vectorizer = pickle.load(f)

        # Load intents data
        with open('chatbot_intents.json', 'r', encoding='utf-8') as f:
            intents_data = json.load(f)

        # Initialize translator
        translator = google_translator() 

        return model, vectorizer, intents_data, lemmatizer, translator

    except FileNotFoundError as e:
        st.error(f"Error loading required files: {e}. Ensure all three files are in the directory.")
        raise
    except Exception as e:
        st.error(f"An error occurred during component loading: {e}")
        raise

try:
    model, vectorizer, intents_data, lemmatizer, translator = load_chatbot_components()
except:
    # The stop message is defined in the loading function, so we just stop here.
    st.stop()

# --- 2. CHATBOT LOGIC FUNCTIONS ---

def get_response(intent_tag):
    for intent in intents_data['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    for intent in intents_data['intents']:
        if intent['tag'] == 'fallback':
              return random.choice(intent['responses'])
    return "I am unable to process your request at the moment."

# The translator functions are using the google_trans_new logic
def is_hindi(text):
    return any('\u0900' <= char <= '\u097F' for char in text)

def translate_to_english(text):
    if not text:
        return "", 'en'
    try:
        translation = translator.translate(text, lang_tgt='en')
        # Simplified source language detection for deployment stability
        detected_src = 'auto' 
        if is_hindi(text):
            return translation, 'hi'
        return translation, detected_src
    except Exception:
        return text, 'en'

def translate_response(text, dest_lang):
    if dest_lang == 'en':
        return text
    try:
        translation = translator.translate(text, lang_tgt=dest_lang)
        return translation
    except Exception:
        return text

def classify_intent(sentence):
    # CRITICAL FIX 3: Pass the raw sentence directly to the vectorizer.
    # The vectorizer's internal tokenizer (get_lemmas_for_training) will handle splitting/lemmatization.
    
    # First, ensure the sentence is lowercased before vectorizing
    sentence_lower = sentence.lower()
    
    X_test = vectorizer.transform([sentence_lower])
    if X_test.nnz == 0:
        return 'fallback'

    prediction = model.predict(X_test)[0]
    probabilities = model.predict_proba(X_test)[0]
    max_proba = np.max(probabilities)

    # Use a reasonable confidence threshold
    if max_proba >= 0.50:
          return prediction

    # Fallback logic should operate on the already-lemmatized/tokenized form if possible,
    # but here we use the raw lowercased input for robust keyword checks since the main model failed.
    if 'ranchi' in sentence_lower or 'waterfall' in sentence_lower or 'रांची' in sentence_lower:
        return 'about_ranchi'
    elif 'jamshedpur' in sentence_lower or 'steel city' in sentence_lower or 'जमशेदपुर' in sentence_lower:
        return 'about_jamshedpur'
    elif 'betla' in sentence_lower or 'safari' in sentence_lower or 'wildlife' in sentence_lower:
        return 'about_betla'
    elif 'deoghar' in sentence_lower or 'baba dham' in sentence_lower or 'jyotirlinga' in sentence_lower:
        return 'about_deoghar'
    elif 'jain' in sentence_lower or 'parasnath' in sentence_lower or 'shikharji' in sentence_lower or 'historic' in sentence_lower or 'culture' in sentence_lower:
        return 'about_parasnath'
    elif 'itinerary' in sentence_lower or 'plan' in sentence_lower or 'suggest' in sentence_lower or 'trip' in sentence_lower:
        return 'itinerary_suggestion'
    elif 'food' in sentence_lower or 'cuisine' in sentence_lower or 'litti' in sentence_lower:
        return 'local_cuisine'
    elif 'transport' in sentence_lower or 'travel' in sentence_lower or 'airport' in sentence_lower:
        return 'transport'
    elif 'hello' in sentence_lower or 'hi' in sentence_lower or 'namaste' in sentence_lower or 'greetings' in sentence_lower:
        return 'greeting'
    elif 'thank' in sentence_lower or 'thanks' in sentence_lower or 'cheers' in sentence_lower:
        return 'thanks'
    elif 'what is famous' in sentence_lower or 'tell me about jharkhand' in sentence_lower or 'why visit jharkhand' in sentence_lower:
        return 'about_state'

    return 'fallback'

# --- 3. STREAMLIT UI CODE (Unchanged) ---

st.set_page_config(page_title="Jharkhand Tourism Chatbot 🤖", layout="centered")
st.title("🤖 Jharkhand Tourism Chatbot")
st.subheader("Your Multilingual Guide to the Land of Forests! 🌳")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! Welcome to Jharkhand Tourism. How can I assist you?"})


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about Ranchi, Deoghar, or local culture..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Process Chatbot Response ---
    with st.spinner('Thinking...'):
        eng_input, source_lang = translate_to_english(prompt)
        if eng_input:
            intent_tag = classify_intent(eng_input)
            english_response = get_response(intent_tag)
            final_response = translate_response(english_response, source_lang)
        else:
            final_response = "Please type a message so I can assist you!"

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(final_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_response})
