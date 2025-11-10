# app.py (This is the file that runs on Streamlit Cloud)

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

# --- CRITICAL FIX 1: DEFINE THE TOKENIZER USED IN TRAINING ---
# This function MUST exist in app.py with the same name as was saved in the .pkl file.
# It must also match the simple logic (split()) we adopted to avoid the NLTK 'punkt' error.
lemmatizer = WordNetLemmatizer()
def get_lemmas_for_training(text):
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
            # The vectorizer loading will now successfully find get_lemmas_for_training
            vectorizer = pickle.load(f)

        # Load intents data
        with open('chatbot_intents.json', 'r', encoding='utf-8') as f:
            intents_data = json.load(f)

        # Initialize tools
        # We don't need to initialize lemmatizer again as it's global for get_lemmas_for_training
        translator = google_translator() 

        # Note: We return lemmatizer just for consistency in the original functions
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

def is_hindi(text):
    return any('\u0900' <= char <= '\u097F' for char in text)

def translate_to_english(text):
    if not text:
        return "", 'en'
    try:
        # google_trans_new returns the translated text directly
        translation = translator.translate(text, lang_tgt='en')
        
        # We use a placeholder for detected source language
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
    # CRITICAL FIX 2: Use the same logic for classification as in the training tokenizer
    # The split() logic is now defined by the get_lemmas_for_training function used in the model.
    # Although the saved vectorizer uses get_lemmas_for_training for its internal tokenizer,
    # we still need to process the input sentence explicitly here for vectorization.
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

    # Robust Keyword Fallback... (rest of your logic)
    if 'ranchi' in sentence_str or 'raanchi' in sentence_str or 'waterfall' in sentence_str or 'रांची' in sentence_str:
        return 'about_ranchi'
    elif 'jamshedpur' in sentence_str or 'steel city' in sentence_str or 'जमशेदपुर' in sentence_str:
        return 'about_jamshedpur'
    elif 'betla' in sentence_str or 'safari' in sentence_str:
        return 'about_betla'
    elif 'deoghar' in sentence_str or 'baba dham' in sentence_str or 'jyotirlinga' in sentence_str:
        return 'about_deoghar'
    elif 'jain' in sentence_str or 'parasnath' in sentence_str or 'shikharji' in sentence_str or 'historic' in sentence_str or 'culture' in sentence_str:
        return 'about_parasnath'
    elif 'itinerary' in sentence_str or 'plan' in sentence_str or 'suggest' in sentence_str or 'trip' in sentence_str:
        return 'itinerary_suggestion'
    elif 'food' in sentence_str or 'cuisine' in sentence_str or 'litti' in sentence_str:
        return 'local_cuisine'
    elif 'transport' in sentence_str or 'travel' in sentence_str or 'airport' in sentence_str:
        return 'transport'
    elif 'hello' in sentence_str or 'hi' in sentence_str or 'namaste' in sentence_str:
        return 'greeting'

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
