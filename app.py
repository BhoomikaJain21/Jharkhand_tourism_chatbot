import streamlit as st
import pickle
import json
import nltk
import random
import numpy as np
import re
from google_trans_new import google_translator 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- 1. SETUP: Load Model and Functions (Streamlit Cache) ---
@st.cache_resource
def load_chatbot_components():
    """Loads all necessary components from saved files."""
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

        with open('chatbot_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('chatbot_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

        with open('chatbot_intents.json', 'r', encoding='utf-8') as f:
            intents_data = json.load(f)

        all_patterns = []
        all_responses = []
        for intent in intents_data['intents']:
            if intent['tag'] not in ['fallback']: 
                all_patterns.extend(intent['patterns'])
                all_responses.extend(intent['responses']) 

        lemmatizer = WordNetLemmatizer()
        translator = google_translator()

        return model, vectorizer, intents_data, lemmatizer, translator, all_patterns, all_responses

    except FileNotFoundError as e:
        st.error(f"Error loading required files: {e.filename}. Please ensure 'train_chatbot.py' was run successfully.")
        raise
    except Exception as e:
        st.error(f"An unexpected error occurred during component loading: {e}")
        raise

try:
    model, vectorizer, intents_data, lemmatizer, translator, ALL_PATTERNS, ALL_RESPONSES = load_chatbot_components()
except:
    st.stop()


# --- 2. CHATBOT LOGIC FUNCTIONS ---

def is_hindi(text):
    """Checks if the text contains Hindi characters for source language determination."""
    return any('\u0900' <= char <= '\u097F' for char in text)

def translate_to_english(text):
    """Translates user input to English and returns the detected language code."""
    if not text:
        return "", 'en'
    try:
        cleaned_text = re.sub(r'[^\w\s\u0900-\u097F]', '', text)
        translation = translator.translate(cleaned_text, lang_tgt='en')
        detected_src = translator.detect(cleaned_text)

        if is_hindi(cleaned_text) or detected_src == 'hi':
            return translation, 'hi'

        return translation, detected_src or 'en'
    except Exception:
        # Fallback to original text in case of translation error
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

def get_best_response_by_similarity(user_input_en, intents_data, model, vectorizer, lemmatizer, threshold=0.3):
    """
    Finds the best answer by predicting the intent and then using a random 
    response from that intent's pool for high confidence questions.
    """
    
    words = user_input_en.split()
    lemmatized_input = " ".join([lemmatizer.lemmatize(word.lower()) for word in words])
    
    user_vec = vectorizer.transform([lemmatized_input])
    
    try:
        predicted_tag = model.predict(user_vec)[0]
        confidence_score = model.predict_proba(user_vec).max()
    except Exception:
        predicted_tag = 'fallback'
        confidence_score = 0.0

    tag_responses = []
    
    for intent in intents_data['intents']:
         if intent['tag'] == predicted_tag:
             tag_responses = intent['responses']
             break

    # Utility intents (greeting, farewell, thanks, out_of_scope) should always return a response
    if predicted_tag in ['greeting', 'farewell', 'thanks', 'out_of_scope']:
         return random.choice(tag_responses)

    if confidence_score >= threshold and tag_responses:
        # Return a random response from the high-confidence predicted intent
        return random.choice(tag_responses)
    else:
        # Fallback if confidence is too low
        for intent in intents_data['intents']:
            if intent['tag'] == 'fallback':
                return random.choice(intent['responses'])

    return "I am currently unable to process your specific question. Please try asking a different way."


# --- 3. STREAMLIT UI CODE ---
st.set_page_config(page_title="Jharkhand Tourism Chatbot 🤖", layout="centered")
st.title("🤖 Jharkhand Tourism Guide")
st.subheader("Your Multilingual Helper for the Land of Forests! 🌳")

if "messages" not in st.session_state:
    st.session_state.messages = []
    initial_message = "Hello! Welcome to Jharkhand Tourism. I can tell you about popular places, waterfalls, culture, food, and more. Feel free to ask in English or Hindi!"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about Jharkhand tourism..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner('Finding the best answer...'):
        eng_input, source_lang = translate_to_english(prompt)

        english_response = get_best_response_by_similarity(
            eng_input, 
            intents_data, 
            model, 
            vectorizer, 
            lemmatizer, 
            threshold=0.3
        )

        final_response = translate_response(english_response, source_lang)

    with st.chat_message("assistant"):
        st.markdown(final_response)

    st.session_state.messages.append({"role": "assistant", "content": final_response})
