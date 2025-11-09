import streamlit as st
import pickle
import json
import nltk
import random
import numpy as np
import os 
from google_trans_new import google_translator
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- 1. SETUP: Load Model and Functions (Streamlit Cache) ---
@st.cache_resource
def load_chatbot_components():
    """Loads all necessary components from saved files and ensures NLTK data is available."""
    try:
        # --- CRITICAL FIX: Explicitly set NLTK data path to a writable directory ---
        # This resolves the LookupError by forcing NLTK to use a writable cloud directory.
        NLTK_DATA_DIR = "/tmp/nltk_data"
        if NLTK_DATA_DIR not in nltk.data.path:
            nltk.data.path.append(NLTK_DATA_DIR)
        
        # Ensure the directory exists before downloading
        if not os.path.exists(NLTK_DATA_DIR):
            os.makedirs(NLTK_DATA_DIR)

        # Ensure all dependencies are downloaded into the new path.
        # Use download_dir=NLTK_DATA_DIR to force installation into the correct path.
        nltk.download('punkt', download_dir=NLTK_DATA_DIR, quiet=True) 
        nltk.download('wordnet', download_dir=NLTK_DATA_DIR, quiet=True) 
        nltk.download('omw-1.4', download_dir=NLTK_DATA_DIR, quiet=True) # Open Multilingual WordNet
        nltk.download('averaged_perceptron_tagger', download_dir=NLTK_DATA_DIR, quiet=True)
        # --------------------------------------------------------

        # Load the saved model and vectorizer
        with open('chatbot_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('chatbot_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load intents data
        with open('chatbot_intents.json', 'r', encoding='utf-8') as f:
            intents_data = json.load(f)

        # Initialize tools
        lemmatizer = WordNetLemmatizer()
        translator = google_translator()
        
        return model, vectorizer, intents_data, lemmatizer, translator
        
    except FileNotFoundError as e:
        st.error(f"Error loading required files: {e}. Ensure all three files are in the directory.")
        # If files are missing, raise to stop execution cleanly
        raise
    except Exception as e:
        st.error(f"An error occurred during component loading: {e}")
        raise

try:
    # This call now includes the comprehensive NLTK download/setup
    model, vectorizer, intents_data, lemmatizer, translator = load_chatbot_components()
except:
    st.stop() # Stop the Streamlit run if the loading fails

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
        translation = translator.translate(text, dest='en')
        detected_src = translation.src
        if is_hindi(text):
            return translation.text, 'hi'
        return translation.text, detected_src
    except Exception:
        return text, 'en' 

def translate_response(text, dest_lang):
    if dest_lang == 'en':
        return text
    try:
        translation = translator.translate(text, dest=dest_lang)
        return translation.text
    except Exception:
        return text 

def classify_intent(sentence):
    # This requires the 'punkt' resource
    sentence_words = nltk.word_tokenize(sentence)
    # This requires the 'wordnet' resource
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
    
    # Robust Keyword Fallback if confidence is low
    if 'ranchi' in sentence_str or 'raanchi' in sentence_str or 'waterfall' in sentence_str or 'झरनो' in sentence_str or 'रांची' in sentence_str:
        return 'about_ranchi'
    elif 'jamshedpur' in sentence_str or 'steel city' in sentence_str or 'जमशेदपुर' in sentence_str:
        return 'about_jamshedpur'
    elif 'betla' in sentence_str or 'safari' in sentence_str:
        return 'about_betla'
    elif 'deoghar' in sentence_str or 'baba dham' in sentence_str or 'jyotirlinga' in sentence_str:
        return 'about_deoghar'
    elif 'jain' in sentence_str or 'parasnath' in sentence_str or 'shikharji' in sentence_str:
        return 'about_parasnath'
    elif 'food' in sentence_str or 'cuisine' in sentence_str or 'litti' in sentence_str:
        return 'local_cuisine'
    elif 'transport' in sentence_str or 'travel' in sentence_str or 'airport' in sentence_str:
        return 'transport'
    elif 'hello' in sentence_str or 'hi' in sentence_str or 'namaste' in sentence_str:
        return 'greeting'

    return 'fallback'

# --- 3. STREAMLIT UI CODE ---

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
