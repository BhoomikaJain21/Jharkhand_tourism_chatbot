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

# --- 1. SETUP: Load Model and Functions ---
@st.cache_resource
def load_chatbot_components():
    """Loads all necessary components from saved files."""
    try:
        nltk.download('punkt', quiet=True) 
        nltk.download('wordnet', quiet=True)
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
        # Raise an exception to stop execution gracefully
        raise
    except Exception as e:
        st.error(f"An error occurred during component loading: {e}")
        raise

try:
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
    sentence_words = nltk.word_tokenize(sentence)
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
    
    # Robust Keyword Fallback
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
        intent_tag = classify_intent(eng_input)
        english_response = get_response(intent_tag)
        final_response = translate_response(english_response, source_lang)
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(final_response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_response})
