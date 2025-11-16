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
        # Download NLTK resources needed for lemmatization
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
            
        # Extract all available patterns (questions) and their corresponding answers
        # This creates a flat list of all questions and a parallel list of all possible responses
        all_patterns = []
        all_responses = []
        for intent in intents_data['intents']:
            if intent['tag'] not in ['greeting', 'fallback']:
                for pattern, response in zip(intent['patterns'], intent['responses']):
                    all_patterns.append(pattern)
                    all_responses.append(response)

        # Initialize tools
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

def get_best_response_by_similarity(user_input_en, patterns, responses, threshold=0.2):
    """
    Finds the best answer by comparing the user's input against ALL known questions
    using the model's vectorizer for semantic similarity.
    """
    # 1. Check for greeting first
    if 'hi' in user_input_en.lower() or 'hello' in user_input_en.lower() or 'hey' in user_input_en.lower():
        for intent in intents_data['intents']:
             if intent['tag'] == 'greeting':
                 return random.choice(intent['responses'])

    # 2. Vectorize the user's input
    user_vec = vectorizer.transform([user_input_en])
    
    # 3. Vectorize all known patterns (questions) if not already done. 
    # NOTE: In a real environment, you should pre-calculate pattern vectors.
    # Here, we transform the training data (patterns) used by the model
    pattern_vecs = vectorizer.transform(patterns) 
    
    # 4. Calculate similarity (dot product between user input and all patterns)
    # This gives us a similarity score (cosine or TF-IDF product)
    similarity_scores = user_vec.dot(pattern_vecs.transpose()).toarray()[0]
    
    # 5. Find the index of the highest similarity score
    best_match_index = np.argmax(similarity_scores)
    best_score = similarity_scores[best_match_index]

    # 6. Return the corresponding response if the score is above the threshold
    if best_score > threshold:
        # Using the actual answer corresponding to the best matching question
        return responses[best_match_index]
    else:
        # If no good match, use the fallback response
        for intent in intents_data['intents']:
            if intent['tag'] == 'fallback':
                return random.choice(intent['responses'])
    
    return "I am unable to process your request at the moment."


# --- 3. STREAMLIT UI CODE ---
st.set_page_config(page_title="Jharkhand Tourism Chatbot 🤖", layout="centered")
st.title("🤖 Jharkhand Tourism Chatbot (V2: Improved Logic)")
st.subheader("Your Multilingual Guide to the Land of Forests! 🌳")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! Welcome to Jharkhand Tourism. I now provide much more specific answers. Ask me, for example: **'What is the capital of Jharkhand?'** or **'Jharkhand ki rajdhani kya hai?'**"})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner('Thinking...'):
        # 1. Translate user prompt to English and get the source language code
        eng_input, source_lang = translate_to_english(prompt)
        
        # 2. Get the response based on vector similarity against ALL patterns (V2 Logic)
        english_response = get_best_response_by_similarity(eng_input, ALL_PATTERNS, ALL_RESPONSES)
        
        # 3. Translate the English response back to the user's source language
        final_response = translate_response(english_response, source_lang)

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(final_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_response})
