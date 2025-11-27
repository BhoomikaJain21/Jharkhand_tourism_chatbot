import streamlit as st
import pickle
import json
import nltk
import numpy as np
import re
from google_trans_new import google_translator
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
# Import the specific exception to fix the deployment error
from nltk.downloader import DownloadError 

# --- Configuration ---
VECTORIZER_FILE = 'qa_vectorizer.pkl'
VECTORS_FILE = 'qa_vectors.pkl'
DATA_FILE = 'qa_data.json'
SIMILARITY_THRESHOLD = 0.35 

# --- Ensure NLTK resources are available ---
# We check and download the necessary resources when the app starts.
try:
    nltk.data.find('tokenizers/punkt')
except DownloadError: # FIXED: Catch the directly imported DownloadError
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except DownloadError: # FIXED: Catch the directly imported DownloadError
    nltk.download('wordnet', quiet=True)


# --- Model Loading and Setup ---
@st.cache_resource
def load_resources():
    """Loads all necessary components and caches them."""
    try:
        # 1. Load QA Data
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            questions = data['questions']
            answers = data['answers']
            st.session_state.qa_map = {q: a for q, a in zip(questions, answers)}

        # 2. Load Vectorizer
        with open(VECTORIZER_FILE, 'rb') as file:
            vectorizer = pickle.load(file)

        # 3. Load QA Vectors (The training data)
        with open(VECTORS_FILE, 'rb') as file:
            qa_vectors = pickle.load(file)
        
        # 4. Initialize Translator and Lemmatizer
        translator = google_translator()
        lemmatizer = WordNetLemmatizer()
        
        return vectorizer, qa_vectors, answers, translator, lemmatizer

    except FileNotFoundError:
        st.error("Model files not found. Please run `train_qa_matcher.py` locally first.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading resources: {e}")
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


def get_best_response_by_similarity(user_input_en, vectorizer, qa_vectors, answers, threshold):
    """
    Finds the best answer by calculating Cosine Similarity against all
    pre-vectorized questions.
    """
    # 1. Preprocess and Vectorize User Input
    lemmatized_input = preprocess_text_for_inference(user_input_en)
    
    if not lemmatized_input.strip():
        return "Please ask a specific question about Jharkhand tourism. 🤔"
        
    user_vec = vectorizer.transform([lemmatized_input])
    
    # 2. Calculate Cosine Similarity
    similarity_scores = cosine_similarity(user_vec, qa_vectors).flatten()
    
    # 3. Find Best Match
    best_match_index = np.argmax(similarity_scores)
    best_score = similarity_scores[best_match_index]
    
    # 4. Determine Response
    if best_score >= threshold:
        english_response = answers[best_match_index]
    else:
        english_response = "I am not sure I understand that query. Could you please rephrase or ask a different question about Jharkhand tourism? 😔"
        
    return english_response

# --- Streamlit App ---
st.title("🗺️ Multilingual Jharkhand Tourism Chatbot (Retrieval Model)")
st.markdown("Ask your question in English, Hindi, or any other major language!")

# Load resources (must be before the main chat logic)
vectorizer, qa_vectors, answers, translator, lemmatizer = load_resources()

# Store in session state for easy access
st.session_state.vectorizer = vectorizer
st.session_state.qa_vectors = qa_vectors
st.session_state.answers = answers
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

    # --- Chatbot Logic (Fixed for Multilingual Q&A Matching) ---
    
    # 1. Translate User Input to English
    eng_input, source_lang = translate_to_english(prompt)
    
    # 2. Get English Response using Similarity Matching
    english_response = get_best_response_by_similarity(
        eng_input,
        st.session_state.vectorizer,
        st.session_state.qa_vectors,
        st.session_state.answers,
        SIMILARITY_THRESHOLD
    )
    
    # 3. Translate Response back to User's Language
    final_response = translate_response(english_response, source_lang)

    # 4. Display and Save Response
    with st.chat_message("assistant"):
        st.markdown(final_response)

    st.session_state.messages.append({"role": "assistant", "content": final_response})
