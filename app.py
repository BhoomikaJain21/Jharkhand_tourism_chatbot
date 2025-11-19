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
        # This setup is kept for flexibility but is not the primary mechanism in V3 logic.
        all_patterns = []
        all_responses = []
        for intent in intents_data['intents']:
            if intent['tag'] not in ['fallback']: 
                all_patterns.extend(intent['patterns'])
                all_responses.extend([random.choice(intent['responses']) for _ in intent['patterns']])

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
        cleaned_text = re.sub(r'[^\w\s\u0900-\u097F]', '', text)
        translation = translator.translate(cleaned_text, lang_tgt='en')
        detected_src = translator.detect(cleaned_text)

        if is_hindi(cleaned_text) or detected_src == 'hi':
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
        return text # Return original English text on error

def get_best_response_by_similarity(user_input_en, intents_data, model, vectorizer, lemmatizer, threshold=0.3):
    """
    Finds the best answer by predicting the intent and then finding the best 
    pattern match within that intent.
    """
    
    # 1. Process and Vectorize User Input (Crucial for consistent features)
    words = user_input_en.split()
    # Use the same exact processing (lower-case and lemmatize) as the training script
    lemmatized_input = " ".join([lemmatizer.lemmatize(word.lower()) for word in words])
    
    # Vectorize the user's input
    user_vec = vectorizer.transform([lemmatized_input])
    
    # 2. Predict the Intent Tag
    try:
        predicted_tag = model.predict(user_vec)[0]
    except Exception as e:
        # Fallback if prediction fails (e.g., feature mismatch)
        predicted_tag = 'fallback'

    # 3. Retrieve Intent Data and Response
    tag_responses = []
    tag_patterns = []
    
    for intent in intents_data['intents']:
         if intent['tag'] == predicted_tag:
             tag_responses = intent['responses']
             tag_patterns = intent['patterns']
             break

    if predicted_tag in ['greeting', 'fallback']:
         return random.choice(tag_responses)

    # 4. Similarity Search within the predicted intent
    
    # Lemmatize and vectorize the patterns for the predicted tag (Crucial step for fix)
    # Ensure patterns are processed the same way as training data.
    lemmatized_tag_patterns = []
    for p in tag_patterns:
         p_words = p.split()
         lemmatized_tag_patterns.append(" ".join([lemmatizer.lemmatize(word.lower()) for word in p_words]))
    
    # Now transform these correctly processed patterns
    pattern_vecs = vectorizer.transform(lemmatized_tag_patterns)

    # Calculate similarity (dot product between user input and predicted intent's patterns)
    similarity_scores = user_vec.dot(pattern_vecs.transpose()).toarray()[0]

    # Find the index of the highest similarity score within the predicted tag
    best_match_index = np.argmax(similarity_scores)
    best_score = similarity_scores[best_match_index]

    # 5. Return the response
    if best_score >= threshold:
        # Return the response corresponding to the best matching question
        # This assumes the Q-A pairs are generally aligned within the intent lists.
        return tag_responses[best_match_index % len(tag_responses)]
    else:
        # If score is too low, use the fallback response
        for intent in intents_data['intents']:
            if intent['tag'] == 'fallback':
                return random.choice(intent['responses'])

    return "I am currently unable to process your specific question. Please try asking a different way."


# --- 3. STREAMLIT UI CODE (Non-technical presentation) ---
st.set_page_config(page_title="Jharkhand Tourism Chatbot 🤖", layout="centered")
st.title("🤖 Jharkhand Tourism Guide")
st.subheader("Your Multilingual Helper for the Land of Forests! 🌳")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Simplified initial message
    initial_message = "Hello! Welcome to Jharkhand Tourism. I can tell you about popular places, waterfalls, culture, food, and more. Feel free to ask in English or Hindi!"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about Jharkhand tourism..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner('Finding the best answer...'):
        # 1. Translate user prompt to English and get the source language code
        eng_input, source_lang = translate_to_english(prompt)

        # 2. Get the response based on intent and similarity
        # Pass necessary components to the function
        english_response = get_best_response_by_similarity(
            eng_input, 
            intents_data, 
            model, 
            vectorizer, 
            lemmatizer, 
            threshold=0.3
        )

        # 3. Translate the English response back to the user's source language
        final_response = translate_response(english_response, source_lang)

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(final_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_response})
