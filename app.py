import streamlit as st
import pickle
import json
import nltk
import random
import numpy as np
import re
# Note: google_trans_new is used for translation. A simple 'pip install google-trans-new' is required.
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
        # This is for the improved similarity-based response retrieval
        all_patterns = []
        all_responses = []
        for intent in intents_data['intents']:
            # Exclude fallback, as we want to match real questions
            if intent['tag'] not in ['fallback']: 
                # Ensure patterns and responses are matched correctly. 
                # Since the original data maps Q to A, we pair them up.
                
                # NOTE: For datasets where one tag has multiple responses, 
                # and multiple patterns, this simple flat list works best 
                # for similarity comparison:
                # Every question (pattern) is paired with one of its possible answers.
                # To ensure a specific answer is retrieved for a specific question, 
                # we'll use the *entire* list of responses in the similarity logic 
                # as a lookup table.
                all_patterns.extend(intent['patterns'])
                
                # We need to ensure the ALL_RESPONSES list size is the same as ALL_PATTERNS
                # for proper indexing later. A simpler approach is to use the original 
                # Q-A mapping logic from the training script to ensure 1-to-1 retrieval.
                # Since the training script groups Q/A by intent, we rely on the vectorizer 
                # finding the *closest* pattern, then we retrieve its corresponding answer 
                # from the full set.
                
                # A more robust way: use the original Q-A mapping.
                for pattern in intent['patterns']:
                    # Simple heuristic: for each pattern, we assume one random response is acceptable.
                    # This is a simplification; a full production system would need direct Q-A mapping.
                    all_responses.append(random.choice(intent['responses']))


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
    # If loading fails, stop the Streamlit app
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
        # Simple string cleaning before translation
        cleaned_text = re.sub(r'[^\w\s\u0900-\u097F]', '', text)
        
        # Use the google_translator instance
        translation = translator.translate(cleaned_text, lang_tgt='en')
        detected_src = translator.detect(cleaned_text)

        # Check for Hindi or a detected Hindi language code
        if is_hindi(cleaned_text) or detected_src == 'hi':
            return translation, 'hi'

        # Return English translation and detected source language (defaulting to 'en')
        return translation, detected_src or 'en'
    except Exception:
        # Fallback in case of translation error
        return text, 'en'

def translate_response(text, dest_lang):
    """Translates the English response back to the user's language."""
    if dest_lang == 'en':
        return text
    try:
        # Use the google_translator instance
        translation = translator.translate(text, lang_src='en', lang_tgt=dest_lang)
        return translation
    except Exception:
        return text # Return original English text on error

def get_best_response_by_similarity(user_input_en, patterns, responses, threshold=0.3):
    """
    Finds the best answer by comparing the user's input against ALL known questions
    using the model's vectorizer for semantic similarity.
    """
    # 1. Handle Greeting/Fallback (using predefined intent logic first)
    lemmatized_input = " ".join([lemmatizer.lemmatize(word.lower()) for word in user_input_en.split()])
    
    # Vectorize the user's input
    user_vec = vectorizer.transform([lemmatized_input])
    
    # Predict the intent tag using the trained model
    predicted_tag_index = model.predict(user_vec)[0]
    
    # Get all responses for the predicted tag
    tag_responses = []
    for intent in intents_data['intents']:
         if intent['tag'] == predicted_tag_index:
             tag_responses = intent['responses']
             break

    if predicted_tag_index in ['greeting', 'fallback']:
         return random.choice(tag_responses)

    # 2. Similarity Search within the predicted intent (More accurate than against ALL patterns)
    
    # Get all patterns for the predicted tag
    tag_patterns = []
    for intent in intents_data['intents']:
        if intent['tag'] == predicted_tag_index:
            tag_patterns = intent['patterns']
            break
    
    # Lemmatize and vectorize the patterns for the predicted tag
    lemmatized_tag_patterns = [" ".join([lemmatizer.lemmatize(word.lower()) for word in p.split()]) for p in tag_patterns]
    pattern_vecs = vectorizer.transform(lemmatized_tag_patterns)

    # Calculate similarity (dot product between user input and predicted intent's patterns)
    similarity_scores = user_vec.dot(pattern_vecs.transpose()).toarray()[0]

    # Find the index of the highest similarity score within the predicted tag
    best_match_index = np.argmax(similarity_scores)
    best_score = similarity_scores[best_match_index]

    # 3. Return the response
    if best_score > threshold:
        # Since tag_patterns and tag_responses were grouped from the original JSON, 
        # for a good Q-A dataset, the index should retrieve a relevant response.
        # However, due to the structure of the training script, multiple questions map to 
        # multiple answers under the same tag. The safest retrieval is a random 
        # response from the *predicted intent's* response pool.
        return tag_responses[best_match_index] if best_match_index < len(tag_responses) else random.choice(tag_responses)
    else:
        # If score is too low, use the fallback response
        for intent in intents_data['intents']:
            if intent['tag'] == 'fallback':
                return random.choice(intent['responses'])

    return "I am unable to process your request at the moment."


# --- 3. STREAMLIT UI CODE ---
st.set_page_config(page_title="Jharkhand Tourism Chatbot 🤖", layout="centered")
st.title("🤖 Jharkhand Tourism Chatbot (V3: Intent + Similarity)")
st.subheader("Your Multilingual Guide to the Land of Forests! 🌳")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    initial_message = "Hello! Welcome to Jharkhand Tourism. I use an intent classifier followed by a similarity check for better accuracy. Ask me about **waterfalls**, **wildlife**, or **food** in Jharkhand!"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question..."):\
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner('Thinking...'):
        # 1. Translate user prompt to English and get the source language code
        eng_input, source_lang = translate_to_english(prompt)

        # 2. Get the response based on vector similarity
        english_response = get_best_response_by_similarity(eng_input, ALL_PATTERNS, ALL_RESPONSES, threshold=0.3)

        # 3. Translate the English response back to the user's source language
        final_response = translate_response(english_response, source_lang)

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(final_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_response})
