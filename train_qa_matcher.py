import pickle
import json
import nltk
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# --- NLTK Downloads ---
print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True) 
print("NLTK downloads complete.")

# --- 1. Load and Deduplicate Data ---
DATA_FILE = 'jharkhand_tourism_qa_150 (1).json'
unique_qa_map = {}

try:
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        qa_dataset = json.load(f)
        
    print(f"Loaded {len(qa_dataset)} total entries from {DATA_FILE}")

    # Deduplicate the dataset 
    for item in qa_dataset:
        question = item['question'].strip()
        answer = item['answer'].strip()
        if question not in unique_qa_map:
            unique_qa_map[question] = answer
            
    # FIX: Add a specific Q&A pair to capture "famous people" queries.
    unique_qa_map["Who are some famous people from Jharkhand?"] = "While Jharkhand has many notable figures, this chatbot focuses exclusively on tourism, culture, and travel information."

    questions = list(unique_qa_map.keys())
    answers = list(unique_qa_map.values())
    
    if not questions:
        raise ValueError("No valid question-answer pairs found after deduplication.")

except FileNotFoundError:
    print(f"Error: The file {DATA_FILE} was not found.")
    exit()
except Exception as e:
    print(f"Error processing data: {e}")
    exit()

# --- 2. Data Preprocessing and Training ---
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Tokenizes, converts to lowercase, and lemmatizes the text."""
    text = re.sub(r'[?]', '', text.lower()) 
    words = nltk.tokenize.wordpunct_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Preprocess all unique questions for training
preprocessed_questions = [preprocess_text(q) for q in questions]

print(f"Training on {len(questions)} unique questions...")

# 3. Train the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_df=0.85, min_df=1, stop_words='english')
qa_vectors = vectorizer.fit_transform(preprocessed_questions)

print("✅ TF-IDF Vectorizer trained successfully.")
print(f"Vector space dimensions: {qa_vectors.shape}")

# --- 4. Saving the Components ---
vectorizer_filename = 'qa_vectorizer.pkl'
vectors_filename = 'qa_vectors.pkl'
data_filename = 'qa_data.json' 

print("\n--- Saving Files ---")

with open(vectorizer_filename, 'wb') as file:
    pickle.dump(vectorizer, file)
print(f"Saved: {vectorizer_filename}")

with open(vectors_filename, 'wb') as file:
    pickle.dump(qa_vectors, file)
print(f"Saved: {vectors_filename}")

# Save the unique questions and answers
with open(data_filename, 'w', encoding='utf-8') as file:
    json.dump({'questions': questions, 'answers': answers}, file, ensure_ascii=False, indent=4)
print(f"Saved: {data_filename} (Contains {len(questions)} unique Q/A pairs)")

print("\nTraining complete. You can now run the Streamlit app.")
