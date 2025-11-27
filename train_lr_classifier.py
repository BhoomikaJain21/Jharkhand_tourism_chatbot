import pickle
import json
import nltk
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# --- NLTK Downloads ---
print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True) 
print("NLTK downloads complete.")

# --- 1. Load Data and Define Intents ---
DATA_FILE = 'jharkhand_tourism_qa_150 (1).json'
unique_qa_pairs = {}
intents_map = {}

# Define the Master Intent Map with a single representative Answer for each category
MASTER_ANSWERS = {
    'overview': "Jharkhand is known for waterfalls, forests, hills, wildlife sanctuaries, rich minerals, and tribal culture.",
    'waterfalls': "Hundru Falls and Dassam Falls are among the most popular waterfalls, like Johna Falls (Gautamdhara Falls).",
    'wildlife': "You can visit Betla National Park, Dalma Wildlife Sanctuary (near Jamshedpur), and Hazaribagh Wildlife Sanctuary.",
    'best_time': "The best time to visit Jharkhand is from October to March.",
    'location': "Specific locations like Dassam Falls, Patratu Valley, and Dalma Sanctuary are scattered, mainly near major cities like Ranchi and Jamshedpur.",
    'logistics': "Travel is mainly by road, and the main entry point is Birsa Munda Airport in Ranchi. Jharkhand is generally safe for tourists.",
    'culture': "The Santhal tribe, Chhau and Jhumair dance forms, and foods like Dhuska and Chilka Roti are central to Jharkhand's culture.",
    'cities': "The capital is Ranchi. Jamshedpur is known as the Steel City. The Subarnarekha River flows through Ranchi.",
    'terrain': "Parasnath Hill is the highest peak and a famous Jain site. The climate is tropical with hot summers and cool winters.",
    'activities': "Adventure activities include trekking (popular in Parasnath Hills and Netarhat), rock climbing, wildlife safari, and boating."
}

# Manually assign Intents to each question in the dataset for supervised training
def assign_intent(question):
    q = question.lower()
    if 'best tourist places' in q or 'famous for' in q: return 'overview'
    if 'waterfall' in q or 'johna falls' in q: return 'waterfalls'
    if 'national park' in q or 'betla' in q or 'dalma' in q or 'hazaribagh' in q: return 'wildlife'
    if 'time to visit' in q or 'season' in q or 'climate' in q: return 'best_time'
    if 'where is' in q or 'located' in q or 'patratu valley' in q: return 'location'
    if 'reach' in q or 'safe' in q or 'airport' in q: return 'logistics'
    if 'tribe' in q or 'food' in q or 'dance form' in q: return 'culture'
    if 'capital' in q or 'steel city' in q or 'river' in q: return 'cities'
    if 'peak' in q or 'parasnath' in q or 'climate' in q: return 'terrain'
    if 'adventure' in q or 'trekking' in q: return 'activities'
    return 'overview' # Default fallback

try:
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        qa_dataset = json.load(f)
        
    for item in qa_dataset:
        question = item['question'].strip()
        if question not in unique_qa_pairs:
            unique_qa_pairs[question] = item['answer'].strip()
            
    questions = list(unique_qa_pairs.keys())
    # Assign intent to each unique question
    intents = [assign_intent(q) for q in questions]

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

print(f"Training Logistic Regression model on {len(questions)} unique questions across {len(set(intents))} intents...")

# 3. Train the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_df=0.85, min_df=1, stop_words='english')
qa_vectors = vectorizer.fit_transform(preprocessed_questions)

# 4. Encode Intents and Train Logistic Regression
le = LabelEncoder()
y_encoded = le.fit_transform(intents)

model = LogisticRegression(solver='liblinear', multi_class='ovr', C=1.0)
model.fit(qa_vectors, y_encoded)

print("✅ Logistic Regression Model and TF-IDF Vectorizer trained successfully.")
print(f"Trained Intents: {le.classes_}")

# --- 5. Saving the Components ---
vectorizer_filename = 'lr_vectorizer.pkl'
model_filename = 'lr_model.pkl'
le_filename = 'lr_label_encoder.pkl'
data_filename = 'lr_intent_answers.json' 

print("\n--- Saving Files ---")

with open(vectorizer_filename, 'wb') as file:
    pickle.dump(vectorizer, file)
print(f"Saved: {vectorizer_filename}")

with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Saved: {model_filename}")

with open(le_filename, 'wb') as file:
    pickle.dump(le, file)
print(f"Saved: {le_filename}")

# Save the master intent-to-answer map
with open(data_filename, 'w', encoding='utf-8') as file:
    json.dump(MASTER_ANSWERS, file, ensure_ascii=False, indent=4)
print(f"Saved: {data_filename} (Master Answers)")

print("\nTraining complete. You can now run the Streamlit app.")
