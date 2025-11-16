import pickle
import json
import nltk
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- NLTK Downloads (FIXED ERROR) ---
print("Downloading NLTK resources...")
# We download them unconditionally to prevent the LookupError
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print("NLTK downloads complete.")

# --- 1. Load and Transform Data ---
try:
    # Use the exact file name provided
    with open('jharkhand_tourism_large_dataset (2).json', 'r', encoding='utf-8') as f:
        state_data = json.load(f)
except FileNotFoundError:
    print("ERROR: 'jharkhand_tourism_large_dataset (2).json' not found. Please ensure it is in the same directory.")
    exit()

lemmatizer = WordNetLemmatizer()
new_intents = []

tag_rules = {
    'capital|ranchi|waterfall|airport': 'about_ranchi',
    'parasnath|hill|jyotirlinga|baidyanath|temple|religious|deoghar': 'religious_places',
    'betla|wildlife|safari|forest': 'about_betla',
    'food|cuisine|eat|dishes|litti': 'local_cuisine',
    'jamshedpur|steel': 'about_jamshedpur',
    'trekking|boating|adventure|hike': 'adventure_activities',
    'culture|dance|festival': 'culture_highlights',
    'climate|time to visit|overview|famous': 'overview',
    'industry|mineral': 'financial_info',
    'safe|languages|reach': 'general_info'
}

qa_tags = {}
for item in state_data['qa_dataset']:
    question = item['question'].lower()
    tag = 'general_info'
    
    processed_question = " ".join([lemmatizer.lemmatize(word) for word in re.findall(r'\b\w+\b', question)])
    
    for rule, intent_tag in tag_rules.items():
        if re.search(r'\b(?:' + rule.replace('|', '|') + r')\b', processed_question):
            tag = intent_tag
            break
            
    if tag not in qa_tags:
        qa_tags[tag] = {'tag': tag, 'patterns': [], 'responses': []}
        
    qa_tags[tag]['patterns'].append(item['question'])
    qa_tags[tag]['responses'].append(item['answer'])

new_intents.extend(list(qa_tags.values()))

new_intents.extend([
    {'tag': 'greeting', 'patterns': ['Hi', 'Hello', 'Is anyone there?', 'Namaste', 'hey'], 
     'responses': ["Hello! Welcome to Jharkhand Tourism. How can I assist you?", "Hi there! What can I tell you about Jharkhand?"]},
    {'tag': 'fallback', 'patterns': ['I do not know', 'nothing', 'general query', 'random text'], 
     'responses': ["I'm sorry, I don't understand that. Can you try asking about a specific place in Jharkhand?", "I need more information. Could you rephrase your question?"]}
])

intents_data = {'intents': new_intents}

# --- 2. Training the Model ---
training_data = []
training_tags = []

for intent in intents_data['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        words = pattern.split()
        processed_pattern = " ".join([lemmatizer.lemmatize(word.lower()) for word in words])
        
        training_data.append(processed_pattern)
        training_tags.append(tag)

print("\n--- Starting Training ---")

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(training_data)
print("✅ Vectorizer trained successfully.")

unique_tags = sorted(list(set(training_tags)))
y_train = [unique_tags.index(tag) for tag in training_tags]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

model.classes_ = np.array(unique_tags)
print("✅ Model trained successfully.")

# --- 3. Saving the Components ---
model_filename = 'chatbot_model.pkl'
vectorizer_filename = 'chatbot_vectorizer.pkl'
intents_filename = 'chatbot_intents.json'

print("\n--- Saving Files ---")

with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Saved: {model_filename}")

with open(vectorizer_filename, 'wb') as file:
    pickle.dump(vectorizer, file)
print(f"Saved: {vectorizer_filename}")

with open(intents_filename, 'w', encoding='utf-8') as file:
    json.dump(intents_data, file, ensure_ascii=False, indent=4)
print(f"Saved: {intents_filename}")

print("\nTraining script updated.")
