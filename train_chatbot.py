import pickle
import json
import nltk
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- NLTK Downloads ---
print("Downloading NLTK resources...")
# We download them unconditionally to prevent the LookupError
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print("NLTK downloads complete.")

# --- 1. Load and Transform Data (USING THE UPLOADED JSON FORMAT) ---
# NOTE: Using the file name 'jharkhand_tourism_qa_150 (1).json'
try:
    with open('jharkhand_tourism_qa_150 (1).json', 'r', encoding='utf-8') as f:
        qa_dataset = json.load(f)
except FileNotFoundError:
    print("ERROR: 'jharkhand_tourism_qa_150 (1).json' not found. Please ensure the file is named correctly.")
    exit()

lemmatizer = WordNetLemmatizer()
new_intents = []

# Rules to categorize questions into intent tags
# These rules map keywords to general topics (intents)
tag_rules = {
    'capital|ranchi|waterfall|airport|subarnarekha|river|patratu': 'about_ranchi_area',
    'parasnath|hill|jyotirlinga|baidyanath|temple|religious|deoghar|gautamdhara|jaina': 'religious_places',
    'betla|wildlife|safari|forest|dalma|hazaribagh|elephant|tiger|national park': 'about_wildlife_parks',
    'food|cuisine|eat|dishes|dhuska|rugra|jhor|roti': 'local_cuisine',
    'jamshedpur|steel|city|tata': 'about_jamshedpur',
    'trekking|boating|adventure|hike|rock climbing': 'adventure_activities',
    'culture|dance|festival|chhau|jhumair|santhal|tribe|tradition': 'culture_highlights',
    'climate|time to visit|overview|famous|best time|special': 'overview',
    'mineral|steel|industrial': 'financial_info',
    'safe|languages|reach|netarhat|transport|lodging': 'general_info'
}

qa_tags = {}
for item in qa_dataset:
    question = item['question'].lower()
    tag = 'general_info' # Default tag

    # Process the question for matching
    processed_question = " ".join([lemmatizer.lemmatize(word) for word in re.findall(r'\b\w+\b', question)])

    # Determine tag based on keywords
    for rule, intent_tag in tag_rules.items():
        if re.search(r'\b(?:' + rule.replace('|', '|') + r')\b', processed_question):
            tag = intent_tag
            break

    # Aggregate patterns and responses for each tag
    if tag not in qa_tags:
        qa_tags[tag] = {'tag': tag, 'patterns': [], 'responses': []}

    qa_tags[tag]['patterns'].append(item['question'])
    qa_tags[tag]['responses'].append(item['answer'])

# Add the extracted QA intents
new_intents.extend(list(qa_tags.values()))

# Add predefined intents
new_intents.extend([
    {'tag': 'greeting', 'patterns': ['Hi', 'Hello', 'Is anyone there?', 'Namaste', 'hey', 'good morning', 'good evening'],
     'responses': ["Hello! Welcome to Jharkhand Tourism. How can I assist you? 🌳", "Hi there! What can I tell you about the Land of Forests?"]},
    {'tag': 'fallback', 'patterns': ['I do not know', 'nothing', 'general query', 'random text', 'not understood'],
     'responses': ["I'm sorry, I don't understand that. Can you try asking about a specific place, a waterfall, or food in Jharkhand?", "I need more information. Could you rephrase your question? Try 'Where is Dassam Falls located?'"]}
])

intents_data = {'intents': new_intents}

# --- 2. Training the Model ---
training_data = []
training_tags = []

for intent in intents_data['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        words = pattern.split()
        # Lemmatize and lower-case the pattern for training
        processed_pattern = " ".join([lemmatizer.lemmatize(word.lower()) for word in words])

        training_data.append(processed_pattern)
        training_tags.append(tag)

print("\n--- Starting Training ---")

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(training_data)
print("✅ Vectorizer trained successfully.")

# Prepare labels
unique_tags = sorted(list(set(training_tags)))
y_train = [unique_tags.index(tag) for tag in training_tags]

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Assign class names for prediction mapping
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

print("\nTraining script finished.")
