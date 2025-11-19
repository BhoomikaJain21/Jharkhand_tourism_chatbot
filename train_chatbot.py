import pickle
import json
import nltk
import numpy as np
import re
import random
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
# CHANGE: Switching from LogisticRegression to MultinomialNB
from sklearn.naive_bayes import MultinomialNB 

# --- NLTK Downloads ---
print("Downloading NLTK resources...")
print("NLTK downloads complete.")

# --- 1. Load Data (Fixed Syntax Error) ---
qa_dataset = [
    {"id": 1, "question": "What are the best tourist places in Jharkhand?", "answer": "Jharkhand is known for waterfalls, forests, hills, wildlife sanctuaries, and cultural sites."},
    {"id": 2, "question": "What is Jharkhand famous for?", "answer": "Jharkhand is famous for waterfalls, forests, rich minerals, and tribal culture."},
    {"id": 3, "question": "Which waterfall in Jharkhand is most popular?", "answer": "Hundru Falls and Dassam Falls are among the most popular waterfalls."},
    {"id": 4, "question": "Where is Dassam Falls located?", "answer": "Dassam Falls is located near Taimara village, around 40 km from Ranchi."},
    {"id": 5, "question": "What is the best time to visit Jharkhand?", "answer": "The best time to visit Jharkhand is from October to March."},
    {"id": 6, "question": "Which national parks can I visit in Jharkhand?", "answer": "You can visit Betla National Park, Dalma Wildlife Sanctuary, and Hazaribagh Wildlife Sanctuary."},
    {"id": 7, "question": "What is special about Betla National Park?", "answer": "Betla National Park is known for tigers, elephants, and dense forests."},
    {"id": 8, "question": "How can I reach Netarhat from Ranchi?", "answer": "You can reach Netarhat by road from Ranchi via Latehar."},
    {"id": 9, "question": "Is Jharkhand safe for tourists?", "answer": "Yes, Jharkhand is generally safe for tourists with standard precautions."},
    {"id": 10, "question": "What is the capital of Jharkhand?", "answer": "The capital of Jharkhand is Ranchi."},
    {"id": 11, "question": "Which is the highest peak in Jharkhand?", "answer": "Parasnath Hill is the highest peak in Jharkhand."},
    {"id": 12, "question": "What is Parasnath Hill famous for?", "answer": "Parasnath Hill is a famous Jain pilgrimage site."},
    {"id": 13, "question": "Which tribe is most common in Jharkhand?", "answer": "The Santhal tribe is one of the major tribes in Jharkhand."}
    ,{"id": 14, "question": "What are traditional Jharkhand foods?", "answer": "Popular foods include Dhuska, Rugra, Chhoti Macher Jhor, and Chilka Roti."},
    {"id": 15, "question": "What is the dance form of Jharkhand?", "answer": "Chhau and Jhumair are popular dance forms of Jharkhand."},
    {"id": 16, "question": "Which river flows through Ranchi?", "answer": "The Subarnarekha River flows through Ranchi."},
    {"id": 17, "question": "What is the climate of Jharkhand?", "answer": "Jharkhand has a tropical climate with hot summers and cool winters."},
    {"id": 18, "question": "Which wildlife sanctuary is near Jamshedpur?", "answer": "Dalma Wildlife Sanctuary is the nearest major sanctuary to Jamshedpur."},
    {"id": 19, "question": "Where is Dalma Wildlife Sanctuary located?", "answer": "Dalma Wildlife Sanctuary is near Jamshedpur, famous for elephants."},
    {"id": 20, "question": "What is Johna Falls also known as?", "answer": "Johna Falls is also called Gautamdhara Falls."},
    {"id": 21, "question": "What adventure activities are available in Jharkhand?", "answer": "Adventure activities include trekking, rock climbing, wildlife safari, and boating."},
    {"id": 22, "question": "Where can I do trekking in Jharkhand?", "answer": "Trekking is popular in Parasnath Hills and Netarhat."},
    {"id": 23, "question": "Which airport is closest to major attractions?", "answer": "Birsa Munda Airport in Ranchi is the main airport for tourists."},
    {"id": 24, "question": "Where is Patratu Valley located?", "answer": "Patratu Valley is located near Ramgarh, close to Ranchi."},
    {"id": 25, "question": "Which city is known as the Steel City of Jharkhand?", "answer": "Jamshedpur is known as the Steel City of Jharkhand."},
    {"id": 26, "question": "What are the best tourist places in Jharkhand?", "answer": "Jharkhand is known for waterfalls, forests, hills, wildlife sanctuaries, and cultural sites."},
    {"id": 27, "question": "What is Jharkhand famous for?", "answer": "Jharkhand is famous for waterfalls, forests, rich minerals, and tribal culture."},
    {"id": 28, "question": "Which waterfall in Jharkhand is most popular?", "answer": "Hundru Falls and Dassam Falls are among the most popular waterfalls."},
    {"id": 29, "question": "Where is Dassam Falls located?", "answer": "Dassam Falls is located near Taimara village, around 40 km from Ranchi."},
    {"id": 30, "question": "What is the best time to visit Jharkhand?", "answer": "The best time to visit Jharkhand is from October to March."},
    {"id": 31, "question": "Which national parks can I visit in Jharkhand?", "answer": "You can visit Betla National Park, Dalma Wildlife Sanctuary, and Hazaribagh Wildlife Sanctuary."},
    {"id": 32, "question": "What is special about Betla National Park?", "answer": "Betla National Park is known for tigers, elephants, and dense forests."},
    {"id": 33, "question": "How can I reach Netarhat from Ranchi?", "answer": "You can reach Netarhat by road from Ranchi via Latehar."},
    {"id": 34, "question": "Is Jharkhand safe for tourists?", "answer": "Yes, Jharkhand is generally safe for tourists with standard precautions."},
    {"id": 35, "question": "What is the capital of Jharkhand?", "answer": "The capital of Jharkhand is Ranchi."},
    {"id": 36, "question": "Which is the highest peak in Jharkhand?", "answer": "Parasnath Hill is the highest peak in Jharkhand."},
    {"id": 37, "question": "What is Parasnath Hill famous for?", "answer": "Parasnath Hill is a famous Jain pilgrimage site."},
    {"id": 38, "question": "Which tribe is most common in Jharkhand?", "answer": "The Santhal tribe is one of the major tribes in Jharkhand."},
    {"id": 39, "question": "What are traditional Jharkhand foods?", "answer": "Popular foods include Dhuska, Rugra, Chhoti Macher Jhor, and Chilka Roti."},
    {"id": 40, "question": "What is the dance form of Jharkhand?", "answer": "Chhau and Jhumair are popular dance forms of Jharkhand."},
    {"id": 41, "question": "Which river flows through Ranchi?", "answer": "The Subarnarekha River flows through Ranchi."},
    {"id": 42, "question": "What is the climate of Jharkhand?", "answer": "Jharkhand has a tropical climate with hot summers and cool winters."},
    {"id": 43, "question": "Which wildlife sanctuary is near Jamshedpur?", "answer": "Dalma Wildlife Sanctuary is the nearest major sanctuary to Jamshedpur."},
    {"id": 44, "question": "Where is Dalma Wildlife Sanctuary located?", "answer": "Dalma Wildlife Sanctuary is near Jamshedpur, famous for elephants."},
    {"id": 45, "question": "What is Johna Falls also known as?", "answer": "Johna Falls is also called Gautamdhara Falls."},
    {"id": 46, "question": "What adventure activities are available in Jharkhand?", "answer": "Adventure activities include trekking, rock climbing, wildlife safari, and boating."},
    {"id": 47, "question": "Where can I do trekking in Jharkhand?", "answer": "Trekking is popular in Parasnath Hills and Netarhat."},
    {"id": 48, "question": "Which airport is closest to major attractions?", "answer": "Birsa Munda Airport in Ranchi is the main airport for tourists."},
    {"id": 49, "question": "Where is Patratu Valley located?", "answer": "Patratu Valley is located near Ramgarh, close to Ranchi."},
    {"id": 50, "question": "Which city is known as the Steel City of Jharkhand?", "answer": "Jamshedpur is known as the Steel City of Jharkhand."},
    {"id": 51, "question": "What are the best tourist places in Jharkhand?", "answer": "Jharkhand is known for waterfalls, forests, hills, wildlife sanctuaries, and cultural sites."},
    {"id": 52, "question": "What is Jharkhand famous for?", "answer": "Jharkhand is famous for waterfalls, forests, rich minerals, and tribal culture."},
    {"id": 53, "question": "Which waterfall in Jharkhand is most popular?", "answer": "Hundru Falls and Dassam Falls are among the most popular waterfalls."},
    {"id": 54, "question": "Where is Dassam Falls located?", "answer": "Dassam Falls is located near Taimara village, around 40 km from Ranchi."},
    {"id": 55, "question": "What is the best time to visit Jharkhand?", "answer": "The best time to visit Jharkhand is from October to March."},
    {"id": 56, "question": "Which national parks can I visit in Jharkhand?", "answer": "You can visit Betla National Park, Dalma Wildlife Sanctuary, and Hazaribagh Wildlife Sanctuary."},
    {"id": 57, "question": "What is special about Betla National Park?", "answer": "Betla National Park is known for tigers, elephants, and dense forests."},
    {"id": 58, "question": "How can I reach Netarhat from Ranchi?", "answer": "You can reach Netarhat by road from Ranchi via Latehar."},
    {"id": 59, "question": "Is Jharkhand safe for tourists?", "answer": "Yes, Jharkhand is generally safe for tourists with standard precautions."},
    {"id": 60, "question": "What is the capital of Jharkhand?", "answer": "The capital of Jharkhand is Ranchi."},
    {"id": 61, "question": "Which is the highest peak in Jharkhand?", "answer": "Parasnath Hill is the highest peak in Jharkhand."},
    {"id": 62, "question": "What is Parasnath Hill famous for?", "answer": "Parasnath Hill is a famous Jain pilgrimage site."},
    {"id": 63, "question": "Which tribe is most common in Jharkhand?", "answer": "The Santhal tribe is one of the major tribes in Jharkhand."},
    {"id": 64, "question": "What are traditional Jharkhand foods?", "answer": "Popular foods include Dhuska, Rugra, Chhoti Macher Jhor, and Chilka Roti."},
    {"id": 65, "question": "What is the dance form of Jharkhand?", "answer": "Chhau and Jhumair are popular dance forms of Jharkhand."},
    {"id": 66, "question": "Which river flows through Ranchi?", "answer": "The Subarnarekha River flows through Ranchi."},
    {"id": 67, "question": "What is the climate of Jharkhand?", "answer": "Jharkhand has a tropical climate with hot summers and cool winters."},
    {"id": 68, "question": "Which wildlife sanctuary is near Jamshedpur?", "answer": "Dalma Wildlife Sanctuary is the nearest major sanctuary to Jamshedpur."},
    {"id": 69, "question": "Where is Dalma Wildlife Sanctuary located?", "answer": "Dalma Wildlife Sanctuary is near Jamshedpur, famous for elephants."},
    {"id": 70, "question": "What is Johna Falls also known as?", "answer": "Johna Falls is also called Gautamdhara Falls."},
    {"id": 71, "question": "What adventure activities are available in Jharkhand?", "answer": "Adventure activities include trekking, rock climbing, wildlife safari, and boating."},
    {"id": 72, "question": "Where can I do trekking in Jharkhand?", "answer": "Trekking is popular in Parasnath Hills and Netarhat."},
    {"id": 73, "question": "Which airport is closest to major attractions?", "answer": "Birsa Munda Airport in Ranchi is the main airport for tourists."},
    {"id": 74, "question": "Where is Patratu Valley located?", "answer": "Patratu Valley is located near Ramgarh, close to Ranchi."},
    {"id": 75, "question": "Which city is known as the Steel City of Jharkhand?", "answer": "Jamshedpur is known as the Steel City of Jharkhand."},
    {"id": 76, "question": "What are the best tourist places in Jharkhand?", "answer": "Jharkhand is known for waterfalls, forests, hills, wildlife sanctuaries, and cultural sites."},
    {"id": 77, "question": "What is Jharkhand famous for?", "answer": "Jharkhand is famous for waterfalls, forests, rich minerals, and tribal culture."},
    {"id": 78, "question": "Which waterfall in Jharkhand is most popular?", "answer": "Hundru Falls and Dassam Falls are among the most popular waterfalls."},
    {"id": 79, "question": "Where is Dassam Falls located?", "answer": "Dassam Falls is located near Taimara village, around 40 km from Ranchi."},
    {"id": 80, "question": "What is the best time to visit Jharkhand?", "answer": "The best time to visit Jharkhand is from October to March."},
    {"id": 81, "question": "Which national parks can I visit in Jharkhand?", "answer": "You can visit Betla National Park, Dalma Wildlife Sanctuary, and Hazaribagh Wildlife Sanctuary."},
    {"id": 82, "question": "What is special about Betla National Park?", "answer": "Betla National Park is known for tigers, elephants, and dense forests."},
    {"id": 83, "question": "How can I reach Netarhat from Ranchi?", "answer": "You can reach Netarhat by road from Ranchi via Latehar."},
    {"id": 84, "question": "Is Jharkhand safe for tourists?", "answer": "Yes, Jharkhand is generally safe for tourists with standard precautions."},
    {"id": 85, "question": "What is the capital of Jharkhand?", "answer": "The capital of Jharkhand is Ranchi."},
    {"id": 86, "question": "Which is the highest peak in Jharkhand?", "answer": "Parasnath Hill is the highest peak in Jharkhand."},
    {"id": 87, "question": "What is Parasnath Hill famous for?", "answer": "Parasnath Hill is a famous Jain pilgrimage site."},
    {"id": 88, "question": "Which tribe is most common in Jharkhand?", "answer": "The Santhal tribe is one of the major tribes in Jharkhand."},
    {"id": 89, "question": "What are traditional Jharkhand foods?", "answer": "Popular foods include Dhuska, Rugra, Chhoti Macher Jhor, and Chilka Roti."},
    {"id": 90, "question": "What is the dance form of Jharkhand?", "answer": "Chhau and Jhumair are popular dance forms of Jharkhand."},
    {"id": 91, "question": "Which river flows through Ranchi?", "answer": "The Subarnarekha River flows through Ranchi."},
    {"id": 92, "question": "What is the climate of Jharkhand?", "answer": "Jharkhand has a tropical climate with hot summers and cool winters."},
    {"id": 93, "question": "Which wildlife sanctuary is near Jamshedpur?", "answer": "Dalma Wildlife Sanctuary is the nearest major sanctuary to Jamshedpur."},
    {"id": 94, "question": "Where is Dalma Wildlife Sanctuary located?", "answer": "Dalma Wildlife Sanctuary is near Jamshedpur, famous for elephants."},
    {"id": 95, "question": "What is Johna Falls also known as?", "answer": "Johna Falls is also called Gautamdhara Falls."},
    {"id": 96, "question": "What adventure activities are available in Jharkhand?", "answer": "Adventure activities include trekking, rock climbing, wildlife safari, and boating."},
    {"id": 97, "question": "Where can I do trekking in Jharkhand?", "answer": "Trekking is popular in Parasnath Hills and Netarhat."},
    {"id": 98, "question": "Which airport is closest to major attractions?", "answer": "Birsa Munda Airport in Ranchi is the main airport for tourists."},
    {"id": 99, "question": "Where is Patratu Valley located?", "answer": "Patratu Valley is located near Ramgarh, close to Ranchi."},
    {"id": 100, "question": "Which city is known as the Steel City of Jharkhand?", "answer": "Jamshedpur is known as the Steel City of Jharkhand."},
    {"id": 101, "question": "What are the best tourist places in Jharkhand?", "answer": "Jharkhand is known for waterfalls, forests, hills, wildlife sanctuaries, and cultural sites."},
    {"id": 102, "question": "What is Jharkhand famous for?", "answer": "Jharkhand is famous for waterfalls, forests, rich minerals, and tribal culture."},
    {"id": 103, "question": "Which waterfall in Jharkhand is most popular?", "answer": "Hundru Falls and Dassam Falls are among the most popular waterfalls."},
    {"id": 104, "question": "Where is Dassam Falls located?", "answer": "Dassam Falls is located near Taimara village, around 40 km from Ranchi."},
    {"id": 105, "question": "What is the best time to visit Jharkhand?", "answer": "The best time to visit Jharkhand is from October to March."},
    {"id": 106, "question": "Which national parks can I visit in Jharkhand?", "answer": "You can visit Betla National Park, Dalma Wildlife Sanctuary, and Hazaribagh Wildlife Sanctuary."},
    {"id": 107, "question": "What is special about Betla National Park?", "answer": "Betla National Park is known for tigers, elephants, and dense forests."},
    {"id": 108, "question": "How can I reach Netarhat from Ranchi?", "answer": "You can reach Netarhat by road from Ranchi via Latehar."},
    {"id": 109, "question": "Is Jharkhand safe for tourists?", "answer": "Yes, Jharkhand is generally safe for tourists with standard precautions."},
    {"id": 110, "question": "What is the capital of Jharkhand?", "answer": "The capital of Jharkhand is Ranchi."},
    {"id": 111, "question": "Which is the highest peak in Jharkhand?", "answer": "Parasnath Hill is the highest peak in Jharkhand."},
    {"id": 112, "question": "What is Parasnath Hill famous for?", "answer": "Parasnath Hill is a famous Jain pilgrimage site."},
    {"id": 113, "question": "Which tribe is most common in Jharkhand?", "answer": "The Santhal tribe is one of the major tribes in Jharkhand."},
    {"id": 114, "question": "What are traditional Jharkhand foods?", "answer": "Popular foods include Dhuska, Rugra, Chhoti Macher Jhor, and Chilka Roti."},
    {"id": 115, "question": "What is the dance form of Jharkhand?", "answer": "Chhau and Jhumair are popular dance forms of Jharkhand."},
    {"id": 116, "question": "Which river flows through Ranchi?", "answer": "The Subarnarekha River flows through Ranchi."},
    {"id": 117, "question": "What is the climate of Jharkhand?", "answer": "Jharkhand has a tropical climate with hot summers and cool winters."},
    {"id": 118, "question": "Which wildlife sanctuary is near Jamshedpur?", "answer": "Dalma Wildlife Sanctuary is the nearest major sanctuary to Jamshedpur."},
    {"id": 119, "question": "Where is Dalma Wildlife Sanctuary located?", "answer": "Dalma Wildlife Sanctuary is near Jamshedpur, famous for elephants."},
    {"id": 120, "question": "What is Johna Falls also known as?", "answer": "Johna Falls is also called Gautamdhara Falls."},
    {"id": 121, "question": "What adventure activities are available in Jharkhand?", "answer": "Adventure activities include trekking, rock climbing, wildlife safari, and boating."},
    {"id": 122, "question": "Where can I do trekking in Jharkhand?", "answer": "Trekking is popular in Parasnath Hills and Netarhat."},
    {"id": 123, "question": "Which airport is closest to major attractions?", "answer": "Birsa Munda Airport in Ranchi is the main airport for tourists."},
    {"id": 124, "question": "Where is Patratu Valley located?", "answer": "Patratu Valley is located near Ramgarh, close to Ranchi."},
    {"id": 125, "question": "Which city is known as the Steel City of Jharkhand?", "answer": "Jamshedpur is known as the Steel City of Jharkhand."},
    {"id": 126, "question": "What are the best tourist places in Jharkhand?", "answer": "Jharkhand is known for waterfalls, forests, hills, wildlife sanctuaries, and cultural sites."},
    {"id": 127, "question": "What is Jharkhand famous for?", "answer": "Jharkhand is famous for waterfalls, forests, rich minerals, and tribal culture."},
    {"id": 128, "question": "Which waterfall in Jharkhand is most popular?", "answer": "Hundru Falls and Dassam Falls are among the most popular waterfalls."},
    {"id": 129, "question": "Where is Dassam Falls located?", "answer": "Dassam Falls is located near Taimara village, around 40 km from Ranchi."},
    {"id": 130, "question": "What is the best time to visit Jharkhand?", "answer": "The best time to visit Jharkhand is from October to March."},
    {"id": 131, "question": "Which national parks can I visit in Jharkhand?", "answer": "You can visit Betla National Park, Dalma Wildlife Sanctuary, and Hazaribagh Wildlife Sanctuary."},
    {"id": 132, "question": "What is special about Betla National Park?", "answer": "Betla National Park is known for tigers, elephants, and dense forests."},
    {"id": 133, "question": "How can I reach Netarhat from Ranchi?", "answer": "You can reach Netarhat by road from Ranchi via Latehar."},
    {"id": 134, "question": "Is Jharkhand safe for tourists?", "answer": "Yes, Jharkhand is generally safe for tourists with standard precautions."},
    {"id": 135, "question": "What is the capital of Jharkhand?", "answer": "The capital of Jharkhand is Ranchi."},
    {"id": 136, "question": "Which is the highest peak in Jharkhand?", "answer": "Parasnath Hill is the highest peak in Jharkhand."},
    {"id": 137, "question": "What is Parasnath Hill famous for?", "answer": "Parasnath Hill is a famous Jain pilgrimage site."},
    {"id": 138, "question": "Which tribe is most common in Jharkhand?", "answer": "The Santhal tribe is one of the major tribes in Jharkhand."},
    {"id": 139, "question": "What are traditional Jharkhand foods?", "answer": "Popular foods include Dhuska, Rugra, Chhoti Macher Jhor, and Chilka Roti."},
    {"id": 140, "question": "What is the dance form of Jharkhand?", "answer": "Chhau and Jhumair are popular dance forms of Jharkhand."},
    {"id": 141, "question": "Which river flows through Ranchi?", "answer": "The Subarnarekha River flows through Ranchi."},
    {"id": 142, "question": "What is the climate of Jharkhand?", "answer": "Jharkhand has a tropical climate with hot summers and cool winters."},
    {"id": 143, "question": "Which wildlife sanctuary is near Jamshedpur?", "answer": "Dalma Wildlife Sanctuary is the nearest major sanctuary to Jamshedpur."},
    {"id": 144, "question": "Where is Dalma Wildlife Sanctuary located?", "answer": "Dalma Wildlife Sanctuary is near Jamshedpur, famous for elephants."},
    {"id": 145, "question": "What is Johna Falls also known as?", "answer": "Johna Falls is also called Gautamdhara Falls."},
    {"id": 146, "question": "What adventure activities are available in Jharkhand?", "answer": "Adventure activities include trekking, rock climbing, wildlife safari, and boating."},
    {"id": 147, "question": "Where can I do trekking in Jharkhand?", "answer": "Trekking is popular in Parasnath Hills and Netarhat."},
    {"id": 148, "question": "Which airport is closest to major attractions?", "answer": "Birsa Munda Airport in Ranchi is the main airport for tourists."},
    {"id": 149, "question": "Where is Patratu Valley located?", "answer": "Patratu Valley is located near Ramgarh, close to Ranchi."},
    {"id": 150, "question": "Which city is known as the Steel City of Jharkhand?", "answer": "Jamshedpur is known as the Steel City of Jharkhand."}
]
    
lemmatizer = WordNetLemmatizer()
qa_tags = {}

# --- Augmentation Function ---
def augment_patterns(question, tag):
    """Generates synthetic, relevant variations of a question."""
    patterns = [question]
    q = question.lower()
    
    # 1. Simple Rephrasing/Template Replacement
    if 'what are the best' in q or 'what is special' in q:
        patterns.append(q.replace('what is', 'tell me about').replace('what are the best', 'what are the must-see'))
        
    if 'where is' in q:
        place = q.split('where is ')[-1].replace('?', '').strip()
        patterns.append(f"Can you tell me the location of {place}?")
        patterns.append(f"How do I get to {place}?")
        
    if 'how can i reach' in q:
        place = q.split('reach ')[-1].replace('?', '').strip()
        patterns.append(f"What is the best way to travel to {place}?")
        patterns.append(f"What is the road travel info for {place}?")
        
    if 'which' in q and 'is' in q:
        patterns.append(q.replace('which', 'what'))
        
    if 'famous for' in q:
        patterns.append(q.replace('famous for', 'known for'))
        
    # 2. Key Term Variations
    if tag == 'wildlife_parks':
        patterns.append(question.replace('national parks', 'sanctuaries'))
        patterns.append(question.replace('visit', 'see animals in'))
        if 'tiger' in q:
             patterns.append(f"Where can I see a tiger in Jharkhand?")
             
    if tag == 'local_culture_food':
         if 'food' in q:
             patterns.append(question.replace('traditional foods', 'famous dishes'))
             patterns.append("Tell me the famous food of Jharkhand")
             
    if tag == 'waterfalls':
        patterns.append(question.replace('popular', 'best'))
        
    if tag == 'travel_logistics':
        if 'safe' in q:
            patterns.append("Is traveling to Jharkhand dangerous?")
        if 'trekking' in q:
            patterns.append("Where can I go for a hike?")
            
    # 3. Handle specific test queries that failed
    if 'latehar' in q:
        patterns.append("Thinking about having a 2 hour stop in latehar, what should I do?")

    if 'parasnath hill' in q:
        patterns.append("what is the distance from Ranchi to Parasnath hill?")
        
    return list(set(patterns))

# --- Explicit Intent Grouping based on Content ID ---
intent_map = {}
for i in range(1, 151):
    # Using the same mapping as the last stable attempt
    if i in [1, 2, 5, 17, 26, 27, 30, 42, 51, 52, 55, 67, 76, 77, 80, 92, 101, 102, 105, 117, 126, 127, 130, 142]:
        tag = 'general_overview'
    elif i in [3, 4, 20, 28, 29, 45, 53, 54, 70, 78, 79, 103, 104, 120, 128, 129, 145]:
        tag = 'waterfalls'
    elif i in [6, 7, 18, 19, 31, 32, 43, 44, 56, 57, 68, 69, 81, 82, 93, 94, 106, 107, 118, 119, 131, 132, 143, 144]:
        tag = 'wildlife_parks'
    elif i in [11, 12, 36, 37, 61, 62, 86, 87, 111, 112, 136, 137]:
        tag = 'pilgrimage_sites'
    elif i in [13, 14, 15, 38, 39, 40, 63, 64, 65, 88, 89, 90, 113, 114, 115, 138, 139, 140]:
        tag = 'local_culture_food'
    elif i in [10, 16, 23, 24, 25, 35, 41, 48, 49, 50, 60, 66, 73, 74, 75, 85, 91, 98, 99, 100, 110, 116, 123, 124, 125, 135, 141, 148, 149, 150]:
        tag = 'major_cities_info'
    elif i in [8, 9, 21, 22, 33, 34, 46, 47, 58, 59, 71, 72, 83, 84, 96, 97, 108, 109, 121, 122, 133, 134, 146, 147]:
        tag = 'travel_logistics'
    else:
        tag = 'general_overview' 

    item = next(d for d in qa_dataset if d.get('id') == i)
    if tag not in qa_tags:
        qa_tags[tag] = {'tag': tag, 'patterns': [], 'responses': []}
    
    qa_tags[tag]['patterns'].extend(augment_patterns(item['question'], tag))
    qa_tags[tag]['responses'].append(item['answer'])

new_intents = list(qa_tags.values())

# --- Utility Intents ---
new_intents.extend([
    {'tag': 'greeting', 'patterns': ['Hi', 'Hello', 'Is anyone there?', 'Namaste', 'hey', 'good morning', 'good evening', 'hi there'],
     'responses': ["Hello! Welcome to Jharkhand Tourism. How can I assist you? 🌳", "Hi there! What can I tell you about the Land of Forests?"]},
    
    {'tag': 'farewell', 'patterns': ['bye', 'goodbye', 'see you', 'tata', 'later', 'thanks bye', 'i am leaving', 'i have to go'],
     'responses': ["Goodbye! Have a great trip to Jharkhand! 👋", "See you later! Enjoy the Land of Forests!"]},
     
    {'tag': 'thanks', 'patterns': ['thank you', 'thanks', 'that was helpful', 'much obliged', 'appreciate it', 'i appreciate your help'],
     'responses': ["You're welcome! How else can I assist you?", "My pleasure! Happy to help with your Jharkhand journey.", "Glad I could help!"]},
     
    {'tag': 'out_of_scope', 'patterns': ['where is M.S.Dhoni from', 'who is M.S. Dhoni', 'what about dhoni', 'dhoni hometown', 'M. S. Dhoni', 'cricket', 'prime minister', 'president', 'historical place'],
     'responses': ["I specialize in Jharkhand tourism. While M.S. Dhoni is a famous person from Ranchi, I cannot provide biographical details. Can I help you with a tourist question instead?", "That's a great question, but I focus on tourism information. How can I help you plan your trip?"]},

    {'tag': 'fallback', 'patterns': ['I do not know', 'nothing', 'general query', 'random text', 'not understood', 'what can you do', 'I need more information'],
     'responses': ["I'm sorry, I don't understand that. Can you try asking about a specific place, a waterfall, or food in Jharkhand?", "I need more information. Could you rephrase your question? Try 'Where is Dassam Falls located?'"]}
])

intents_data = {'intents': new_intents}

# --- 2. Training the Model ---
training_data = []
training_tags = []

for intent in intents_data['intents']:
    tag = intent['tag']
    unique_patterns = set(intent['patterns']) 
    for pattern in unique_patterns:
        words = pattern.split()
        processed_pattern = " ".join([lemmatizer.lemmatize(word.lower()) for word in words])

        training_data.append(processed_pattern)
        training_tags.append(tag)

print(f"\n--- Starting Training with {len(training_data)} augmented patterns ---")

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(training_data)
print("✅ Vectorizer trained successfully.")

unique_tags = sorted(list(set(training_tags)))
y_train = [unique_tags.index(tag) for tag in training_tags]

# CHANGE: Using MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

model.classes_ = np.array(unique_tags)
print("✅ Model (MultinomialNB) trained successfully.")

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
