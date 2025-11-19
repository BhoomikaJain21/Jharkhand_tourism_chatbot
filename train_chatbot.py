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
# NLTK resources are assumed to be available or downloaded in the environment
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True)
print("NLTK downloads complete.")

# --- 1. Load and Transform Data ---
try:
    # Load the QA dataset from the provided file (assuming it's available in the environment)
    # Using the content provided in the user's last turn for reliability.
    qa_dataset = [
    {
        "id": 1,
        "question": "What are the best tourist places in Jharkhand?",
        "answer": "Jharkhand is known for waterfalls, forests, hills, wildlife sanctuaries, and cultural sites."
    },
    {
        "id": 2,
        "question": "What is Jharkhand famous for?",
        "answer": "Jharkhand is famous for waterfalls, forests, rich minerals, and tribal culture."
    },
    {
        "id": 3,
        "question": "Which waterfall in Jharkhand is most popular?",
        "answer": "Hundru Falls and Dassam Falls are among the most popular waterfalls."
    },
    {
        "id": 4,
        "question": "Where is Dassam Falls located?",
        "answer": "Dassam Falls is located near Taimara village, around 40 km from Ranchi."
    },
    {
        "id": 5,
        "question": "What is the best time to visit Jharkhand?",
        "answer": "The best time to visit Jharkhand is from October to March."
    },
    {
        "id": 6,
        "question": "Which national parks can I visit in Jharkhand?",
        "answer": "You can visit Betla National Park, Dalma Wildlife Sanctuary, and Hazaribagh Wildlife Sanctuary."
    },
    {
        "id": 7,
        "question": "What is special about Betla National Park?",
        "answer": "Betla National Park is known for tigers, elephants, and dense forests."
    },
    {
        "id": 8,
        "question": "How can I reach Netarhat from Ranchi?",
        "answer": "You can reach Netarhat by road from Ranchi via Latehar."
    },
    {
        "id": 9,
        "question": "Is Jharkhand safe for tourists?",
        "answer": "Yes, Jharkhand is generally safe for tourists with standard precautions."
    },
    {
        "id": 10,
        "question": "What is the capital of Jharkhand?",
        "answer": "The capital of Jharkhand is Ranchi."
    },
    {
        "id": 11,
        "question": "Which is the highest peak in Jharkhand?",
        "answer": "Parasnath Hill is the highest peak in Jharkhand."
    },
    {
        "id": 12,
        "question": "What is Parasnath Hill famous for?",
        "answer": "Parasnath Hill is a famous Jain pilgrimage site."
    },
    {
        "id": 13,
        "question": "Which tribe is most common in Jharkhand?",
        "answer": "The Santhal tribe is one of the major tribes in Jharkhand."
    },
    {
        "id": 14,
        "question": "What are traditional Jharkhand foods?",
        "answer": "Popular foods include Dhuska, Rugra, Chhoti Macher Jhor, and Chilka Roti."
    },
    {
        "id": 15,
        "question": "What is the dance form of Jharkhand?",
        "answer": "Chhau and Jhumair are popular dance forms of Jharkhand."
    },
    {
        "id": 16,
        "question": "Which river flows through Ranchi?",
        "answer": "The Subarnarekha River flows through Ranchi."
    },
    {
        "id": 17,
        "question": "What is the climate of Jharkhand?",
        "answer": "Jharkhand has a tropical climate with hot summers and cool winters."
    },
    {
        "id": 18,
        "question": "Which wildlife sanctuary is near Jamshedpur?",
        "answer": "Dalma Wildlife Sanctuary is the nearest major sanctuary to Jamshedpur."
    },
    {
        "id": 19,
        "question": "Where is Dalma Wildlife Sanctuary located?",
        "answer": "Dalma Wildlife Sanctuary is near Jamshedpur, famous for elephants."
    },
    {
        "id": 20,
        "question": "What is Johna Falls also known as?",
        "answer": "Johna Falls is also called Gautamdhara Falls."
    },
    {
        "id": 21,
        "question": "What adventure activities are available in Jharkhand?",
        "answer": "Adventure activities include trekking, rock climbing, wildlife safari, and boating."
    },
    {
        "id": 22,
        "question": "Where can I do trekking in Jharkhand?",
        "answer": "Trekking is popular in Parasnath Hills and Netarhat."
    },
    {
        "id": 23,
        "question": "Which airport is closest to major attractions?",
        "answer": "Birsa Munda Airport in Ranchi is the main airport for tourists."
    },
    {
        "id": 24,
        "question": "Where is Patratu Valley located?",
        "answer": "Patratu Valley is located near Ramgarh, close to Ranchi."
    },
    {
        "id": 25,
        "question": "Which city is known as the Steel City of Jharkhand?",
        "answer": "Jamshedpur is known as the Steel City of Jharkhand."
    },
    {
        "id": 26,
        "question": "What are the best tourist places in Jharkhand?",
        "answer": "Jharkhand is known for waterfalls, forests, hills, wildlife sanctuaries, and cultural sites."
    },
    {
        "id": 27,
        "question": "What is Jharkhand famous for?",
        "answer": "Jharkhand is famous for waterfalls, forests, rich minerals, and tribal culture."
    },
    {
        "id": 28,
        "question": "Which waterfall in Jharkhand is most popular?",
        "answer": "Hundru Falls and Dassam Falls are among the most popular waterfalls."
    },
    {
        "id": 29,
        "question": "Where is Dassam Falls located?",
        "answer": "Dassam Falls is located near Taimara village, around 40 km from Ranchi."
    },
    {
        "id": 30,
        "question": "What is the best time to visit Jharkhand?",
        "answer": "The best time to visit Jharkhand is from October to March."
    },
    {
        "id": 31,
        "question": "Which national parks can I visit in Jharkhand?",
        "answer": "You can visit Betla National Park, Dalma Wildlife Sanctuary, and Hazaribagh Wildlife Sanctuary."
    },
    {
        "id": 32,
        "question": "What is special about Betla National Park?",
        "answer": "Betla National Park is known for tigers, elephants, and dense forests."
    },
    {
        "id": 33,
        "question": "How can I reach Netarhat from Ranchi?",
        "answer": "You can reach Netarhat by road from Ranchi via Latehar."
    },
    {
        "id": 34,
        "question": "Is Jharkhand safe for tourists?",
        "answer": "Yes, Jharkhand is generally safe for tourists with standard precautions."
    },
    {
        "id": 35,
        "question": "What is the capital of Jharkhand?",
        "answer": "The capital of Jharkhand is Ranchi."
    },
    {
        "id": 36,
        "question": "Which is the highest peak in Jharkhand?",
        "answer": "Parasnath Hill is the highest peak in Jharkhand."
    },
    {
        "id": 37,
        "question": "What is Parasnath Hill famous for?",
        "answer": "Parasnath Hill is a famous Jain pilgrimage site."
    },
    {
        "id": 38,
        "question": "Which tribe is most common in Jharkhand?",
        "answer": "The Santhal tribe is one of the major tribes in Jharkhand."
    },
    {
        "id": 39,
        "question": "What are traditional Jharkhand foods?",
        "answer": "Popular foods include Dhuska, Rugra, Chhoti Macher Jhor, and Chilka Roti."
    },
    {
        "id": 40,
        "question": "What is the dance form of Jharkhand?",
        "answer": "Chhau and Jhumair are popular dance forms of Jharkhand."
    },
    {
        "id": 41,
        "question": "Which river flows through Ranchi?",
        "answer": "The Subarnarekha River flows through Ranchi."
    },
    {
        "id": 42,
        "question": "What is the climate of Jharkhand?",
        "answer": "Jharkhand has a tropical climate with hot summers and cool winters."
    },
    {
        "id": 43,
        "question": "Which wildlife sanctuary is near Jamshedpur?",
        "answer": "Dalma Wildlife Sanctuary is the nearest major sanctuary to Jamshedpur."
    },
    {
        "id": 44,
        "question": "Where is Dalma Wildlife Sanctuary located?",
        "answer": "Dalma Wildlife Sanctuary is near Jamshedpur, famous for elephants."
    },
    {
        "id": 45,
        "question": "What is Johna Falls also known as?",
        "answer": "Johna Falls is also called Gautamdhara Falls."
    },
    {
        "id": 46,
        "question": "What adventure activities are available in Jharkhand?",
        "answer": "Adventure activities include trekking, rock climbing, wildlife safari, and boating."
    },
    {
        "id": 47,
        "question": "Where can I do trekking in Jharkhand?",
        "answer": "Trekking is popular in Parasnath Hills and Netarhat."
    },
    {
        "id": 48,
        "question": "Which airport is closest to major attractions?",
        "answer": "Birsa Munda Airport in Ranchi is the main airport for tourists."
    },
    {
        "id": 49,
        "question": "Where is Patratu Valley located?",
        "answer": "Patratu Valley is located near Ramgarh, close to Ranchi."
    },
    {
        "id": 50,
        "question": "Which city is known as the Steel City of Jharkhand?",
        "answer": "Jamshedpur is known as the Steel City of Jharkhand."
    },
    {
        "id": 51,
        "question": "What are the best tourist places in Jharkhand?",
        "answer": "Jharkhand is known for waterfalls, forests, hills, wildlife sanctuaries, and cultural sites."
    },
    {
        "id": 52,
        "question": "What is Jharkhand famous for?",
        "answer": "Jharkhand is famous for waterfalls, forests, rich minerals, and tribal culture."
    },
    {
        "id": 53,
        "question": "Which waterfall in Jharkhand is most popular?",
        "answer": "Hundru Falls and Dassam Falls are among the most popular waterfalls."
    },
    {
        "id": 54,
        "question": "Where is Dassam Falls located?",
        "answer": "Dassam Falls is located near Taimara village, around 40 km from Ranchi."
    },
    {
        "id": 55,
        "question": "What is the best time to visit Jharkhand?",
        "answer": "The best time to visit Jharkhand is from October to March."
    },
    {
        "id": 56,
        "question": "Which national parks can I visit in Jharkhand?",
        "answer": "You can visit Betla National Park, Dalma Wildlife Sanctuary, and Hazaribagh Wildlife Sanctuary."
    },
    {
        "id": 57,
        "question": "What is special about Betla National Park?",
        "answer": "Betla National Park is known for tigers, elephants, and dense forests."
    },
    {
        "id": 58,
        "question": "How can I reach Netarhat from Ranchi?",
        "answer": "You can reach Netarhat by road from Ranchi via Latehar."
    },
    {
        "id": 59,
        "question": "Is Jharkhand safe for tourists?",
        "answer": "Yes, Jharkhand is generally safe for tourists with standard precautions."
    },
    {
        "id": 60,
        "question": "What is the capital of Jharkhand?",
        "answer": "The capital of Jharkhand is Ranchi."
    },
    {
        "id": 61,
        "question": "Which is the highest peak in Jharkhand?",
        "answer": "Parasnath Hill is the highest peak in Jharkhand."
    },
    {
        "id": 62,
        "question": "What is Parasnath Hill famous for?",
        "answer": "Parasnath Hill is a famous Jain pilgrimage site."
    },
    {
        "id": 63,
        "question": "Which tribe is most common in Jharkhand?",
        "answer": "The Santhal tribe is one of the major tribes in Jharkhand."
    },
    {
        "id": 64,
        "question": "What are traditional Jharkhand foods?",
        "answer": "Popular foods include Dhuska, Rugra, Chhoti Macher Jhor, and Chilka Roti."
    },
    {
        "id": 65,
        "question": "What is the dance form of Jharkhand?",
        "answer": "Chhau and Jhumair are popular dance forms of Jharkhand."
    },
    {
        "id": 66,
        "question": "Which river flows through Ranchi?",
        "answer": "The Subarnarekha River flows through Ranchi."
    },
    {
        "id": 67,
        "question": "What is the climate of Jharkhand?",
        "answer": "Jharkhand has a tropical climate with hot summers and cool winters."
    },
    {
        "id": 68,
        "question": "Which wildlife sanctuary is near Jamshedpur?",
        "answer": "Dalma Wildlife Sanctuary is the nearest major sanctuary to Jamshedpur."
    },
    {
        "id": 69,
        "question": "Where is Dalma Wildlife Sanctuary located?",
        "answer": "Dalma Wildlife Sanctuary is near Jamshedpur, famous for elephants."
    },
    {
        "id": 70,
        "question": "What is Johna Falls also known as?",
        "answer": "Johna Falls is also called Gautamdhara Falls."
    },
    {
        "id": 71,
        "question": "What adventure activities are available in Jharkhand?",
        "answer": "Adventure activities include trekking, rock climbing, wildlife safari, and boating."
    },
    {
        "id": 72,
        "question": "Where can I do trekking in Jharkhand?",
        "answer": "Trekking is popular in Parasnath Hills and Netarhat."
    },
    {
        "id": 73,
        "question": "Which airport is closest to major attractions?",
        "answer": "Birsa Munda Airport in Ranchi is the main airport for tourists."
    },
    {
        "id": 74,
        "question": "Where is Patratu Valley located?",
        "answer": "Patratu Valley is located near Ramgarh, close to Ranchi."
    },
    {
        "id": 75,
        "question": "Which city is known as the Steel City of Jharkhand?",
        "answer": "Jamshedpur is known as the Steel City of Jharkhand."
    },
    {
        "id": 76,
        "question": "What are the best tourist places in Jharkhand?",
        "answer": "Jharkhand is known for waterfalls, forests, hills, wildlife sanctuaries, and cultural sites."
    },
    {
        "id": 77,
        "question": "What is Jharkhand famous for?",
        "answer": "Jharkhand is famous for waterfalls, forests, rich minerals, and tribal culture."
    },
    {
        "id": 78,
        "question": "Which waterfall in Jharkhand is most popular?",
        "answer": "Hundru Falls and Dassam Falls are among the most popular waterfalls."
    },
    {
        "id": 79,
        "question": "Where is Dassam Falls located?",
        "answer": "Dassam Falls is located near Taimara village, around 40 km from Ranchi."
    },
    {
        "id": 80,
        "question": "What is the best time to visit Jharkhand?",
        "answer": "The best time to visit Jharkhand is from October to March."
    },
    {
        "id": 81,
        "question": "Which national parks can I visit in Jharkhand?",
        "answer": "You can visit Betla National Park, Dalma Wildlife Sanctuary, and Hazaribagh Wildlife Sanctuary."
    },
    {
        "id": 82,
        "question": "What is special about Betla National Park?",
        "answer": "Betla National Park is known for tigers, elephants, and dense forests."
    },
    {
        "id": 83,
        "question": "How can I reach Netarhat from Ranchi?",
        "answer": "You can reach Netarhat by road from Ranchi via Latehar."
    },
    {
        "id": 84,
        "question": "Is Jharkhand safe for tourists?",
        "answer": "Yes, Jharkhand is generally safe for tourists with standard precautions."
    },
    {
        "id": 85,
        "question": "What is the capital of Jharkhand?",
        "answer": "The capital of Jharkhand is Ranchi."
    },
    {
        "id": 86,
        "question": "Which is the highest peak in Jharkhand?",
        "answer": "Parasnath Hill is the highest peak in Jharkhand."
    },
    {
        "id": 87,
        "question": "What is Parasnath Hill famous for?",
        "answer": "Parasnath Hill is a famous Jain pilgrimage site."
    },
    {
        "id": 88,
        "question": "Which tribe is most common in Jharkhand?",
        "answer": "The Santhal tribe is one of the major tribes in Jharkhand."
    },
    {
        "id": 89,
        "question": "What are traditional Jharkhand foods?",
        "answer": "Popular foods include Dhuska, Rugra, Chhoti Macher Jhor, and Chilka Roti."
    },
    {
        "id": 90,
        "question": "What is the dance form of Jharkhand?",
        "answer": "Chhau and Jhumair are popular dance forms of Jharkhand."
    },
    {
        "id": 91,
        "question": "Which river flows through Ranchi?",
        "answer": "The Subarnarekha River flows through Ranchi."
    },
    {
        "id": 92,
        "question": "What is the climate of Jharkhand?",
        "answer": "Jharkhand has a tropical climate with hot summers and cool winters."
    },
    {
        "id": 93,
        "question": "Which wildlife sanctuary is near Jamshedpur?",
        "answer": "Dalma Wildlife Sanctuary is the nearest major sanctuary to Jamshedpur."
    },
    {
        "id": 94,
        "question": "Where is Dalma Wildlife Sanctuary located?",
        "answer": "Dalma Wildlife Sanctuary is near Jamshedpur, famous for elephants."
    },
    {
        "id": 95,
        "question": "What is Johna Falls also known as?",
        "answer": "Johna Falls is also called Gautamdhara Falls."
    },
    {
        "id": 96,
        "question": "What adventure activities are available in Jharkhand?",
        "answer": "Adventure activities include trekking, rock climbing, wildlife safari, and boating."
    },
    {
        "id": 97,
        "question": "Where can I do trekking in Jharkhand?",
        "answer": "Trekking is popular in Parasnath Hills and Netarhat."
    },
    {
        "id": 98,
        "question": "Which airport is closest to major attractions?",
        "answer": "Birsa Munda Airport in Ranchi is the main airport for tourists."
    },
    {
        "id": 99,
        "question": "Where is Patratu Valley located?",
        "answer": "Patratu Valley is located near Ramgarh, close to Ranchi."
    },
    {
        "id": 100,
        "question": "Which city is known as the Steel City of Jharkhand?",
        "answer": "Jamshedpur is known as the Steel City of Jharkhand."
    },
    {
        "id": 101,
        "question": "What are the best tourist places in Jharkhand?",
        "answer": "Jharkhand is known for waterfalls, forests, hills, wildlife sanctuaries, and cultural sites."
    },
    {
        "id": 102,
        "question": "What is Jharkhand famous for?",
        "answer": "Jharkhand is famous for waterfalls, forests, rich minerals, and tribal culture."
    },
    {
        "id": 103,
        "question": "Which waterfall in Jharkhand is most popular?",
        "answer": "Hundru Falls and Dassam Falls are among the most popular waterfalls."
    },
    {
        "id": 104,
        "question": "Where is Dassam Falls located?",
        "answer": "Dassam Falls is located near Taimara village, around 40 km from Ranchi."
    },
    {
        "id": 105,
        "question": "What is the best time to visit Jharkhand?",
        "answer": "The best time to visit Jharkhand is from October to March."
    },
    {
        "id": 106,
        "question": "Which national parks can I visit in Jharkhand?",
        "answer": "You can visit Betla National Park, Dalma Wildlife Sanctuary, and Hazaribagh Wildlife Sanctuary."
    },
    {
        "id": 107,
        "question": "What is special about Betla National Park?",
        "answer": "Betla National Park is known for tigers, elephants, and dense forests."
    },
    {
        "id": 108,
        "question": "How can I reach Netarhat from Ranchi?",
        "answer": "You can reach Netarhat by road from Ranchi via Latehar."
    },
    {
        "id": 109,
        "question": "Is Jharkhand safe for tourists?",
        "answer": "Yes, Jharkhand is generally safe for tourists with standard precautions."
    },
    {
        "id": 110,
        "question": "What is the capital of Jharkhand?",
        "answer": "The capital of Jharkhand is Ranchi."
    },
    {
        "id": 111,
        "question": "Which is the highest peak in Jharkhand?",
        "answer": "Parasnath Hill is the highest peak in Jharkhand."
    },
    {
        "id": 112,
        "question": "What is Parasnath Hill famous for?",
        "answer": "Parasnath Hill is a famous Jain pilgrimage site."
    },
    {
        "id": 113,
        "question": "Which tribe is most common in Jharkhand?",
        "answer": "The Santhal tribe is one of the major tribes in Jharkhand."
    },
    {
        "id": 114,
        "question": "What are traditional Jharkhand foods?",
        "answer": "Popular foods include Dhuska, Rugra, Chhoti Macher Jhor, and Chilka Roti."
    },
    {
        "id": 115,
        "question": "What is the dance form of Jharkhand?",
        "answer": "Chhau and Jhumair are popular dance forms of Jharkhand."
    },
    {
        "id": 116,
        "question": "Which river flows through Ranchi?",
        "answer": "The Subarnarekha River flows through Ranchi."
    },
    {
        "id": 117,
        "question": "What is the climate of Jharkhand?",
        "answer": "Jharkhand has a tropical climate with hot summers and cool winters."
    },
    {
        "id": 118,
        "question": "Which wildlife sanctuary is near Jamshedpur?",
        "answer": "Dalma Wildlife Sanctuary is the nearest major sanctuary to Jamshedpur."
    },
    {
        "id": 119,
        "question": "Where is Dalma Wildlife Sanctuary located?",
        "answer": "Dalma Wildlife Sanctuary is near Jamshedpur, famous for elephants."
    },
    {
        "id": 120,
        "question": "What is Johna Falls also known as?",
        "answer": "Johna Falls is also called Gautamdhara Falls."
    },
    {
        "id": 121,
        "question": "What adventure activities are available in Jharkhand?",
        "answer": "Adventure activities include trekking, rock climbing, wildlife safari, and boating."
    },
    {
        "id": 122,
        "question": "Where can I do trekking in Jharkhand?",
        "answer": "Trekking is popular in Parasnath Hills and Netarhat."
    },
    {
        "id": 123,
        "question": "Which airport is closest to major attractions?",
        "answer": "Birsa Munda Airport in Ranchi is the main airport for tourists."
    },
    {
        "id": 124,
        "question": "Where is Patratu Valley located?",
        "answer": "Patratu Valley is located near Ramgarh, close to Ranchi."
    },
    {
        "id": 125,
        "question": "Which city is known as the Steel City of Jharkhand?",
        "answer": "Jamshedpur is known as the Steel City of Jharkhand."
    },
    {
        "id": 126,
        "question": "What are the best tourist places in Jharkhand?",
        "answer": "Jharkhand is known for waterfalls, forests, hills, wildlife sanctuaries, and cultural sites."
    },
    {
        "id": 127,
        "question": "What is Jharkhand famous for?",
        "answer": "Jharkhand is famous for waterfalls, forests, rich minerals, and tribal culture."
    },
    {
        "id": 128,
        "question": "Which waterfall in Jharkhand is most popular?",
        "answer": "Hundru Falls and Dassam Falls are among the most popular waterfalls."
    },
    {
        "id": 129,
        "question": "Where is Dassam Falls located?",
        "answer": "Dassam Falls is located near Taimara village, around 40 km from Ranchi."
    },
    {
        "id": 130,
        "question": "What is the best time to visit Jharkhand?",
        "answer": "The best time to visit Jharkhand is from October to March."
    },
    {
        "id": 131,
        "question": "Which national parks can I visit in Jharkhand?",
        "answer": "You can visit Betla National Park, Dalma Wildlife Sanctuary, and Hazaribagh Wildlife Sanctuary."
    },
    {
        "id": 132,
        "question": "What is special about Betla National Park?",
        "answer": "Betla National Park is known for tigers, elephants, and dense forests."
    },
    {
        "id": 133,
        "question": "How can I reach Netarhat from Ranchi?",
        "answer": "You can reach Netarhat by road from Ranchi via Latehar."
    },
    {
        "id": 134,
        "question": "Is Jharkhand safe for tourists?",
        "answer": "Yes, Jharkhand is generally safe for tourists with standard precautions."
    },
    {
        "id": 135,
        "question": "What is the capital of Jharkhand?",
        "answer": "The capital of Jharkhand is Ranchi."
    },
    {
        "id": 136,
        "question": "Which is the highest peak in Jharkhand?",
        "answer": "Parasnath Hill is the highest peak in Jharkhand."
    },
    {
        "id": 137,
        "question": "What is Parasnath Hill famous for?",
        "answer": "Parasnath Hill is a famous Jain pilgrimage site."
    },
    {
        "id": 138,
        "question": "Which tribe is most common in Jharkhand?",
        "answer": "The Santhal tribe is one of the major tribes in Jharkhand."
    },
    {
        "id": 139,
        "question": "What are traditional Jharkhand foods?",
        "answer": "Popular foods include Dhuska, Rugra, Chhoti Macher Jhor, and Chilka Roti."
    },
    {
        "id": 140,
        "question": "What is the dance form of Jharkhand?",
        "answer": "Chhau and Jhumair are popular dance forms of Jharkhand."
    },
    {
        "id": 141,
        "question": "Which river flows through Ranchi?",
        "answer": "The Subarnarekha River flows through Ranchi."
    },
    {
        "id": 142,
        "question": "What is the climate of Jharkhand?",
        "answer": "Jharkhand has a tropical climate with hot summers and cool winters."
    },
    {
        "id": 143,
        "question": "Which wildlife sanctuary is near Jamshedpur?",
        "answer": "Dalma Wildlife Sanctuary is the nearest major sanctuary to Jamshedpur."
    },
    {
        "id": 144,
        "question": "Where is Dalma Wildlife Sanctuary located?",
        "answer": "Dalma Wildlife Sanctuary is near Jamshedpur, famous for elephants."
    },
    {
        "id": 145,
        "question": "What is Johna Falls also known as?",
        "answer": "Johna Falls is also called Gautamdhara Falls."
    },
    {
        "id": 146,
        "question": "What adventure activities are available in Jharkhand?",
        "answer": "Adventure activities include trekking, rock climbing, wildlife safari, and boating."
    },
    {
        "id": 147,
        "question": "Where can I do trekking in Jharkhand?",
        "answer": "Trekking is popular in Parasnath Hills and Netarhat."
    },
    {
        "id": 148,
        "question": "Which airport is closest to major attractions?",
        "answer": "Birsa Munda Airport in Ranchi is the main airport for tourists."
    },
    {
        "id": 149,
        "question": "Where is Patratu Valley located?",
        "answer": "Patratu Valley is located near Ramgarh, close to Ranchi."
    },
    {
        "id": 150,
        "question": "Which city is known as the Steel City of Jharkhand?",
        "answer": "Jamshedpur is known as the Steel City of Jharkhand."
    }
]
    
except Exception as e:
    print(f"ERROR: Could not load QA dataset: {e}")
    exit()

lemmatizer = WordNetLemmatizer()
qa_tags = {}

# Simplified Keyword-based Intent Grouping
def determine_intent_tag(question):
    q = question.lower()
    
    if 'waterfall|dassam|hundru|johna|gautamdhara' in q:
        return 'waterfalls'
    if 'national park|wildlife|betla|dalma|hazaribagh|tiger|elephant' in q:
        return 'wildlife_parks'
    if 'parasnath|hill|peak|jain|deoghar|mandir' in q:
        return 'religious_places'
    if 'food|cuisine|traditional|dhuska|rugra|roti' in q:
        return 'local_cuisine'
    if 'jamshedpur|steel city' in q:
        return 'about_jamshedpur'
    if 'trekking|adventure|activities' in q:
        return 'adventure'
    if 'tribe|santhal|culture|dance|chhau|jhumair' in q:
        return 'culture'
    if 'capital|ranchi|subarnarekha|airport|patratu|river|latehar' in q:
        return 'about_ranchi_area'
    if 'best time|season|climate|famous' in q:
        return 'overview'
    if 'safe|reach|netarhat|road|distance|transport|lodging' in q:
        return 'general_info'

    return 'general_info' # Default to general info for less specific questions

# Populate QA Intents
for item in qa_dataset:
    tag = determine_intent_tag(item['question'])

    if tag not in qa_tags:
        qa_tags[tag] = {'tag': tag, 'patterns': [], 'responses': []}

    qa_tags[tag]['patterns'].append(item['question'])
    qa_tags[tag]['responses'].append(item['answer'])

new_intents = list(qa_tags.values())

# ADDED/FIXED UTILITY AND OUT-OF-SCOPE INTENTS
new_intents.extend([
    {'tag': 'greeting', 'patterns': ['Hi', 'Hello', 'Is anyone there?', 'Namaste', 'hey', 'good morning', 'good evening'],
     'responses': ["Hello! Welcome to Jharkhand Tourism. How can I assist you? 🌳", "Hi there! What can I tell you about the Land of Forests?"]},
    
    # FIXED: The model was confusing 'bye' with other intents. Added more patterns.
    {'tag': 'farewell', 'patterns': ['bye', 'goodbye', 'see you', 'tata', 'later', 'thanks bye'],
     'responses': ["Goodbye! Have a great trip to Jharkhand! 👋", "See you later! Enjoy the Land of Forests!"]},
     
    # WORKING: Thanks intent was fine, but added more responses.
    {'tag': 'thanks', 'patterns': ['thank you', 'thanks', 'that was helpful', 'much obliged', 'appreciate it'],
     'responses': ["You're welcome! How else can I assist you?", "My pleasure! Happy to help with your Jharkhand journey.", "Glad I could help!"]},
     
    # WORKING: Out of scope intent was fine.
    {'tag': 'out_of_scope', 'patterns': ['where is M.S.Dhoni from', 'who is M.S. Dhoni', 'what about dhoni', 'dhoni hometown', 'cricket', 'prime minister', 'president'],
     'responses': ["I specialize in Jharkhand tourism. While M.S. Dhoni is a famous person from Ranchi, I cannot provide biographical details. Can I help you with a tourist question instead?", "That's a great question, but I focus on tourism information. How can I help you plan your trip?"]},

    {'tag': 'fallback', 'patterns': ['I do not know', 'nothing', 'general query', 'random text', 'not understood', 'what can you do'],
     'responses': ["I'm sorry, I don't understand that. Can you try asking about a specific place, a waterfall, or food in Jharkhand?", "I need more information. Could you rephrase your question? Try 'Where is Dassam Falls located?'"]}
])

intents_data = {'intents': new_intents}

# --- 2. Training the Model (Unchanged) ---
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

# --- 3. Saving the Components (Unchanged) ---
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
