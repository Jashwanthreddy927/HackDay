import pandas as pd
import numpy as np
import pickle
import streamlit as st
import spacy
from spacy.cli import download
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
from fuzzywuzzy import process

# Load dataset
file_path = 'symbipredict_2022.csv'
df = pd.read_csv(file_path)

# Convert symptoms to lowercase and clean column names
df.columns = df.columns.str.lower().str.replace('_', ' ')
df.iloc[:, :-1] = df.iloc[:, :-1].applymap(lambda x: str(x).strip().lower() if isinstance(x, str) else x)

# Select symptoms (all columns except the last) and disease (last column)
X = df.iloc[:, :-1]  # Symptoms columns
y = df.iloc[:, -1]   # Disease column

# Get all symptoms as a list
all_symptoms = X.columns.tolist()

# Synonym mapping for symptoms
symptom_synonyms = {
    "runny nose": "nasal discharge",
    "stomach pain": "abdominal pain",
    "high temperature": "fever",
    "sore throat": "throat pain"
}

# Encode symptoms
mlb = MultiLabelBinarizer()
X_encoded = mlb.fit_transform(X.apply(lambda row: row[row == 1].index.tolist(), axis=1))

# Encode diseases
disease_encoder = LabelEncoder()
y_encoded = disease_encoder.fit_transform(y)
disease_mapping = {idx: disease for idx, disease in enumerate(disease_encoder.classes_)}

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Save model
with open('symptom_checker_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('disease_encoder.pkl', 'wb') as f:
    pickle.dump(disease_encoder, f)
with open('mlb.pkl', 'wb') as f:
    pickle.dump(mlb, f)

# Load spaCy NLP model

    nlp = spacy.load('en_core_web_sm')

# Streamlit UI with custom styling
st.markdown(
    """
    <style>
        body {
            background-color: #f0f8ff;
            font-family: 'Arial', sans-serif;
        }
        .stTextArea, .stButton {
            border-radius: 10px;
        }
        .stButton > button {
            background-color: #0073e6;
            color: white;
            font-size: 16px;
            padding: 10px;
        }
        .stTitle {
            color: #004d99;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 AI Symptom Checker ")
st.write("Describe your symptoms, and our AI will predict possible diseases.")

# User Input: Chatbot Interface
user_input = st.text_area("Enter your symptoms in natural language:")

def extract_symptoms(text):
    """Extract and match symptoms from user input."""
    doc = nlp(text.lower())
    extracted = set()
    for token in doc:
        match = process.extractOne(token.text, all_symptoms, score_cutoff=50)
        if match:
            extracted.add(match[0])
        elif token.text in symptom_synonyms:
            extracted.add(symptom_synonyms[token.text])
    return list(extracted)

if st.button("🔍 Predict Disease"):
    if not user_input.strip():
        st.warning("⚠ Please enter your symptoms.")
    else:
        extracted_symptoms = extract_symptoms(user_input)
        if not extracted_symptoms:
            st.error("❌ No recognizable symptoms found. Try rephrasing.")
        else:
            st.success(f"✅ Extracted Symptoms: {', '.join(extracted_symptoms)}")
            input_encoded = mlb.transform([extracted_symptoms])
            probabilities = model.predict_proba(input_encoded)[0]
            non_zero_indices = [i for i, prob in enumerate(probabilities) if prob > 0.01]
            if not non_zero_indices:
                st.error("❌ No strong matches found.")
            else:
                sorted_indices = sorted(non_zero_indices, key=lambda i: probabilities[i], reverse=True)[:3]
                predicted_diseases = [disease_mapping[idx] for idx in sorted_indices]
                st.write("### Most Likely Diseases:")
                for i, disease in enumerate(predicted_diseases, start=1):
                    confidence = probabilities[sorted_indices[i-1]] * 100
                    st.write(f"{i}. {disease} - ({confidence:.2f}% confidence)")
