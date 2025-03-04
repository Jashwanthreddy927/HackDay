import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import pickle
import streamlit as st  
import os

# Load dataset
file_path = 'symbipredict_2022.csv'
df = pd.read_csv(file_path)

# Convert all symptoms in dataset to lowercase, strip spaces, and replace underscores
df.columns = df.columns.str.lower().str.replace('_', ' ')
df.iloc[:, :-1] = df.iloc[:, :-1].applymap(lambda x: str(x).strip().lower() if isinstance(x, str) else x)

# Select symptoms (all columns except the last) and disease (last column)
X = df.iloc[:, :-1]  # Symptoms columns
y = df.iloc[:, -1]   # Disease column

# Get all symptoms as a list (column names except the last column)
all_symptoms = X.columns.tolist()

# Encode symptoms using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
X_encoded = mlb.fit_transform(X.apply(lambda row: row[row == 1].index.tolist(), axis=1))

# Encode disease labels
disease_encoder = LabelEncoder()
y_encoded = disease_encoder.fit_transform(y)

# Create a mapping of encoded labels to actual disease names
disease_mapping = {idx: disease for idx, disease in enumerate(disease_encoder.classes_)}

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Save model and encoders
model_path = os.path.join(os.path.dirname(__file__), 'symptom_checker_model.pkl')
disease_encoder_path = os.path.join(os.path.dirname(__file__), 'disease_encoder.pkl')
mlb_path = os.path.join(os.path.dirname(__file__), 'mlb.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

with open(disease_encoder_path, 'wb') as f:
    pickle.dump(disease_encoder, f)

with open(mlb_path, 'wb') as f:
    pickle.dump(mlb, f)

# Streamlit UI Styling
st.markdown("""
    <style>
        body {
            background-color: #f4f4f9;
        }
        .main-container {
            text-align: center;
            padding: 30px;
        }
        .title {
            font-size: 32px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 20px;
        }
        .dropdown-container {
            text-align: left;
            margin: 10px auto;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .predict-button {
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            font-size: 18px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: 0.3s;
        }
        .predict-button:hover {
            background-color: #2980b9;
        }
        .result-container {
            text-align: left;
            margin-top: 20px;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="title">AI-Powered Symptom Checker</div>', unsafe_allow_html=True)
st.markdown('<p>Select symptoms to predict possible diseases</p>', unsafe_allow_html=True)

# Dropdown for symptoms selection
selected_symptoms = st.multiselect("Select symptoms from the list:", options=all_symptoms)
st.markdown("<br>", unsafe_allow_html=True)

# Predict button with styling
if st.button("Predict Disease", key='predict', help="Click to get predictions"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        input_encoded = mlb.transform([selected_symptoms])
        probabilities = model.predict_proba(input_encoded)[0]

        # Filter out diseases with 0% probability
        non_zero_indices = [i for i, prob in enumerate(probabilities) if prob > 0.01]  # Ignore probabilities <1%

        if not non_zero_indices:
            st.error("No strong matches found for the given symptoms.")
        else:
            sorted_indices = sorted(non_zero_indices, key=lambda i: probabilities[i], reverse=True)[:3]
            predicted_diseases = [disease_mapping[idx] for idx in sorted_indices]

            # Display results
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.markdown("<h4>Most likely diseases:</h4>", unsafe_allow_html=True)
            for i, disease in enumerate(predicted_diseases, start=1):
                confidence = probabilities[sorted_indices[i-1]] * 100  # Convert to percentage
                st.markdown(f"<p>{i}. <strong>{disease}</strong> ({confidence:.2f}% confidence)</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

print("Model trained and saved successfully!")
