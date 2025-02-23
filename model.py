import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
file_path = 'symbipredict_2022.csv'
df = pd.read_csv(file_path)

# Preprocess dataset
df.columns = df.columns.str.lower().str.replace('_', ' ')
df.iloc[:, :-1] = df.iloc[:, :-1].applymap(lambda x: str(x).strip().lower() if isinstance(x, str) else x)

X = df.iloc[:, :-1]  
y = df.iloc[:, -1]   

all_symptoms = X.columns.tolist()

mlb = MultiLabelBinarizer()
X_encoded = mlb.fit_transform(X.apply(lambda row: row[row == 1].index.tolist(), axis=1))

disease_encoder = LabelEncoder()
y_encoded = disease_encoder.fit_transform(y)

disease_mapping = {idx: disease for idx, disease in enumerate(disease_encoder.classes_)}

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
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

# Streamlit UI
st.title("🔍 AI-Powered Symptom Checker")

# Inject Custom HTML & CSS
custom_html = """
<style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f9f9f9;
        text-align: center;
    }
    .title {
        color: #2C3E50;
        font-size: 26px;
        font-weight: bold;
    }
    .dropdown {
        padding: 10px;
        font-size: 16px;
        border-radius: 5px;
        border: 1px solid #ccc;
        width: 80%;
        margin: 10px auto;
        display: block;
    }
    .predict-btn {
        background-color: #3498DB;
        color: white;
        border: none;
        padding: 12px 20px;
        font-size: 18px;
        border-radius: 5px;
        cursor: pointer;
        transition: 0.3s;
    }
    .predict-btn:hover {
        background-color: #217DBB;
    }
    .result-box {
        background-color: #ECF0F1;
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
</style>
"""

components.html(custom_html, height=10)  # Inject the CSS styling

st.write("## Select symptoms to predict possible diseases")
selected_symptoms = st.multiselect("Symptoms:", options=all_symptoms)

if st.button("🔎 Predict Disease", key="predict"):
    if not selected_symptoms:
        st.write("❗ Please select at least one symptom.")
    else:
        input_encoded = mlb.transform([selected_symptoms])
        probabilities = model.predict_proba(input_encoded)[0]

        non_zero_indices = [i for i, prob in enumerate(probabilities) if prob > 0.01]

        if not non_zero_indices:
            st.write("⚠️ No strong matches found for the given symptoms.")
        else:
            sorted_indices = sorted(non_zero_indices, key=lambda i: probabilities[i], reverse=True)[:3]
            predicted_diseases = [disease_mapping[idx] for idx in sorted_indices]

            results_html = '<div class="result-box"><h3>🩺 Most likely diseases:</h3>'
            for i, disease in enumerate(predicted_diseases, start=1):
                confidence = probabilities[sorted_indices[i-1]] * 100
                results_html += f"<p><b>{i}. {disease}</b> ({confidence:.2f}% confidence)</p>"
            results_html += "</div>"

            components.html(results_html, height=200)
