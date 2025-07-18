import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# === Load model and encoders ===
MODEL_PATH = "rf_model.pkl"  # changed from models/rf_model.pkl
ENCODER_DIR = "." 

@st.cache_resource
def load_model_and_encoders():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please train the model first.")
        st.stop()

    model = joblib.load(MODEL_PATH)

    # Load all encoders dynamically
    categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    encoders = {}
    for col in categorical_cols:
        safe_col = col.replace("/", "_")
        encoders[col] = joblib.load(os.path.join(ENCODER_DIR, f"encoder_{safe_col}.pkl"))

    return model, encoders

model, encoders = load_model_and_encoders()

# === App Title ===
st.title("🎓 Student Performance Predictor")
st.write("Predict whether a student is likely to **Pass or Fail** based on their details.")

# === Input Section ===
gender = st.selectbox("Gender", encoders['gender'].classes_)
race = st.selectbox("Race/Ethnicity", encoders['race/ethnicity'].classes_)
parent_edu = st.selectbox("Parental Level of Education", encoders['parental level of education'].classes_)
lunch = st.selectbox("Lunch", encoders['lunch'].classes_)
prep = st.selectbox("Test Preparation Course", encoders['test preparation course'].classes_)

math = st.slider("Math Score", 0, 100, 50)
reading = st.slider("Reading Score", 0, 100, 50)
writing = st.slider("Writing Score", 0, 100, 50)

# === Predict Button ===
if st.button("Predict"):
    try:
        # Create input dict
        input_data = {
            'gender': gender,
            'race/ethnicity': race,
            'parental level of education': parent_edu,
            'lunch': lunch,
            'test preparation course': prep,
            'math score': math,
            'reading score': reading,
            'writing score': writing
        }

        # Encode categorical features
        for col in encoders:
            le = encoders[col]
            if input_data[col] not in le.classes_:
                st.error(f"Invalid value for {col}.")
                st.stop()
            input_data[col] = le.transform([input_data[col]])[0]

        # Convert to DataFrame
        features_order = ['gender', 'race/ethnicity', 'parental level of education',
                          'lunch', 'test preparation course', 'math score', 'reading score', 'writing score']
        input_df = pd.DataFrame([input_data])[features_order]

        # Prediction
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]

        # Display results
        if pred == 1:
            st.success(f"🎉 Likely to Pass! (Confidence: {prob[1]*100:.2f}%)")
        else:
            st.error(f"Likely to Fail. (Confidence: {prob[0]*100:.2f}%)")

        # === Feature Importance ===
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": features_order,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance_df.set_index("Feature"))

    except Exception as e:
        st.error(f"Error during prediction: {e}")
