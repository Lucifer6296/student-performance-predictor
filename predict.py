import pandas as pd
import joblib

def predict_new(data_dict):
    model = joblib.load("models/rf_model.pkl")

    categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    encoders = {}
    for col in categorical_cols:
        safe_col_name = col.replace("/", "_")
        encoders[col] = joblib.load(f"models/encoder_{safe_col_name}.pkl")

    # Encode categorical features using loaded encoders
    for col in categorical_cols:
        le = encoders[col]
        if data_dict[col] in le.classes_:
            data_dict[col] = le.transform([data_dict[col]])[0]
        else:
            raise ValueError(f"Unknown category '{data_dict[col]}' for column '{col}'")

    # Create DataFrame with correct feature order
    model_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 
                      'test preparation course', 'math score', 'reading score', 'writing score']

    input_df = pd.DataFrame([data_dict])
    input_df = input_df[model_features]

    pred = model.predict(input_df)
    return "Pass" if pred[0] == 1 else "Fail"


if __name__ == "__main__":
    sample_data = {
        'gender': 'female',
        'race/ethnicity': 'group B',
        'parental level of education': "bachelor's degree",
        'lunch': 'standard',
        'test preparation course': 'none',
        'math score': 70,
        'reading score': 65,
        'writing score': 72
    }
    result = predict_new(sample_data)
    print("Predicted result:", result)
