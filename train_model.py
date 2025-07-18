import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import numpy as np

# === Load Dataset ===
DATA_PATH = "data/StudentsPerformance.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# === Feature Engineering ===
df['average'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
df['result'] = df['average'].apply(lambda x: 1 if x >= 60 else 0)

categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

# Encode categorical columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders
for col, le in encoders.items():
    safe_col_name = col.replace("/", "_")
    joblib.dump(le, os.path.join(MODEL_DIR, f"encoder_{safe_col_name}.pkl"))

# Features and Target
X = df.drop(['average', 'result'], axis=1)
y = df['result']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Random Forest with Hyperparameter Tuning ===
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20, scoring='accuracy', cv=3, n_jobs=-1, random_state=42)
search.fit(X_train, y_train)

# Best Model
model = search.best_estimator_
print(f"Best Parameters: {search.best_params_}")

# Evaluate
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save Model
joblib.dump(model, MODEL_PATH)

# === Save Feature Importance ===
feature_importance = model.feature_importances_
features = X.columns
importance_dict = {features[i]: feature_importance[i] for i in range(len(features))}
importance_df = pd.DataFrame(list(importance_dict.items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=False)
importance_df.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)

print("\nModel and encoders saved in 'models/' directory.")
print("Feature importance saved as 'feature_importance.csv'.")
