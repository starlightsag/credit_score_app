# credit_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load the dataset
df = pd.read_csv("dummy_credit_data.csv")

# Encode categorical features
le = LabelEncoder()
for col in ['education_level', 'marital_status', 'residence_type', 'loan_purpose']:
    df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop("loan_approved", axis=1)
y = df["loan_approved"]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "credit_model.pkl")
joblib.dump(scaler, "scaler.pkl")
