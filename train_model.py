# ==========================================
# Student Placement Prediction
# Logistic Regression Training
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("dataset/placementdata.csv")

print("\nFirst 5 Records")
print(df.head())

print("\nDataset Shape :", df.shape)

# -------------------------------
# Drop StudentID
# -------------------------------
df.drop("StudentID", axis=1, inplace=True)

# -------------------------------
# Encode Categorical Columns
# -------------------------------
le_extra = LabelEncoder()
le_training = LabelEncoder()
le_target = LabelEncoder()

df["ExtracurricularActivities"] = le_extra.fit_transform(
    df["ExtracurricularActivities"]
)

df["PlacementTraining"] = le_training.fit_transform(
    df["PlacementTraining"]
)

df["PlacementStatus"] = le_target.fit_transform(
    df["PlacementStatus"]
)

# -------------------------------
# Features and Target
# -------------------------------
X = df.drop("PlacementStatus", axis=1)
y = df["PlacementStatus"]

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -------------------------------
# Train Logistic Regression Model
# -------------------------------
model = LogisticRegression(max_iter=3000)

model.fit(X_train, y_train)

# -------------------------------
# Prediction
# -------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", round(accuracy * 100, 2), "%")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# Save Model
# -------------------------------
joblib.dump(model, "model.pkl")

print("\nModel Saved Successfully!")