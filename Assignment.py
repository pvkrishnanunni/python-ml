
import numpy as hp
import pandas as pd
from sklearn. linear model import LogisticRegression
from sklearn.metrics
  import (
  accuracy_score, classification_report, confusion_ matrix, f1_score, precision_score, recall_score,
from
sklearn.model_selection import train_test split
from sklearn.preprocessing import StandardScaler

# ======================
# PART 1: DATA PREPARATION & EDA
# ======================
# 1. Load dataset
df = pd. read_csv("diabetes.csv")
print("--- Dataset Exploration ---")
print(f"Shape: {df. shape)")
print(f"Column Names: {df. columns, tolist()}")
print(" \nData Types: \n", df,dtypes)
print(" Missing Values (Explicit); \n", df.isnu11),sume)
print("InBasic Statistical Summary: \n", df. describe())
# 2. Separate Features (X) and Target (y)
x = df.drop(columns= ["Outcome"])
y = df["Outcome"]
# 3. Train-Test Split (80/20)
X_train, X_test, y_train, y test - train test_split(
X, y, test_size=0.20, random_state=42
)

# 4. Handle Hidden Missing Values (Zeros in clinical metrics)
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero cols:
median_val = X_train. loc[_train[col] |= 0, col] median()
X_train[col] = X_train[col] replace(0, median_val)
x_test[col] = X_test[col].replace(0, medianval)

# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"\nTraining set size: {X_train.shape[0]} | Testing set size:" f" ("_test. shape[0]}")

# =======================================
# PART 2 & 3: MODEL BUILDING & TRAINING
# =======================================
# Instantiate Logistic Regression
model = LogisticRegression(random_state=42)
# Fit on training data
model.fit(X_train_scaled, y_train)
# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# =======================================
# PART 4: MODEL EVALUATION & PERFORMANCE
# =======================================
train_acc = accuracy_score(y_train, y_train_pred) 
test_acc = accuracy_score(y_test, y_test_pred) 
precision = precision_score(y_test,y_test_pred) 
recall = recall_score(y_test, y_test_pred) 
f1 = f1_score(y_test, y_test_pred)
conf _matrix = confusion_matrixy_test, y_test_pred)

print("\n--- Model Performance ---")
print(f"Training Accuracy : (train_acc: .4f) ({train_acc * 100: .2f}%) ")
print(f"Testing Accuracy : (test_acc: .4f} ({test_acc * 100: .2f]%) ")
print(f"Precision: {precision: .4f}")
print(f"Recall: {recall: .4fJ")
print(f"F1-Score: {f1: .4f]")
print("\nConfusion Matrix:")
print(conf _matrix)
print("\nFull Classification Report:")
print(classification_report(y_test, y_test_pred))
