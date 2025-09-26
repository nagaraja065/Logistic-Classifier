# --- Step 1: Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print("--- Logistic Regression on Breast Cancer Dataset ---")

# --- Step 2: Load and Preprocess the Data ---
try:
    df = pd.read_csv('data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: data.csv not found. Make sure the file is in the same directory.")
    exit()

# Drop the unnecessary 'id' and 'Unnamed: 32' columns
df = df.drop(['id', 'Unnamed: 32'], axis=1)

# Convert the 'diagnosis' target column to numbers (M=1, B=0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
print("Data preprocessing complete.")

# --- Step 3: Define Features (X) and Target (y) ---
# The target 'y' is the 'diagnosis' column
y = df['diagnosis']
# The features 'X' are all the other columns
X = df.drop('diagnosis', axis=1)

print(f"\nTarget (y) is: 'diagnosis'")
print(f"Number of features (X): {len(X.columns)}")

# --- Step 4: Split Data and Apply Feature Scaling ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"\nData split into training ({len(X_train)} rows) and testing ({len(X_test)} rows) sets.")

# Feature Scaling is crucial for this dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Features have been scaled using StandardScaler.")

# --- Step 5: Train the Logistic Regression Model ---
print("\n--- Training the Model ---")
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print("Logistic Regression model trained successfully.")

# --- Step 6: Evaluate the Model ---
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
# Update target names to match the new dataset
print(classification_report(y_test, y_pred, target_names=['Benign (B)', 'Malignant (M)']))

# --- Step 7: Visualize the Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
# Update labels for the heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign (B)', 'Malignant (M)'],
            yticklabels=['Benign (B)', 'Malignant (M)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()