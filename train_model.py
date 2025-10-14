# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# --- DATA PREPARATION ---
# Step 2: Load the dataset
print("Loading dataset...")
data = pd.read_csv('fake_accounts_dataset.csv')

# Step 3: Define features (X) and the target (y)
# We are trying to predict the 'is_fake' column based on the other features.
X = data.drop('is_fake', axis=1)
y = data['is_fake']

# Step 4: Split the data into training and testing sets
# This helps us see how well our model performs on data it hasn't seen before.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODEL TRAINING ---
# Step 5: Scale the features
# Scaling standardizes the data, which helps the model train better.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Initialize and train the Logistic Regression model
print("Training the model...")
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# --- EVALUATION & SAVING ---
# Step 7: Evaluate the model (optional, but good practice)
print("Evaluating the model...")
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 8: Save the trained model and the scaler to files
print("Saving model and scaler...")
with open('fake_account_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Done! Model and scaler have been saved.")