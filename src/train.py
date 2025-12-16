import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

# Load dataset
data = pd.read_csv("data/dataset.csv")

# Example: simple model predicting 'score' from 'age'
X = data[['age']]  # features
y = data['score']  # target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a small model
model = LinearRegression()
model.fit(X_train, y_train)

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the model
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved to models/model.pkl")
