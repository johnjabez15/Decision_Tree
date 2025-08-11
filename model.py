# model.py
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Paths
DATA_PATH = os.path.join("dataset", "customer_purchase_dataset.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "decision_tree_model.pkl")

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop("Purchase", axis=1)
y = df["Purchase"]

# Encode categorical variables
label_encoders = {}
for column in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Save label encoders for future use
with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "wb") as f:
    pickle.dump(label_encoders, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Save trained model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {MODEL_PATH}")
