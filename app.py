from flask import Flask, render_template, request
import pickle
import os
import numpy as np

app = Flask(__name__)

# Paths
MODEL_PATH = os.path.join("model", "decision_tree_model.pkl")
ENCODERS_PATH = os.path.join("model", "label_encoders.pkl")

# Load model and label encoders
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODERS_PATH, "rb") as f:
    label_encoders = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get raw form inputs
        age = int(request.form["age"])
        gender = request.form["gender"]
        annual_income = float(request.form["annual_income"])
        spending_score = int(request.form["spending_score"])
        prior_purchases = int(request.form["prior_purchases"])
        product_category = request.form["product_category"]
        loyalty_member = request.form["loyalty_member"]

        # Encode categorical features using saved encoders
        gender_encoded = label_encoders["Gender"].transform([gender])[0]
        product_category_encoded = label_encoders["Product_Category"].transform([product_category])[0]
        loyalty_member_encoded = label_encoders["Loyalty_Member"].transform([loyalty_member])[0]

        # Create feature array in the exact same order as training
        ordered_input = np.array([
            age,
            gender_encoded,
            annual_income,
            spending_score,
            prior_purchases,
            product_category_encoded,
            loyalty_member_encoded
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(ordered_input)[0]
        result = "Likely to Purchase" if prediction == 1 else "Unlikely to Purchase"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
