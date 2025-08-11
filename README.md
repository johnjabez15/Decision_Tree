# Decision Tree Project – Customer Purchase Prediction  

## 📌 Overview  
This project implements a **Decision Tree Classifier** to predict whether a customer is **likely to make a purchase** based on demographic, spending, and loyalty data.  
It uses **scikit-learn** for model training and a simple **Flask** web application to allow user input and real-time predictions.  

---

## 💻 Tech Stack  
- **Python 3.x** – Core programming language  
- **scikit-learn** – Machine learning model training  
- **pandas** – Data manipulation and preprocessing  
- **numpy** – Numerical computations  
- **Flask** – Web framework for the application backend  
- **HTML5, CSS3** – Frontend design and styling  
- **Pickle** – Model serialization and loading  

---

## 📂 Project Structure  
```
Data Science/
│
├── Decision Tree/
│ ├── dataset/
│ │ └── customer_purchase_dataset.csv
│ ├── model/
│ │ ├── decision_tree_model.pkl
│ │ └── label_encoders.pkl
│ ├── static/
│ │ └── style.css
│ ├── templates/
│ │ ├── index.html
│ │ └── result.html
│ ├── train_model.py
│ ├── app.py
│ ├── requirements.txt
│ └── README.md

```

---

## ⚙️ Installation & Setup  

**1️⃣ Clone the repository**  
```
git clone <your-repo-url>
cd "DataScience/Decision Tree"
```
**2️⃣ Create a virtual environment (recommended)**
```
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

**3️⃣ Install dependencies**
```
pip install -r requirements.txt
```
___

**📊 Dataset**

We use a customer purchase dataset containing the following features:

Age – Customer's age (integer)

Gender – Male/Female

Annual_Income – Annual income in USD (float)

Spending_Score – Score between 0–100 (integer)

Prior_Purchases – Number of past purchases (integer)

Product_Category – Product type (e.g., Electronics, Clothing, etc.)

Loyalty_Member – Yes/No (categorical)

Purchased – Target (1 = Likely to purchase, 0 = Unlikely to purchase)
___

## 🚀 How to Run
**1️⃣ Train the Model**
```
python train_model.py
```
This will create:

decision_tree_model.pkl

label_encoders.pkl

**2️⃣ Run the Flask App**
```
python app.py
Visit http://127.0.0.1:5000/ in your browser.
```

**🖥️ Frontend Input Example**
On the form, you can enter:

Age: 30
Gender: Male
Annual Income: 45000
Spending Score: 80
Prior Purchases: 5
Product Category: Electronics
Loyalty Member: Yes

##  📌 Future Scope
🔹 Deploy the model on Heroku or Render for public access.

🔹 Improve accuracy by hyperparameter tuning and feature engineering.

🔹 Add data visualization dashboard using Plotly or Matplotlib.

🔹 Integrate with real-time customer analytics systems.

🔹 Create a REST API for integration with other e-commerce applications.

🔹 Enhance UI/UX for better user experience.

___

## Screen Shots

**Home Page**
<img width="1920" height="1080" alt="Screenshot (17)" src="https://github.com/user-attachments/assets/9ae81607-572a-4a0d-a285-58e8691ae5c5" />


**Result Page**
<img width="1920" height="1080" alt="Screenshot (18)" src="https://github.com/user-attachments/assets/178b9e70-50df-4979-ac56-0f4f29796e16" />
