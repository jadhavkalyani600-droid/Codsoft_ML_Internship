# Customer Churn Prediction 📉

## 📌 Project Overview
The goal of this project is to predict which customers are likely to leave a service (Churn) based on their account information, demographics, and transaction history.



## 📊 Dataset
I used a 'Churn Modelling' dataset containing:
- **Customer Info:** Credit Score, Geography, Gender, Age.
- **Account Info:** Tenure, Balance, Number of Products.

## 🛠️ Tech Stack
- **Language:** Python 3.14
- **Algorithm:** Gradient Boosting / Random Forest.
- **Libraries:** Scikit-learn, Pandas, NumPy.

## 🤖 Approach
1. **Feature Engineering:** Converting categorical data (Gender/Geography) into numerical data.
2. **Model Training:** Training the classifier to identify "At-Risk" customers.
3. **Visualization:** Plotting feature importance to see what causes churn.

## 📁 Files in this Task
- `customer_churn.py`: Prediction logic script.
- `CUSTOMER_CHURN_RESULTS.csv`: Output with churn probabilities.

## 🚀 How to Run
Run the command: `python customer_churn.py`