# Credit Card Fraud Detection 💳

## 📌 Project Overview
This project aims to build a machine learning model to identify fraudulent credit card transactions. The main challenge addressed here is 'Data Imbalance', as legitimate transactions far outnumber fraudulent ones.



## 📊 Dataset
The dataset contains transactions made by credit cards. 
- **Features:** PCA transformed numerical variables (V1-V28), 'Amount', and 'Time'.
- **Target:** 'Class' (1 for fraud, 0 for genuine).

## 🛠️ Tech Stack
- **Language:** Python 3.14
- **Libraries:** Pandas, Scikit-learn, Seaborn, Matplotlib.
- **Algorithm:** Random Forest / Logistic Regression.

## 🤖 Approach
1. **Data Preprocessing:** Scaling the 'Amount' feature.
2. **Handling Imbalance:** Using sampling techniques to balance the classes.
3. **Evaluation:** Measuring performance using a Confusion Matrix and Precision-Recall scores.

## 📁 Files in this Task
- `fraud_detection.py`: Main script for detection.
- `FRAUD_DETECTION_RESULTS.csv`: Final output predictions.

## 🚀 How to Run
1. Install dependencies: `pip install pandas scikit-learn seaborn`
2. Run: `python fraud_detection.py`