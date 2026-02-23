import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the Dataset
print("--- Step 1: Loading Data... ---")
df = pd.read_csv('Churn_Modelling.csv')

# 2. Drop unnecessary columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# 3. Convert Categorical data to Numbers (Label Encoding)
le = LabelEncoder()
df['Geography'] = le.fit_transform(df['Geography'])
df['Gender'] = le.fit_transform(df['Gender'])

# 4. Separate Features (X) and Target (y)
X = df.drop('Exited', axis=1)
y = df['Exited']

# 5. Split Data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model 1: Random Forest Training
print("--- Step 2: Training Random Forest... ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 7. Model 2: Logistic Regression Training
print("--- Step 3: Training Logistic Regression... ---")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# 8. Compare Accuracy Results
print("\n" + "="*45)
print(f"✅ Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)*100:.2f}%")
print(f"✅ Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log)*100:.2f}%")
print("="*45)

# 9. Manual Testing for a New Customer
print("\n--- Step 4: Testing for a New Customer ---")
# Data: CreditScore: 600, Geo: France(0), Gen: Male(1), Age: 40, Tenure: 3, Balance: 60000, Products: 2, HasCard: 1, Active: 1, Salary: 50000
new_cust = [[600, 0, 1, 40, 3, 60000, 2, 1, 1, 50000]]
test_pred = rf_model.predict(new_cust)

if test_pred[0] == 1:
    print("RESULT: This customer is likely to CHURN (Leave).")
else:
    print("RESULT: This customer is likely to STAY.")

# 10. Visualization (Feature Importance) with English Labels
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).sort_values(ascending=True).plot(kind='barh', color='skyblue')
plt.title('Top Factors Influencing Customer Churn')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
# --- स्टेप ५: निकालांची CSV फाईल तयार करणे ---
# आपण टेस्ट डेटाचे प्रेडिक्शन्स एका फाईलमध्ये सेव्ह करूया
results_df = pd.DataFrame({
    'Actual_Value': y_test.values,
    'Model_Prediction': y_pred_rf  # आपण Random Forest चे रिझल्ट्स वापरत आहोत कारण ते जास्त ऍक्युरेट आहेत
})

results_df.to_csv('CHURN_FINAL_REPORT.csv', index=False)

print("\n" + "="*45)
print("✅ SUCCESS! 'CHURN_FINAL_REPORT.csv' तयार झाली आहे.")
print("तुमच्या फोल्डरमध्ये ही फाईल तपासा.")
print("="*45)