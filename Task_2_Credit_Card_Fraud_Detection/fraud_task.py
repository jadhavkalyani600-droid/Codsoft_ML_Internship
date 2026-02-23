import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Data Loading
print("--- step 1: data is loading... ---")
# taking 50000 rows
train_df = pd.read_csv('fraudTrain.csv', nrows=50000)
test_df = pd.read_csv('fraudTest.csv', nrows=10000)

# 2. data cleaning function 
def clean_data(df):
    # removing unwanted columns
    cols_to_drop = ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'trans_num', 'unix_time', 'job', 'dob']
    df = df.drop(columns=cols_to_drop)
    
    # removing 'Hour' from timing
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df = df.drop(columns=['trans_date_trans_time'])
    
    # coverting words into the numbers
    le = LabelEncoder()
    df['category'] = le.fit_transform(df['category'])
    df['gender'] = le.fit_transform(df['gender'])
    df['merchant'] = le.fit_transform(df['merchant'])
    return df

print("--- step 2: data cleaning is in progress... ---")
train_df = clean_data(train_df)
test_df = clean_data(test_df)

# 3. differentiate features and target
X_train = train_df.drop(columns=['is_fraud'])
y_train = train_df['is_fraud']
X_test = test_df.drop(columns=['is_fraud'])
y_test = test_df['is_fraud']

# 4. model training (Random Forest)
print("--- step 3: model training is going on (Random Forest)... ---")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. prediction and result
print("--- step 4: checking the result... ---")
y_pred = model.predict(X_test)

print("\n" + "="*45)
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred))
print("="*45)
import matplotlib.pyplot as plt
import seaborn as sns

# Feature Importance Graph
features = X_train.columns
importances = model.feature_importances_
indices = importances.argsort()

plt.figure(figsize=(10, 6))
plt.title('which column is important? (Feature Importances)')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Importance Score')
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Confusion Matrix Graph
print("\n--- showing result in graph... ---")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Fraud Detection')
plt.show()

# 2. Feature Importance Graph
importances = model.feature_importances_
features = X_train.columns
feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind='barh', color='teal')
plt.title('Which columns proved to be important for identifying fraud?')
plt.xlabel('Importance Score')
plt.show()
# collecting final result in a table
results_df = pd.DataFrame({
    'Actual_Is_Fraud': y_test.values,
    'Model_Prediction': y_pred
})

# saving this file in csv format
results_df.to_csv('FRAUD_DETECTION_RESULTS.csv', index=False)

print("\n" + "="*45)
print("✅ congragulations! 'FRAUD_DETECTION_RESULTS.csv' is created.")
print("Now you can go to your folder and view this file in Excel.")
print("="*45)