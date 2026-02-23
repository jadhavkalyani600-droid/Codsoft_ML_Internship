import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Data Load करणे
print("--- स्टेप 1: डेटा लोड होत आहे... ---")
# PC हँग होऊ नये म्हणून आपण सध्या 50,000 रोज घेऊया
train_df = pd.read_csv('fraudTrain.csv', nrows=50000)
test_df = pd.read_csv('fraudTest.csv', nrows=10000)

# 2. डेटा क्लीनिंग फंक्शन (दोन्ही फाईल्ससाठी एकच लॉजिक)
def clean_data(df):
    # नको असलेले कॉलम्स काढणे
    cols_to_drop = ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'trans_num', 'unix_time', 'job', 'dob']
    df = df.drop(columns=cols_to_drop)
    
    # वेळेतून 'Hour' काढणे
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df = df.drop(columns=['trans_date_trans_time'])
    
    # शब्दांना नंबर्समध्ये बदलणे
    le = LabelEncoder()
    df['category'] = le.fit_transform(df['category'])
    df['gender'] = le.fit_transform(df['gender'])
    df['merchant'] = le.fit_transform(df['merchant'])
    return df

print("--- स्टेप 2: डेटा क्लीनिंग सुरू आहे... ---")
train_df = clean_data(train_df)
test_df = clean_data(test_df)

# 3. Features आणि Target वेगळे करणे
X_train = train_df.drop(columns=['is_fraud'])
y_train = train_df['is_fraud']
X_test = test_df.drop(columns=['is_fraud'])
y_test = test_df['is_fraud']

# 4. मॉडेल ट्रेनिंग (Random Forest)
print("--- स्टेप 3: मॉडेल ट्रेनिंग सुरू आहे (Random Forest)... ---")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. प्रेडिक्शन आणि रिझल्ट
print("--- स्टेप 4: निकाल तपासत आहे... ---")
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
plt.title('Konta Column Mahatvacha Ahe? (Feature Importances)')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Importance Score')
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Confusion Matrix Graph
print("\n--- निकाल ग्राफमध्ये दाखवत आहे... ---")
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
plt.title('कौणते कॉलम्स फ्रॉड ओळखण्यासाठी महत्त्वाचे ठरले?')
plt.xlabel('Importance Score')
plt.show()
# फायनल रिझल्ट्स एका टेबलमध्ये गोळा करूया
results_df = pd.DataFrame({
    'Actual_Is_Fraud': y_test.values,
    'Model_Prediction': y_pred
})

# ही फाईल CSV फॉरमॅटमध्ये सेव्ह करू
results_df.to_csv('FRAUD_DETECTION_RESULTS.csv', index=False)

print("\n" + "="*45)
print("✅ अभिनंदन! 'FRAUD_DETECTION_RESULTS.csv' तयार झाली आहे.")
print("आता तुम्ही तुमच्या फोल्डरमध्ये जाऊन ही फाईल Excel मध्ये पाहू शकता.")
print("="*45)