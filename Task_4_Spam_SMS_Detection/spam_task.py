import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the Dataset
print("--- Step 1: Loading SMS Data... ---")
# Using latin-1 encoding to avoid reading errors
df = pd.read_csv('spam.csv', encoding='latin-1')

# 2. Clean Data (Removing extra columns and renaming)
df = df.iloc[:, :2]
df.columns = ['label', 'message']

# 3. Encoding labels (ham: 0, spam: 1)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# 4. Split Data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)

# 5. Text Vectorization (Converting words to numbers using TF-IDF)
print("--- Step 2: Preprocessing Text Data... ---")
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 6. Train the Model (Naive Bayes)
print("--- Step 3: Training Naive Bayes Model... ---")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 7. Model Evaluation
y_pred = model.predict(X_test_tfidf)
print("\n" + "="*45)
print(f"✅ Spam Detection Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("="*45)

# 8. Save Predictions to CSV
results_df = pd.DataFrame({
    'Message': X_test,
    'Actual_Label': y_test.map({0: 'ham', 1: 'spam'}),
    'Predicted_Label': pd.Series(y_pred, index=X_test.index).map({0: 'ham', 1: 'spam'})
})
results_df.to_csv('SPAM_DETECTION_RESULTS.csv', index=False)
print("\n✅ SUCCESS: 'SPAM_DETECTION_RESULTS.csv' created!")

# 9. Professional Visualization (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix for Spam SMS Detection')
plt.tight_layout()
plt.show()
