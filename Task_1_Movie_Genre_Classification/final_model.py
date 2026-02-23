import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Data Load
train_df = pd.read_csv('train_data.txt', sep=':::', engine='python', names=['ID', 'Title', 'Genre', 'Description'])
test_sol_df = pd.read_csv('test_data_solution.txt', sep=':::', engine='python', names=['ID', 'Title', 'Genre', 'Description'])

# 2. Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
X_train = tfidf.fit_transform(train_df['Description'])
y_train = train_df['Genre'].str.strip()
X_test = tfidf.transform(test_sol_df['Description'])
y_test = test_sol_df['Genre'].str.strip()

# 3. Training
print("Model Finalize hot ahe...")
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# 4. Detailed Evaluation
y_pred = model.predict(X_test)
print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred))

# 5. Visualization (Top 5 Genres Accuracy)
plt.figure(figsize=(10,6))
sns.countplot(x=y_train, order=y_train.value_counts().index[:10])
plt.title('Top 10 Genres in Dataset')
plt.xticks(rotation=45)
plt.show() # Ha command graph dakhvel

# 6. Prediction File Banvane (Final Submission sathi)
output_df = pd.DataFrame({'Movie': test_sol_df['Title'], 'Actual': y_test, 'Predicted': y_pred})
output_df.to_csv('movie_predictions.csv', index=False)
print("✅ Results 'movie_predictions.csv' madhe save zale ahet!")