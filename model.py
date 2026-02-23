import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Data Load karne
print("Data load hot ahe...")
# Train data madhe ID, Title, Genre ani Description ahe [cite: 4]
train_df = pd.read_csv('train_data.txt', sep=':::', engine='python', names=['ID', 'Title', 'Genre', 'Description'])
# Test data madhe Genre nahiye [cite: 4]
test_df = pd.read_csv('test_data.txt', sep=':::', engine='python', names=['ID', 'Title', 'Description'])
# Test solution madhun apan accuracy check karu shakto [cite: 3]
test_solution_df = pd.read_csv('test_data_solution.txt', sep=':::', engine='python', names=['ID', 'Title', 'Genre', 'Description'])

print("Data successfully load zala!")

# 2. Text Preprocessing (TF-IDF)
# Plot summary (Description) la numbers madhe convert karne
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

X_train = tfidf.fit_transform(train_df['Description'])
y_train = train_df['Genre'].strip() # Spaces kadhun taknya sathi

X_test = tfidf.transform(test_df['Description'])
y_test = test_solution_df['Genre'].strip()

# 3. Model Training (Logistic Regression)
print("Model training suru ahe... thoda vel thamba.")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Model Testing
y_pred = model.predict(X_test)

# 5. Results baghne
print("\n--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Swataha test kara (Prediction Function)
def predict_genre(my_plot):
    vector = tfidf.transform([my_plot])
    genre = model.predict(vector)
    return genre[0]

# Example test
sample_plot = "A group of astronauts travel through a wormhole in space to ensure humanity's survival."
print(f"\nSample Plot: {sample_plot}")
print(f"Predicted Genre: {predict_genre(sample_plot)}")