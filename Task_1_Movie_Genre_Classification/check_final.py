import pandas as pd
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("--- Step 1: Script Start Zhali ---")

# 1. Files check kara
if not os.path.exists('train_data.txt'):
    print("❌ ERROR: train_data.txt sapdli nahi! Krupaya file ya folder madhe taka.")
    sys.exit()

# 2. Data Load & Train (Fast version sathi features kami kele ahet)
print("--- Step 2: Model Train hot ahe... ---")
train_df = pd.read_csv('train_data.txt', sep=':::', engine='python', names=['ID', 'Title', 'Genre', 'Description'])
test_sol_df = pd.read_csv('test_data_solution.txt', sep=':::', engine='python', names=['ID', 'Title', 'Genre', 'Description'])

tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
X_train = tfidf.fit_transform(train_df['Description'])
y_train = train_df['Genre'].str.strip()

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 3. CSV Save karne (Full Error Handling)
try:
    y_pred = model.predict(tfidf.transform(test_sol_df['Description']))
    output_df = pd.DataFrame({'Movie': test_sol_df['Title'], 'Genre': y_pred})
    
    file_name = "MAZA_REPORT.csv"
    output_df.to_csv(file_name, index=False)
    
    print("\n" + "="*50)
    print("✅ SUCCESS! TUZI FILE TAYAR ZALI AHE.")
    print(f"📍 FILE CHA NAAV: {file_name}")
    print(f"📍 LOCATION: {os.getcwd()}")
    print("="*50)
except Exception as e:
    print(f"❌ ERROR: File save karta yeat nahiye. Karan: {e}")

print("--- Step 3: Script Purna Zhali ---")