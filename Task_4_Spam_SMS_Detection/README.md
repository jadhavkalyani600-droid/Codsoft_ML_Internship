# Spam SMS Detection 🛡️

## 📌 Project Overview
This project uses Natural Language Processing (NLP) to classify SMS messages as either **Spam** or **Ham** (Legitimate). It helps in filtering out unwanted advertisements and phishing links.



## 📊 Dataset
Contains thousands of SMS messages labeled as 'spam' or 'ham'.

## 🛠️ Tech Stack
- **Algorithm:** Multinomial Naive Bayes (Best for text data).
- **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency).
- **Libraries:** Pandas, Scikit-learn, NLTK.

## 🤖 Approach
1. **Text Cleaning:** Removing punctuation and stop words.
2. **Vectorization:** Converting words into math vectors.
3. **Classification:** Running the Naive Bayes algorithm for high accuracy.

## 📁 Files in this Task
- `spam_task.py`: SMS classifier code.
- `SPAM_DETECTION_RESULTS.csv`: Final message classification.

## 🚀 How to Run
Run: `python spam_task.py`