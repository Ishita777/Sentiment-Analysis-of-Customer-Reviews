import os
import nltk
import pickle
import tkinter as tk
from tkinter import messagebox
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

# Load CSV dataset
def load_dataset(file_path):
    df = pd.read_csv(r'Dataset-SA.csv')
    return df['Review'].tolist(), df['Sentiment'].tolist()

# Load or train the model
model_file = 'sentiment_model.pkl'
if os.path.exists(model_file):
    # Load pre-trained model
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
else:
    messagebox.showinfo("Error", "Model file not found.")
    exit()

# Load dataset
dataset_file = 'your_dataset.csv'
reviews, sentiments = load_dataset(dataset_file)

# Preprocess the dataset
preprocessed_reviews = [preprocess_text(review) for review in reviews]

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(preprocessed_reviews)

# Train the model
model = LogisticRegression()
model.fit(X, sentiments)

# Function to analyze sentiment
def analyze_sentiment():
    user_input = text_entry.get("1.0", "end").strip()
    preprocessed_input = preprocess_text(user_input)
    input_tfidf = tfidf_vectorizer.transform([preprocessed_input])
    prediction = model.predict(input_tfidf)
    result_label.config(text="Predicted sentiment: " + prediction[0])

# Create tkinter window
window = tk.Tk()
window.title("Sentiment Analysis")
window.geometry("400x200")

# Create text entry widget
text_entry = tk.Text(window, height=5, width=50)
text_entry.pack(pady=10)

# Create analyze button
analyze_button = tk.Button(window, text="Analyze Sentiment", command=analyze_sentiment)
analyze_button.pack()

# Create result label
result_label = tk.Label(window, text="")
result_label.pack()

# Run tkinter event loop
window.mainloop()
