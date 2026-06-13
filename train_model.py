import os
import zipfile
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Constants
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "smsspamcollection.zip")
DATA_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "spam_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

def download_and_extract_data():
    """Downloads the SMS Spam Collection dataset from UCI and extracts it."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    if not os.path.exists(DATA_FILE):
        print("Downloading dataset from UCI Machine Learning Repository...")
        response = requests.get(DATASET_URL)
        with open(ZIP_PATH, 'wb') as f:
            f.write(response.content)
            
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Dataset ready!")
    else:
        print("Dataset already exists.")

def load_data():
    """Loads the dataset into a Pandas DataFrame."""
    # The dataset is tab-separated with no headers. The columns are 'label' and 'message'
    df = pd.read_csv(DATA_FILE, sep='\t', header=None, names=['label', 'message'])
    
    # Convert labels to binary format: 'spam' -> 1, 'ham' -> 0
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    return df

def train_and_evaluate_model():
    """Trains the Naive Bayes model and prints evaluation metrics."""
    df = load_data()
    
    # 1. Train-Test Split
    # We split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2, random_state=42
    )
    
    # 2. Text Preprocessing and Vectorization (TF-IDF)
    # TfidfVectorizer converts raw text into a matrix of TF-IDF features.
    # It handles lowercasing, tokenization, and removing common English stop words.
    print("\nVectorizing text data using TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit the vectorizer on the training data and transform the text into numbers
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Transform the test data using the fitted vectorizer
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 3. Train the Model
    # We use Multinomial Naive Bayes, which is highly effective for text classification
    print("Training Naive Bayes classifier...")
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)
    
    # 4. Evaluate the Model
    print("\nEvaluating model on test data...")
    predictions = classifier.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Ham (0)', 'Spam (1)']))
    
    # 5. Save the Model and Vectorizer
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    joblib.dump(classifier, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")

if __name__ == "__main__":
    print("=== SMS Spam Classifier Training Pipeline ===")
    download_and_extract_data()
    train_and_evaluate_model()
    print("Pipeline completed successfully.")
