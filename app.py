from flask import Flask, render_template, request, jsonify
import joblib
import os

app = Flask(__name__)

# Paths to the saved model and vectorizer
MODEL_PATH = os.path.join("models", "spam_classifier.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")

# Load model and vectorizer at startup
classifier = None
vectorizer = None

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    classifier = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    print("Warning: Model or Vectorizer not found. Please run train_model.py first.")

@app.route("/")
def home():
    """Renders the beautiful UI."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to predict if a message is spam or ham."""
    if classifier is None or vectorizer is None:
        return jsonify({"error": "Model not trained yet."}), 500

    data = request.json
    message = data.get("message", "")
    
    if not message.strip():
        return jsonify({"error": "Message is empty."}), 400

    # Vectorize the input message
    message_tfidf = vectorizer.transform([message])
    
    # Predict (1 for spam, 0 for ham)
    prediction = classifier.predict(message_tfidf)[0]
    
    # Get probability scores (optional, but good for confidence)
    probabilities = classifier.predict_proba(message_tfidf)[0]
    confidence = probabilities[prediction] * 100
    
    result = "Spam" if prediction == 1 else "Not Spam"
    
    return jsonify({
        "prediction": result,
        "confidence": f"{confidence:.2f}%"
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
