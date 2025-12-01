import joblib
from flask import Flask, request, jsonify, render_template
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os

# Ensure NLTK data is available
try:
    stopwords.words('english')
    WordNetLemmatizer()
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load the trained model and TF-IDF vectorizer
mnb_model = joblib.load('mnb_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function (must be identical to training phase)
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = re.sub(r'\\d+', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json(force=True)
    message = data.get('message', '')

    if not message:
        return jsonify({"error": "'message' field is required"}), 400

    # Preprocess the input message
    cleaned_message = preprocess_text(message)

    # Transform the cleaned message using the fitted TF-IDF vectorizer
    message_tfidf = tfidf_vectorizer.transform([cleaned_message])

    # Make prediction
    prediction = mnb_model.predict(message_tfidf)[0]

    # Convert prediction to human-readable label
    label = 'spam' if prediction == 1 else 'ham'

    return jsonify({"prediction": label})

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
