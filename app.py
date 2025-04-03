from flask import Flask, request, render_template
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the trained model and tokenizer
model = joblib.load("model.h5")
tokenizer = joblib.load("tokenizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Placeholder for authentication logic
        return render_template('input.html')
    return render_template('login.html')

@app.route('/submit_review', methods=['POST'])
def submit_review():
    review = request.form['review']
    prediction = predictive_system(review)
    return render_template('output.html', result=prediction)

def predictive_system(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequences = pad_sequences(sequences, maxlen=200)
    prediction = model.predict(padded_sequences)
    return "Positive Review" if prediction > 0.5 else "Negative Review"

if __name__ == '_main_':
    app.run(debug=True)