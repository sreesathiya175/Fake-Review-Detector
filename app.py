# app.py

from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load vectorizer and model
model_path = os.path.join(os.path.dirname(__file__), 'fake_review_model.pkl')
vectorizer, model = pickle.load(open(model_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    if review:
        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)[0]
        label = '🟢 Genuine Review' if prediction == 0 else '🔴 Fake Review'
        return render_template('index.html', review=review, prediction=label)
    else:
        return render_template('index.html', review="", prediction="⚠️ Please enter a review.")

if __name__ == '__main__':
    app.run(debug=True)
