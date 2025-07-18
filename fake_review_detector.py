import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Sample dataset (you can replace with real data)
data = {
    'review': [
        "Great product! Totally worth the money!",
        "Buy this now!!! Amazing!!!",
        "Not good. It broke in two days.",
        "Fake product. Don't buy.",
        "I love it!!! Will buy again!!",
        "Excellent quality. Happy with it.",
        "Worst product ever! Fake!",
        "Works as expected. Satisfied.",
        "Best thing ever! So cheap and awesome!",
        "Terrible experience. Never again."
    ],
    'label': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]  # 1 = fake, 0 = genuine
}

df = pd.DataFrame(data)

# Clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

df['cleaned'] = df['review'].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model and vectorizer
with open('fake_review_model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)

print("✅ Model saved as fake_review_model.pkl")
