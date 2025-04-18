# Week-2Task
Week 2task



---

Requirements

Make sure you have these Python packages installed:

pip install pandas scikit-learn


---

Dataset Format (news.csv)

Create a CSV file named news.csv in the same folder as your script with two columns:

text,label
"This is a genuine news article.",REAL
"This is a fake news article!",FAKE
...


---

Full Program (fake_news_classifier.py)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_and_train_model(filename):
    try:
        df = pd.read_csv(filename)
        df.dropna(inplace=True)
        df['label'] = df['label'].map({'REAL': 1, 'FAKE': 0})

        X = df['text']
        y = df['label']

        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        X_vec = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"Model trained with accuracy: {accuracy:.2f}")

        return model, vectorizer
    except Exception as e:
        print("Error loading or training model:", e)
        return None, None

def predict(text, model, vectorizer):
    if model and vectorizer:
        text_vec = vectorizer.transform([text])
        result = model.predict(text_vec)[0]
        return "REAL" if result == 1 else "FAKE"
    else:
        return "Model not loaded properly."

def main():
    model, vectorizer = load_and_train_model("news.csv")
    if not model:
        return

    while True:
        print("\nEnter news text (or type 'q' to quit):")
        user_input = input("> ").strip()
        if user_input.lower() == 'q':
            break
        result = predict(user_input, model, vectorizer)
        print(f"Prediction: {result}")

if __name__ == "__main__":
    main()


---

