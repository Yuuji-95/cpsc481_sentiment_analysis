import pickle

# Load model
with open("../data/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load TF-IDF vectorizer
with open("../data/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_sentiment(review_text):
    # Vectorize input
    review_vec = vectorizer.transform([review_text])

    # Predict
    prediction = model.predict(review_vec)[0]

    return prediction

# Demo
if __name__ == "__main__":
    while True:
        text = input("\nEnter a movie review (or type 'exit'): ")

        if text.lower() == "exit":
            break

        result = predict_sentiment(text)
        print(" Sentiment:", result)
