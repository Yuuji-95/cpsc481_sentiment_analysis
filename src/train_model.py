import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load preprocessed dataset
df = pd.read_csv("../data/imdb_clean.csv")

# Features & labels
X = df["clean_review"]
y = df["sentiment"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert text -> TF-IDF numerical vectors
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Evaluate model
preds = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))

# Save model + vectorizer for later use
with open("../data/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../data/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved!")
