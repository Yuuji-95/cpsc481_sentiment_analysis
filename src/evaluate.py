import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load cleaned data
df = pd.read_csv("../data/imdb_clean.csv")
X = df["clean_review"]
y = df["sentiment"]

# 2. Same split as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Load model + vectorizer
with open("../data/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("../data/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# 4. Vectorize and predict
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

# 5. Print metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# 6. Confusion matrix figure
cm = confusion_matrix(y_test, y_pred, labels=["negative", "positive"])

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["negative", "positive"],
    yticklabels=["negative", "positive"],
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix – IMDB Sentiment")

# Save to figures folder
plt.tight_layout()
plt.savefig("../figures/confusion_matrix.png")
print("\nSaved confusion matrix to figures/confusion_matrix.png")
