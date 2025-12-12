import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("../data/imdb_reviews.csv")

# Initialize tools
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)        # remove HTML
    text = re.sub(r"[^a-zA-Z ]", "", text)  # remove punctuation & numbers
    text = text.lower()                     # lowercase
    text = text.split()                     # tokenize
    text = [ps.stem(word) for word in text if word not in stop_words]
    return " ".join(text)

print("Cleaning text... This may take ~30 seconds.")

df["clean_review"] = df["review"].apply(clean_text)

# Save cleaned dataset
df.to_csv("../data/imdb_clean.csv", index=False)

print("Done! Saved cleaned file to data/imdb_clean.csv")
