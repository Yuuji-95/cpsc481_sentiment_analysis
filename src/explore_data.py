import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../data/imdb_reviews.csv")

print("\n---- FIRST 5 ROWS ----")
print(df.head())

print("\n---- DATASET INFO ----")
print(df.info())

print("\n---- CLASS DISTRIBUTION ----")
print(df['sentiment'].value_counts())

# Plot class distribution
df['sentiment'].value_counts().plot(kind='bar', title="Sentiment Distribution")
plt.show()
