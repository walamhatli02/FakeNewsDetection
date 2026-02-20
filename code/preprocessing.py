import pandas as pd
import re
import nltk # type: ignore
from nltk.corpus import stopwords # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore

nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
df = pd.read_csv("../data/raw/fake_news.csv")
# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub(r"[^a-z\s]", "", text)  # remove special chars
    return text

df["clean_text"] = df["text"].apply(clean_text)

# Remove stopwords + lemmatization
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df["processed_text"] = df["clean_text"].apply(preprocess)

# Save cleaned dataset
df.to_csv("../data/processed/train_clean.csv", index=False)

print("Preprocessing done!")
print(df.head())