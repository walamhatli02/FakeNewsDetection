from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
dataset = load_dataset("mrm8488/fake-news")

# Convert to pandas
df = pd.DataFrame(dataset["train"])

# Save to CSV
df.to_csv("data/raw/fake_news.csv", index=False)

print("Dataset downloaded and saved!")
print(df.head())

# Info
print(df.info())
print(df["label"].value_counts())

# Plot distribution
sns.countplot(x="label", data=df)
plt.title("Fake vs Real News Distribution")
plt.show()