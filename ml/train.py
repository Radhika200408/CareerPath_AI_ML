import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def train_model(csv_path):
    df = pd.read_csv(csv_path)

    # Ensure missing values are empty strings so concatenation works
    df = df.fillna("")

    df["content"] = (
        df["Job Title"].astype(str) + " " +
        df["Role"].astype(str) + " " +
        df["Industry"].astype(str) + " " +
        df["Required Skills"].astype(str)
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(df["content"])

    pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
    pickle.dump(matrix, open("models/matrix.pkl", "wb"))
    df.to_pickle("models/careers.pkl")

    return len(df)
