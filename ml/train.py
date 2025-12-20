import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def train_model(csv_path):
    try:
        print(f"Reading file: {csv_path}")
        if csv_path.lower().endswith('.xlsx'):
            df = pd.read_excel(csv_path)
        else:
            df = pd.read_csv(csv_path)
        print(f"Columns in file: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading file: {e}")
        raise

    # Ensure missing values are empty strings so concatenation works
    df = df.fillna("")

    # Use all columns except the index for content
    content_columns = [col for col in df.columns if col not in df.index.names]
    df["content"] = df[content_columns].astype(str).agg(" ".join, axis=1)

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(df["content"])

    pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
    pickle.dump(matrix, open("models/matrix.pkl", "wb"))
    df.to_pickle("models/careers.pkl")

    return len(df)
