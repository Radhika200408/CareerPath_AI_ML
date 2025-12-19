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

    required_columns = ["Job Title", "Role", "Industry", "Key Skills"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["content"] = (
        df["Job Title"].astype(str) + " " +
        df["Role"].astype(str) + " " +
        df["Industry"].astype(str) + " " +
        df["Key Skills"].astype(str)
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(df["content"])

    pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
    pickle.dump(matrix, open("models/matrix.pkl", "wb"))
    df.to_pickle("models/careers.pkl")

    return len(df)
