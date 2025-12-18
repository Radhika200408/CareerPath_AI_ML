import pickle
from sklearn.metrics.pairwise import cosine_similarity

def recommend_careers(user_skills):
    vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
    matrix = pickle.load(open("models/matrix.pkl", "rb"))
    df = pickle.load(open("models/careers.pkl", "rb"))

    user_vector = vectorizer.transform([" ".join(user_skills)])
    similarity_scores = cosine_similarity(user_vector, matrix)[0]

    df["match_score"] = similarity_scores
    return df.sort_values(by="match_score", ascending=False).head(5)
