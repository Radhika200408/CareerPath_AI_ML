from fastapi import FastAPI, UploadFile, File
from ml.train import train_model
from ml.recommender import recommend_careers

app = FastAPI(title="CareerPath AI")


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload-dataset")
def upload_dataset(file: UploadFile = File(...)):
    path = f"data/{file.filename}"
    with open(path, "wb") as f:
        f.write(file.file.read())

    records = train_model(path)
    return {"message": "Model trained", "records": records}

@app.post("/recommend")
def recommend(data: dict):
    return recommend_careers(data["skills"]).to_dict(orient="records")
