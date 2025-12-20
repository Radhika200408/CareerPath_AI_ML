from fastapi import FastAPI, UploadFile, File, Request
from ml.train import train_model
from ml.recommender import recommend_careers

app = FastAPI(title="CareerPath AI")


@app.get("/health")
def health():
    return {"status": "ok"}

from fastapi import status
from fastapi.responses import JSONResponse

@app.post("/upload-dataset")
async def upload_dataset(request: Request, file: UploadFile = File(None)):
    import os
    if not file or not file.filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "No file uploaded or filename missing."}
        )
    os.makedirs("data", exist_ok=True)
    path = f"data/{file.filename}"
    try:
        contents = await file.read()
        with open(path, "wb") as f:
            f.write(contents)
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Failed to save file: {e}"}
        )
    try:
        records = train_model(path)
    except ImportError as e:
        if "openpyxl" in str(e):
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Missing required dependency 'openpyxl'. Please install it with 'pip install openpyxl' and restart the server."
                }
            )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Failed to train model: {e}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Failed to train model: {e}"}
        )
    return {"message": "Model trained", "records": records}

import os

@app.post("/recommend")
def recommend(data: dict):
    skills = data.get("skills")
    if not skills:
        return {"error": "Missing required key: 'skills'"}, 400
    if not os.path.exists("models/vectorizer.pkl"):
        return {"error": "Model file 'models/vectorizer.pkl' not found. Please upload dataset or train the model first."}, 500
    try:
        return recommend_careers(skills).to_dict(orient="records")
    except Exception as e:
        return {"error": f"Failed to generate recommendations: {str(e)}"}, 500

