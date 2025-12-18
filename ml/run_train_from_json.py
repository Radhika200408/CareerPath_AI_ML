from pathlib import Path
import json
import pandas as pd
import sys



def main():
    root = Path(__file__).resolve().parent.parent
    # make project root importable so `ml` package can be imported
    sys.path.insert(0, str(root))

    from ml.train import train_model
    from ml.recommender import recommend_careers

    json_path = root / "data" / "careers.json"
    csv_path = root / "data" / "careers_converted.csv"

    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    rows = []
    for it in items:
        role = it.get("role", "")
        skills = it.get("skills", [])
        rows.append({
            "Job Title": "",
            "Role": role,
            "Industry": "",
            "Required Skills": " ".join(skills)
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    print(f"Wrote converted CSV to: {csv_path}")
    # ensure models directory exists
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    count = train_model(str(csv_path))
    print(f"Trained model on {count} records")

    # Test recommender
    sample = ["python", "sql"]
    print("Sample recommendation for skills:", sample)
    recs = recommend_careers(sample)
    print(recs.to_dict(orient="records"))


if __name__ == "__main__":
    main()
