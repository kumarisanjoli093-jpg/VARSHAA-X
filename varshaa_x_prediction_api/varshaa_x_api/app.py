from pathlib import Path
import pickle
from flask import Flask, request, jsonify

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent.parent / "varshaa_x_model_package" / "varshaa_x_model" / "varshaa_x_random_forest.pkl"

with MODEL_PATH.open("rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
FEATURES = bundle["features"]
THRESHOLD_MM_H = float(bundle.get("threshold_mm_h", 50.0))
LEAD_HOURS = float(bundle.get("lead_hours", 3))

app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/predict", methods=["OPTIONS"])
def predict_options():
    return jsonify({"status": "ok"})

def validate_features(data):
    missing = [name for name in FEATURES if name not in data]
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)}")

    values = []
    for name in FEATURES:
        try:
            value = float(data[name])
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be numeric")
        values.append(value)

    return values


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "VARSHAA-X Random Forest",
        "lead_hours": LEAD_HOURS,
        "threshold_mm_h": THRESHOLD_MM_H,
        "features": FEATURES,
    })


@app.post("/predict")
def predict():
    try:
        data = request.get_json(silent=True) or {}
        values = validate_features(data)

        probability = float(model.predict_proba([values])[0][1])

        # Prototype risk bands. The actual probability remains available.
        if probability >= 0.66:
            risk = "HIGH"
        elif probability >= 0.33:
            risk = "MODERATE"
        else:
            risk = "LOW"

        return jsonify({
            "probability": round(probability, 4),
            "probability_percent": round(probability * 100, 1),
            "risk": risk,
            "lead_hours": LEAD_HOURS,
            "threshold_mm_h": THRESHOLD_MM_H,
            "target": f"rainfall >= {THRESHOLD_MM_H:g} mm/h within next {LEAD_HOURS:g} hours",
        })

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "Prediction failed", "detail": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
