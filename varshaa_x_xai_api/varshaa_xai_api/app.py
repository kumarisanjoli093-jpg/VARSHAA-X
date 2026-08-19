from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Hide only the known sklearn persistence warning for the prototype API.
warnings.filterwarnings(
    "ignore",
    message="Trying to unpickle estimator .* from version 1\\.8\\.0 when using version 1\\.9\\.0",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR.parent.parent
    / "varshaa_x_model_package"
    / "varshaa_x_model"
    / "varshaa_x_random_forest.pkl"
)

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
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        values.append(value)

    return values


def model_explanation(values):
    """Model-native local explanation using Random Forest tree votes.

    For each feature, compare the average predicted probability after
    replacing that feature with its training-distribution median. This is a
    perturbation explanation: it answers which inputs move this particular
    prediction most, without pretending to be a calibrated causal effect.
    """
    x = pd.DataFrame([values], columns=FEATURES)
    base = float(model.predict_proba(x)[0, 1])

    # RandomForest does not retain the original training matrix, so use
    # representative physical medians from the model's training data captured
    # in the API defaults. These are deliberately conservative prototype
    # baselines and are labelled as perturbation effects.
    baseline = {
        "rain_current_max_mm_h": 0.0,
        "rain_current_mean_mm_h": 0.0,
        "cloud_top_pressure_hpa": 700.0,
        "cloud_top_temperature_k": 260.0,
        "effective_cloud_emissivity": 0.85,
    }

    effects = []
    for i, name in enumerate(FEATURES):
        altered = values.copy()
        altered[i] = baseline[name]
        altered_prob = float(model.predict_proba(
            pd.DataFrame([altered], columns=FEATURES)
        )[0, 1])
        delta = base - altered_prob
        effects.append({
            "feature": name,
            "value": float(values[i]),
            "baseline": float(baseline[name]),
            "probability_change": round(delta, 4),
            "direction": "increases risk" if delta > 0 else "decreases risk",
            "absolute_effect": abs(delta),
        })

    effects.sort(key=lambda z: z["absolute_effect"], reverse=True)

    labels = {
        "rain_current_max_mm_h": "Current maximum rainfall",
        "rain_current_mean_mm_h": "Current mean rainfall",
        "cloud_top_pressure_hpa": "Cloud-top pressure",
        "cloud_top_temperature_k": "Cloud-top temperature",
        "effective_cloud_emissivity": "Cloud emissivity",
    }

    top = []
    for e in effects[:3]:
        top.append({
            "feature": labels[e["feature"]],
            "value": e["value"],
            "direction": e["direction"],
            "probability_change": e["probability_change"],
        })

    return top, effects


@app.get("/")
def root():
    return jsonify({
        "service": "VARSHAA-X Prediction API",
        "status": "online",
        "endpoints": ["/health", "/predict"],
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "VARSHAA-X Random Forest",
        "lead_hours": LEAD_HOURS,
        "threshold_mm_h": THRESHOLD_MM_H,
        "features": FEATURES,
        "xai": "enabled",
    })


@app.post("/predict")
def predict():
    try:
        data = request.get_json(silent=True) or {}
        values = validate_features(data)

        probability = float(model.predict_proba([values])[0][1])

        if probability >= 0.66:
            risk = "HIGH"
        elif probability >= 0.33:
            risk = "MODERATE"
        else:
            risk = "LOW"

        explanations, detailed = model_explanation(values)

        return jsonify({
            "probability": round(probability, 4),
            "probability_percent": round(probability * 100, 1),
            "risk": risk,
            "lead_hours": LEAD_HOURS,
            "threshold_mm_h": THRESHOLD_MM_H,
            "target": f"rainfall >= {THRESHOLD_MM_H:g} mm/h within next {LEAD_HOURS:g} hours",
            "explanation": explanations,
            "explanation_method": "feature perturbation against prototype baselines",
            "detailed_explanation": detailed,
        })

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "Prediction failed", "detail": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
